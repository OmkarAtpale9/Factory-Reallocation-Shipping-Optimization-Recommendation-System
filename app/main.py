from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
import yaml

from app.components.filters import apply_filters, render_filters
from app.components.layout import kpi_row, set_page_config
from src.analysis import (
    compute_kpis,
    destination_bottlenecks,
    leaderboard_best_worst,
    monthly_trend,
    route_aggregation,
    ship_mode_comparison,
)
from src.data_preprocessing import load_csv, preprocess_shipments
from src.feature_engineering import add_features
from src.model import build_delay_model
from src.utils import load_settings, resolve_data_path, setup_logging
from src.viz import fig_destination_heatmap, fig_monthly_trend, fig_route_bar, fig_ship_mode_compare


logger = logging.getLogger(__name__)


@st.cache_data(show_spinner=False)
def _load_dataset_from_path(path: str) -> pd.DataFrame:
    return load_csv(path)


@st.cache_data(show_spinner=False)
def _load_dataset_from_upload(uploaded) -> pd.DataFrame:
    return pd.read_csv(uploaded)


def _get_schema(settings) -> dict:
    return dict(settings.raw.get("schema", {}))


def _norm_col(s: str) -> str:
    return str(s).strip().lower()


def _auto_match_column(df: pd.DataFrame, desired: str) -> Optional[str]:
    """
    Best-effort column resolver:
    - exact match
    - case/whitespace-insensitive match
    """
    if desired in df.columns:
        return desired
    desired_n = _norm_col(desired)
    mapping = {_norm_col(c): c for c in df.columns}
    return mapping.get(desired_n)


def _auto_match_any(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for cand in candidates:
        if not isinstance(cand, str) or not cand.strip():
            continue
        m = _auto_match_column(df, cand)
        if m:
            return m
    return None


def _dataset_signature(source_label: str, df: pd.DataFrame) -> str:
    cols = "|".join(sorted([_norm_col(c) for c in df.columns.astype(str).tolist()]))
    raw = f"{_norm_col(source_label)}::{cols}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _overrides_path() -> Path:
    return Path("config/schema_overrides.yaml")


def _load_schema_overrides() -> dict:
    p = _overrides_path()
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_schema_overrides(overrides: dict) -> None:
    p = _overrides_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(overrides, f, sort_keys=True, allow_unicode=True)


def _auto_schema(df: pd.DataFrame, base_schema: dict) -> dict:
    """
    Auto-detect schema using defaults + common aliases.
    """
    aliases = {
        "order_date": [base_schema.get("order_date", ""), "OrderDate", "Order_Date", "Order", "Order Placed Date"],
        "ship_date": [base_schema.get("ship_date", ""), "ShipDate", "Ship_Date", "Shipping Date", "Dispatch Date"],
        "factory": [
            base_schema.get("factory", ""),
            "Factory Name",
            "Plant",
            "Plant Name",
            "Origin",
            "Origin Location",
            "Warehouse",
            "Fulfillment Center",
            "Source",
        ],
        "ship_mode": [
            base_schema.get("ship_mode", ""),
            "Shipping Mode",
            "Mode",
            "ShipMethod",
            "Ship Method",
            "Transport Mode",
            "Carrier Mode",
        ],
        "state": [base_schema.get("state", ""), "State/Province", "Province", "State Name", "Destination State"],
        "region": [base_schema.get("region", ""), "Area", "Zone", "Territory", "Destination Region"],
        "sales": [base_schema.get("sales", ""), "Revenue", "Amount", "Order Amount", "Total Sales", "Sales Amount"],
        "units": [base_schema.get("units", ""), "Quantity", "Qty", "Order Quantity", "Units Sold"],
    }

    out = dict(base_schema)
    for key, cands in aliases.items():
        m = _auto_match_any(df, list(cands))
        if m:
            out[key] = m
    return out


def _render_schema_mapper(df: pd.DataFrame, schema_in: dict, *, source_label: str) -> dict:
    """
    Sidebar UI to map required logical fields to actual CSV columns.
    This prevents KeyError when users upload files with different headers.
    """
    cols = [str(c) for c in df.columns.tolist()]
    sig = _dataset_signature(source_label, df)
    overrides = _load_schema_overrides()
    remembered = overrides.get(sig, {}).get("schema")
    schema_in = dict(remembered) if isinstance(remembered, dict) and remembered else dict(schema_in)

    with st.sidebar.expander("Column mapping (auto + optional manual override)", expanded=False):
        st.caption("Auto-detected columns are preselected. Override only if needed.")
        remember_toggle = st.checkbox("Remember mapping for this dataset", value=bool(remembered), key="remember_schema")

        def pick(key: str, label: str, required: bool = True) -> Optional[str]:
            desired = str(schema_in.get(key, ""))
            default = _auto_match_column(df, desired) if desired else None
            options = ["(none)"] + cols
            idx = options.index(default) if default in options else 0
            choice = st.selectbox(label, options=options, index=idx, key=f"map_{key}")
            if choice == "(none)":
                if required:
                    st.warning(f"Missing required mapping: {label}")
                return None
            return str(choice)

        mapped = dict(schema_in)
        mapped["order_date"] = pick("order_date", "Order Date column", required=True)
        mapped["ship_date"] = pick("ship_date", "Ship Date column", required=True)
        mapped["factory"] = pick("factory", "Factory column", required=True)
        mapped["ship_mode"] = pick("ship_mode", "Ship Mode column", required=True)
        mapped["state"] = pick("state", "State column (optional)", required=False)
        mapped["region"] = pick("region", "Region column (optional)", required=False)
        mapped["sales"] = pick("sales", "Sales column (optional)", required=False)
        mapped["units"] = pick("units", "Units column (optional)", required=False)

        # Keep only valid strings
        mapped_clean = {k: v for k, v in mapped.items() if isinstance(v, str) and v}

        if remember_toggle:
            overrides[sig] = {
                "saved_at": datetime.utcnow().isoformat() + "Z",
                "source_label": source_label,
                "schema": mapped_clean,
            }
            _save_schema_overrides(overrides)
        else:
            if sig in overrides:
                overrides.pop(sig, None)
                _save_schema_overrides(overrides)

        return mapped_clean


def _pipeline(df_raw: pd.DataFrame, settings, delay_threshold_days: int, schema: dict) -> tuple[pd.DataFrame, dict]:
    proc_cfg = settings.raw.get("processing", {})
    feat_cfg = settings.raw.get("features", {})

    proc = preprocess_shipments(
        df_raw,
        schema=schema,
        drop_invalid_ship_before_order=bool(proc_cfg.get("drop_invalid_ship_before_order", True)),
        missing_strategy=str(proc_cfg.get("missing", {}).get("strategy", "drop_critical")),
        critical_columns=list(proc_cfg.get("missing", {}).get("critical_columns", [])),
    )

    fe = add_features(
        proc.df,
        schema=schema,
        delay_threshold_days=int(delay_threshold_days),
        route_separator=str(feat_cfg.get("route_separator", " → ")),
        route_destination_preference=str(feat_cfg.get("route_destination_preference", "StateThenRegion")),
    )

    meta = {
        "schema": schema,
        "lead_time_col": fe.lead_time_col,
        "route_col": fe.route_col,
        "delay_col": fe.delay_col,
        "order_date_col": schema["order_date"],
        "ship_mode_col": schema.get("ship_mode", "Ship Mode"),
        "dropped_invalid_records": proc.dropped_invalid_records,
    }
    return fe.df, meta


def _page_overview(df: pd.DataFrame, meta: dict) -> None:
    st.subheader("Overview Dashboard")
    kpis = compute_kpis(df, lead_time_col=meta["lead_time_col"], delay_col=meta["delay_col"])
    kpi_row(avg_lead_time=kpis.avg_lead_time, delay_frequency=kpis.delay_frequency, total_shipments=kpis.total_shipments)

    c1, c2 = st.columns([2, 1])
    with c1:
        trend = monthly_trend(df, lead_time_col=meta["lead_time_col"], delay_col=meta["delay_col"])
        st.plotly_chart(fig_monthly_trend(trend), use_container_width=True)
    with c2:
        st.caption("Data quality")
        st.write(
            {
                "rows_after_pipeline": int(len(df)),
                "dropped_invalid_ship_before_order": int(meta.get("dropped_invalid_records", 0)),
            }
        )

    st.divider()
    st.subheader("Sample (filtered) records")
    st.dataframe(df.head(200), use_container_width=True)


def _page_leaderboard(df: pd.DataFrame, meta: dict, settings) -> None:
    st.subheader("Route Efficiency Leaderboard")
    cfg = settings.raw.get("analysis", {})
    route_df = route_aggregation(
        df,
        route_col=meta["route_col"],
        lead_time_col=meta["lead_time_col"],
        delay_col=meta["delay_col"],
        min_volume=int(cfg.get("min_route_volume", 5)),
    )
    best, worst = leaderboard_best_worst(route_df, size=int(cfg.get("leaderboard_size", 10)))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top best routes**")
        st.dataframe(best, use_container_width=True)
    with c2:
        st.markdown("**Top worst routes**")
        st.dataframe(worst, use_container_width=True)

    st.divider()
    st.plotly_chart(fig_route_bar(route_df, route_col=meta["route_col"]), use_container_width=True)


def _page_geo(df: pd.DataFrame, meta: dict, settings) -> None:
    st.subheader("Geographic Analysis (Bottlenecks)")
    cfg = settings.raw.get("analysis", {})
    dest_df = destination_bottlenecks(
        df,
        destination_col="_Destination",
        lead_time_col=meta["lead_time_col"],
        delay_col=meta["delay_col"],
        min_volume=int(cfg.get("min_route_volume", 5)),
    )
    st.dataframe(dest_df, use_container_width=True)
    st.plotly_chart(fig_destination_heatmap(dest_df, destination_col="_Destination", value_col="AvgLeadTime"), use_container_width=True)


def _page_ship_mode(df: pd.DataFrame, meta: dict) -> None:
    st.subheader("Ship Mode Comparison")
    mode_df = ship_mode_comparison(
        df,
        ship_mode_col=meta["ship_mode_col"],
        lead_time_col=meta["lead_time_col"],
        delay_col=meta["delay_col"],
    )
    st.dataframe(mode_df, use_container_width=True)
    st.plotly_chart(fig_ship_mode_compare(mode_df, ship_mode_col=meta["ship_mode_col"]), use_container_width=True)


def _page_drilldown(df: pd.DataFrame, meta: dict, settings) -> None:
    st.subheader("Drill-down (Order-level)")
    st.caption("Use the sidebar filters to narrow down. Then sort/search in the table below.")

    show_cols = [c for c in [
        meta["schema"].get("order_date"),
        meta["schema"].get("ship_date"),
        meta["schema"].get("factory"),
        "_Destination",
        meta["schema"].get("ship_mode"),
        meta["schema"].get("sales"),
        meta["schema"].get("units"),
        meta["lead_time_col"],
        meta["delay_col"],
        meta["route_col"],
    ] if c and c in df.columns]

    st.dataframe(df[show_cols].sort_values(meta["lead_time_col"], ascending=False).head(2000), use_container_width=True)

    if bool(settings.raw.get("ml", {}).get("enabled", True)):
        st.divider()
        st.subheader("Delay Prediction (RandomForest)")
        with st.expander("Train/evaluate model on filtered data", expanded=False):
            if len(df) < 50:
                st.warning("Not enough rows to train a meaningful model. Try widening your filters.")
                return
            ml_cfg = settings.raw.get("ml", {})
            rf_params = dict(ml_cfg.get("random_forest", {}))
            res = build_delay_model(
                df,
                schema=meta["schema"],
                delay_col=meta["delay_col"],
                test_size=float(ml_cfg.get("test_size", 0.2)),
                random_state=int(ml_cfg.get("random_state", 42)),
                rf_params=rf_params,
            )
            st.write({"accuracy": res.accuracy, "features_used": res.feature_columns})
            cm = res.confusion_matrix
            cm_df = pd.DataFrame(cm, index=["Actual:0", "Actual:1"], columns=["Pred:0", "Pred:1"])
            st.dataframe(cm_df, use_container_width=False)


def main() -> None:
    settings = load_settings()
    log_cfg = settings.raw.get("logging", {})
    setup_logging(level=str(log_cfg.get("level", "INFO")), fmt=str(log_cfg.get("format", None)))

    set_page_config(settings.app_name)
    st.title(settings.app_name)

    st.sidebar.subheader("Data source")
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    data_path = resolve_data_path(None, settings.default_data_path)
    st.sidebar.caption(f"Default path: `{data_path.as_posix()}`")

    if uploaded is not None:
        df_raw = _load_dataset_from_upload(uploaded)
        source_label = f"Upload: {uploaded.name}"
    else:
        if not Path(data_path).exists():
            st.warning(
                "No dataset found. Upload a CSV from the sidebar or place one at the default path shown in the sidebar."
            )
            st.stop()
        df_raw = _load_dataset_from_path(str(data_path))
        source_label = str(data_path)

    st.caption(f"Source: `{source_label}` | Rows: {len(df_raw):,}")

    base_schema = _get_schema(settings)
    auto = _auto_schema(df_raw, base_schema)
    schema = _render_schema_mapper(df_raw, auto, source_label=source_label)
    ship_mode_col = schema.get("ship_mode", base_schema.get("ship_mode", "Ship Mode"))

    if "order_date" not in schema or "ship_date" not in schema or "factory" not in schema or "ship_mode" not in schema:
        st.error("Please map the required columns in the sidebar (Order Date, Ship Date, Factory, Ship Mode).")
        st.stop()

    filters = render_filters(
        df_raw,
        order_date_col=schema["order_date"],
        ship_mode_col=ship_mode_col,
        delay_threshold_default=int(settings.raw.get("features", {}).get("delay_threshold_days_default", 7)),
    )

    # Run pipeline using the selected threshold, then apply additional filters to the engineered dataset.
    df_feat, meta = _pipeline(df_raw, settings, delay_threshold_days=filters.delay_threshold_days, schema=schema)
    df_filtered = apply_filters(
        df_feat,
        filters,
        order_date_col=meta["order_date_col"],
        ship_mode_col=meta["ship_mode_col"],
    )

    st.sidebar.divider()
    page = st.sidebar.radio(
        "Pages",
        options=[
            "Overview Dashboard",
            "Route Efficiency Leaderboard",
            "Geographic Analysis (heatmap)",
            "Ship Mode Comparison",
            "Drill-down (order-level data)",
        ],
    )

    if page == "Overview Dashboard":
        _page_overview(df_filtered, meta)
    elif page == "Route Efficiency Leaderboard":
        _page_leaderboard(df_filtered, meta, settings)
    elif page == "Geographic Analysis (heatmap)":
        _page_geo(df_filtered, meta, settings)
    elif page == "Ship Mode Comparison":
        _page_ship_mode(df_filtered, meta)
    else:
        _page_drilldown(df_filtered, meta, settings)


if __name__ == "__main__":
    main()

