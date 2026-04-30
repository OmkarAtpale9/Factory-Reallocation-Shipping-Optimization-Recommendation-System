from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class FilterState:
    date_start: Optional[pd.Timestamp]
    date_end: Optional[pd.Timestamp]
    destinations: list[str]
    ship_modes: list[str]
    delay_threshold_days: int


def render_filters(
    df: pd.DataFrame,
    *,
    order_date_col: str,
    destination_col: str = "_Destination",
    ship_mode_col: str,
    delay_threshold_default: int,
) -> FilterState:
    st.sidebar.subheader("Filters")

    ddf = df.copy()
    ddf[order_date_col] = pd.to_datetime(ddf[order_date_col], errors="coerce")
    min_d = ddf[order_date_col].min()
    max_d = ddf[order_date_col].max()

    if pd.isna(min_d) or pd.isna(max_d):
        min_d, max_d = None, None

    if min_d is not None and max_d is not None:
        start, end = st.sidebar.date_input(
            "Order date range",
            value=(min_d.date(), max_d.date()),
            min_value=min_d.date(),
            max_value=max_d.date(),
        )
        date_start = pd.Timestamp(start)
        date_end = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    else:
        st.sidebar.caption("Date filters unavailable (missing/invalid dates).")
        date_start, date_end = None, None

    dest_values: Sequence[str] = (
        sorted([str(x) for x in ddf[destination_col].dropna().unique().tolist()])
        if destination_col in ddf.columns
        else []
    )
    selected_dest = st.sidebar.multiselect("State / Region", options=dest_values, default=[])

    mode_values: Sequence[str] = (
        sorted([str(x) for x in ddf[ship_mode_col].dropna().unique().tolist()])
        if ship_mode_col in ddf.columns
        else []
    )
    selected_modes = st.sidebar.multiselect("Ship mode", options=mode_values, default=[])

    threshold = st.sidebar.slider(
        "Delay threshold (days)",
        min_value=1,
        max_value=60,
        value=int(delay_threshold_default),
        step=1,
    )

    return FilterState(
        date_start=date_start,
        date_end=date_end,
        destinations=selected_dest,
        ship_modes=selected_modes,
        delay_threshold_days=int(threshold),
    )


def apply_filters(
    df: pd.DataFrame,
    filters: FilterState,
    *,
    order_date_col: str,
    destination_col: str = "_Destination",
    ship_mode_col: str,
) -> pd.DataFrame:
    out = df.copy()
    out[order_date_col] = pd.to_datetime(out[order_date_col], errors="coerce")

    if filters.date_start is not None and filters.date_end is not None:
        out = out.loc[out[order_date_col].notna()].copy()
        out = out.loc[(out[order_date_col] >= filters.date_start) & (out[order_date_col] <= filters.date_end)].copy()

    if filters.destinations and destination_col in out.columns:
        out = out.loc[out[destination_col].astype(str).isin(filters.destinations)].copy()

    if filters.ship_modes and ship_mode_col in out.columns:
        out = out.loc[out[ship_mode_col].astype(str).isin(filters.ship_modes)].copy()

    return out

