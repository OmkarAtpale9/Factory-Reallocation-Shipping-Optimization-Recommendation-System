from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .utils import normalize_min_max


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KPIResult:
    avg_lead_time: float
    delay_frequency: float
    total_shipments: int


def compute_kpis(df: pd.DataFrame, *, lead_time_col: str, delay_col: str) -> KPIResult:
    total = int(len(df))
    if total == 0:
        return KPIResult(avg_lead_time=0.0, delay_frequency=0.0, total_shipments=0)
    return KPIResult(
        avg_lead_time=float(df[lead_time_col].mean()),
        delay_frequency=float(df[delay_col].mean()),
        total_shipments=total,
    )


def route_aggregation(
    df: pd.DataFrame,
    *,
    route_col: str,
    lead_time_col: str,
    delay_col: str,
    min_volume: int = 5,
) -> pd.DataFrame:
    g = df.groupby(route_col, dropna=False)
    out = g.agg(
        Volume=(route_col, "size"),
        AvgLeadTime=(lead_time_col, "mean"),
        StdLeadTime=(lead_time_col, "std"),
        DelayRate=(delay_col, "mean"),
    ).reset_index()
    out["StdLeadTime"] = out["StdLeadTime"].fillna(0.0)
    out = out.loc[out["Volume"] >= int(min_volume)].copy()

    # Efficiency score: lower lead time and delay rate => higher score
    # Normalize both and invert.
    out["_LeadNorm"] = normalize_min_max(out["AvgLeadTime"])
    out["_DelayNorm"] = normalize_min_max(out["DelayRate"])
    out["RouteEfficiencyScore"] = 1.0 - (0.7 * out["_LeadNorm"] + 0.3 * out["_DelayNorm"])
    out["RouteEfficiencyScore"] = out["RouteEfficiencyScore"].clip(0.0, 1.0)
    out = out.drop(columns=["_LeadNorm", "_DelayNorm"])
    return out.sort_values(["RouteEfficiencyScore", "Volume"], ascending=[False, False])


def leaderboard_best_worst(
    route_df: pd.DataFrame,
    *,
    size: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    size = int(size)
    best = route_df.head(size).copy()
    worst = route_df.tail(size).sort_values("RouteEfficiencyScore", ascending=True).copy()
    return best, worst


def destination_bottlenecks(
    df: pd.DataFrame,
    *,
    destination_col: str = "_Destination",
    lead_time_col: str,
    delay_col: str,
    min_volume: int = 5,
) -> pd.DataFrame:
    g = df.groupby(destination_col, dropna=False)
    out = g.agg(
        Volume=(destination_col, "size"),
        AvgLeadTime=(lead_time_col, "mean"),
        StdLeadTime=(lead_time_col, "std"),
        DelayRate=(delay_col, "mean"),
    ).reset_index()
    out["StdLeadTime"] = out["StdLeadTime"].fillna(0.0)
    out = out.loc[out["Volume"] >= int(min_volume)].copy()
    out = out.sort_values(["AvgLeadTime", "DelayRate", "Volume"], ascending=[False, False, False])
    return out


def ship_mode_comparison(
    df: pd.DataFrame,
    *,
    ship_mode_col: str,
    lead_time_col: str,
    delay_col: str,
) -> pd.DataFrame:
    g = df.groupby(ship_mode_col, dropna=False)
    out = g.agg(
        Volume=(ship_mode_col, "size"),
        AvgLeadTime=(lead_time_col, "mean"),
        StdLeadTime=(lead_time_col, "std"),
        DelayRate=(delay_col, "mean"),
    ).reset_index()
    out["StdLeadTime"] = out["StdLeadTime"].fillna(0.0)
    return out.sort_values(["AvgLeadTime", "DelayRate"], ascending=[True, True])


def monthly_trend(df: pd.DataFrame, *, month_col: str = "_OrderMonth", lead_time_col: str, delay_col: str) -> pd.DataFrame:
    if month_col not in df.columns:
        raise KeyError(f"Month column '{month_col}' not found (expected from feature engineering)")
    g = df.groupby(month_col)
    out = g.agg(
        Shipments=(month_col, "size"),
        AvgLeadTime=(lead_time_col, "mean"),
        DelayRate=(delay_col, "mean"),
    ).reset_index().sort_values(month_col)
    return out

