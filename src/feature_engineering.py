from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureResult:
    df: pd.DataFrame
    lead_time_col: str
    route_col: str
    delay_col: str


def _pick_destination(
    df: pd.DataFrame,
    schema: Dict[str, str],
    preference: str,
) -> str:
    state = schema.get("state", "State")
    region = schema.get("region", "Region")

    state_exists = state in df.columns
    region_exists = region in df.columns

    if preference == "StateOnly":
        if not state_exists:
            raise KeyError(f"Configured State column '{state}' not found")
        return state
    if preference == "RegionOnly":
        if not region_exists:
            raise KeyError(f"Configured Region column '{region}' not found")
        return region

    if preference == "RegionThenState":
        if region_exists:
            return region
        if state_exists:
            return state
        raise KeyError("Neither State nor Region column found")

    # Default: StateThenRegion
    if state_exists:
        return state
    if region_exists:
        return region
    raise KeyError("Neither State nor Region column found")


def add_features(
    df: pd.DataFrame,
    schema: Dict[str, str],
    *,
    delay_threshold_days: int,
    route_separator: str = " → ",
    route_destination_preference: str = "StateThenRegion",
    lead_time_col: str = "Lead Time (Days)",
    route_col: str = "Route",
    delay_col: str = "Is Delayed",
) -> FeatureResult:
    """
    Adds:
    - Lead Time (Days)
    - Route (Factory -> destination)
    - Is Delayed (LeadTime > threshold)
    """
    df = df.copy()

    order_date = schema["order_date"]
    ship_date = schema["ship_date"]
    factory = schema["factory"]
    destination_col = _pick_destination(df, schema, route_destination_preference)

    if order_date not in df.columns or ship_date not in df.columns:
        raise KeyError("Order Date and Ship Date columns must exist to compute lead time")

    if factory not in df.columns:
        raise KeyError(f"Factory column '{factory}' not found")

    lead_days = (df[ship_date] - df[order_date]).dt.total_seconds() / 86400.0
    df[lead_time_col] = lead_days

    # Keep only valid non-negative lead times (preprocessing should have removed ship<order,
    # but this also removes NaT-derived NaNs).
    df = df.loc[df[lead_time_col].notna() & (df[lead_time_col] >= 0)].copy()

    df[route_col] = df[factory].astype(str).str.strip() + route_separator + df[destination_col].astype(str).str.strip()

    df[delay_col] = (df[lead_time_col] > float(delay_threshold_days)).astype(int)

    # Optional helper columns for easier filtering/aggregation
    df["_Destination"] = df[destination_col]

    # Create a month key for trends
    df["_OrderMonth"] = pd.to_datetime(df[order_date]).dt.to_period("M").dt.to_timestamp()

    logger.info(
        "Feature engineering complete: %s rows, delay_threshold_days=%s",
        len(df),
        delay_threshold_days,
    )

    return FeatureResult(df=df, lead_time_col=lead_time_col, route_col=route_col, delay_col=delay_col)

