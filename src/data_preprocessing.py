from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessResult:
    df: pd.DataFrame
    dropped_invalid_records: int


def load_csv(path: str, encoding: Optional[str] = None) -> pd.DataFrame:
    logger.info("Loading CSV: %s", path)
    return pd.read_csv(path, encoding=encoding) if encoding else pd.read_csv(path)


def _to_datetime_inplace(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=False)


def preprocess_shipments(
    df: pd.DataFrame,
    schema: Dict[str, str],
    *,
    drop_invalid_ship_before_order: bool = True,
    missing_strategy: str = "drop_critical",
    critical_columns: Optional[list[str]] = None,
) -> PreprocessResult:
    """
    Preprocess shipments DataFrame.

    - Parse dates
    - Remove invalid records (Ship Date < Order Date)
    - Handle missing values
    """
    df = df.copy()

    order_date = schema["order_date"]
    ship_date = schema["ship_date"]

    _to_datetime_inplace(df, order_date)
    _to_datetime_inplace(df, ship_date)

    dropped_invalid = 0
    if drop_invalid_ship_before_order and order_date in df.columns and ship_date in df.columns:
        invalid_mask = df[ship_date].notna() & df[order_date].notna() & (df[ship_date] < df[order_date])
        dropped_invalid = int(invalid_mask.sum())
        if dropped_invalid:
            logger.warning("Dropping %s invalid records where Ship Date < Order Date", dropped_invalid)
            df = df.loc[~invalid_mask].copy()

    if missing_strategy not in {"drop_critical", "fill"}:
        raise ValueError(f"Unsupported missing strategy: {missing_strategy}")

    critical_columns = critical_columns or []
    # Only drop critical columns that actually exist in df (schema may be customized).
    crit_existing = [c for c in critical_columns if c in df.columns]
    if missing_strategy == "drop_critical" and crit_existing:
        before = len(df)
        df = df.dropna(subset=crit_existing).copy()
        after = len(df)
        if before != after:
            logger.info("Dropped %s rows due to missing critical fields", before - after)

    if missing_strategy == "fill":
        # Conservative: fill numeric with median, categorical with mode, keep date NaT as-is.
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                med = df[col].median()
                df[col] = df[col].fillna(med)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                continue
            else:
                mode = df[col].mode(dropna=True)
                fill_val: Any = mode.iloc[0] if not mode.empty else "Unknown"
                df[col] = df[col].fillna(fill_val)

    return PreprocessResult(df=df, dropped_invalid_records=dropped_invalid)

