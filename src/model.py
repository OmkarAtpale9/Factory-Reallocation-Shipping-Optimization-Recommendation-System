from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MLResult:
    model: Pipeline
    accuracy: float
    confusion_matrix: np.ndarray
    feature_columns: list[str]
    target_column: str


def build_delay_model(
    df: pd.DataFrame,
    *,
    schema: Dict[str, str],
    delay_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    rf_params: Optional[Dict] = None,
) -> MLResult:
    """
    Train a delay classifier.

    Required (if present in data): Sales, Units, Ship Mode, Region.
    If a configured column is missing, it's silently excluded from features.
    """
    rf_params = rf_params or {}

    candidate_features = [
        schema.get("sales", "Sales"),
        schema.get("units", "Units"),
        schema.get("ship_mode", "Ship Mode"),
        schema.get("region", "Region"),
    ]
    feature_cols = [c for c in candidate_features if c in df.columns]
    if not feature_cols:
        raise ValueError("No usable feature columns found for ML. Check schema in config/settings.yaml.")
    if delay_col not in df.columns:
        raise KeyError(f"Delay column '{delay_col}' not found.")

    X = df[feature_cols].copy()
    y = df[delay_col].astype(int).copy()

    # Identify types
    numeric = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in feature_cols if c not in numeric]

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    clf = RandomForestClassifier(
        random_state=random_state,
        **rf_params,
    )

    pipe = Pipeline(steps=[("preprocess", pre), ("model", clf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=y if y.nunique() > 1 else None,
    )

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    acc = float(accuracy_score(y_test, preds))
    cm = confusion_matrix(y_test, preds, labels=[0, 1])

    logger.info("Model trained. Accuracy=%.4f", acc)
    return MLResult(
        model=pipe,
        accuracy=acc,
        confusion_matrix=cm,
        feature_columns=feature_cols,
        target_column=delay_col,
    )


def save_model(model: Pipeline, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved model to %s", path)


def load_model(path: str | Path) -> Pipeline:
    path = Path(path)
    return joblib.load(path)

