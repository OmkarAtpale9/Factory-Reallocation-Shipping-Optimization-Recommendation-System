from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    raw: Dict[str, Any]

    @property
    def app_name(self) -> str:
        return str(self.raw.get("app", {}).get("name", "Shipping Analytics"))

    @property
    def default_data_path(self) -> str:
        return str(self.raw.get("app", {}).get("default_data_path", "data/shipments.csv"))


def load_yaml(path: str | os.PathLike) -> Dict[str, Any]:
    path = str(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_settings(settings_path: str | os.PathLike = "config/settings.yaml") -> Settings:
    raw = load_yaml(settings_path)
    return Settings(raw=raw)


def setup_logging(level: str = "INFO", fmt: str | None = None) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl, format=fmt)


def project_root() -> Path:
    # Assumes this file lives at <root>/src/utils.py
    return Path(__file__).resolve().parents[1]


def resolve_data_path(candidate: Optional[str], default_path: str) -> Path:
    root = project_root()
    if candidate:
        p = Path(candidate)
        return p if p.is_absolute() else (root / p)
    return (root / default_path)


def normalize_min_max(series) -> Any:
    s = series.astype(float)
    min_v = float(s.min())
    max_v = float(s.max())
    if max_v - min_v == 0:
        return s * 0.0
    return (s - min_v) / (max_v - min_v)

