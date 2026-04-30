from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import FastAPI

from src.utils import load_settings, setup_logging


logger = logging.getLogger(__name__)

settings = load_settings()
log_cfg = settings.raw.get("logging", {})
setup_logging(level=str(log_cfg.get("level", "INFO")), fmt=str(log_cfg.get("format", None)))

app = FastAPI(title=settings.app_name)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "app": settings.app_name}


@app.get("/config")
def config() -> Dict[str, Any]:
    # Safe subset; avoid secrets (none are expected in this template).
    return {"app": settings.raw.get("app", {}), "schema": settings.raw.get("schema", {})}

