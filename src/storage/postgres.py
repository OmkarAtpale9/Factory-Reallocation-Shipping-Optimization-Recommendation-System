from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .base import DataQuery, ShipmentRepository


@dataclass(frozen=True)
class PostgresConfig:
    dsn: str
    table_name: str = "shipments"


class PostgresShipmentRepository(ShipmentRepository):
    """
    Placeholder for PostgreSQL integration.

    Intentionally does not include a runtime dependency on SQLAlchemy/psycopg in
    this template. Add your preferred driver and implement fetch logic.
    """

    def __init__(self, cfg: PostgresConfig):
        self._cfg = cfg

    def fetch_shipments(self, query: Optional[DataQuery] = None) -> pd.DataFrame:
        raise NotImplementedError(
            "PostgreSQL integration is a placeholder. "
            "Add SQLAlchemy/psycopg and implement this repository."
        )

