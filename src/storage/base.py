from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class DataQuery:
    start_date: str | None = None
    end_date: str | None = None
    destination: str | None = None
    ship_mode: str | None = None


class ShipmentRepository(Protocol):
    def fetch_shipments(self, query: DataQuery | None = None) -> pd.DataFrame: ...

