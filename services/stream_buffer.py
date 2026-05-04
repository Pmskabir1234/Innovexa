"""Rolling buffer of sensor rows for windowed real-time inference."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List

import pandas as pd


class RollingSensorBuffer:
    """Fixed-length FIFO of sensor dicts aligned to the assistant window size."""

    def __init__(self, window_size: int) -> None:
        """Allocate a deque capped at ``window_size`` rows."""

        if window_size < 1:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self._rows: Deque[Dict[str, Any]] = deque(maxlen=window_size)

    def append(self, row: Dict[str, Any]) -> None:
        """Ingest one sensor record (mutates internal state)."""

        self._rows.append(dict(row))

    def is_ready(self) -> bool:
        """Return True when the buffer holds a full context window."""

        return len(self._rows) >= self.window_size

    def to_dataframe(self) -> pd.DataFrame:
        """Materialize the current window as a dataframe for ML pipelines."""

        if not self._rows:
            return pd.DataFrame()
        return pd.DataFrame(list(self._rows))

    def __len__(self) -> int:
        """Return how many rows are currently buffered."""

        return len(self._rows)
