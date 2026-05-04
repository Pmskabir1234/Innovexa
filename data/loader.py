"""Load persisted sensor data and simulate streaming batches."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import pandas as pd


def load_sensor_csv(path: str | Path, parse_dates: bool = True) -> pd.DataFrame:
    """Load sensor history from CSV into a typed dataframe."""

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Sensor file not found: {p}")
    df = pd.read_csv(p)
    if parse_dates and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def stream_batches_from_dataframe(
    df: pd.DataFrame,
    window_size: int,
    step: int = 1,
    max_batches: Optional[int] = None,
) -> Iterator[pd.DataFrame]:
    """Slice a dataframe into overlapping windows for pseudo-streaming replay."""

    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if step <= 0:
        raise ValueError("step must be positive")
    start = window_size
    batches = 0
    while start <= len(df):
        yield df.iloc[start - window_size : start].copy()
        start += step
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
