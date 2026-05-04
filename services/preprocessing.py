"""Sensor window validation and light cleaning."""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def validate_sensor_window(df: pd.DataFrame, required_columns: Sequence[str]) -> pd.DataFrame:
    """Ensure required columns exist and numeric sensor columns contain finite values."""

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required sensor columns: {missing}")
    if df.empty:
        raise ValueError("Sensor window is empty")
    subset = df[list(required_columns)].apply(pd.to_numeric, errors="coerce")
    if subset.isna().any().any():
        subset = subset.ffill().bfill()
    if subset.isna().any().any():
        raise ValueError("Unable to coerce sensor readings to numeric values")
    out = df.copy()
    for col in required_columns:
        out[col] = subset[col]
    return out


def enforce_minimum_window(df: pd.DataFrame, min_rows: int) -> None:
    """Raise if the window is shorter than the minimum context length."""

    if len(df) < min_rows:
        raise ValueError(f"Need at least {min_rows} samples, received {len(df)}")
