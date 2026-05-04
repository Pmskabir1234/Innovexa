"""Feature engineering for multivariate sensor windows."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


class FeaturePipeline:
    """Compute fixed-length feature vectors from rolling sensor context."""

    def __init__(self, sensor_columns: List[str]) -> None:
        """Store the ordered sensor columns used for training and inference."""

        self.sensor_columns = list(sensor_columns)

    def feature_names(self) -> List[str]:
        """Return human-readable feature names aligned to transform output."""

        names: List[str] = []
        for col in self.sensor_columns:
            for stat in ("mean", "std", "min", "max", "last"):
                names.append(f"{col}_{stat}")
        return names

    def transform_window(self, df: pd.DataFrame) -> np.ndarray:
        """Aggregate a sensor window into a single feature vector."""

        row: List[float] = []
        for col in self.sensor_columns:
            series = df[col].astype(float)
            row.extend(
                [
                    float(series.mean()),
                    float(series.std(ddof=0)),
                    float(series.min()),
                    float(series.max()),
                    float(series.iloc[-1]),
                ]
            )
        return np.array(row, dtype=float).reshape(1, -1)
