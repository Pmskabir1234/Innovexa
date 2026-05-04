"""Isolation Forest wrapper for multivariate anomaly scoring."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    """Train and score multivariate feature vectors for anomalies."""

    def __init__(self, contamination: float, n_estimators: int, random_state: int = 42) -> None:
        """Configure the underlying isolation forest."""

        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._model: Optional[IsolationForest] = None

    def fit(self, features: np.ndarray) -> "AnomalyDetector":
        """Fit the detector on inlier-heavy training features."""

        self._model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(features)
        return self

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        """Return anomaly scores; lower values indicate stronger anomalies."""

        if self._model is None:
            raise RuntimeError("AnomalyDetector must be fitted before scoring")
        return self._model.decision_function(features)

    def predict_label(self, features: np.ndarray) -> np.ndarray:
        """Return model labels (-1 anomaly, 1 normal)."""

        if self._model is None:
            raise RuntimeError("AnomalyDetector must be fitted before predict")
        return self._model.predict(features)
