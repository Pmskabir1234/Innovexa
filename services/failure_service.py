"""Failure risk classifier trained on engineered features."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class FailurePredictor:
    """Estimate near-term failure probability from feature vectors."""

    def __init__(self, random_state: int = 42, n_estimators: int = 300) -> None:
        """Configure the ensemble classifier."""

        self.random_state = random_state
        self.n_estimators = n_estimators
        self._model: Optional[RandomForestClassifier] = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "FailurePredictor":
        """Fit the classifier using supervised failure labels."""

        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
        self._model.fit(features, labels)
        return self

    def predict_proba_failure(self, features: np.ndarray) -> np.ndarray:
        """Return probability of the positive (failure) class per row."""

        if self._model is None:
            raise RuntimeError("FailurePredictor must be fitted first")
        probs = self._model.predict_proba(features)
        classes = list(self._model.classes_)
        if len(classes) < 2:
            return np.zeros(features.shape[0], dtype=float)
        pos_col = int(np.where(np.array(classes) == 1)[0][0])
        return probs[:, pos_col]

    def feature_importances(self) -> np.ndarray:
        """Return normalized feature importances from the forest."""

        if self._model is None:
            raise RuntimeError("FailurePredictor must be fitted first")
        return self._model.feature_importances_
