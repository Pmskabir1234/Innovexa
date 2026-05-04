"""Lightweight explanations for anomaly and failure models."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .failure_service import FailurePredictor
from .features import FeaturePipeline


def explain_failure_importances(
    predictor: FailurePredictor,
    feature_pipeline: FeaturePipeline,
    top_k: int = 6,
) -> List[Tuple[str, float]]:
    """Return the strongest random forest feature contributions."""

    names = feature_pipeline.feature_names()
    scores = predictor.feature_importances()
    order = np.argsort(scores)[::-1][:top_k]
    return [(names[i], float(scores[i])) for i in order]


def explain_anomaly_drivers(
    feature_vector: np.ndarray,
    reference_mean: np.ndarray,
    reference_std: np.ndarray,
    feature_pipeline: FeaturePipeline,
    top_k: int = 5,
) -> List[Dict[str, float]]:
    """Highlight features with largest z-score deviation from training statistics."""

    names = feature_pipeline.feature_names()
    vec = feature_vector.reshape(-1)
    eps = 1e-6
    z = np.abs((vec - reference_mean) / (reference_std + eps))
    order = np.argsort(z)[::-1][:top_k]
    return [
        {
            "feature": names[i],
            "z_score": float(z[i]),
            "value": float(vec[i]),
        }
        for i in order
    ]
