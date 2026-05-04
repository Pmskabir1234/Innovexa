"""End-to-end orchestration for predictions, decisions, and explanations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from data.synthetic import SENSOR_COLUMNS, IndustrialSensorSimulator, SimulationParams
from utils.config import Settings, ensure_runtime_dirs, get_settings

from .anomaly_service import AnomalyDetector
from .decision_engine import DecisionEngine, DecisionResult
from .decision_log import log_decision_record
from .explainability import explain_anomaly_drivers, explain_failure_importances
from .failure_service import FailurePredictor
from .features import FeaturePipeline
from .preprocessing import enforce_minimum_window, validate_sensor_window
from .rca import RootCauseAnalyzer, RootCauseInsight


@dataclass
class PredictionBundle:
    """Model outputs for a single sensor window."""

    failure_probability: float
    anomaly_score: float
    anomaly_flag: bool
    isolation_label: int


class TechnicalAssistant:
    """Production-style facade tying data, ML, RCA, and decisions together."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Wire subcomponents using centralized settings."""

        self.settings = ensure_runtime_dirs(settings or get_settings())
        self.sensor_columns = list(SENSOR_COLUMNS)
        self.window_size = 64
        self.feature_pipeline = FeaturePipeline(self.sensor_columns)
        self.anomaly_detector = AnomalyDetector(
            contamination=self.settings.anomaly_contamination,
            n_estimators=self.settings.isolation_n_estimators,
            random_state=self.settings.simulation_seed,
        )
        self.failure_predictor = FailurePredictor(random_state=self.settings.simulation_seed)
        self.rca = RootCauseAnalyzer(self.sensor_columns)
        self.decision_engine = DecisionEngine.from_settings(self.settings)
        self._reference_mean: Optional[np.ndarray] = None
        self._reference_std: Optional[np.ndarray] = None

    def fit_from_synthetic(self) -> None:
        """Train detectors using freshly simulated industrial data."""

        params = SimulationParams(
            n_steps=self.settings.training_rows,
            seed=self.settings.simulation_seed,
            failure_horizon_steps=self.settings.failure_horizon_steps,
            wear_failure_threshold=self.settings.wear_failure_threshold,
        )
        simulator = IndustrialSensorSimulator(params)
        df = simulator.generate_history()
        features, labels = _sliding_window_dataset(df, self.feature_pipeline, self.window_size)
        self.anomaly_detector.fit(features)
        self.failure_predictor.fit(features, labels)
        self._reference_mean = features.mean(axis=0)
        self._reference_std = features.std(axis=0)

    def predict_window(self, df_window: pd.DataFrame) -> PredictionBundle:
        """Run ML models on a validated sensor context window."""

        cleaned = validate_sensor_window(df_window, self.sensor_columns)
        enforce_minimum_window(cleaned, self.window_size)
        features = self.feature_pipeline.transform_window(cleaned)
        anomaly_score = float(self.anomaly_detector.decision_function(features)[0])
        label = int(self.anomaly_detector.predict_label(features)[0])
        failure_probability = float(self.failure_predictor.predict_proba_failure(features)[0])
        anomaly_flag = label == -1
        return PredictionBundle(
            failure_probability=failure_probability,
            anomaly_score=anomaly_score,
            anomaly_flag=anomaly_flag,
            isolation_label=label,
        )

    def analyze_rca(self, df_window: pd.DataFrame) -> RootCauseInsight:
        """Return structured RCA insights for the latest window."""

        cleaned = validate_sensor_window(df_window, self.sensor_columns)
        enforce_minimum_window(cleaned, self.window_size)
        return self.rca.analyze(cleaned)

    def decide(self, df_window: pd.DataFrame) -> DecisionResult:
        """Produce a fused maintenance decision."""

        preds = self.predict_window(df_window)
        rca = self.analyze_rca(df_window)
        return self.decision_engine.decide(
            failure_probability=preds.failure_probability,
            anomaly_score=preds.anomaly_score,
            anomaly_flag=preds.anomaly_flag,
            rca=rca,
        )

    def explain(self, df_window: pd.DataFrame) -> Dict[str, Any]:
        """Return global and local explanations for stakeholders."""

        cleaned = validate_sensor_window(df_window, self.sensor_columns)
        enforce_minimum_window(cleaned, self.window_size)
        features = self.feature_pipeline.transform_window(cleaned)
        preds = self.predict_window(cleaned)
        failure_explanations = explain_failure_importances(self.failure_predictor, self.feature_pipeline)
        if self._reference_mean is None or self._reference_std is None:
            raise RuntimeError("Assistant must be fitted before explanations")
        anomaly_explanations = explain_anomaly_drivers(
            features,
            self._reference_mean,
            self._reference_std,
            self.feature_pipeline,
        )
        return {
            "failure_probability": preds.failure_probability,
            "anomaly_score": preds.anomaly_score,
            "top_failure_features": [{"feature": n, "importance": v} for n, v in failure_explanations],
            "anomaly_feature_deviations": anomaly_explanations,
        }

    def what_if(self, df_window: pd.DataFrame, deltas: Dict[str, float]) -> Dict[str, Any]:
        """Re-evaluate decisions after applying additive deltas to sensor columns."""

        adjusted = validate_sensor_window(df_window, self.sensor_columns).copy()
        enforce_minimum_window(adjusted, self.window_size)
        for column, delta in deltas.items():
            if column not in self.sensor_columns:
                raise ValueError(f"Unknown sensor column for what-if: {column}")
            adjusted[column] = adjusted[column] + float(delta)
        baseline = self.decide(df_window)
        scenario = self.decide(adjusted)
        base_preds = self.predict_window(df_window)
        scen_preds = self.predict_window(adjusted)
        return {
            "baseline": {
                "action": baseline.action,
                "risk_level": baseline.risk_level,
                "risk_score": baseline.risk_score,
                "confidence": baseline.confidence,
                "confidence_breakdown": baseline.confidence_breakdown,
                "explanation": baseline.explanation,
                "failure_probability": base_preds.failure_probability,
                "anomaly_flag": base_preds.anomaly_flag,
                "anomaly_score": base_preds.anomaly_score,
            },
            "scenario": {
                "action": scenario.action,
                "risk_level": scenario.risk_level,
                "risk_score": scenario.risk_score,
                "confidence": scenario.confidence,
                "confidence_breakdown": scenario.confidence_breakdown,
                "explanation": scenario.explanation,
                "failure_probability": scen_preds.failure_probability,
                "anomaly_flag": scen_preds.anomaly_flag,
                "anomaly_score": scen_preds.anomaly_score,
            },
            "applied_deltas": deltas,
        }

    def log_decision(self, record: Dict[str, Any]) -> None:
        """Persist a decision record when logging is enabled."""

        if not self.settings.log_decisions:
            return
        path = self.settings.logs_dir / "decisions.jsonl"
        log_decision_record(path, record)


def _sliding_window_dataset(
    df: pd.DataFrame,
    pipeline: FeaturePipeline,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a fixed window across history to build supervised training rows."""

    rows: List[np.ndarray] = []
    labels: List[int] = []
    for end in range(window, len(df)):
        segment = df.iloc[end - window : end]
        rows.append(pipeline.transform_window(segment).ravel())
        labels.append(int(df.iloc[end - 1]["failure_within_h"]))
    return np.vstack(rows), np.array(labels, dtype=int)


def dataframe_from_records(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert API payloads into a dataframe with canonical column ordering."""

    df = pd.DataFrame(records)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df
