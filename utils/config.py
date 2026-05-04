"""Application configuration loaded from environment with typed defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central settings for API, models, and logging paths."""

    model_config = SettingsConfigDict(
        env_prefix="TA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    models_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "models")
    logs_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "logs")
    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data")

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    simulation_seed: int = 42
    training_rows: int = 8000
    failure_horizon_steps: int = 48
    wear_failure_threshold: float = 0.82
    anomaly_contamination: float = 0.05
    isolation_n_estimators: int = 200
    decision_failure_prob_high: float = 0.65
    decision_failure_prob_med: float = 0.35
    decision_anomaly_score_low: float = -0.25

    decision_weight_failure: float = 0.46
    decision_weight_anomaly: float = 0.34
    decision_weight_rca: float = 0.20
    decision_risk_low_below: float = 0.28
    decision_risk_medium_below: float = 0.55
    decision_anomaly_severity_span: float = 0.45
    decision_confidence_c0: float = 0.42
    decision_confidence_c1: float = 0.48
    decision_confidence_gamma: float = 0.82
    decision_confidence_evidence_bonus: float = 0.10
    decision_contradiction_failure_max: float = 0.15
    decision_contradiction_anomaly_min: float = 0.55
    decision_contradiction_penalty: float = 0.12
    decision_confidence_floor: float = 0.40
    decision_confidence_ceiling: float = 0.95
    decision_signal_threshold_failure: float = 0.25
    decision_signal_threshold_anomaly: float = 0.25
    decision_signal_threshold_rca: float = 0.35
    decision_rca_nominal_damping: float = 0.18

    log_decisions: bool = True
    model_artifact_prefix: str = "assistant"

    stream_tick_ms_default: int = 250
    stream_default_max_ticks: int = 400
    stream_max_ticks_cap: int = 50_000
    stream_drift_horizon_steps: int = 12_000
    stream_load_period_steps: int = 5_000


def get_settings() -> Settings:
    """Return settings singleton for dependency injection."""

    return Settings()


def ensure_runtime_dirs(settings: Optional[Settings] = None) -> Settings:
    """Create models and logs directories if missing."""

    s = settings or get_settings()
    s.models_dir.mkdir(parents=True, exist_ok=True)
    s.logs_dir.mkdir(parents=True, exist_ok=True)
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return s
