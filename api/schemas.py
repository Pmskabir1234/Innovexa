"""Pydantic models shared across FastAPI routes."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    """Single multivariate sensor observation."""

    timestamp: Optional[datetime] = None
    motor_current_a: float
    vibration_rms: float
    bearing_temp_c: float
    inlet_pressure_bar: float = Field(..., alias="pressure_bar")
    flow_rate_l_min: float
    valve_position_pct: float = Field(default=50.0)  # Make optional with default
    ambient_temp_c: Optional[float] = None
    humidity_percent: Optional[float] = None
    rpm: Optional[float] = None
    voltage_v: Optional[float] = None
    torque_nm: Optional[float] = None

    model_config = {"populate_by_name": True}


class SensorWindowRequest(BaseModel):
    """Batch of readings forming a temporal context window."""

    asset_id: Optional[str] = Field(
        default=None, alias="machine_id", description="Optional asset tag for logging"
    )
    readings: List[SensorReading] = Field(
        ...,
        min_length=1,
        description="At least 1 sample is required. Windows < 64 will be inflated.",
    )

    model_config = {"populate_by_name": True}


class InferenceRequest(BaseModel):
    """Simple wrapper for frontend single-point calls."""

    asset_id: str = Field(..., alias="machine_id")
    parameters: SensorReading

    model_config = {"populate_by_name": True}


class HistoryItem(BaseModel):
    """A record of a past decision/analysis."""

    logged_at_utc: datetime
    machine_id: str
    risk_category: str
    failure_probability_percent: float
    created_at: datetime


class HistoryResponse(BaseModel):
    """List of history items."""

    items: List[HistoryItem]


class SimulationRequest(BaseModel):
    """Payload for what-if simulation from the frontend."""

    asset_id: str = Field(..., alias="machine_id")
    base_parameters: SensorReading
    overrides: SensorReading

    model_config = {"populate_by_name": True}


class SimulationResponse(BaseModel):
    """Results of a what-if simulation."""

    base_failure_probability_percent: float
    base_risk: str
    simulated_failure_probability_percent: float
    simulated_risk: str
    impact_summary: str


class PredictionResponse(BaseModel):
    """ML outputs for the provided window."""

    failure_probability: float
    anomaly_score: float
    anomaly_flag: bool
    isolation_label: int

    # Frontend-centric fields
    failure_probability_percent: Optional[float] = None
    risk_category: Optional[str] = None
    feature_importance: Optional[List[Dict[str, Any]]] = None


class RCAEvidence(BaseModel):
    """Narrative and quantitative RCA payload."""

    primary_hypothesis: str
    confidence: float
    evidence: List[str]
    correlated_pairs: List[Dict[str, Any]]


class ParameterStatusCard(BaseModel):
    """One telemetry channel with bands and interpretation (presentation layer)."""

    key: str
    label: str
    current_value: float
    unit: str
    normal_min: float
    normal_max: float
    critical_low: float
    critical_high: float
    status: Literal["normal", "warning", "critical"]
    status_emoji: str
    explanation: str


class AnalyzeResponse(BaseModel):
    """ML artifacts plus human-centric summaries and visualization."""

    # Backend-centric fields
    prediction: PredictionResponse
    rca: RCAEvidence
    human_readable_summary: str = Field(
        ...,
        description="Primary operator-facing markdown combining all structured sections",
    )
    system_summary: str
    key_observations: str
    root_cause_narrative: str
    recommended_actions: str
    confidence_risk_narrative: str
    ranked_factors_markdown: str
    decision_explanation_mode: str = Field(
        ...,
        description="Why models surfaced these conclusions",
    )
    engineering_memory_note: str
    parameter_table: List[ParameterStatusCard]
    alert_severity: Literal["low", "medium", "high"]
    trend_notes: List[str]
    ranked_influencing_factors: List[Dict[str, Any]]
    chart_image_png_base64: Optional[str] = Field(
        default=None,
        description="PNG bytes Base64 for matplotlib trio chart",
    )

    # Frontend-centric fields (added for compatibility)
    failure_probability_percent: float
    anomaly_score: float
    decision_priority: str
    risk_category: str
    health_score: float
    parameter_diagnostics: List[Dict[str, Any]]
    feature_importance: List[Dict[str, Any]]
    trend_insights: List[Dict[str, Any]]
    comparison_note: Optional[str] = None
    engineering_report: Optional[str] = None
    structured_analysis: Optional[Dict[str, Any]] = None


class DecisionResponse(BaseModel):
    """Operational decision with structured rationale."""

    action: str
    risk_level: str
    risk_score: float = Field(..., description="Composite risk R in [0, 1] from weighted signals")
    confidence: float
    confidence_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Terms of the confidence formula for auditability",
    )
    rationale: List[str]
    explanation: List[str] = Field(
        default_factory=list,
        description="How risk tier and confidence were derived from inputs",
    )


class ExplainResponse(BaseModel):
    """Explanation artifacts for model outputs."""

    failure_probability: float
    anomaly_score: float
    top_failure_features: List[Dict[str, Any]]
    anomaly_feature_deviations: List[Dict[str, Any]]


class WhatIfRequest(SensorWindowRequest):
    """Sensor window plus hypothetical adjustments."""

    deltas: Dict[str, float] = Field(default_factory=dict)
