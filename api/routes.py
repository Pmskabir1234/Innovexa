"""FastAPI routers for inference, analysis, and explanation."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, Request

from api.schemas import (
    AnalyzeResponse,
    DecisionResponse,
    ExplainResponse,
    ParameterStatusCard,
    PredictionResponse,
    RCAEvidence,
    SensorWindowRequest,
    WhatIfRequest,
)
from services.assistant import TechnicalAssistant, dataframe_from_records
from services.enrichment import ensure_presentation_columns
from services.insight_presenter import build_analyze_presentation, build_what_if_narrative

router = APIRouter()


def get_assistant(request: Request) -> TechnicalAssistant:
    """Resolve the assistant instance from application state."""

    return request.app.state.assistant


@router.post("/predict", response_model=PredictionResponse)
def predict(
    payload: SensorWindowRequest,
    assistant: TechnicalAssistant = Depends(get_assistant),
) -> PredictionResponse:
    """Score the latest window for anomalies and failure risk."""

    df = dataframe_from_records([r.model_dump() for r in payload.readings])
    bundle = assistant.predict_window(df)
    return PredictionResponse(
        failure_probability=bundle.failure_probability,
        anomaly_score=bundle.anomaly_score,
        anomaly_flag=bundle.anomaly_flag,
        isolation_label=bundle.isolation_label,
    )


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(
    payload: SensorWindowRequest,
    assistant: TechnicalAssistant = Depends(get_assistant),
) -> AnalyzeResponse:
    """Return ML scores alongside RCA plus narratives, parameter bands, and a chart."""

    df = dataframe_from_records([r.model_dump(exclude_none=True) for r in payload.readings])
    preds = assistant.predict_window(df)
    rca = assistant.analyze_rca(df)
    explain_body = assistant.explain(df)
    df_enriched = ensure_presentation_columns(df.copy())
    ux = build_analyze_presentation(df, df_enriched, preds, rca, explain_body)
    cards = [ParameterStatusCard(**row) for row in ux["parameter_table"]]
    return AnalyzeResponse(
        prediction=PredictionResponse(
            failure_probability=preds.failure_probability,
            anomaly_score=preds.anomaly_score,
            anomaly_flag=preds.anomaly_flag,
            isolation_label=preds.isolation_label,
        ),
        rca=RCAEvidence(
            primary_hypothesis=rca.primary_hypothesis,
            confidence=rca.confidence,
            evidence=rca.evidence,
            correlated_pairs=[
                {"sensor_a": a, "sensor_b": b, "score": s} for a, b, s in rca.correlated_pairs
            ],
        ),
        human_readable_summary=ux["human_readable_summary"],
        system_summary=ux["system_summary"],
        key_observations=ux["key_observations"],
        root_cause_narrative=ux["root_cause_narrative"],
        recommended_actions=ux["recommended_actions"],
        confidence_risk_narrative=ux["confidence_risk_narrative"],
        ranked_factors_markdown=ux["ranked_factors_markdown"],
        decision_explanation_mode=ux["decision_explanation_mode"],
        engineering_memory_note=ux["engineering_memory_note"],
        parameter_table=cards,
        alert_severity=ux["alert_severity"],
        trend_notes=ux["trend_notes"],
        ranked_influencing_factors=ux["ranked_influencing_factors"],
        chart_image_png_base64=ux["chart_image_png_base64"],
    )


@router.post("/decision", response_model=DecisionResponse)
def decision(
    payload: SensorWindowRequest,
    assistant: TechnicalAssistant = Depends(get_assistant),
) -> DecisionResponse:
    """Emit a fused maintenance decision and optional audit log."""

    df = dataframe_from_records([r.model_dump() for r in payload.readings])
    result = assistant.decide(df)
    preds = assistant.predict_window(df)
    log_payload: Dict[str, Any] = {
        "endpoint": "/decision",
        "asset_id": payload.asset_id,
        "failure_probability": preds.failure_probability,
        "anomaly_score": preds.anomaly_score,
        "decision": result.action,
        "risk_level": result.risk_level,
        "risk_score": result.risk_score,
        "confidence": result.confidence,
        "confidence_breakdown": result.confidence_breakdown,
    }
    assistant.log_decision(log_payload)
    return DecisionResponse(
        action=result.action,
        risk_level=result.risk_level,
        risk_score=result.risk_score,
        confidence=result.confidence,
        confidence_breakdown=result.confidence_breakdown,
        rationale=result.rationale,
        explanation=result.explanation,
    )


@router.post("/explain", response_model=ExplainResponse)
def explain(
    payload: SensorWindowRequest,
    assistant: TechnicalAssistant = Depends(get_assistant),
) -> ExplainResponse:
    """Surface feature-level drivers behind the latest assessment."""

    df = dataframe_from_records([r.model_dump() for r in payload.readings])
    body = assistant.explain(df)
    return ExplainResponse(
        failure_probability=body["failure_probability"],
        anomaly_score=body["anomaly_score"],
        top_failure_features=body["top_failure_features"],
        anomaly_feature_deviations=body["anomaly_feature_deviations"],
    )


@router.post("/whatif")
def what_if(
    payload: WhatIfRequest,
    assistant: TechnicalAssistant = Depends(get_assistant),
) -> Dict[str, Any]:
    """Contrast baseline and perturbed decisions for simple planning."""

    df = dataframe_from_records([r.model_dump(exclude_none=True) for r in payload.readings])
    payload_dict = assistant.what_if(df, payload.deltas)
    payload_dict["human_readable_summary"] = build_what_if_narrative(payload_dict)
    return payload_dict
