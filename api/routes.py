"""FastAPI routers for inference, analysis, and explanation."""

from __future__ import annotations

from typing import Any, Dict, List, Union
import logging
import json
from pathlib import Path
import pandas as pd

from fastapi import APIRouter, Depends, Request, Query

from api.schemas import (
    AnalyzeResponse,
    DecisionResponse,
    ExplainResponse,
    ParameterStatusCard,
    PredictionResponse,
    RCAEvidence,
    SensorWindowRequest,
    WhatIfRequest,
    InferenceRequest,
    HistoryResponse,
    HistoryItem,
    SimulationRequest,
    SimulationResponse,
)
from services.assistant import TechnicalAssistant, dataframe_from_records
from services.enrichment import ensure_presentation_columns
from services.insight_presenter import build_analyze_presentation, build_what_if_narrative

router = APIRouter()
logger = logging.getLogger("assistant")


def get_assistant(request: Request) -> TechnicalAssistant:
    """Resolve the assistant instance from application state."""

    return request.app.state.assistant


def _prepare_dataframe(payload: Union[SensorWindowRequest, InferenceRequest]) -> Any:
    """Extract readings and inflate if necessary to meet window requirements."""
    
    if isinstance(payload, InferenceRequest):
        readings = [payload.parameters.model_dump()]
    else:
        readings = [r.model_dump() for r in payload.readings]
    
    df = dataframe_from_records(readings)
    
    # Inflate if single point or too small
    if len(df) < 64:
        # Duplicate the last row until we have 64
        last_row = df.iloc[[-1]]
        padding = 64 - len(df)
        df = pd.concat([df] + [last_row] * padding, ignore_index=True)
        
    return df


@router.post("/predict", response_model=PredictionResponse)
def predict(
    payload: Union[SensorWindowRequest, InferenceRequest],
    assistant: TechnicalAssistant = Depends(get_assistant),
) -> PredictionResponse:
    """Score the latest window for anomalies and failure risk."""
    logger.info(f"POST /predict payload: {payload.model_dump_json()}")

    df = _prepare_dataframe(payload)
    bundle = assistant.predict_window(df)
    return PredictionResponse(
        failure_probability=bundle.failure_probability,
        anomaly_score=bundle.anomaly_score,
        anomaly_flag=bundle.anomaly_flag,
        isolation_label=bundle.isolation_label,
    )


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(
    payload: Union[SensorWindowRequest, InferenceRequest],
    assistant: TechnicalAssistant = Depends(get_assistant),
) -> AnalyzeResponse:
    """Return ML scores alongside RCA plus narratives, parameter bands, and a chart."""
    logger.info(f"POST /analyze payload: {payload.model_dump_json()}")

    df = _prepare_dataframe(payload)
    preds = assistant.predict_window(df)
    rca = assistant.analyze_rca(df)
    explain_body = assistant.explain(df)
    df_enriched = ensure_presentation_columns(df.copy())
    ux = build_analyze_presentation(df, df_enriched, preds, rca, explain_body)
    cards = [ParameterStatusCard(**row) for row in ux["parameter_table"]]
    
    # Log the decision for history
    asset_id = payload.asset_id if hasattr(payload, "asset_id") else "UNKNOWN"
    assistant.log_decision({
        "machine_id": asset_id,
        "risk_category": ux["alert_severity"].capitalize(),
        "failure_probability_percent": preds.failure_probability * 100,
        "created_at": pd.Timestamp.now().isoformat()
    })

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


@router.get("/history", response_model=HistoryResponse)
def get_history(
    machine_id: str = Query(..., description="Filter history by machine ID"),
    limit: int = Query(10, ge=1, le=100),
    assistant: TechnicalAssistant = Depends(get_assistant),
) -> HistoryResponse:
    """Retrieve recent analysis history for a given machine."""
    
    log_path = assistant.settings.logs_dir / "decisions.jsonl"
    items = []
    if log_path.exists():
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get("machine_id") == machine_id:
                        items.append(HistoryItem(
                            logged_at_utc=data["logged_at_utc"],
                            machine_id=data["machine_id"],
                            risk_category=data["risk_category"],
                            failure_probability_percent=data["failure_probability_percent"],
                            created_at=data.get("created_at") or data["logged_at_utc"]
                        ))
                except (json.JSONDecodeError, KeyError):
                    continue
    
    # Return latest first
    items.sort(key=lambda x: x.logged_at_utc, reverse=True)
    return HistoryResponse(items=items[:limit])


@router.post("/simulate", response_model=SimulationResponse)
def simulate(
    payload: SimulationRequest,
    assistant: TechnicalAssistant = Depends(get_assistant),
) -> SimulationResponse:
    """Run a what-if simulation comparing base vs override parameters."""
    logger.info(f"POST /simulate payload: {payload.model_dump_json()}")

    base_df = _prepare_dataframe(InferenceRequest(asset_id=payload.asset_id, parameters=payload.base_parameters))
    over_df = _prepare_dataframe(InferenceRequest(asset_id=payload.asset_id, parameters=payload.overrides))
    
    base_preds = assistant.predict_window(base_df)
    over_preds = assistant.predict_window(over_df)
    
    base_decision = assistant.decide(base_df)
    over_decision = assistant.decide(over_df)
    
    delta = (over_preds.failure_probability - base_preds.failure_probability) * 100
    summary = f"Adjusting parameters leads to a {abs(delta):.1f}pp {'increase' if delta > 0 else 'decrease'} in failure probability. "
    summary += f"The system risk level shifts from {base_decision.risk_level} to {over_decision.risk_level}."

    return SimulationResponse(
        base_failure_probability_percent=base_preds.failure_probability * 100,
        base_risk=base_decision.risk_level,
        simulated_failure_probability_percent=over_preds.failure_probability * 100,
        simulated_risk=over_decision.risk_level,
        impact_summary=summary
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
