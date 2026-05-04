"""Rule-based and correlation-driven root cause hints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class RootCauseInsight:
    """Structured RCA output for downstream explanation."""

    primary_hypothesis: str
    confidence: float
    evidence: List[str]
    correlated_pairs: List[Tuple[str, str, float]]


class RootCauseAnalyzer:
    """Blend expert rules with simple correlation signals."""

    def __init__(self, sensor_columns: List[str]) -> None:
        """Remember sensor ordering for reporting."""

        self.sensor_columns = list(sensor_columns)

    def analyze(self, df_window: pd.DataFrame) -> RootCauseInsight:
        """Produce ranked hypotheses from the latest context window."""

        evidence: List[str] = []
        corr_pairs = _top_correlations(df_window, self.sensor_columns, top_k=3)
        for a, b, val in corr_pairs:
            evidence.append(f"{a} ↔ {b} correlation {val:.2f}")

        vibration = float(df_window["vibration_rms"].iloc[-5:].mean())
        temp = float(df_window["bearing_temp_c"].iloc[-5:].mean())
        flow = float(df_window["flow_rate_l_min"].iloc[-5:].mean())
        current = float(df_window["motor_current_a"].iloc[-5:].mean())

        hypothesis = "Within normal operating envelope"
        confidence = 0.35

        if vibration > 6.5 and temp > 72.0:
            hypothesis = "Bearing degradation or misalignment (thermal + vibration coupling)"
            confidence = 0.78
            evidence.append("Elevated vibration with rising bearing temperature")
        elif flow < 95.0 and current > 26.0:
            hypothesis = "Hydraulic restriction or impeller wear reducing throughput"
            confidence = 0.72
            evidence.append("High motor current with depressed flow")
        elif vibration > 7.8:
            hypothesis = "Mechanical imbalance or looseness"
            confidence = 0.68
            evidence.append("Isolation in vibration RMS trend")

        return RootCauseInsight(
            primary_hypothesis=hypothesis,
            confidence=float(confidence),
            evidence=evidence,
            correlated_pairs=corr_pairs,
        )


def _top_correlations(df: pd.DataFrame, cols: List[str], top_k: int) -> List[Tuple[str, str, float]]:
    """Return strongest absolute correlations between sensors."""

    if len(cols) < 2:
        return []
    sub = df[cols].astype(float)
    if len(sub) < 3:
        return []
    c = sub.corr().abs()
    pairs: List[Tuple[str, str, float]] = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            pairs.append((a, b, float(c.loc[a, b])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]
