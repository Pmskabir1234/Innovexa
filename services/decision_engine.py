"""Fuse model outputs, thresholds, and rules into actionable guidance.

Risk and confidence (MVP calibration):

1) Normalized signals in ``[0, 1]``:

   - **Failure signal** ``s_f = clip(p, 0, 1)`` where ``p`` is failure probability.

   - **Anomaly signal** ``s_a = clip((τ - a) / σ_a, 0, 1)`` where ``a`` is the
     isolation-forest decision function (lower = more anomalous), ``τ`` is
     ``anomaly_score_threshold``, and ``σ_a`` is ``anomaly_severity_span``.

   - **RCA signal** ``s_r``: hypothesis confidence, damped when the narrative is
     the nominal "within normal" hypothesis so benign RCA does not inflate risk.

2) **Composite risk** (weighted sum):

   ``R = w_f·s_f + w_a·s_a + w_r·s_r`` with ``w_f + w_a + w_r = 1``.

3) **Risk tier** (``low`` / ``medium`` / ``high``):

   - If ``p ≥ p_high`` → **high** (failure model override).
   - Elif ``p ≥ p_med`` → tier at least **medium**.
   - Else tier from ``R``: **low** if ``R < R₁``, **medium** if ``R₁ ≤ R < R₂``,
     **high** if ``R ≥ R₂``.

4) **Confidence** in ``[c_min, c_max]``:

   ``C = clip( c₀ + c₁·R^γ + c₂·(K/3) - δ·𝟙[contradiction], c_min, c_max )``

   where ``K`` counts how many of ``{s_f≥θ_f, s_a≥θ_a, s_r≥θ_r}`` fire,
   and **contradiction** is ``s_f < 0.15`` with ``s_a > 0.55`` (strong anomaly,
   weak failure estimate).

Explainability strings summarize ``R``, weights, tier boundaries, and ``C``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pow
from typing import Dict, List, Tuple

from utils.config import Settings 

from .rca import RootCauseInsight


@dataclass
class DecisionResult:
    """Structured operational decision for API and logging layers."""

    action: str
    risk_level: str
    risk_score: float
    confidence: float
    confidence_breakdown: Dict[str, float]
    rationale: List[str]
    explanation: List[str]


@dataclass
class _SignalSnapshot:
    """Internal bundle of normalized contributors."""

    failure: float
    anomaly: float
    rca: float


class DecisionEngine:
    """Translate scores into maintenance decisions with calibrated risk and confidence."""

    def __init__(
        self,
        failure_prob_high: float,
        failure_prob_med: float,
        anomaly_score_threshold: float,
        weight_failure: float,
        weight_anomaly: float,
        weight_rca: float,
        risk_low_below: float,
        risk_medium_below: float,
        anomaly_severity_span: float,
        confidence_c0: float,
        confidence_c1: float,
        confidence_gamma: float,
        confidence_evidence_bonus: float,
        contradiction_failure_max: float,
        contradiction_anomaly_min: float,
        contradiction_penalty: float,
        confidence_floor: float,
        confidence_ceiling: float,
        signal_threshold_failure: float,
        signal_threshold_anomaly: float,
        signal_threshold_rca: float,
        rca_nominal_damping: float,
    ) -> None:
        """Persist thresholds, weights, and confidence coefficients."""

        self.failure_prob_high = failure_prob_high
        self.failure_prob_med = failure_prob_med
        self.anomaly_score_threshold = anomaly_score_threshold
        wf, wa, wr = weight_failure, weight_anomaly, weight_rca
        total = wf + wa + wr
        if total <= 0:
            raise ValueError("Decision weights must sum to a positive value")
        self.weight_failure = wf / total
        self.weight_anomaly = wa / total
        self.weight_rca = wr / total
        self.risk_low_below = risk_low_below
        self.risk_medium_below = risk_medium_below
        self.anomaly_severity_span = max(anomaly_severity_span, 1e-6)
        self.confidence_c0 = confidence_c0
        self.confidence_c1 = confidence_c1
        self.confidence_gamma = confidence_gamma
        self.confidence_evidence_bonus = confidence_evidence_bonus
        self.contradiction_failure_max = contradiction_failure_max
        self.contradiction_anomaly_min = contradiction_anomaly_min
        self.contradiction_penalty = contradiction_penalty
        self.confidence_floor = confidence_floor
        self.confidence_ceiling = confidence_ceiling
        self.signal_threshold_failure = signal_threshold_failure
        self.signal_threshold_anomaly = signal_threshold_anomaly
        self.signal_threshold_rca = signal_threshold_rca
        self.rca_nominal_damping = rca_nominal_damping

    @classmethod
    def from_settings(cls, settings: "Settings") -> "DecisionEngine":
        """Build an engine from application settings."""
        
        # here was the same line mentioned
        if not isinstance(settings, Settings):
            raise TypeError("settings must be a Settings instance")
        return cls(
            failure_prob_high=settings.decision_failure_prob_high,
            failure_prob_med=settings.decision_failure_prob_med,
            anomaly_score_threshold=settings.decision_anomaly_score_low,
            weight_failure=settings.decision_weight_failure,
            weight_anomaly=settings.decision_weight_anomaly,
            weight_rca=settings.decision_weight_rca,
            risk_low_below=settings.decision_risk_low_below,
            risk_medium_below=settings.decision_risk_medium_below,
            anomaly_severity_span=settings.decision_anomaly_severity_span,
            confidence_c0=settings.decision_confidence_c0,
            confidence_c1=settings.decision_confidence_c1,
            confidence_gamma=settings.decision_confidence_gamma,
            confidence_evidence_bonus=settings.decision_confidence_evidence_bonus,
            contradiction_failure_max=settings.decision_contradiction_failure_max,
            contradiction_anomaly_min=settings.decision_contradiction_anomaly_min,
            contradiction_penalty=settings.decision_contradiction_penalty,
            confidence_floor=settings.decision_confidence_floor,
            confidence_ceiling=settings.decision_confidence_ceiling,
            signal_threshold_failure=settings.decision_signal_threshold_failure,
            signal_threshold_anomaly=settings.decision_signal_threshold_anomaly,
            signal_threshold_rca=settings.decision_signal_threshold_rca,
            rca_nominal_damping=settings.decision_rca_nominal_damping,
        )

    def decide(
        self,
        failure_probability: float,
        anomaly_score: float,
        anomaly_flag: bool,
        rca: RootCauseInsight,
    ) -> DecisionResult:
        """Combine ML signals and RCA into a decision with explicit scoring."""

        signals = self._normalize_signals(failure_probability, anomaly_score, rca)
        risk_score = self._composite_risk(signals)
        risk_level = self._classify_risk(risk_score, failure_probability)
        confidence, breakdown = self._compute_confidence(risk_score, signals, failure_probability)
        action = self._select_action(risk_level, failure_probability, anomaly_flag, anomaly_score, signals)
        rationale = self._build_rationale(
            failure_probability, anomaly_score, anomaly_flag, rca, risk_level, signals
        )
        explanation = self._build_explanation(
            risk_score, risk_level, signals, confidence, breakdown, failure_probability, anomaly_score
        )

        return DecisionResult(
            action=action,
            risk_level=risk_level,
            risk_score=risk_score,
            confidence=confidence,
            confidence_breakdown=breakdown,
            rationale=rationale,
            explanation=explanation,
        )

    def _normalize_signals(
        self,
        failure_probability: float,
        anomaly_score: float,
        rca: RootCauseInsight,
    ) -> _SignalSnapshot:
        """Map raw model outputs to comparable severity in ``[0, 1]``."""

        s_f = _clip(failure_probability, 0.0, 1.0)
        raw_anom = (self.anomaly_score_threshold - anomaly_score) / self.anomaly_severity_span
        s_a = _clip(raw_anom, 0.0, 1.0)
        damp = self.rca_nominal_damping if "within normal" in rca.primary_hypothesis.lower() else 1.0
        s_r = _clip(rca.confidence * damp, 0.0, 1.0)
        return _SignalSnapshot(failure=s_f, anomaly=s_a, rca=s_r)

    def _composite_risk(self, signals: _SignalSnapshot) -> float:
        """Weighted linear combination of normalized signals."""

        r = (
            self.weight_failure * signals.failure
            + self.weight_anomaly * signals.anomaly
            + self.weight_rca * signals.rca
        )
        return float(_clip(r, 0.0, 1.0))

    def _classify_risk(self, risk_score: float, failure_probability: float) -> str:
        """Assign ``low``, ``medium``, or ``high`` with failure-model overrides."""

        if failure_probability >= self.failure_prob_high:
            return "high"
        if risk_score >= self.risk_medium_below:
            tier = "high"
        elif risk_score >= self.risk_low_below:
            tier = "medium"
        else:
            tier = "low"
        if failure_probability >= self.failure_prob_med and tier == "low":
            tier = "medium"
        return tier

    def _compute_confidence(
        self,
        risk_score: float,
        signals: _SignalSnapshot,
        failure_probability: float,
    ) -> Tuple[float, Dict[str, float]]:
        """Apply the documented confidence formula and return interpretable parts."""

        k = 0
        if signals.failure >= self.signal_threshold_failure:
            k += 1
        if signals.anomaly >= self.signal_threshold_anomaly:
            k += 1
        if signals.rca >= self.signal_threshold_rca:
            k += 1

        contradiction = (
            signals.failure <= self.contradiction_failure_max
            and signals.anomaly >= self.contradiction_anomaly_min
        )
        strength = pow(max(risk_score, 0.0), self.confidence_gamma)
        penalty = self.contradiction_penalty if contradiction else 0.0
        raw = (
            self.confidence_c0
            + self.confidence_c1 * strength
            + self.confidence_evidence_bonus * (k / 3.0)
            - penalty
        )
        confidence = float(_clip(raw, self.confidence_floor, self.confidence_ceiling))

        breakdown: Dict[str, float] = {
            "composite_risk_R": round(risk_score, 4),
            "signal_failure_sf": round(signals.failure, 4),
            "signal_anomaly_sa": round(signals.anomaly, 4),
            "signal_rca_sr": round(signals.rca, 4),
            "weight_failure_wf": round(self.weight_failure, 4),
            "weight_anomaly_wa": round(self.weight_anomaly, 4),
            "weight_rca_wr": round(self.weight_rca, 4),
            "evidence_count_K": float(k),
            "strength_term_R_gamma": round(self.confidence_c1 * strength, 4),
            "evidence_bonus": round(self.confidence_evidence_bonus * (k / 3.0), 4),
            "contradiction_penalty_applied": round(penalty, 4),
            "confidence_raw": round(raw, 4),
        }
        return confidence, breakdown

    def _select_action(
        self,
        risk_level: str,
        failure_probability: float,
        anomaly_flag: bool,
        anomaly_score: float,
        signals: _SignalSnapshot,
    ) -> str:
        """Pick a concrete maintenance action from tier and dominant signals."""

        if risk_level == "low":
            return "Continue monitoring with current sampling plan"

        if risk_level == "medium":
            if failure_probability >= self.failure_prob_med:
                return "Plan maintenance within 48 hours; increase sampling cadence and verify lubrication"
            if signals.anomaly >= 0.35:
                return "Run targeted diagnostics on sensors with elevated joint deviation"
            return "Review trends and schedule inspection if any signal worsens over the next shifts"

        if failure_probability >= self.failure_prob_high:
            return "Schedule immediate inspection; prepare for controlled shutdown if vibration or temperature accelerates"
        if anomaly_flag or anomaly_score < self.anomaly_score_threshold or signals.anomaly >= 0.55:
            return "Escalate to on-call engineering: multivariate anomaly with high operational risk"
        return "Immediate risk posture: coordinate inspection and reduce load until cleared"

    def _build_rationale(
        self,
        failure_probability: float,
        anomaly_score: float,
        anomaly_flag: bool,
        rca: RootCauseInsight,
        risk_level: str,
        signals: _SignalSnapshot,
    ) -> List[str]:
        """Operational bullets for runbooks and dashboards."""

        lines: List[str] = []
        lines.append(
            f"Risk tier {risk_level.upper()} (failure={failure_probability:.2f}, "
            f"anomaly_score={anomaly_score:.3f}, RCA confidence={rca.confidence:.2f})"
        )
        if failure_probability >= self.failure_prob_med:
            lines.append("Failure model indicates elevated near-term risk")
        if anomaly_flag or signals.anomaly >= 0.25:
            lines.append("Multivariate profile deviates from learned healthy manifold")
        if rca.confidence >= 0.55 and "within normal" not in rca.primary_hypothesis.lower():
            lines.append(f"RCA hypothesis: {rca.primary_hypothesis}")
        if not lines:
            lines.append("All monitored signals within expected bands")
        return lines

    def _build_explanation(
        self,
        risk_score: float,
        risk_level: str,
        signals: _SignalSnapshot,
        confidence: float,
        breakdown: Dict[str, float],
        failure_probability: float,
        anomaly_score: float,
    ) -> List[str]:
        """Transparent derivation of tier and confidence for audit trails."""

        r_low, r_mid = self.risk_low_below, self.risk_medium_below
        tier_rule = (
            f"Tier from composite R={risk_score:.3f}: low if R<{r_low:.2f}, "
            f"medium if {r_low:.2f}<=R<{r_mid:.2f}, high if R>={r_mid:.2f}, "
            f"with failure-probability overrides at {self.failure_prob_med:.2f}/"
            f"{self.failure_prob_high:.2f}."
        )
        weight_rule = (
            f"R = {self.weight_failure:.2f}*s_f + {self.weight_anomaly:.2f}*s_a + "
            f"{self.weight_rca:.2f}*s_r with s_f={signals.failure:.3f}, "
            f"s_a={signals.anomaly:.3f}, s_r={signals.rca:.3f}."
        )
        conf_rule = (
            f"Confidence {confidence:.3f} = clip(c0 + c1*R^gamma + c2*K/3 - penalty, "
            f"[{self.confidence_floor:.2f}, {self.confidence_ceiling:.2f}]); "
            f"K={int(breakdown['evidence_count_K'])} active signal bands."
        )
        ctx = (
            f"Raw inputs: p_fail={failure_probability:.3f}, isolation decision_function={anomaly_score:.4f} "
            f"(lower is more anomalous vs training)."
        )
        return [tier_rule, weight_rule, conf_rule, ctx]


def _clip(value: float, low: float, high: float) -> float:
    """Clamp a scalar to a closed interval."""

    return max(low, min(high, value))


