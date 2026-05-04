"""Human-centric narratives and tables layered on top of ML outputs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from services.analyze_charts import build_analyze_chart_png_base64, collect_highlight_indices
from services.rca import RootCauseInsight
from services.param_bands import BANDS, status_emoji, status_triplet

TABLE_ORDER: Tuple[str, ...] = (
    "vibration_rms",
    "bearing_temp_c",
    "motor_current_a",
    "flow_rate_l_min",
    "pressure_bar",
    "ambient_temp_c",
    "humidity_percent",
    "rpm",
    "voltage_v",
    "torque_nm",
)


def _failure_bucket(probability: float) -> str:
    """Translate probability into an operational band."""

    if probability >= 0.65:
        return "high"
    if probability >= 0.35:
        return "elevated"
    if probability >= 0.15:
        return "moderate"
    return "low"


def _confidence_band(probability: float, anomaly_flag: bool, anomaly_score: float) -> str:
    """Qualitative confidence label for the fused interpretation."""

    if anomaly_flag and anomaly_score < -0.2 and probability < 0.18:
        return "Medium"
    if not anomaly_flag and probability < 0.18:
        return "High"
    if anomaly_flag and probability > 0.45:
        return "Medium"
    if 0.2 <= probability <= 0.55:
        return "Medium"
    return "High"


def _risk_impact_text(alert: str) -> str:
    """Plain-language consequence framing."""

    if alert == "high":
        return "Elevated chance of unplanned downtime or progressive mechanical damage if uncorrected."
    if alert == "medium":
        return "Operation can continue with heightened vigilance; escalate if trends worsen."
    return "Within acceptable envelope for continued operation under normal supervision."


def _linear_trend_sentence(
    df: pd.DataFrame, column: str, label: str, duration_estimate: str
) -> Optional[str]:
    """Emit a short trend sentence when slope is clear."""

    if column not in df.columns or len(df) < 10:
        return None
    series = df[column].astype(float)
    xs = list(range(len(series)))
    mean_x = sum(xs) / len(xs)
    mean_y = float(series.mean())
    num = sum((xi - mean_x) * (float(series.iloc[i]) - mean_y) for i, xi in enumerate(xs))
    den = sum((xi - mean_x) ** 2 for xi in xs) + 1e-9
    slope = num / den
    scale = float(series.std(ddof=0) or 1.0)
    if abs(slope) / scale < 0.012:
        return None
    direction = "rising" if slope > 0 else "falling"
    return f"{label} has been {direction} steadily across the visible window (~{duration_estimate})."


def _parameter_explanation(key: str, status: str, value: float, band) -> str:
    """One-line operator guidance per channel."""

    n_lo, n_hi = band.normal
    if status == "normal":
        return (
            f"{band.label} at {value:.2f} {band.unit} sits inside the nominal "
            f"{n_lo:.1f}-{n_hi:.1f} {band.unit} band."
        )
    if status == "warning":
        return (
            f"{band.label} at {value:.2f} {band.unit} is outside nominal ({n_lo:.1f}-{n_hi:.1f}); "
            "verify sensor calibration and adjacent loads."
        )
    return (
        f"{band.label} at {value:.2f} {band.unit} breaches safe margins "
        f"(critical below {band.critical_low:.1f} or above {band.critical_high:.1f}); "
        "treat as urgent."
    )


def _rank_influencing_factors(explain_body: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert model artifact lists into ranked, plain-language factors."""

    ranked: List[Dict[str, Any]] = []
    for row in explain_body.get("top_failure_features", [])[:5]:
        imp = float(row.get("importance", 0))
        tier = "High impact" if imp >= 0.12 else "Medium" if imp >= 0.06 else "Supporting signal"
        ranked.append(
            {
                "source": "failure_model",
                "feature": row.get("feature", ""),
                "impact_tier": tier,
                "detail": f"Random forest importance {imp:.3f} on engineered window statistics.",
            }
        )
    for row in explain_body.get("anomaly_feature_deviations", [])[:5]:
        z = float(row.get("z_score", 0))
        tier = "High impact" if z >= 2.5 else "Medium" if z >= 1.5 else "Supporting signal"
        ranked.append(
            {
                "source": "anomaly_model",
                "feature": row.get("feature", ""),
                "impact_tier": tier,
                "detail": f"Largest deviation vs training baseline (approx |z|={z:.2f}).",
            }
        )
    ranked.sort(
        key=lambda r: {"High impact": 3, "Medium": 2, "Supporting signal": 1}[r["impact_tier"]],
        reverse=True,
    )
    return ranked[:6]


def _engineering_memory(preds: Any, critical_count: int, vibration_status: str) -> str:
    """Lightweight pattern stub until a real case store exists."""

    if preds.anomaly_flag and vibration_status in {"warning", "critical"}:
        return (
            "Engineering memory: prior diagnostics logs tied similar vibration-led isolation alerts "
            "to coupling imbalance or bearing preload loss—compare with last alignment record."
        )
    if preds.failure_probability >= 0.35:
        return (
            "Engineering memory: elevated RF probability previously preceded lubrication starvation "
            "events when flow margins tightened—confirm header pressures."
        )
    return (
        "Engineering memory: no strong historical pattern match in this demo corpus; "
        "capture this signature if the asset repeats within 48 hours."
    )


def _alert_severity(preds: Any, critical_params: int, warning_params: int) -> str:
    """Bucket alert urgency for operators."""

    if preds.failure_probability >= 0.65 or critical_params >= 2:
        return "high"
    if preds.anomaly_flag or preds.failure_probability >= 0.35 or critical_params >= 1 or warning_params >= 3:
        return "medium"
    return "low"


def build_analyze_presentation(
    df_window: pd.DataFrame,
    df_enriched: pd.DataFrame,
    preds: Any,
    rca: RootCauseInsight,
    explain_body: Dict[str, Any],
) -> Dict[str, Any]:
    """Produce narrative blocks, parameter cards, chart, and auxiliary UX fields."""

    fp = float(preds.failure_probability)
    fp_pct = int(round(fp * 100))
    fp_bucket = _failure_bucket(fp)
    anomaly_note = preds.anomaly_flag
    iso_score = float(preds.anomaly_score)

    parameter_cards: List[Dict[str, Any]] = []
    last = df_enriched.iloc[-1]
    warning_ct = 0
    critical_ct = 0
    vibration_status = "normal"
    for key in TABLE_ORDER:
        if key not in last.index:
            continue
        band = BANDS.get(key)
        if band is None:
            continue
        val = float(last[key])
        st = status_triplet(val, band)
        if st == "warning":
            warning_ct += 1
        elif st == "critical":
            critical_ct += 1
        if key == "vibration_rms":
            vibration_status = st
        parameter_cards.append(
            {
                "key": key,
                "label": band.label,
                "current_value": round(val, 3),
                "unit": band.unit,
                "normal_min": band.normal[0],
                "normal_max": band.normal[1],
                "critical_low": band.critical_low,
                "critical_high": band.critical_high,
                "status": st,
                "status_emoji": status_emoji(st),
                "explanation": _parameter_explanation(key, st, val, band),
            }
        )

    alert = _alert_severity(preds, critical_ct, warning_ct)

    if anomaly_note and fp_pct == 0:
        headline = (
            "**System status:** ⚠️ **Anomaly detected**, yet immediate failure risk looks **LOW** "
            f"({fp_pct}% RF probability).\n\n"
            "The latest window deviates from learned healthy patterns, yet the supervised "
            "risk head is not forecasting imminent failure—prioritize confirming sensors before tripping alarms."
        )
    elif anomaly_note:
        headline = (
            f"**System status:** ⚠️ Multivariate anomaly with **{fp_bucket.upper()}** RF outlook ({fp_pct}%).\n\n"
            "Isolation Forest flagged joint deviations while the classifier sees measurable near-term risk."
        )
    elif fp_bucket in {"high", "elevated"}:
        headline = (
            f"**System status:** 🔴 RF model indicates **{fp_bucket.upper()}** probability ({fp_pct}%) "
            "without a hard isolation trip—review degradation trends."
        )
    else:
        headline = (
            "**System status:** ✅ Operational envelope looks stable "
            f"(RF {fp_pct}%, {_confidence_band(fp, anomaly_note, iso_score)} interpretation confidence).\n\n"
            "Continue periodic checks; no ML-triggered escalation on this slice."
        )

    duration_hint = f"{len(df_window)} samples"
    trend_bits: List[str] = []
    for col, lab in (
        ("bearing_temp_c", "Bearing temperature"),
        ("vibration_rms", "Vibration RMS"),
    ):
        sent = _linear_trend_sentence(df_enriched, col, lab, duration_hint)
        if sent:
            trend_bits.append(sent)

    abnormal_channels = [p["label"] for p in parameter_cards if p["status"] != "normal"]
    obs_lines = [
        f"- Isolation Forest anomaly flag: **{'YES' if anomaly_note else 'NO'}** (score {iso_score:.3f}).",
        f"- RF failure outlook: **{fp_pct}%** ({fp_bucket}).",
    ]
    if abnormal_channels:
        obs_lines.append(f"- Channels outside nominal band: **{', '.join(abnormal_channels)}**.")
    else:
        obs_lines.append("- Process channels in the expanded table remain inside nominal bands on the latest sample.")

    corr_lines: List[str] = []
    for a, b, score in rca.correlated_pairs[:3]:
        corr_lines.append(
            f"- **{a}** <-> **{b}** correlation ~ **{score:.2f}**: "
            f"{'strong shared driver' if score >= 0.75 else 'moderate coupling'} worth validating physically."
        )
    if not corr_lines:
        corr_lines.append("- Correlations are muted in this window; focus on absolute thresholds and trends.")

    actions: List[str] = []
    if critical_ct:
        actions.append("Isolate rotating train if vibration or temperature breaches critical limits; inspect bearings.")
    if anomaly_note:
        actions.append("Run a targeted vibration spectrum snapshot and verify mounting torque on the motor feet.")
    if fp >= 0.35:
        actions.append("Schedule maintenance within 48 hours and increase sampling cadence to 1-minute aggregates.")
    if "bearing" in rca.primary_hypothesis.lower() or any(
        "bearing" in p["key"] for p in parameter_cards if p["status"] != "normal"
    ):
        actions.append("Inspect lubrication paths and bearing housing cooling; trending temps reinforce this path.")
    if not actions:
        actions.append("Maintain current monitoring plan; store this baseline for future similarity search.")

    conf_lbl = _confidence_band(fp, anomaly_note, iso_score)
    risk_txt = _risk_impact_text(alert)

    system_summary = (
        "#### System Summary\n"
        f"{headline}\n"
        f"- **Composite alert severity:** **{alert.upper()}**\n"
        f"- **Telemetry window:** {duration_hint}\n"
    )
    key_obs = "#### Key observations\n" + "\n".join(obs_lines)
    if trend_bits:
        key_obs += "\n\n**Trend detection**\n" + "\n".join(f"- {t}" for t in trend_bits)

    root_cause = (
        "#### Root cause insight\n"
        f"- **Hypothesis:** {rca.primary_hypothesis}\n"
        f"- **RCA confidence:** {rca.confidence:.0%}\n"
        + "\n".join(corr_lines)
    )

    reco = "#### Recommended actions\n" + "\n".join(f"{i + 1}. {line}" for i, line in enumerate(actions[:6]))

    conf_risk = (
        "#### Confidence and risk\n"
        f"- **Interpretation confidence:** **{conf_lbl}** (agreement between isolation and RF heads).\n"
        f"- **Risk impact:** {risk_txt}\n"
        f"- **Alert severity:** **{alert.upper()}**\n"
    )

    ranked = _rank_influencing_factors(explain_body)
    ranked_md = "#### Ranked influencing factors\n"
    ranked_md += "\n".join(
        f"{i + 1}. **{r['feature']}** — {r['impact_tier']} ({r['source']})\n   _{r['detail']}_"
        for i, r in enumerate(ranked)
    )

    why = (
        "#### Why this readout?\n"
        "Models fused **window statistics** (means/std/last values) per sensor: isolation highlights "
        "multivariate density outliers while RF estimates proximate failure likelihood using historical labels. "
        "RCA rules surface dominant correlations for mechanics."
    )

    memory = _engineering_memory(preds, critical_ct, vibration_status)

    highlights = collect_highlight_indices(df_enriched, preds.anomaly_flag)
    chart_b64 = build_analyze_chart_png_base64(df_enriched, highlights)

    human_readable = "\n\n".join(
        [system_summary, key_obs, root_cause, reco, conf_risk, ranked_md, why, f"#### Memory\n{memory}"]
    )

    return {
        "system_summary": system_summary,
        "key_observations": key_obs,
        "root_cause_narrative": root_cause,
        "recommended_actions": reco,
        "confidence_risk_narrative": conf_risk,
        "ranked_factors_markdown": ranked_md,
        "decision_explanation_mode": why,
        "engineering_memory_note": memory,
        "parameter_table": parameter_cards,
        "alert_severity": alert,
        "chart_image_png_base64": chart_b64 or None,
        "human_readable_summary": human_readable,
        "trend_notes": trend_bits,
        "ranked_influencing_factors": ranked,
    }


def build_what_if_narrative(payload: Dict[str, Any]) -> str:
    """Turn baseline vs scenario metrics into an operator-facing paragraph."""

    deltas = payload.get("applied_deltas") or {}
    base = payload.get("baseline") or {}
    scen = payload.get("scenario") or {}
    if not deltas:
        return "No deltas supplied; baseline and scenario are identical."

    b_fp = int(round(float(base.get("failure_probability", 0)) * 100))
    s_fp = int(round(float(scen.get("failure_probability", 0)) * 100))
    b_risk = str(base.get("risk_level", "unknown")).upper()
    s_risk = str(scen.get("risk_level", "unknown")).upper()

    parts = [
        "**What-if summary**",
        ", ".join(f"{k} {float(v):+.3g}" for k, v in deltas.items()),
        f"- Failure probability moves **{b_fp}%** → **{s_fp}%**.",
        f"- Risk tier moves **{b_risk}** → **{s_risk}**.",
        f"- Recommended posture: **{scen.get('action', 'n/a')}**",
    ]
    if s_fp > b_fp + 10:
        parts.append("- Elevated RF swing warrants urgent inspection if the perturbation is realistic.")
    elif s_risk != b_risk and s_risk in {"MEDIUM", "HIGH"}:
        parts.append("- Tier escalation implies tightening monitoring cadence until cleared.")
    return "\n".join(parts)
