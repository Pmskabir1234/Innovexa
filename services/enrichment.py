"""Derive extended plant parameters for UX without changing ML feature vectors."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ensure_presentation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineering parameters inferred from core sensors when absent."""

    out = df.copy()
    if "inlet_pressure_bar" in out.columns and "pressure_bar" not in out.columns:
        out["pressure_bar"] = out["inlet_pressure_bar"].astype(float)

    cur = out["motor_current_a"].astype(float) if "motor_current_a" in out.columns else None
    temp = out["bearing_temp_c"].astype(float) if "bearing_temp_c" in out.columns else None
    flow = out["flow_rate_l_min"].astype(float) if "flow_rate_l_min" in out.columns else None

    if cur is None or temp is None:
        return out

    temp_ref = float(temp.median())
    if "ambient_temp_c" not in out.columns or out["ambient_temp_c"].isna().all():
        out["ambient_temp_c"] = 22.0 + 0.12 * (temp - temp_ref) + 0.15 * (cur - float(cur.median())) / max(
            float(cur.std(ddof=0) or 1.0), 0.5
        )

    if "humidity_percent" not in out.columns or out["humidity_percent"].isna().all():
        out["humidity_percent"] = np.clip(
            52.0 - 0.35 * (temp - 48.0) + np.linspace(-1.5, 1.5, len(out)),
            18.0,
            78.0,
        )

    if "rpm" not in out.columns or out["rpm"].isna().all():
        load_proxy = cur / max(float(cur.quantile(0.9)), 24.0)
        out["rpm"] = 1680.0 + 140.0 * np.clip(load_proxy, 0.3, 1.15)

    if "voltage_v" not in out.columns or out["voltage_v"].isna().all():
        out["voltage_v"] = 398.0 + 0.55 * (cur - 18.0) + np.linspace(-2.0, 2.0, len(out))

    if "torque_nm" not in out.columns or out["torque_nm"].isna().all():
        rpm_safe = out["rpm"].astype(float).replace(0, np.nan).fillna(1750.0)
        out["torque_nm"] = np.clip(9550.0 * cur * 0.42 / rpm_safe, 15.0, 140.0)

    if flow is not None and "pressure_bar" in out.columns:
        coupling = (flow / max(float(flow.median()), 90.0)) * (out["pressure_bar"].astype(float) / 4.8)
        out["humidity_percent"] = np.clip(
            out["humidity_percent"].astype(float) + 0.08 * (coupling - 1.0) * 10.0,
            18.0,
            78.0,
        )

    return out


def latest_row(df: pd.DataFrame) -> pd.Series:
    """Return the most recent observation."""

    return df.iloc[-1]


def linear_trend_direction(series: pd.Series, threshold_ratio: float = 0.015):
    """Classify recent slope as rising, falling, or stable."""

    y = series.astype(float).values
    if len(y) < 8:
        return None
    x = np.arange(len(y), dtype=float)
    x = x - x.mean()
    yc = y - y.mean()
    denom = float(np.dot(x, x)) + 1e-9
    slope = float(np.dot(x, yc) / denom)
    scale = max(float(np.std(y)), 1e-6)
    if slope / scale > threshold_ratio:
        return "rising"
    if slope / scale < -threshold_ratio:
        return "falling"
    return "stable"
