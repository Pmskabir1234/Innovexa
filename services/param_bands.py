"""Nominal operating bands for UX status (does not affect ML training)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class ParamBand:
    """Normal band and outer critical thresholds for one telemetry channel."""

    label: str
    unit: str
    normal: Tuple[float, float]
    critical_low: float
    critical_high: float


BANDS: Dict[str, ParamBand] = {
    "vibration_rms": ParamBand("Vibration (RMS)", "mm/s", (0.8, 6.8), 0.4, 10.5),
    "bearing_temp_c": ParamBand("Bearing temperature", "°C", (38.0, 72.0), 32.0, 88.0),
    "motor_current_a": ParamBand("Motor current", "A", (11.0, 28.0), 9.0, 34.0),
    "flow_rate_l_min": ParamBand("Flow rate", "L/min", (72.0, 205.0), 55.0, 230.0),
    "pressure_bar": ParamBand("Inlet pressure", "bar", (3.85, 5.45), 3.25, 6.1),
    "ambient_temp_c": ParamBand("Ambient temperature", "°C", (18.0, 32.0), 12.0, 40.0),
    "humidity_percent": ParamBand("Relative humidity", "%", (28.0, 62.0), 15.0, 82.0),
    "rpm": ParamBand("Shaft speed", "RPM", (1580.0, 1880.0), 1480.0, 1980.0),
    "voltage_v": ParamBand("Supply voltage", "V", (387.0, 418.0), 372.0, 435.0),
    "torque_nm": ParamBand("Drive torque", "N·m", (28.0, 105.0), 18.0, 125.0),
}


CHART_KEYS: Tuple[str, str, str] = ("vibration_rms", "bearing_temp_c", "motor_current_a")


def status_triplet(value: float, band: ParamBand) -> str:
    """Return normal / warning / critical."""

    n_lo, n_hi = band.normal
    if value <= band.critical_low or value >= band.critical_high:
        return "critical"
    if value < n_lo or value > n_hi:
        return "warning"
    return "normal"


def status_emoji(status: str) -> str:
    """Map status tier to a scanning cue."""

    if status == "critical":
        return "🔴"
    if status == "warning":
        return "⚠️"
    return "✅"
