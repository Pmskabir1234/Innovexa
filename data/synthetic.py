"""Synthetic industrial multivariate time series for demos and training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd


@dataclass
class SimulationParams:
    """Parameters controlling synthetic signal generation."""

    n_steps: int = 8000
    freq: str = "1min"
    asset_id: str = "pump-01"
    seed: int = 42
    anomaly_probability: float = 0.015
    spike_scale: float = 3.5
    drift_scale: float = 0.002
    wear_increment_mu: float = 0.00015
    wear_increment_sigma: float = 0.00008
    failure_horizon_steps: int = 48
    wear_failure_threshold: float = 0.82


SENSOR_COLUMNS: List[str] = [
    "motor_current_a",
    "vibration_rms",
    "bearing_temp_c",
    "inlet_pressure_bar",
    "flow_rate_l_min",
    "valve_position_pct",
]


class IndustrialSensorSimulator:
    """Generate correlated industrial sensor streams with wear and anomalies."""

    def __init__(self, params: Optional[SimulationParams] = None) -> None:
        """Initialize simulator with optional parameter overrides."""

        self.params = params or SimulationParams()

    def generate_history(self) -> pd.DataFrame:
        """Build a full historical dataframe with labels for supervised training."""

        return _generate_frame(self.params)

    def stream_windows(
        self,
        window_size: int = 64,
        max_batches: Optional[int] = None,
    ) -> Iterator[pd.DataFrame]:
        """Yield contiguous windows of rows to mimic online ingestion."""

        df = _generate_frame(self.params)
        start = window_size
        batches = 0
        while start <= len(df):
            yield df.iloc[start - window_size : start].copy()
            start += max(1, window_size // 4)
            batches += 1
            if max_batches is not None and batches >= max_batches:
                break


def _generate_frame(params: SimulationParams) -> pd.DataFrame:
    """Assemble wear-driven dynamics, correlated noise, and anomaly injections."""

    rng = np.random.default_rng(params.seed)
    n = params.n_steps
    idx = pd.date_range("2024-01-01", periods=n, freq=params.freq)
    wear = np.zeros(n, dtype=float)
    for t in range(1, n):
        wear[t] = np.clip(
            wear[t - 1] + rng.normal(params.wear_increment_mu, params.wear_increment_sigma),
            0.0,
            1.0,
        )
    noise = rng.multivariate_normal(
        mean=np.zeros(6),
        cov=_sensor_covariance(),
        size=n,
    )
    load = 0.55 + 0.25 * np.sin(np.linspace(0, 12 * np.pi, n)) + rng.normal(0, 0.02, n)
    load = np.clip(load, 0.2, 1.0)
    vibration_base = 1.2 + 2.8 * load + 6.0 * wear**1.35
    current_base = 12.0 + 18.0 * load + 9.0 * wear
    temp_base = 38.0 + 22.0 * load + 28.0 * wear**1.2
    pressure_base = 4.5 + 0.8 * load - 0.4 * wear
    flow_base = 180.0 * load * (1.0 - 0.35 * wear)
    valve_base = 40.0 + 35.0 * load

    vibration = vibration_base + noise[:, 1] * 0.35
    current = current_base + noise[:, 0] * 0.45
    temp = temp_base + noise[:, 2] * 0.5
    pressure = pressure_base + noise[:, 3] * 0.06
    flow = flow_base + noise[:, 4] * 4.0
    valve = np.clip(valve_base + noise[:, 5] * 2.0, 5.0, 95.0)

    anomaly_flag = np.zeros(n, dtype=int)
    for t in range(n):
        if rng.random() < params.anomaly_probability:
            kind = rng.integers(0, 4)
            anomaly_flag[t] = 1
            if kind == 0:
                vibration[t] += params.spike_scale * rng.normal(2.5, 0.4)
                temp[t] += params.spike_scale * rng.normal(1.2, 0.2)
            elif kind == 1:
                current[t] *= 1.0 + params.spike_scale * 0.08
            elif kind == 2:
                flow[t] *= 0.65
                pressure[t] -= 0.35
            else:
                vibration[t] += params.drift_scale * (n - t)
                temp[t] += params.drift_scale * 0.8 * (n - t)

    data = {
        "timestamp": idx,
        "asset_id": params.asset_id,
        "motor_current_a": current,
        "vibration_rms": vibration,
        "bearing_temp_c": temp,
        "inlet_pressure_bar": pressure,
        "flow_rate_l_min": flow,
        "valve_position_pct": valve,
        "wear_index": wear,
        "anomaly_flag": anomaly_flag,
    }
    df = pd.DataFrame(data)
    failure_within_h = _failure_labels(wear, params.failure_horizon_steps, params.wear_failure_threshold)
    df["failure_within_h"] = failure_within_h
    return df


def _failure_labels(wear: np.ndarray, horizon: int, threshold: float) -> np.ndarray:
    """Mark timesteps whose near-future wear trajectory crosses the critical threshold."""

    n = len(wear)
    labels = np.zeros(n, dtype=int)
    for t in range(n):
        end = min(n, t + horizon + 1)
        future = wear[t + 1 : end]
        if future.size > 0 and np.max(future) >= threshold:
            labels[t] = 1
    return labels


def _sensor_covariance() -> np.ndarray:
    """Return a PSD covariance sketch so sensors co-move realistically."""

    base = np.array([0.8, 0.35, 0.25, 0.1, 0.2, 0.05], dtype=float)
    c = np.outer(base, base)
    np.fill_diagonal(c, np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    return c
