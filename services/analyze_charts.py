"""Matplotlib charts for analyze UX (non-interactive PNG)."""

from __future__ import annotations

import base64
import io
from typing import List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from services.param_bands import BANDS, CHART_KEYS


def collect_highlight_indices(df: pd.DataFrame, model_anomaly_last: bool) -> List[int]:
    """Merge simulator flags, model anomaly on latest sample, and vibration spikes."""

    idx_set: set[int] = set()
    if "anomaly_flag" in df.columns:
        for pos in range(len(df)):
            try:
                if int(df["anomaly_flag"].iloc[pos]) == 1:
                    idx_set.add(pos)
            except (ValueError, TypeError):
                continue
    if model_anomaly_last:
        idx_set.add(len(df) - 1)
    if "vibration_rms" in df.columns and len(df) > 3:
        v = df["vibration_rms"].astype(float).values
        mu = float(np.mean(v))
        sigma = float(np.std(v)) + 1e-9
        for pos in range(len(v)):
            if abs(v[pos] - mu) / sigma > 2.4:
                idx_set.add(pos)
    return sorted(idx_set)


def build_analyze_chart_png_base64(df: pd.DataFrame, highlight_indices: Sequence[int]) -> str:
    """Render three telemetry strips with nominal bands and anomaly markers."""

    if df.empty or len(df) < 2:
        return ""
    x_arr = np.arange(len(df))
    highlights = sorted(set(int(i) for i in highlight_indices if 0 <= int(i) < len(df)))
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.2), sharex=True, constrained_layout=True)
    labeled_anomaly = False

    for ax, key in zip(axes, CHART_KEYS):
        band = BANDS.get(key)
        series = df[key].astype(float) if key in df.columns else None
        if series is None or band is None:
            ax.set_visible(False)
            continue
        ax.plot(x_arr, series.values, color="#1f4e79", linewidth=1.4, label="Observed")
        n_lo, n_hi = band.normal
        ax.axhspan(n_lo, n_hi, color="#c6efce", alpha=0.35, label="Nominal band")
        ax.axhline(band.critical_low, color="#ffc7ce", linestyle="--", linewidth=0.9, alpha=0.9)
        ax.axhline(band.critical_high, color="#ffc7ce", linestyle="--", linewidth=0.9, alpha=0.9)
        for hi in highlights:
            lbl = "Anomaly cue" if not labeled_anomaly else ""
            ax.scatter(
                x_arr[hi],
                float(series.iloc[hi]),
                color="#c00000",
                s=42,
                zorder=5,
                label=lbl,
            )
            labeled_anomaly = True
        ax.set_ylabel(f"{band.label}\n({band.unit})")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Sample index (recent window, oldest→newest)")
    axes[0].set_title("Key parameters vs nominal envelopes")
    handles, labels = axes[0].get_legend_handles_labels()
    if labels:
        by_label = dict(zip(labels, handles))
        axes[0].legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")
