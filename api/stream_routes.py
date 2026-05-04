"""Real-time anomaly streaming over WebSocket and NDJSON."""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from starlette.responses import StreamingResponse

from data.stream_simulator import IndustrialRealtimeSimulator
from services.assistant import TechnicalAssistant
from services.stream_buffer import RollingSensorBuffer
from utils.config import get_settings

router = APIRouter(tags=["streaming"])


def _parse_int(q: Optional[str], default: int, min_v: int = 1, max_v: Optional[int] = None) -> int:
    """Parse a positive integer from a query string with safe bounds."""

    if q is None or q == "":
        return default
    try:
        v = int(q)
    except ValueError:
        return default
    v = max(min_v, v)
    if max_v is not None:
        v = min(v, max_v)
    return v


def _simulator_for_stream(seed: int, drift_horizon: int, load_period: int) -> IndustrialRealtimeSimulator:
    """Build a realtime simulator with streaming-specific defaults."""

    from data.synthetic import SimulationParams

    params = SimulationParams(seed=seed)
    return IndustrialRealtimeSimulator(
        params=params,
        drift_horizon=drift_horizon,
        load_period_steps=load_period,
    )


async def _anomaly_stream_events(
    assistant: TechnicalAssistant,
    tick_ms: int,
    max_ticks: int,
    seed: int,
    drift_horizon: int,
    load_period: int,
) -> AsyncIterator[str]:
    """Yield NDJSON lines: buffering progress, then per-tick anomaly scores."""

    tick_s = max(0.005, tick_ms / 1000.0)
    sim = _simulator_for_stream(seed, drift_horizon, load_period)
    buf = RollingSensorBuffer(assistant.window_size)
    for idx in range(1, max_ticks + 1):
        row = sim.step()
        buf.append(row)
        await asyncio.sleep(tick_s)
        if not buf.is_ready():
            payload: Dict[str, Any] = {
                "type": "buffering",
                "tick": idx,
                "buffered": len(buf),
                "window_size": assistant.window_size,
            }
            yield json.dumps(payload, default=str) + "\n"
            continue
        df = buf.to_dataframe()
        df_ml = df.drop(columns=["sim_anomaly_flag"], errors="ignore")
        preds = assistant.predict_window(df_ml)
        ts = row["timestamp"]
        ts_out = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        payload = {
            "type": "scores",
            "tick": idx,
            "timestamp": ts_out,
            "anomaly_score": preds.anomaly_score,
            "anomaly_flag": preds.anomaly_flag,
            "failure_probability": preds.failure_probability,
            "isolation_label": preds.isolation_label,
            "sim_injected_anomaly": int(row.get("sim_anomaly_flag", 0)),
        }
        yield json.dumps(payload, default=str) + "\n"


@router.get("/stream/anomaly-ndjson")
async def stream_anomaly_ndjson(
    request: Request,
    tick_ms: Optional[int] = None,
    max_ticks: Optional[int] = None,
    seed: Optional[int] = None,
) -> StreamingResponse:
    """Simulate a live sensor feed and emit one JSON object per line (NDJSON)."""

    settings = get_settings()
    cap = settings.stream_max_ticks_cap
    t_ms = tick_ms if tick_ms is not None else settings.stream_tick_ms_default
    t_ms = max(5, min(t_ms, 60_000))
    m_ticks = max_ticks if max_ticks is not None else settings.stream_default_max_ticks
    m_ticks = max(1, min(m_ticks, cap))
    rng_seed = seed if seed is not None else settings.simulation_seed + 1337
    assistant: TechnicalAssistant = request.app.state.assistant

    async def body() -> AsyncIterator[bytes]:
        """Encode NDJSON chunks for the streaming response."""

        async for line in _anomaly_stream_events(
            assistant,
            tick_ms=t_ms,
            max_ticks=m_ticks,
            seed=int(rng_seed),
            drift_horizon=settings.stream_drift_horizon_steps,
            load_period=settings.stream_load_period_steps,
        ):
            yield line.encode("utf-8")

    return StreamingResponse(body(), media_type="application/x-ndjson")


@router.websocket("/ws/anomaly-stream")
async def websocket_anomaly_stream(websocket: WebSocket) -> None:
    """Push simulated sensor ticks and anomaly scores over a WebSocket."""

    await websocket.accept()
    settings = get_settings()
    assistant: TechnicalAssistant = websocket.app.state.assistant
    tick_ms = _parse_int(websocket.query_params.get("tick_ms"), settings.stream_tick_ms_default, 5, 60_000)
    max_ticks = _parse_int(
        websocket.query_params.get("max_ticks"),
        settings.stream_default_max_ticks,
        1,
        settings.stream_max_ticks_cap,
    )
    seed_q = websocket.query_params.get("seed")
    seed = int(seed_q) if seed_q not in (None, "") else settings.simulation_seed + 1337

    try:
        async for line in _anomaly_stream_events(
            assistant,
            tick_ms=tick_ms,
            max_ticks=max_ticks,
            seed=seed,
            drift_horizon=settings.stream_drift_horizon_steps,
            load_period=settings.stream_load_period_steps,
        ):
            await websocket.send_text(line.strip())
    except WebSocketDisconnect:
        return
