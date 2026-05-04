"""Streamlit chat-style UI that drives the FastAPI assistant."""

from __future__ import annotations

import sys
from pathlib import Path

# Running `python streamlit_app\app.py` puts `streamlit_app` on sys.path, not the repo root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import base64
from io import BytesIO
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
import streamlit as st

from data.synthetic import IndustrialSensorSimulator, SimulationParams


def _build_payload(df: pd.DataFrame, asset_id: str) -> Dict[str, Any]:
    """Serialize the latest sensor window for HTTP transport."""

    optional_cols = (
        "ambient_temp_c",
        "humidity_percent",
        "rpm",
        "voltage_v",
        "torque_nm",
    )
    readings: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        ts = row["timestamp"]
        iso = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        reading: Dict[str, Any] = {
            "timestamp": iso,
            "motor_current_a": float(row["motor_current_a"]),
            "vibration_rms": float(row["vibration_rms"]),
            "bearing_temp_c": float(row["bearing_temp_c"]),
            "inlet_pressure_bar": float(row["inlet_pressure_bar"]),
            "flow_rate_l_min": float(row["flow_rate_l_min"]),
            "valve_position_pct": float(row["valve_position_pct"]),
        }
        for col in optional_cols:
            if col in row.index and pd.notna(row[col]):
                reading[col] = float(row[col])
        readings.append(reading)
    return {"asset_id": asset_id, "readings": readings}


def _render_analyze_response(caption: str, data_out: Dict[str, Any]) -> None:
    """Primary narrative, matplotlib artifact, parameter grid, then collapsible JSON."""

    if caption:
        st.markdown(caption)
    summary = data_out.get("human_readable_summary")
    if summary:
        st.markdown(summary)
    if data_out.get("chart_image_png_base64"):
        try:
            raw = base64.standard_b64decode(data_out["chart_image_png_base64"])
            st.image(BytesIO(raw), caption="Key parameters vs nominal bands (red = anomaly cues)")
        except (ValueError, TypeError):
            st.warning("Could not decode chart payload.")
    if data_out.get("parameter_table"):
        st.subheader("Parameter snapshot")
        pt = pd.DataFrame(data_out["parameter_table"])
        display_cols = [c for c in ("status_emoji", "label", "current_value", "unit", "status") if c in pt.columns]
        if display_cols:
            st.dataframe(pt[display_cols], use_container_width=True, hide_index=True)
    with st.expander("Advanced view (full JSON)"):
        st.json(data_out)


def _render_whatif_response(caption: str, data_out: Dict[str, Any]) -> None:
    """Narrative deltas plus structured comparison."""

    if caption:
        st.markdown(caption)
    summary = data_out.get("human_readable_summary")
    if summary:
        st.markdown(summary)
    if data_out.get("baseline") and data_out.get("scenario"):
        st.subheader("Structured comparison")
        st.json(
            {
                "baseline": data_out["baseline"],
                "scenario": data_out["scenario"],
                "applied_deltas": data_out.get("applied_deltas"),
            }
        )
    with st.expander("Advanced view (full JSON)"):
        st.json(data_out)


def _sample_window(seed: int) -> pd.DataFrame:
    """Create a deterministic demo slice from the simulator."""

    sim = IndustrialSensorSimulator(SimulationParams(n_steps=400, seed=seed))
    frame = sim.generate_history()
    return frame.iloc[-64:].copy()


def _post_json(base_url: str, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """POST JSON to the assistant API and return the response body."""

    url = base_url.rstrip("/") + path
    with httpx.Client(timeout=30.0) as client:
        response = client.post(url, json=body)
        response.raise_for_status()
        return response.json()


def main() -> None:
    """Render the chat interface and dispatch user intents."""

    st.set_page_config(page_title="Technical Assistant", layout="wide")
    st.title("Smart Engineering Assistant(demo)")
    st.caption("holaaaaaa")

    api_base = st.sidebar.text_input("API base URL", value="http://127.0.0.1:8000")
    seed = st.sidebar.number_input("Simulation seed", value=42, step=1)
    st.sidebar.markdown("### What-if deltas (additive)")
    delta_vibration = st.sidebar.number_input("vibration_rms Δ", value=0.0, format="%.3f")
    delta_temp = st.sidebar.number_input("bearing_temp_c Δ", value=0.0, format="%.3f")
    delta_flow = st.sidebar.number_input("flow_rate_l_min Δ", value=0.0, format="%.3f")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and isinstance(message.get("data"), dict):
                data = message["data"]
                if "prediction" in data and "parameter_table" in data:
                    _render_analyze_response(message["content"], data)
                elif "baseline" in data and "scenario" in data:
                    _render_whatif_response(message["content"], data)
                else:
                    st.markdown(message["content"])
                    st.json(message["data"])
            else:
                st.markdown(message["content"])
                if message.get("data") is not None:
                    st.json(message["data"])

    prompt = st.chat_input("Try: analyze, decision, explain, predict, whatif, help")
    if not prompt:
        return

    user_text = prompt.strip()
    st.session_state.messages.append({"role": "user", "content": user_text})

    if user_text.lower() in {"help", "?", "commands"}:
        help_text = (
            "Commands: **analyze** (narrative + parameter table + chart + RCA), **decision** (action), "
            "**explain** (feature drivers), **predict** (raw ML outputs), "
            "**whatif** (sidebar deltas + narrative summary)."
        )
        st.session_state.messages.append({"role": "assistant", "content": help_text, "data": None})
        st.rerun()

    df = _sample_window(int(seed))
    payload = _build_payload(df, asset_id="streamlit-demo")

    assistant_reply = ""
    data_out: Optional[Dict[str, Any]] = None
    lowered = user_text.lower()
    try:
        if "whatif" in lowered:
            deltas = {
                "vibration_rms": float(delta_vibration),
                "bearing_temp_c": float(delta_temp),
                "flow_rate_l_min": float(delta_flow),
            }
            body = {**payload, "deltas": deltas}
            data_out = _post_json(api_base, "/whatif", body)
            assistant_reply = "Baseline vs scenario after applying sidebar deltas."
        elif "decision" in lowered:
            data_out = _post_json(api_base, "/decision", payload)
            assistant_reply = "Maintenance decision with fused model + rule signals."
        elif "explain" in lowered:
            data_out = _post_json(api_base, "/explain", payload)
            assistant_reply = "Feature importances and anomaly deviations."
        elif "predict" in lowered:
            data_out = _post_json(api_base, "/predict", payload)
            assistant_reply = "Raw failure probability and anomaly score."
        else:
            data_out = _post_json(api_base, "/analyze", payload)
            assistant_reply = "Latest ML assessment plus structured RCA narrative."
    except Exception as exc:  # noqa: BLE001 - demo surface
        assistant_reply = f"API call failed: {exc}"
        data_out = None

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply, "data": data_out})
    st.rerun()


main()
