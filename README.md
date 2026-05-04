# Technical decision assistant (MVP)

Python MVP for anomaly detection, failure risk estimation, root-cause hints, and explainable maintenance decisions. Includes synthetic industrial sensor data, FastAPI endpoints, and a Streamlit chat shell.

## Setup

```bash
cd technical-assistant
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the API

From the `technical-assistant` directory:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Health check: `GET http://127.0.0.1:8000/health`

## Run the Streamlit UI

In a second terminal (same directory and virtualenv):

```bash
streamlit run streamlit_app/app.py
```

## Configuration

Optional environment variables (prefix `TA_`):

| Variable | Meaning |
| --- | --- |
| `TA_TRAINING_ROWS` | Synthetic history length for fitting models |
| `TA_FAILURE_HORIZON_STEPS` | Horizon for failure labels |
| `TA_ANOMALY_CONTAMINATION` | Isolation Forest contamination |
| `TA_DECISION_FAILURE_PROB_HIGH` | High-risk probability cutoff |
| `TA_LOG_DECISIONS` | `true`/`false` to toggle JSONL logging |

Logs append to `logs/decisions.jsonl`.

## Sample API calls

Replace the `readings` array with at least 64 samples (fields per `api/schemas.py`).

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @sample_payload.json
```

Generate `sample_payload.json` quickly with a short Python snippet:

```python
import json, pandas as pd
from data.synthetic import IndustrialSensorSimulator, SimulationParams

df = IndustrialSensorSimulator(SimulationParams(n_steps=400, seed=7)).generate_history().iloc[-64:]
readings = []
for _, row in df.iterrows():
    readings.append({
        "timestamp": row["timestamp"].isoformat(),
        "motor_current_a": float(row["motor_current_a"]),
        "vibration_rms": float(row["vibration_rms"]),
        "bearing_temp_c": float(row["bearing_temp_c"]),
        "inlet_pressure_bar": float(row["inlet_pressure_bar"]),
        "flow_rate_l_min": float(row["flow_rate_l_min"]),
        "valve_position_pct": float(row["valve_position_pct"]),
    })
json.dump({"asset_id": "curl-demo", "readings": readings}, open("sample_payload.json", "w"))
```

Then:

```bash
curl -X POST http://127.0.0.1:8000/analyze -H "Content-Type: application/json" -d @sample_payload.json
curl -X POST http://127.0.0.1:8000/decision -H "Content-Type: application/json" -d @sample_payload.json
curl -X POST http://127.0.0.1:8000/explain -H "Content-Type: application/json" -d @sample_payload.json
```

What-if requests reuse the same `readings` array plus a `deltas` map, for example:

```json
{
  "asset_id": "demo",
  "readings": [ ... 64+ samples ... ],
  "deltas": { "vibration_rms": 2.0, "bearing_temp_c": 4.0 }
}
```

POST that JSON to `http://127.0.0.1:8000/whatif`.

## Project layout

- `data/` — synthetic generator and CSV helpers  
- `services/` — preprocessing, features, ML, RCA, decisions, explanations  
- `api/` — FastAPI schemas and routes  
- `streamlit_app/` — demo UI  
- `utils/config.py` — centralized settings  
