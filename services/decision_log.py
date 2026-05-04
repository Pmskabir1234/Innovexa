"""Append-only JSON logging for assistant decisions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def log_decision_record(log_path: Path, record: Dict[str, Any]) -> None:
    """Serialize a decision payload as one JSON line with UTC timestamp."""

    payload = {"logged_at_utc": datetime.now(timezone.utc).isoformat(), **record}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")
