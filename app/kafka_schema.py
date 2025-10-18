import json
from dataclasses import dataclass, asdict
from collections import deque
from pathlib import Path
from typing import Dict, Any
from config import PERCENTILE_WINDOW, DATA_DIR

class RollingPercentiles:
    def __init__(self, window=PERCENTILE_WINDOW, path: Path | None = None):
        self.window = window
        # Persist inside DATA_DIR by default so it survives container restarts
        self.path = path if path is not None else (DATA_DIR / "percentiles.json")
        self.buf = deque(maxlen=window)
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text())
                self.buf = deque(raw.get("values", []), maxlen=self.window)
            except Exception:
                self.buf = deque(maxlen=self.window)

    def _save(self):
        try:
            self.path.write_text(json.dumps({"values": list(self.buf)}))
        except Exception:
            pass

    def update(self, value: float):
        self.buf.append(float(value))
        self._save()

    def percentile(self, q: float) -> float:
        if not self.buf:
            return 0.0
        arr = sorted(self.buf)
        k = int(round((q / 100.0) * (len(arr) - 1)))
        return float(arr[k])

def risk_tier(chd_prob: float, roll: RollingPercentiles) -> str:
    p75 = roll.percentile(75)
    p50 = roll.percentile(50)
    p25 = roll.percentile(25)
    if chd_prob >= p75: return "P1"
    if chd_prob >= p50: return "P2"
    if chd_prob >= p25: return "P3"
    return "P4"

from dataclasses import dataclass

@dataclass
class InferRequest:
    patient_id: str
    site_id: str
    features: Dict[str, Any]

@dataclass
class RiskClassification:
    patient_id: str
    site_id: str
    probs: Dict[str, float]  # {"chd": 0.37, "cvd": 0.52, ...}
    risk_group: str          # derived from chd prob by default
    model_version: str

def to_json(obj) -> bytes:
    return json.dumps(asdict(obj)).encode("utf-8")
