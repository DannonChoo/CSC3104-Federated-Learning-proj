import json
from dataclasses import dataclass, asdict, field
from collections import deque
from pathlib import Path
from typing import Dict, Any, List, Tuple

from config import PERCENTILE_WINDOW, DATA_DIR, HEADS

# ============================
# Rolling percentiles (per head)
# ============================
class MultiRollingPercentiles:
    # Maintain a rolling window of prediction probabilities per disease head.
    # Persists to DATA_DIR/percentiles.json so it survives container restarts.
    def __init__(
        self,
        heads: List[str] | None = None,
        window: int = PERCENTILE_WINDOW,
        path: Path | None = None
    ):
        self.heads = heads or ["chd"]
        self.window = int(window)
        self.path = path if path is not None else (DATA_DIR / "percentiles.json")
        # map head -> deque
        self.buf: Dict[str, deque] = {h: deque(maxlen=self.window) for h in self.heads}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text())
                vals: Dict[str, List[float]] = raw.get("values", {})
                for h in self.heads:
                    self.buf[h].clear()
                    for v in vals.get(h, []):
                        try:
                            self.buf[h].append(float(v))
                        except Exception:
                            continue
            except Exception:
                # if corrupted, re-init empty deques
                self.buf = {h: deque(maxlen=self.window) for h in self.heads}

    def _save(self) -> None:
        try:
            serial = {h: list(self.buf[h]) for h in self.heads}
            self.path.write_text(json.dumps({"values": serial}))
        except Exception:
            # best-effort persistence; ignore file errors
            pass

    def update(self, head: str, value: float) -> None:
        if head not in self.buf:
            self.buf[head] = deque(maxlen=self.window)
        self.buf[head].append(float(value))
        self._save()

    # allow batch updating all heads from a probs dict
    def update_all(self, probs: Dict[str, float]) -> None:
        for h, v in probs.items():
            try:
                self.update(h, float(v))
            except Exception:
                continue

    def percentile(self, head: str, q: float) -> float:
        dq = self.buf.get(head)
        if not dq or len(dq) == 0:
            return 0.0
        arr = sorted(dq)
        # clamp q in [0,100]
        q = max(0.0, min(100.0, q))
        k = int(round((q / 100.0) * (len(arr) - 1)))
        return float(arr[k])

# ============================
# Tiering logic
# ============================
_TIER_ORDER = {"P1": 3, "P2": 2, "P3": 1, "P4": 0}   # P1 = highest risk

def tier_for_head(prob: float, roll: MultiRollingPercentiles, head: str) -> str:
    p75 = roll.percentile(head, 75)
    p50 = roll.percentile(head, 50)
    p25 = roll.percentile(head, 25)
    if prob >= p75: return "P1"
    if prob >= p50: return "P2"
    if prob >= p25: return "P3"
    return "P4"

# UPDATED: compute per-head tiers and an overall (worst) tier
def risk_tiers(probs: Dict[str, float], roll: MultiRollingPercentiles) -> Tuple[str, Dict[str, str]]:
    # Return (overall_tier, per_head_tiers).
    # Per-head tiers are computed vs each head's rolling percentiles.
    # Overall is the 'worst' (highest risk) tier across heads present in `probs`.
    per_head: Dict[str, str] = {}
    for h, p in probs.items():
        per_head[h] = tier_for_head(float(p), roll, h)

    # overall = worst (max) by tier order among available heads
    if per_head:
        overall = max(per_head.values(), key=lambda t: _TIER_ORDER.get(t, 0))
    else:
        overall = "P4"
    return overall, per_head

# ============================
# Message schemas
# ============================
@dataclass
class InferRequest:
    patient_id: str
    site_id: str
    features: Dict[str, Any]

# add disease_groups for per-disease tiers
@dataclass
class RiskClassification:
    patient_id: str
    site_id: str
    probs: Dict[str, float] # {"chd": 0.37, "cvd": 0.52, ...}
    risk_group: str # overall tier (worst across diseases)
    model_version: str
    disease_groups: Dict[str, str] = field(default_factory=dict) # {"chd":"P2","cvd":"P3",...}

def to_json(obj) -> bytes:
    return json.dumps(asdict(obj)).encode("utf-8")
