
# app/scripts/sample_producer.py
"""
Produce inference requests to Kafka using real rows from DATA.csv.

Usage (inside Compose):
    docker compose -f docker/compose.yml run --rm flower-server \
      python scripts/sample_producer.py --n 3

Env/config:
  - CSV_PATH (default: DATA_DIR/DATA.csv)
  - DATA_DIR (default: app_data)
  - KAFKA_BOOTSTRAP (inside containers set to kafka:9092 by compose)
  - TOPIC_INFER_REQUESTS (default: ai.infer.requests)
  - SITE_ID (default: "test")
"""

import os
import json
import uuid
import argparse
import random
from pathlib import Path

import pandas as pd
from confluent_kafka import Producer

def default_bootstrap():
    if os.getenv("KAFKA_BOOTSTRAP"):
        return os.getenv("KAFKA_BOOTSTRAP")
    if Path("/.dockerenv").exists():
        return "kafka:9092"
    return "localhost:29092"

KAFKA_BOOTSTRAP = default_bootstrap()
DATA_DIR = Path(os.getenv("DATA_DIR", "app_data"))
DEFAULT_CSV = DATA_DIR / "DATA.csv"
TOPIC = os.getenv("TOPIC_INFER_REQUESTS", "ai.infer.requests")

def load_feature_spec():
    spec_path = DATA_DIR / "feature_spec.json"
    if not spec_path.exists():
        raise FileNotFoundError(f"Missing {spec_path}. Run data_prep.py first.")
    with open(spec_path, "r", encoding="utf-8") as f:
        return json.load(f)

def coerce_value(v):
    if pd.isna(v):
        return None
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            pass
    return v

def select_features(row, feature_cols):
    out = {}
    for c in feature_cols:
        out[c] = coerce_value(row.get(c, None))
    return out

def pick_rows(df, n):
    if n <= 0 or len(df) == 0:
        return []
    if n >= len(df):
        return df.sample(n=len(df), random_state=42).to_dict(orient="records")
    # random sample without replacement
    return df.sample(n=n, random_state=random.randint(0, 10_000)).to_dict(orient="records")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1, help="Number of messages to send")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV (defaults to DATA_DIR/DATA.csv)")
    parser.add_argument("--site-id", type=str, default=os.getenv("SITE_ID", "test"))
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else Path(os.getenv("CSV_PATH", str(DEFAULT_CSV)))
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}. Set --csv or CSV_PATH.")

    # Load feature spec and CSV
    spec = load_feature_spec()
    feat_cols = spec["feature_cols"]

    # Load only needed columns when possible (fallback to full and select)
    try:
        df = pd.read_csv(csv_path, usecols=lambda c: (c in feat_cols))
    except Exception:
        df = pd.read_csv(csv_path)

    rows = pick_rows(df, args.n)

    p = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP})
    sent = 0
    for r in rows:
        features = select_features(r, feat_cols)
        payload = {
            "patient_id": str(uuid.uuid4()),
            "site_id": args.site_id,
            "features": features,
        }
        p.produce(TOPIC, json.dumps(payload).encode("utf-8"))
        sent += 1

    p.flush()
    print(f"Produced {sent} message(s) to {TOPIC} @ {KAFKA_BOOTSTRAP}")
    if sent and len(rows) > 0:
        # Preview a truncated feature dict
        sample = select_features(rows[0], feat_cols)
        print("Sample payload:", json.dumps({
            "patient_id": "...",
            "site_id": args.site_id,
            "features": {k: sample[k] for k in list(sample)[:min(8, len(sample))]}
        }, ensure_ascii=False))

if __name__ == "__main__":
    main()
