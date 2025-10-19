import json
import traceback
from pathlib import Path

import pandas as pd
from joblib import load
from tensorflow.keras.models import load_model
from confluent_kafka import Consumer, Producer

from kafka_schema import (
    InferRequest,
    RiskClassification,
    to_json,
    MultiRollingPercentiles,
    risk_tiers,
)
from config import (
    DATA_DIR,
    MODELS_DIR,
    KAFKA_BOOTSTRAP,
    TOPIC_INFER_REQUESTS,
    TOPIC_RISK_CLASSIFICATIONS,
    HEADS,
)

def build_transformer():
    # Load the fitted preprocessor and the feature column spec
    pre = load(DATA_DIR / "preprocess.joblib")
    spec = json.loads((DATA_DIR / "feature_spec.json").read_text())
    feat_cols = spec["feature_cols"]
    return pre, feat_cols

def load_any_model():
    # Load the latest global model from disk.
    # Supports both single-task (CHD) and multitask checkpoints, in .keras or .h5.
    candidates = [
        "global_multitask_model.keras",
        "global_multitask_model.h5",
        "global_chd_model.keras",
        "global_chd_model.h5",
    ]
    for name in candidates:
        p = MODELS_DIR / name
        if p.exists():
            return load_model(p), name
    tried = ", ".join(candidates)
    raise FileNotFoundError(f"No model found in {MODELS_DIR}")

def main():
    pre, feat_cols = build_transformer()
    model, model_name = load_any_model()
    model_version = f"loaded:{model_name}"

    roll = MultiRollingPercentiles(heads=HEADS)

    consumer = Consumer(
        {
            "bootstrap.servers": KAFKA_BOOTSTRAP,
            "group.id": "risk-infer",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
        }
    )
    consumer.subscribe([TOPIC_INFER_REQUESTS])
    producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP})

    print(
        f"Kafka inference service started. "
        f"model={model_name} | bootstrap={KAFKA_BOOTSTRAP} | in={TOPIC_INFER_REQUESTS} | out={TOPIC_RISK_CLASSIFICATIONS}"
    )

    while True:
        try:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print(f"Consumer error: {msg.error()}")
                continue

            payload = json.loads(msg.value().decode("utf-8"))
            req = InferRequest(**payload)

            # Build a one-row DataFrame with exact feature columns
            row = {k: payload.get("features", {}).get(k, None) for k in feat_cols}
            X = pre.transform(pd.DataFrame([row], columns=feat_cols))

            # multitask models often return a dict of heads, single-head returns a 2D array
            yhat = model.predict(X, verbose=0)

            if isinstance(yhat, dict):
                probs = {h: float(yhat[h][0][0]) for h in HEADS if h in yhat}
            else:
                probs = {"chd": float(yhat[0][0])}

            # update rolling percentiles for ALL heads, compute per-disease and overall tiers
            roll.update_all(probs)
            overall_tier, per_head_tiers = risk_tiers(probs, roll)

            out = RiskClassification(
                patient_id=req.patient_id,
                site_id=req.site_id,
                probs=probs,
                risk_group=overall_tier, # overall 'worst' tier across diseases
                model_version=model_version,
                disease_groups=per_head_tiers # per-disease tiers
            )
            producer.produce(TOPIC_RISK_CLASSIFICATIONS, value=to_json(out))
            producer.flush()

        except Exception as e:
            print("Inference error:\n" + "".join(traceback.format_exception(e)))

if __name__ == "__main__":
    main()
