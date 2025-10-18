# app/infer_kafka_service.py
import json
import pandas as pd
from joblib import load
from tensorflow.keras.models import load_model
from confluent_kafka import Consumer, Producer

from kafka_schema import InferRequest, RiskClassification, to_json, RollingPercentiles, risk_tier
from config import (DATA_DIR, MODELS_DIR, KAFKA_BOOTSTRAP,
                    TOPIC_INFER_REQUESTS, TOPIC_RISK_CLASSIFICATIONS, HEADS)

def build_transformer():
    pre = load(DATA_DIR / "preprocess.joblib")
    spec = json.loads((DATA_DIR / "feature_spec.json").read_text())
    feat_cols = spec["feature_cols"]
    return pre, feat_cols

def load_any_model():
    for name in ["global_multitask_model.keras", "global_multitask_model.h5",
                 "global_chd_model.keras", "global_chd_model.h5"]:
        p = MODELS_DIR / name
        if p.exists():
            return load_model(p), name
    raise FileNotFoundError(f"No model file in {MODELS_DIR} (.keras/.h5).")

def main():
    pre, feat_cols = build_transformer()
    model, model_name = load_any_model()
    model_version = f"loaded:{model_name}"
    roll = RollingPercentiles()

    consumer = Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "group.id": "risk-infer",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
    })
    consumer.subscribe([TOPIC_INFER_REQUESTS])
    producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP})

    print(f"Kafka inference service started. model={model_name} bootstrap={KAFKA_BOOTSTRAP}")
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            print(f"Consumer error: {msg.error()}")
            continue
        try:
            payload = json.loads(msg.value().decode("utf-8"))
            req = InferRequest(**payload)

            # exact column order DataFrame -> pre.transform
            row = {k: payload["features"].get(k, None) for k in feat_cols}
            X = pre.transform(pd.DataFrame([row], columns=feat_cols))

            yhat = model.predict(X, verbose=0)

            # Handle multi-task dict OR single-task array
            if isinstance(yhat, dict):
                probs = {h: float(yhat[h][0][0]) for h in HEADS if h in yhat}
            else:
                # fallback single-head -> assume CHD
                probs = {"chd": float(yhat[0][0])}

            chd_p = probs.get("chd", 0.0)
            roll.update(chd_p)
            tier = risk_tier(chd_p, roll)

            out = RiskClassification(
                patient_id=req.patient_id,
                site_id=req.site_id,
                probs=probs,
                risk_group=tier,
                model_version=model_version,
            )
            producer.produce(TOPIC_RISK_CLASSIFICATIONS, value=to_json(out))
            producer.flush()
        except Exception as e:
            # keep going, but show why we dropped a message
            print(f"Inference error: {e}")

if __name__ == "__main__":
    main()
