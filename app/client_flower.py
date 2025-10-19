import argparse
import numpy as np
import flwr as fl
import os
from tensorflow.keras.callbacks import EarlyStopping

from model import build_model
from config import DATA_DIR, MODELS_DIR, LEARNING_RATE, LOCAL_EPOCHS, BATCH_SIZE, HEADS

def compute_sample_weights_multi(Ym: np.ndarray):
    """Return sample_weight dict per head for imbalanced binary labels.
    For head h, positives get weight = n_neg / max(n_pos,1), negatives get 1.0.
    """
    sw = {}
    n = Ym.shape[0]
    for i, h in enumerate(HEADS):
        y = Ym[:, i].astype(int)
        n_pos = int((y == 1).sum())
        n_neg = n - n_pos
        if n_pos == 0 or n_neg == 0:
            # Single-class shard: uniform weights
            sw[h] = np.ones(n, dtype="float32")
        else:
            w_pos = n_neg / float(n_pos)
            w_neg = 1.0
            w = np.where(y == 1, w_pos, w_neg).astype("float32")
            sw[h] = w
    return sw

class MultiTaskClient(fl.client.NumPyClient):
    def __init__(self, site_id: int):
        data = np.load(DATA_DIR / f"site_{site_id}.npz")
        X = data["X"]
        Ym = data["y"]  # shape (N, 5)
        n = X.shape[0]
        # deterministic split (last 10% as val)
        n_val = max(1, int(0.1 * n))
        idx = np.arange(n)
        # you can shuffle once with a fixed seed if you prefer
        train_idx, val_idx = idx[:-n_val], idx[-n_val:]

        self.X_tr, self.X_val = X[train_idx], X[val_idx]
        self.Ym_tr, self.Ym_val = Ym[train_idx], Ym[val_idx]
        self.y_tr = {h: self.Ym_tr[:, i] for i, h in enumerate(HEADS)}
        self.y_val = {h: self.Ym_val[:, i] for i, h in enumerate(HEADS)}

        self.model = build_model(input_dim=X.shape[1], lr=LEARNING_RATE)
        self.site_id = site_id
        self.sample_weight = compute_sample_weights_multi(self.Ym_tr)

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        es = EarlyStopping(monitor="val_chd_auc", mode="max", patience=2, restore_best_weights=True)
        history = self.model.fit(
            self.X_tr, self.y_tr,
            epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(self.X_val, self.y_val),
            verbose=0,
            sample_weight=self.sample_weight,
            callbacks=[es],
        )
        val_chd_auc = float(max(history.history.get("val_chd_auc", [0.0])))
        return self.model.get_weights(), len(self.X_tr), {"val_chd_auc": val_chd_auc}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        results = self.model.evaluate(self.X_val, self.y_val, verbose=0, return_dict=True)
        metrics = {k: float(v) for k, v in results.items() if k.endswith("_auc") or k.endswith("_acc")}
        loss = float(results["loss"])
        return loss, len(self.X_val), metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--site_id", type=int, required=True)
    args = parser.parse_args()

    client = MultiTaskClient(site_id=args.site_id)
    server_addr = os.getenv("SERVER_ADDRESS", "127.0.0.1:8080")
    fl.client.start_client(server_address=server_addr, client=client.to_client())
