# server_flower.py
import flwr as fl
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple

from flwr.common import parameters_to_ndarrays, Parameters
from model import build_model
from config import DATA_DIR, MODELS_DIR, HEADS, ROUNDS

def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    total = sum(n for n, _ in metrics) or 1
    keys = set().union(*[m.keys() for _, m in metrics]) if metrics else set()
    return {k: sum(n * m.get(k, 0.0) for n, m in metrics) / total for k in keys}

def _to_weights(parameters: Any) -> List[np.ndarray]:
    # Accept either a list of ndarrays (newer Flower) or a Parameters object (older)
    if isinstance(parameters, list):
        return parameters
    if isinstance(parameters, Parameters):
        return parameters_to_ndarrays(parameters)
    return list(parameters)

def _load_holdout():
    hold = np.load(DATA_DIR / "holdout.npz")
    Xh, Yh = hold["X"], hold["y"]
    y_dict = {h: Yh[:, i] for i, h in enumerate(HEADS)}
    return Xh, y_dict

class SaveFedAvg(fl.server.strategy.FedAvg):
    # FedAvg that saves the aggregated (global) model on the final round
    def __init__(self, input_dim: int, num_rounds: int, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_rounds = num_rounds

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            params, _ = aggregated  # params can be flwr.common.Parameters
            # Save a checkpoint on the final round
            if server_round == self.num_rounds:
                self._save_global(params, MODELS_DIR / "global_multitask_model.keras")
        return aggregated

    def _save_global(self, params: Any, path: Path) -> None:
        weights = _to_weights(params)
        model = build_model(input_dim=self.input_dim)
        model.set_weights(weights)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
        print(f"[server] Saved aggregated global model to: {path}")

def get_evaluate_fn(input_dim: int):
    Xh, y_dict = _load_holdout()

    def evaluate(server_round: int, parameters: Any, config: Dict[str, Any]):
        # Handle both list-of-ndarrays and Parameters
        weights = _to_weights(parameters)
        model = build_model(input_dim=input_dim)
        model.set_weights(weights)
        results = model.evaluate(Xh, y_dict, verbose=0, return_dict=True)
        loss = float(results["loss"])
        # Log the per-head AUC/ACC
        metrics = {k: float(v) for k, v in results.items() if k.endswith("_auc") or k.endswith("_acc")}
        return loss, metrics

    return evaluate

if __name__ == "__main__":
    # Determine input_dim from holdout
    Xh, _ = _load_holdout()
    input_dim = Xh.shape[1]

    strategy = SaveFedAvg(
        input_dim=input_dim,
        num_rounds=ROUNDS,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(input_dim),
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
    )
