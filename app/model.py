# multi-task, multi-head
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import HEADS, LEARNING_RATE

def build_model(input_dim: int, lr: float = LEARNING_RATE) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,), name="features")
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(64, activation="relu")(x)

    # One sigmoid head per target
    outputs = {
        "chd":  layers.Dense(1, activation="sigmoid", name="chd")(x),
        "cvd":  layers.Dense(1, activation="sigmoid", name="cvd")(x),
        "ckd":  layers.Dense(1, activation="sigmoid", name="ckd")(x),
        "dm":   layers.Dense(1, activation="sigmoid", name="dm")(x),
        "hmod": layers.Dense(1, activation="sigmoid", name="hmod")(x),
    }

    model = keras.Model(inputs=inputs, outputs=outputs, name="tabular_multitask_dnn")

    # Per-head loss and metrics
    losses  = {h: "binary_crossentropy" for h in HEADS}
    metrics = {
        h: [
            keras.metrics.AUC(name=f"{h}_auc"),
            keras.metrics.Precision(name=f"{h}_prec"),
            keras.metrics.Recall(name=f"{h}_rec"),
            keras.metrics.BinaryAccuracy(name=f"{h}_acc"),
        ]
        for h in HEADS
    }

    # emphasize CHD
    loss_weights = {"chd": 1.0, "cvd": 0.8, "ckd": 0.8, "dm": 0.8, "hmod": 0.8}

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics,
    )
    return model
