"""
model.py
========
LSTM model architecture for binary crypto move classification.

Architecture:
    Input  (window, n_features)
    → LSTM(128, return_sequences=True) + Dropout(0.3)
    → LSTM(64,  return_sequences=False) + Dropout(0.3)
    → Dense(32, relu) + BatchNormalization
    → Dense(1, sigmoid)

Loss     : binary_crossentropy
Optimizer: Adam
Metrics  : accuracy, AUC

Usage:
    from model import build_lstm_model
    model = build_lstm_model(n_timesteps=50, n_features=38)
    model.summary()
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_lstm_model(
    n_timesteps:    int,
    n_features:     int,
    lstm_units:     tuple[int, int] = (128, 64),
    dense_units:    int             = 32,
    dropout_rate:   float           = 0.30,
    recurrent_drop: float           = 0.10,
    learning_rate:  float           = 1e-3,
    l2_reg:         float           = 1e-4,
) -> keras.Model:
    """
    Build and compile a stacked LSTM binary classifier.

    Parameters
    ----------
    n_timesteps     : sequence length (window size, e.g. 50)
    n_features      : number of input features per timestep
    lstm_units      : units in each of the two LSTM layers
    dense_units     : units in the intermediate Dense layer
    dropout_rate    : dropout fraction after each LSTM layer
    recurrent_drop  : recurrent dropout inside LSTM cells
    learning_rate   : Adam learning rate
    l2_reg          : L2 regularisation on Dense weights

    Returns
    -------
    Compiled keras.Model
    """
    reg = keras.regularizers.l2(l2_reg)

    inp = keras.Input(shape=(n_timesteps, n_features), name="sequence")

    # ── First LSTM ─────────────────────────────────────────────────────────
    x = layers.LSTM(
        lstm_units[0],
        return_sequences=True,
        recurrent_dropout=recurrent_drop,
        kernel_regularizer=reg,
        name="lstm_1",
    )(inp)
    x = layers.Dropout(dropout_rate, name="drop_1")(x)

    # ── Second LSTM ────────────────────────────────────────────────────────
    x = layers.LSTM(
        lstm_units[1],
        return_sequences=False,
        recurrent_dropout=recurrent_drop,
        kernel_regularizer=reg,
        name="lstm_2",
    )(x)
    x = layers.Dropout(dropout_rate, name="drop_2")(x)

    # ── Feed-forward head ──────────────────────────────────────────────────
    x = layers.Dense(dense_units, activation="relu",
                     kernel_regularizer=reg, name="dense_1")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)

    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    # ── Compile ────────────────────────────────────────────────────────────
    model = keras.Model(inp, out, name="LSTM_Classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


# ---------------------------------------------------------------------------
# Callbacks factory
# ---------------------------------------------------------------------------

def make_callbacks(
    save_path:    str,
    patience:     int   = 10,
    min_delta:    float = 1e-4,
    reduce_lr_patience: int = 5,
) -> list[keras.callbacks.Callback]:
    """
    Standard callback set:
      - EarlyStopping (restore best weights)
      - ModelCheckpoint (save best val_loss)
      - ReduceLROnPlateau (halve LR when stuck)
    """
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=save_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


# ---------------------------------------------------------------------------
# Quick architecture preview
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    m = build_lstm_model(n_timesteps=50, n_features=38)
    m.summary(line_length=72)
    print(f"\n  Trainable params: {m.count_params():,}")
