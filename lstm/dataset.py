"""
dataset.py
==========
Sequence construction, chronological train/val/test split, and scaling.

Public API:
    build_sequences(feat_df, label_s, window, return_indices)
        → X, y  [, orig_indices]
    time_split(X, y)
        → X_train, X_val, X_test, y_train, y_val, y_test
    fit_and_scale(X_train, X_val, X_test)
        → scaled versions + fitted MinMaxScaler

Design notes:
  - Scaler is ONLY fit on X_train → no future leakage
  - Rows with NaN (rolling warm-up OR label tail) are dropped before windowing
  - return_indices=True lets backtest.py look up original OHLCV prices
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Hyper-parameters (match model.py / train.py)
# ---------------------------------------------------------------------------
WINDOW     = 50     # look-back candles per sequence
TRAIN_FRAC = 0.70   # chronological train split
VAL_FRAC   = 0.15   # chronological validation split
# TEST_FRAC  = 0.15  (remainder)


# ---------------------------------------------------------------------------
# Sequence builder
# ---------------------------------------------------------------------------

def build_sequences(
    feature_df:   pd.DataFrame,
    label_series: pd.Series,
    window:       int  = WINDOW,
    return_indices: bool = False,
) -> tuple:
    """
    Sliding-window sequence construction.

    For each valid timestep i (after dropping NaN rows):
        X[k] = feature_df[i-window : i]   shape (window, n_features)
        y[k] = label_series[i]

    Parameters
    ----------
    feature_df     : DataFrame of features (NaN in leading rows expected)
    label_series   : Binary label Series (NaN in trailing rows expected)
    window         : Look-back window size (default 50)
    return_indices : If True, also return orig_indices (row numbers in
                     the original df_5m) for each prediction point.
                     Used by backtest.py to look up future OHLCV prices.

    Returns
    -------
    X  : np.ndarray  shape (n_samples, window, n_features)  float32
    y  : np.ndarray  shape (n_samples,)                     float32
    [orig_indices : np.ndarray  shape (n_samples,)  int]    optional
    """
    # Join features + label, drop any row with NaN in either
    combined = feature_df.copy()
    combined["__y__"] = label_series.values
    combined = combined.dropna()

    # Original row numbers in df_5m (RangeIndex — numeric positions)
    orig_idx = combined.index.to_numpy(dtype=np.intp)

    feat_vals  = combined.drop(columns="__y__").values.astype(np.float32)
    label_vals = combined["__y__"].values.astype(np.float32)

    n = len(feat_vals)
    if n <= window:
        raise ValueError(
            f"Not enough valid rows ({n}) for window size ({window}). "
            "Try reducing window or increasing data history."
        )

    X_list, y_list, idx_list = [], [], []
    for i in range(window, n):
        X_list.append(feat_vals[i - window : i])
        y_list.append(label_vals[i])
        idx_list.append(orig_idx[i])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    if return_indices:
        return X, y, np.array(idx_list, dtype=np.intp)
    return X, y


# ---------------------------------------------------------------------------
# Train / Validation / Test split (chronological — NO shuffle)
# ---------------------------------------------------------------------------

def time_split(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = TRAIN_FRAC,
    val_frac:   float = VAL_FRAC,
) -> tuple[np.ndarray, ...]:
    """
    Chronological split: Train → Validation → Test.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    n     = len(X)
    n_tr  = int(n * train_frac)
    n_val = int(n * val_frac)
    n_te  = n - n_tr - n_val

    splits = {
        "Train" : (X[:n_tr],                 y[:n_tr]),
        "Val"   : (X[n_tr : n_tr + n_val],   y[n_tr : n_tr + n_val]),
        "Test"  : (X[n_tr + n_val:],         y[n_tr + n_val:]),
    }

    print(f"  {'Split':<8}  {'Samples':>8}  {'Pct':>6}  {'Positives':>10}")
    print(f"  {'-----':<8}  {'-------':>8}  {'---':>6}  {'---------':>10}")
    for name, (Xs, ys) in splits.items():
        pos_rate = ys.mean() * 100 if len(ys) else 0.0
        print(
            f"  {name:<8}  {len(Xs):>8,}  "
            f"{len(Xs)/n*100:>5.1f}%  {pos_rate:>9.1f}%"
        )

    X_train, y_train = splits["Train"]
    X_val,   y_val   = splits["Val"]
    X_test,  y_test  = splits["Test"]

    return X_train, X_val, X_test, y_train, y_val, y_test


def split_indices(
    total:      int,
    train_frac: float = TRAIN_FRAC,
    val_frac:   float = VAL_FRAC,
) -> tuple[int, int, int]:
    """Return (n_train, n_val, n_test) counts for a dataset of size total."""
    n_tr  = int(total * train_frac)
    n_val = int(total * val_frac)
    return n_tr, n_val, total - n_tr - n_val


# ---------------------------------------------------------------------------
# Feature scaling  (MinMaxScaler fit on TRAIN ONLY)
# ---------------------------------------------------------------------------

def fit_and_scale(
    X_train: np.ndarray,
    X_val:   np.ndarray,
    X_test:  np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Fit MinMaxScaler(0, 1) on X_train, then transform all three splits.

    IMPORTANT: scaler is NEVER fit on val/test data to prevent leakage.

    Parameters
    ----------
    X_train, X_val, X_test : shape (n_samples, timesteps, features)

    Returns
    -------
    X_train_sc, X_val_sc, X_test_sc, fitted_scaler
    """
    n_tr, n_steps, n_feats = X_train.shape
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))

    # Reshape to 2D (samples*timesteps, features), fit on train only
    scaler.fit(X_train.reshape(-1, n_feats))

    def _transform(X: np.ndarray) -> np.ndarray:
        s, t, f = X.shape
        return scaler.transform(X.reshape(-1, f)).reshape(s, t, f).astype(np.float32)

    return _transform(X_train), _transform(X_val), _transform(X_test), scaler


def scale_X_with_fitted(X: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """Scale a new X array using an already-fitted scaler."""
    s, t, f = X.shape
    return scaler.transform(X.reshape(-1, f)).reshape(s, t, f).astype(np.float32)
