"""
train.py
========
End-to-end LSTM training pipeline for all three prediction horizons.

Pipeline per horizon:
  1. Merge 5m + 15m features
  2. Build sequences  (window=50)
  3. Chronological split  (70% / 15% / 15%)
  4. Fit MinMaxScaler on train only
  5. Train with EarlyStopping + ReduceLROnPlateau
  6. Evaluate: Accuracy / Precision / Recall / F1 / Confusion Matrix
  7. Save model (.keras) + scaler (.pkl) + metadata (.json)

Usage:
  cd lstm/
  python train.py                        # train all 3 horizons
  python train.py --horizon 1            # train horizon 1 only
  python train.py --force-download       # re-download data first
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from data_fetch import fetch_ohlcv, SYMBOL, TF_LOOKBACK_DAYS
from dataset    import build_sequences, time_split, fit_and_scale, WINDOW
from features   import merge_timeframes, make_labels
from model      import build_lstm_model, make_callbacks

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
MODELS_DIR = ROOT / "saved_models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Training hyper-parameters ─────────────────────────────────────────────────
HORIZONS         = [1, 2, 3]
BATCH_SIZE       = 32
MAX_EPOCHS       = 100
EARLY_STOP_PAT   = 10
THRESHOLD_PRED   = 0.50     # classification cut-off for metrics


# ---------------------------------------------------------------------------
# Single-horizon training
# ---------------------------------------------------------------------------

def train_horizon(
    df_5m:     pd.DataFrame,
    df_15m:    pd.DataFrame,
    horizon:   int,
    symbol:    str | None = None,
    verbose:   int = 1,
) -> dict:
    """
    Train one LSTM model for a given prediction horizon.

    Returns dict with evaluation metrics.
    """
    if symbol is None:
        symbol = SYMBOL
    divider = "=" * 62
    print(f"\n{divider}")
    print(f"  TRAINING  —  {symbol}  horizon={horizon}  (5m + 15m LSTM)")
    print(divider)

    # ── Features ─────────────────────────────────────────────────────────────
    feat_df  = merge_timeframes(df_5m, df_15m)
    label_df = make_labels(df_5m, horizons=[horizon], threshold=0.005)
    label_s  = label_df[f"label_{horizon}"]

    n_features = feat_df.shape[1]
    pos_rate   = label_s.dropna().mean()
    print(f"  Features   : {n_features}")
    print(f"  Label rate : {pos_rate:.1%}  (fraction of up-moves ≥ 0.5%)")

    # ── Sequences ─────────────────────────────────────────────────────────────
    X, y = build_sequences(feat_df, label_s, window=WINDOW)
    print(f"  Sequences  : X={X.shape}  y={y.shape}")

    # ── Split ─────────────────────────────────────────────────────────────────
    print()
    X_train, X_val, X_test, y_train, y_val, y_test = time_split(X, y)

    # ── Scale (scaler fit on train only) ──────────────────────────────────────
    X_train, X_val, X_test, scaler = fit_and_scale(X_train, X_val, X_test)

    # ── Class weights (compensate for imbalance) ──────────────────────────────
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    w_pos = n_neg / max(n_pos, 1)
    w_neg = 1.0
    class_weight = {0: w_neg, 1: w_pos}
    print(f"\n  Class weights  → 0: {w_neg:.2f},  1: {w_pos:.2f}")

    # ── Build model ───────────────────────────────────────────────────────────
    _, n_steps, n_feats = X_train.shape
    model = build_lstm_model(n_steps, n_feats)

    if verbose > 1:
        model.summary(line_length=70, print_fn=lambda s: print("  " + s))
    else:
        total_params = model.count_params()
        print(f"  Model params : {total_params:,}")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    save_path = str(MODELS_DIR / f"lstm_{symbol}_{horizon}bar.keras")
    callbacks = make_callbacks(
        save_path         = save_path,
        patience          = EARLY_STOP_PAT,
        reduce_lr_patience= 5,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n  Training (batch={BATCH_SIZE}, max_epochs={MAX_EPOCHS}) …")
    history = model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        epochs          = MAX_EPOCHS,
        batch_size      = BATCH_SIZE,
        class_weight    = class_weight,
        callbacks       = callbacks,
        verbose         = verbose,
    )

    epochs_run = len(history.history["loss"])
    best_val   = min(history.history["val_loss"])
    print(f"  Finished: {epochs_run} epochs  |  best val_loss={best_val:.5f}")

    # ── Test set evaluation ────────────────────────────────────────────────────
    print(f"\n  ── Test Set Evaluation ────────────────────────────────")
    y_pred_prob = model.predict(X_test, batch_size=256, verbose=0).ravel()
    y_pred      = (y_pred_prob >= THRESHOLD_PRED).astype(int)

    acc              = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    print(f"  Threshold  : {THRESHOLD_PRED}")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  Precision  : {prec:.4f}")
    print(f"  Recall     : {rec:.4f}")
    print(f"  F1-Score   : {f1:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]:6,}   FP={cm[0,1]:6,}")
    print(f"    FN={cm[1,0]:6,}   TP={cm[1,1]:6,}")
    print()
    print(classification_report(
        y_test, y_pred, digits=4,
        target_names=["no_move", "up≥0.5%"],
    ))

    # ── Save scaler and metadata ───────────────────────────────────────────────
    scaler_path = MODELS_DIR / f"scaler_{symbol}_{horizon}bar.pkl"
    joblib.dump(scaler, scaler_path)

    result = {
        "symbol":          symbol,
        "horizon":         horizon,
        "window":          WINDOW,
        "n_features":      n_feats,
        "n_timesteps":     n_steps,
        "threshold":       THRESHOLD_PRED,
        "epochs_run":      epochs_run,
        "best_val_loss":   round(best_val, 6),
        "accuracy":        round(float(acc),  4),
        "precision":       round(float(prec), 4),
        "recall":          round(float(rec),  4),
        "f1":              round(float(f1),   4),
        "confusion_matrix": cm.tolist(),
        "train_samples":   len(X_train),
        "val_samples":     len(X_val),
        "test_samples":    len(X_test),
    }

    meta_path = MODELS_DIR / f"meta_{symbol}_{horizon}bar.json"
    meta_path.write_text(json.dumps(result, indent=2))

    print(f"  Model  saved → {save_path}")
    print(f"  Scaler saved → {scaler_path.name}")
    print(f"  Meta   saved → {meta_path.name}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train LSTM models for SIRENUSDT 5m+15m"
    )
    parser.add_argument("--horizon",       type=int, default=None,
                        help="Train single horizon (1|2|3). Default: all three.")
    parser.add_argument("--force-download", action="store_true",
                        help="Re-download OHLCV data even if cached.")
    parser.add_argument("--verbose",       type=int, default=1,
                        choices=[0, 1, 2],
                        help="Keras verbose level (0=silent, 1=bar, 2=one-line)")
    args = parser.parse_args()

    # ── Download data ────────────────────────────────────────────────────────
    print(f"\n  {'='*62}")
    print(f"  LSTM TRAINING PIPELINE  —  {SYMBOL}")
    print(f"  {'='*62}")
    print(f"\n  Loading data …")

    df_5m  = fetch_ohlcv(SYMBOL, "5m",  days=TF_LOOKBACK_DAYS["5m"],
                          force_download=args.force_download)
    df_15m = fetch_ohlcv(SYMBOL, "15m", days=TF_LOOKBACK_DAYS["15m"],
                          force_download=args.force_download)

    horizons = [args.horizon] if args.horizon else HORIZONS

    # ── Train ────────────────────────────────────────────────────────────────
    results = []
    for h in horizons:
        try:
            r = train_horizon(df_5m, df_15m, horizon=h, verbose=args.verbose)
            results.append(r)
        except Exception as exc:
            print(f"\n  [ERROR] horizon={h}: {exc}")
            import traceback; traceback.print_exc()

    # ── Summary table ─────────────────────────────────────────────────────────
    if results:
        print(f"\n\n  {'='*62}")
        print(f"  SUMMARY  —  {SYMBOL}  (test set)")
        print(f"  {'='*62}")
        print(f"  {'Hor':>4}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'Epochs':>7}")
        print(f"  {'---':>4}  {'---':>7}  {'----':>7}  {'---':>7}  {'--':>7}  {'------':>7}")
        for r in results:
            print(
                f"  {r['horizon']:>4}  {r['accuracy']:>7.4f}  "
                f"{r['precision']:>7.4f}  {r['recall']:>7.4f}  "
                f"{r['f1']:>7.4f}  {r['epochs_run']:>7}"
            )
        print()


if __name__ == "__main__":
    main()
