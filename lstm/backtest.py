"""
backtest.py
===========
Simulate trading on the held-out TEST SET using the trained LSTM models.

Rules:
  Entry  : model probability > 0.60  (long only — upward move prediction)
  Exit strategy A  (take-profit / stop-loss):
    - Take Profit : +0.50% from entry
    - Stop Loss   : -0.30% from entry
    - Max hold    : horizon bars (fallback exit at close[entry + horizon])
  Exit strategy B  (horizon exit):
    - Exit at close[entry + horizon] (matches label construction)
    Both strategies are evaluated and compared.

Metrics reported:
  - Total return (%)
  - Win rate (%)
  - Max drawdown (%)
  - Profit factor
  - Sharpe ratio (approximate, daily returns)
  - Trade count

Usage:
  cd lstm/
  python backtest.py                    # backtest all 3 horizons
  python backtest.py --horizon 2
  python backtest.py --threshold 0.65   # stricter entry threshold
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from data_fetch import fetch_ohlcv, SYMBOL, TF_LOOKBACK_DAYS
from dataset    import build_sequences, time_split, scale_X_with_fitted, WINDOW, TRAIN_FRAC, VAL_FRAC
from features   import merge_timeframes, make_labels

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
MODELS_DIR = ROOT / "saved_models"

# ── Trading constants ─────────────────────────────────────────────────────────
FEE_RATE     = 0.0006    # 0.06% per side (Bitget maker ~0.02%, taker ~0.06%)
TAKE_PROFIT  = 0.0050    # +0.50%
STOP_LOSS    = -0.0030   # -0.30%
INIT_CAPITAL = 10_000.0  # USDT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _max_drawdown(capital_curve: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown in %."""
    peak = np.maximum.accumulate(capital_curve)
    dd   = (capital_curve - peak) / peak * 100.0
    return float(dd.min())


def _profit_factor(returns: list[float]) -> float:
    gains  = sum(r for r in returns if r > 0)
    losses = abs(sum(r for r in returns if r < 0))
    return gains / losses if losses > 0 else float("inf")


def _sharpe(returns: list[float], periods_per_year: int = 105_120) -> float:
    """
    Annualised Sharpe (risk-free = 0).
    periods_per_year ≈ 365 * 24 * 12  for 5-minute bars.
    """
    r = np.array(returns)
    if r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(periods_per_year))


# ---------------------------------------------------------------------------
# Core backtest engine
# ---------------------------------------------------------------------------

def _simulate(
    df_5m:       pd.DataFrame,
    probs:       np.ndarray,
    orig_idx:    np.ndarray,
    horizon:     int,
    entry_thr:   float,
    use_tp_sl:   bool = True,
) -> dict:
    """
    Simulate trades from model signals on test-set slice.

    Parameters
    ----------
    df_5m       : full 5m OHLCV DataFrame (original, unscaled)
    probs       : predicted probabilities for test set  shape (n_test,)
    orig_idx    : original row indices in df_5m for each test prediction
    horizon     : hold bars if no TP/SL hit
    entry_thr   : probability threshold to enter a trade
    use_tp_sl   : if True — exit at TP/SL; if False — exit at close[i+horizon]

    Returns dict with all metrics.
    """
    closes = df_5m["close"].values

    capital = INIT_CAPITAL
    capital_curve: list[float] = [INIT_CAPITAL]
    trade_returns:  list[float] = []
    trades: list[dict] = []

    for k, (prob, i) in enumerate(zip(probs, orig_idx)):
        if prob < entry_thr:
            continue

        # Entry price = open of next bar
        entry_bar = i + 1
        if entry_bar >= len(closes):
            continue
        entry_price = float(df_5m["open"].values[entry_bar])

        # Determine exit
        pnl_pct = None

        if use_tp_sl:
            # Walk forward bar-by-bar until TP/SL or horizon expiry
            for bar_offset in range(1, horizon + 1):
                bar = entry_bar + bar_offset
                if bar >= len(closes):
                    break
                high = float(df_5m["high"].values[bar])
                low  = float(df_5m["low"].values[bar])

                gross_tp = (1.0 + TAKE_PROFIT)
                gross_sl = (1.0 + STOP_LOSS)

                if high >= entry_price * gross_tp:
                    pnl_pct = TAKE_PROFIT - 2 * FEE_RATE
                    break
                if low  <= entry_price * gross_sl:
                    pnl_pct = STOP_LOSS  - 2 * FEE_RATE
                    break

            if pnl_pct is None:
                # Horizon expiry fallback
                exit_bar = entry_bar + horizon
                if exit_bar >= len(closes):
                    continue
                exit_price = float(closes[exit_bar])
                gross      = (exit_price - entry_price) / entry_price
                pnl_pct    = gross - 2 * FEE_RATE

        else:
            # Pure horizon exit
            exit_bar = entry_bar + horizon
            if exit_bar >= len(closes):
                continue
            exit_price = float(closes[exit_bar])
            gross      = (exit_price - entry_price) / entry_price
            pnl_pct    = gross - 2 * FEE_RATE

        # Book trade
        capital    *= (1.0 + pnl_pct)
        capital_curve.append(capital)
        trade_returns.append(pnl_pct * 100)
        trades.append({
            "entry_bar": int(entry_bar),
            "prob":      round(float(prob), 4),
            "pnl_pct":   round(pnl_pct * 100, 4),
            "capital":   round(capital, 2),
            "win":       pnl_pct > 0,
        })

    n      = len(trades)
    if n == 0:
        return {
            "trades": 0,
            "win_rate": 0.0, "total_return": 0.0,
            "max_drawdown": 0.0, "profit_factor": 0.0,
            "sharpe": 0.0,
        }

    arr    = np.array(capital_curve)
    wins   = sum(1 for t in trades if t["win"])
    tot_ret = (capital - INIT_CAPITAL) / INIT_CAPITAL * 100

    return {
        "trades":        n,
        "win_rate":      round(wins / n * 100, 2),
        "total_return":  round(tot_ret, 4),
        "max_drawdown":  round(_max_drawdown(arr), 4),
        "profit_factor": round(_profit_factor(trade_returns), 4),
        "sharpe":        round(_sharpe(trade_returns), 4),
        "final_capital": round(capital, 2),
        "trade_list":    trades,
    }


# ---------------------------------------------------------------------------
# Full backtest for one horizon
# ---------------------------------------------------------------------------

def backtest_horizon(
    df_5m:       pd.DataFrame,
    df_15m:      pd.DataFrame,
    horizon:     int,
    entry_thr:   float = 0.60,
) -> None:
    """Load saved model, predict on test set, run simulation."""
    import tensorflow as tf  # deferred import (slow)

    model_path  = MODELS_DIR / f"lstm_{SYMBOL}_{horizon}bar.keras"
    scaler_path = MODELS_DIR / f"scaler_{SYMBOL}_{horizon}bar.pkl"
    meta_path   = MODELS_DIR / f"meta_{SYMBOL}_{horizon}bar.json"

    for p in [model_path, scaler_path, meta_path]:
        if not p.exists():
            print(f"  [SKIP] {p.name} not found — run train.py first")
            return

    model  = tf.keras.models.load_model(str(model_path))
    scaler = joblib.load(scaler_path)

    print(f"\n{'='*62}")
    print(f"  BACKTEST  —  {SYMBOL}  horizon={horizon}  threshold={entry_thr:.0%}")
    print("="*62)

    # ── Rebuild test sequences with original indices ──────────────────────────
    feat_df  = merge_timeframes(df_5m, df_15m)
    label_s  = make_labels(df_5m, [horizon])[f"label_{horizon}"]
    X, y, orig_idx = build_sequences(feat_df, label_s, window=WINDOW, return_indices=True)

    # Recover test slice (same chronological split as train.py)
    n       = len(X)
    n_tr    = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)
    X_test  = X[n_tr + n_val:]
    idx_test = orig_idx[n_tr + n_val:]

    # Scale using fitted scaler (no refit!)
    X_test_sc = scale_X_with_fitted(X_test, scaler)

    # ── Predict ───────────────────────────────────────────────────────────────
    probs = model.predict(X_test_sc, batch_size=256, verbose=0).ravel()
    print(f"  Test samples  : {len(probs):,}")
    print(f"  Signal rate   : {(probs >= entry_thr).mean():.1%}")
    print()

    # ── Run both strategies ───────────────────────────────────────────────────
    for use_tpsl, label in [(True, "TP/SL exits"), (False, "Horizon exits")]:
        res = _simulate(df_5m, probs, idx_test, horizon, entry_thr, use_tpsl)
        print(f"  ── {label} ──────────────────────────────────────────")
        if res["trades"] == 0:
            print(f"  No trades at threshold {entry_thr:.0%}. Try a lower threshold.")
            continue
        print(f"  Trades        : {res['trades']}")
        print(f"  Win Rate      : {res['win_rate']:.2f}%")
        print(f"  Total Return  : {res['total_return']:+.4f}%")
        print(f"  Max Drawdown  : {res['max_drawdown']:.4f}%")
        print(f"  Profit Factor : {res['profit_factor']:.4f}")
        print(f"  Sharpe (ann.) : {res['sharpe']:.4f}")
        print(f"  Final Capital : {res['final_capital']:,.2f} USDT")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest LSTM models on held-out test set"
    )
    parser.add_argument("--horizon",   type=int,   default=None,
                        help="1|2|3 — default: all three")
    parser.add_argument("--threshold", type=float, default=0.60,
                        help="Entry probability threshold (default 0.60)")
    args = parser.parse_args()

    print(f"\n  {'='*62}")
    print(f"  LSTM BACKTEST  —  {SYMBOL}")
    print(f"  {'='*62}")

    df_5m  = fetch_ohlcv(SYMBOL, "5m",  days=TF_LOOKBACK_DAYS["5m"])
    df_15m = fetch_ohlcv(SYMBOL, "15m", days=TF_LOOKBACK_DAYS["15m"])

    horizons = [args.horizon] if args.horizon else [1, 2, 3]
    for h in horizons:
        try:
            backtest_horizon(df_5m, df_15m, h, entry_thr=args.threshold)
        except Exception as exc:
            print(f"  [ERROR] horizon={h}: {exc}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
