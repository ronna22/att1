"""
run_all.py
==========
One-command pipeline: download → train (all 3 horizons) → backtest.

Usage:
  cd lstm/
  python run_all.py                     # full pipeline
  python run_all.py --force-download    # force re-download first
  python run_all.py --horizon 2         # single horizon only
  python run_all.py --threshold 0.65    # custom backtest threshold

Requires:
  pip install tensorflow pandas numpy scikit-learn joblib requests
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full LSTM pipeline")
    parser.add_argument("--horizon",        type=int,   default=None)
    parser.add_argument("--threshold",      type=float, default=0.60)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--verbose",        type=int,   default=1)
    args = parser.parse_args()

    t0 = time.time()
    print("\n" + "=" * 62)
    print("  LSTM PIPELINE  —  SIRENUSDT  (5m + 15m)")
    print("=" * 62)

    # ── Step 1: Download data ─────────────────────────────────────────────────
    print("\n  STEP 1 / 3  —  Data fetch")
    from data_fetch import fetch_ohlcv, SYMBOL, TF_LOOKBACK_DAYS
    df_5m  = fetch_ohlcv(SYMBOL, "5m",  days=TF_LOOKBACK_DAYS["5m"],
                          force_download=args.force_download)
    df_15m = fetch_ohlcv(SYMBOL, "15m", days=TF_LOOKBACK_DAYS["15m"],
                          force_download=args.force_download)

    # ── Step 2: Train ─────────────────────────────────────────────────────────
    print("\n  STEP 2 / 3  —  Training")
    from train import train_horizon, HORIZONS
    horizons = [args.horizon] if args.horizon else HORIZONS
    results  = []
    for h in horizons:
        try:
            r = train_horizon(df_5m, df_15m, h, verbose=args.verbose)
            results.append(r)
        except Exception as exc:
            print(f"  [ERROR] training horizon={h}: {exc}")

    # ── Step 3: Backtest ──────────────────────────────────────────────────────
    print("\n  STEP 3 / 3  —  Backtesting")
    from backtest import backtest_horizon
    for h in horizons:
        try:
            backtest_horizon(df_5m, df_15m, h, entry_thr=args.threshold)
        except Exception as exc:
            print(f"  [ERROR] backtest horizon={h}: {exc}")

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 62)
    print(f"  Done!  Total time: {elapsed/60:.1f} min")
    if results:
        print(f"\n  {'Hor':>4}  {'Acc':>7}  {'F1':>7}  {'Prec':>7}  {'Rec':>7}")
        print(f"  {'---':>4}  {'---':>7}  {'--':>7}  {'----':>7}  {'---':>7}")
        for r in results:
            print(
                f"  {r['horizon']:>4}  {r['accuracy']:>7.4f}  "
                f"{r['f1']:>7.4f}  {r['precision']:>7.4f}  {r['recall']:>7.4f}"
            )
    print("=" * 62)


if __name__ == "__main__":
    main()
