"""
rule_scan.py  —  Pure rule-based backtest: ATR7 + MA10 slope/dist, no ML model.

Usage:
    python rule_scan.py --symbol NOMUSDT --timeframe 1m
    python rule_scan.py --symbol 4USDT   --timeframe 15m
    python rule_scan.py --symbol NOMUSDT --timeframe 1m 5m 15m
"""

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from ml_model import build_features, DATA_DIR, TARGET_PCT, FEE_RATE

HOLD_BARS_LIST = [1, 2, 3]
MIN_TRADES     = 8           # ignore combos with fewer trades


# ---------------------------------------------------------------------------
def load_live_data(symbol: str, timeframe: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}_{timeframe}_live.csv"
    if not path.exists():
        raise FileNotFoundError(f"Nav live datu: {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def rule_backtest(df: pd.DataFrame, feat: pd.DataFrame,
                  hold_bars: int, target_pct: float, fee_rate: float,
                  atr_min: float, slope_min: float,
                  dist_min: float = 0.0,
                  direction: str = "both") -> dict:
    """
    Signal at bar[i]: ATR7% >= atr_min  AND  |ma10_slope%| >= slope_min
                      AND  |ma10_dist%| >= dist_min
    Long  if above_ma100==1 (or direction=='long')
    Short if above_ma100==0 (or direction=='short')
    Entry: open[i+1], Exit: close[i+hold_bars]
    Returns dict with trades, win_rate, total_return, pf, max_dd
    """
    c  = df["close"].values
    o  = df["open"].values
    n  = len(df)

    atr7_pct   = feat["atr7"].values * 100
    slope_pct  = feat["ma10_slope"].values * 100
    dist_pct   = feat["ma10_dist"].values * 100
    above_ma100 = feat["above_ma100"].values

    pnls = []
    for i in range(n - hold_bars - 1):
        # --- signal conditions ---
        if atr7_pct[i] < atr_min:
            continue
        if abs(slope_pct[i]) < slope_min:
            continue
        if abs(dist_pct[i]) < dist_min:
            continue

        # --- direction ---
        is_long  = (above_ma100[i] == 1)
        is_short = (above_ma100[i] == 0)
        if direction == "long"  and not is_long:
            continue
        if direction == "short" and not is_short:
            continue

        # --- slope direction aligned with trade direction ---
        # long  → want upward slope (slope > 0) OR allow any? test both.
        # For now: require slope direction to match trade direction.
        if is_long  and slope_pct[i] < 0:
            continue
        if is_short and slope_pct[i] > 0:
            continue

        entry_px = o[i + 1]
        exit_px  = c[i + hold_bars]

        if is_long:
            pnl = exit_px / entry_px - 1 - 2 * fee_rate
        else:
            pnl = entry_px / exit_px - 1 - 2 * fee_rate

        pnls.append(pnl)

    if not pnls:
        return {"trades": 0}

    arr = np.array(pnls)
    wins   = arr[arr > 0]
    losses = arr[arr <= 0]
    total_ret = arr.sum() * 100

    gross_win  = wins.sum()  if len(wins)  else 0.0
    gross_loss = abs(losses.sum()) if len(losses) else 1e-9
    pf = gross_win / gross_loss if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)

    # max drawdown (cumulative)
    equity = np.cumsum(arr)
    running_max = np.maximum.accumulate(equity)
    dd = (running_max - equity).max() * 100

    return {
        "trades":     len(arr),
        "win_rate":   len(wins) / len(arr) * 100,
        "total_ret":  total_ret,
        "pf":         pf,
        "max_dd":     dd,
        "avg_ret":    arr.mean() * 100,
    }


def scan_thresholds(symbol: str, timeframe: str):
    print(f"\n{'='*70}")
    print(f"  RULE SCAN  ---  {symbol}  {timeframe}  (bez ML modela)")
    print(f"  Signals: ATR7% >= X  AND  ma10_slope% >= Y (viena virzien ar MA100)")
    print(f"  Entry: open[i+1]  Exit: close[i+hold_bars]  Target: {TARGET_PCT*100:.1f}%")
    print(f"{'='*70}")

    df   = load_live_data(symbol, timeframe)
    feat = build_features(df)
    feat = feat.fillna(0)

    atr_vals   = feat["atr7"].values   * 100
    slope_vals = feat["ma10_slope"].values * 100

    # threshold grids: percentiles of absolute values
    atr_pcts   = [0, 25, 40, 50, 60, 70, 80]
    slope_pcts = [0, 25, 40, 50, 60, 70, 80]

    atr_thresholds   = [0.0] + [float(np.percentile(atr_vals[atr_vals > 0], p))
                                 for p in atr_pcts[1:]]
    slope_thresholds = [0.0] + [float(np.percentile(np.abs(slope_vals[slope_vals != 0]), p))
                                 for p in slope_pcts[1:]]

    # remove duplicates and sort
    atr_thresholds   = sorted(set(round(x, 4) for x in atr_thresholds))
    slope_thresholds = sorted(set(round(x, 4) for x in slope_thresholds))

    all_results = []

    for hb in HOLD_BARS_LIST:
        for atr_min, slope_min in product(atr_thresholds, slope_thresholds):
            r = rule_backtest(df, feat, hb, TARGET_PCT, FEE_RATE,
                              atr_min=atr_min, slope_min=slope_min)
            if r["trades"] < MIN_TRADES:
                continue
            all_results.append({
                "hb": hb, "atr_min": atr_min, "slope_min": slope_min,
                **r
            })

    if not all_results:
        print("  Nav nevienas kombinacijas ar pietiekami daudz darijumiem.")
        return

    # sort by PF descending
    all_results.sort(key=lambda x: x["pf"], reverse=True)

    print(f"\n  TOP 20 kombinacijas (no {len(all_results)} ar >={MIN_TRADES} trades):\n")
    print(f"  {'Bar':>3}  {'ATR%':>7}  {'Slp%':>7}  {'Tr':>5}  {'Win%':>6}  {'Ret%':>8}  {'DD%':>6}  {'PF':>6}")
    print(f"  {'-'*3}  {'-'*7}  {'-'*7}  {'-'*5}  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*6}")

    for r in all_results[:20]:
        sign = "+" if r["pf"] >= 1.0 else " "
        print(f"  {sign}{r['hb']:>2}b  {r['atr_min']:>7.4f}  {r['slope_min']:>7.4f}"
              f"  {r['trades']:>5}  {r['win_rate']:>5.1f}%"
              f"  {r['total_ret']:>+7.2f}%  {r['max_dd']:>6.2f}%  {r['pf']:>6.2f}")

    # --- per hold_bars best ---
    print(f"\n  LABAKAIS katram hold_bars:\n")
    for hb in HOLD_BARS_LIST:
        sub = [r for r in all_results if r["hb"] == hb]
        if not sub:
            print(f"  {hb}bar - nav")
            continue
        best = sub[0]
        print(f"  {hb}bar:  ATR>={best['atr_min']:.4f}%  |slope|>={best['slope_min']:.4f}%"
              f"  =>  {best['trades']} trades  {best['win_rate']:.1f}%  "
              f"{best['total_ret']:+.2f}%  PF={best['pf']:.2f}")

    # --- baseline (no filters) ---
    print(f"\n  BASELINE (bez filtriem, slope virziena filtrs):")
    for hb in HOLD_BARS_LIST:
        r0 = rule_backtest(df, feat, hb, TARGET_PCT, FEE_RATE,
                           atr_min=0, slope_min=0)
        if r0["trades"] == 0:
            print(f"  {hb}bar: nav darijumu")
        else:
            print(f"  {hb}bar:  {r0['trades']} trades  {r0['win_rate']:.1f}%  "
                  f"{r0['total_ret']:+.2f}%  PF={r0['pf']:.2f}  DD={r0['max_dd']:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",    required=True)
    parser.add_argument("--timeframe", nargs="+", default=["5m", "15m"])
    args = parser.parse_args()

    for tf in args.timeframe:
        try:
            scan_thresholds(args.symbol, tf)
        except FileNotFoundError as e:
            print(f"\n  [SKIP] {e}")


if __name__ == "__main__":
    main()
