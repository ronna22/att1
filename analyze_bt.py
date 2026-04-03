"""
Quick backtest analysis for the 3 active simulator configs.
Run from: python analyze_bt.py
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import json
from pathlib import Path

CONFIGS = [
    ("5m",  "xgb", 2, "5m-2bar"),
    ("5m",  "xgb", 3, "5m-3bar"),
    ("15m", "xgb", 2, "15m-2bar"),
]
base = Path("ml_results")


# ── Helper: compute ATR on candle df ─────────────────────────────────────────
def compute_atr(df: pd.DataFrame, period: int = 7) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ── Helper: load candle CSV, compute ATR, return indexed by rounded datetime ──
def load_candles_with_atr(symbol: str, timeframe: str, live: bool = True) -> pd.DataFrame:
    suffix = "_live" if live else ""
    # try current dir, then parent, then any subdir
    candidates = [
        Path(f"{symbol}_{timeframe}{suffix}.csv"),
        Path(f"data/{symbol}_{timeframe}{suffix}.csv"),
        *Path(".").rglob(f"{symbol}_{timeframe}{suffix}.csv"),
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"{symbol}_{timeframe}{suffix}.csv not found")
    df = pd.read_csv(path)
    # normalise column names
    df.columns = [c.lower() for c in df.columns]
    if "datetime" not in df.columns and "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    else:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    df["atr7"]  = compute_atr(df, 7)
    df["atr14"] = compute_atr(df, 14)
    # atr relative to price
    df["atr7_pct"]  = df["atr7"]  / df["close"] * 100
    df["atr14_pct"] = df["atr14"] / df["close"] * 100
    # roc10
    df["roc10"] = df["close"].pct_change(10) * 100   # in %
    # ma10_slope: 10-bar MA, pct change over 3 bars (same as build_features)
    ma10 = df["close"].rolling(10).mean()
    df["ma10_slope"] = ma10.pct_change(3) * 100       # in %
    df = df.set_index("datetime")
    return df


# ── Helper: ATR threshold scan ────────────────────────────────────────────────
def threshold_scan(trd: pd.DataFrame, candles: pd.DataFrame,
                   label: str, col: str,
                   signed: bool = False) -> None:
    """
    For each threshold, show trades kept, win%, total return, PF.
    signed=False → min threshold (ATR: higher = more volatile, better)
    signed=True  → split into positive / negative zones (roc10, ma10_slope)
    """
    trd = trd.copy()
    trd["entry_dt"] = pd.to_datetime(trd["entry_time"], utc=True)

    col_df = candles[[col]].reset_index().rename(columns={"datetime": "entry_dt"})
    trd = pd.merge_asof(
        trd.sort_values("entry_dt"),
        col_df.sort_values("entry_dt"),
        on="entry_dt", direction="nearest"
    )

    total = len(trd)
    base_wins = (trd["pnl_pct"] > 0).sum()
    base_ret  = trd["pnl_pct"].sum()

    def _pf(sub):
        gw = sub[sub["pnl_pct"] > 0]["pnl_pct"].sum()
        gl = abs(sub[sub["pnl_pct"] < 0]["pnl_pct"].sum())
        return gw / gl if gl > 0 else 99.0

    def _row(thresh_label, sub, marker=""):
        n = len(sub)
        if n < 5:
            return
        w   = (sub["pnl_pct"] > 0).sum()
        ret = sub["pnl_pct"].sum()
        avg = sub["pnl_pct"].mean()
        pf  = _pf(sub)
        flag = " <--" if (pf > 1.5 and w/n > 0.52) else ""
        print(f"  {thresh_label:>14}  {n:>5}  {w/n*100:>6.1f}%  "
              f"{ret:>+8.2f}%  {avg:>+7.3f}%  {total-n:>8}  {pf:>5.2f}{flag}")

    print(f"\n  {label}  —  '{col}' filter scan")
    print(f"  Baseline: {total} trades  win={base_wins/total*100:.1f}%  "
          f"total_ret={base_ret:+.2f}%")
    print()
    print(f"  {'Threshold':>14}  {'Kept':>5}  {'Win%':>6}  "
          f"{'TotRet':>8}  {'AvgRet':>7}  {'Filtered':>8}  {'PF':>5}")
    print(f"  {'-'*14}  {'-'*5}  {'-'*6}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*5}")

    if not signed:
        # simple min threshold
        percentiles = np.arange(10, 91, 10)
        thresholds  = sorted(set(np.percentile(trd[col].dropna(), percentiles)))
        for t in thresholds:
            _row(f">= {t:.4f}", trd[trd[col] >= t])
    else:
        # directional: positive zone vs negative zone
        # also show abs-value thresholds to find "strong momentum" trades
        print(f"  --- Positive zone (trend UP) ---")
        pos = trd[trd[col] > 0]
        if len(pos) >= 5:
            pcts = np.arange(10, 91, 20)
            for t in sorted(set(np.percentile(pos[col].dropna(), pcts))):
                _row(f">= +{t:.4f}", trd[trd[col] >= t])

        print(f"  --- Negative zone (trend DOWN) ---")
        neg = trd[trd[col] < 0]
        if len(neg) >= 5:
            pcts = np.arange(10, 91, 20)
            for t in sorted(set(np.percentile(neg[col].dropna(), pcts))):
                _row(f"<= {t:.4f}", trd[trd[col] <= t])

        print(f"  --- Abs strength filter ---")
        abs_col = col + "_abs"
        trd[abs_col] = trd[col].abs()
        pcts = np.arange(20, 81, 20)
        for t in sorted(set(np.percentile(trd[abs_col].dropna(), pcts))):
            _row(f"|v|>= {t:.4f}", trd[trd[abs_col] >= t])

    # Distribution: wins vs losses
    print()
    print(f"  '{col}' distribution  (wins vs losses):")
    w_vals = trd[trd["pnl_pct"] > 0][col]
    l_vals = trd[trd["pnl_pct"] < 0][col]
    for pct in [25, 50, 75]:
        print(f"    p{pct:02d}  wins={np.percentile(w_vals, pct):+.4f}  "
              f"losses={np.percentile(l_vals, pct):+.4f}")


# ── Helper: combined ATR + roc10 + ma10_slope grid scan ──────────────────────
BEST_THRESHOLDS = {
    "5m-2bar":  {"atr7_pct": [3.18, 4.14],  "roc10_abs": [2.13, 3.72],  "slope_abs": [0.80, 1.16]},
    "5m-3bar":  {"atr7_pct": [2.60, 3.80],  "roc10_abs": [1.05, 2.49],  "slope_abs": [0.56, 1.26]},
    "15m-2bar": {"atr7_pct": [3.67, 4.62],  "roc10_abs": [3.56, 6.81],  "slope_abs": [0.40, 0.74]},
}

def combined_scan(trd: pd.DataFrame, candles: pd.DataFrame, label: str) -> None:
    trd = trd.copy()
    trd["entry_dt"] = pd.to_datetime(trd["entry_time"], utc=True)

    cols = ["atr7_pct", "roc10", "ma10_slope"]
    col_df = candles[cols].reset_index().rename(columns={"datetime": "entry_dt"})
    trd = pd.merge_asof(
        trd.sort_values("entry_dt"),
        col_df.sort_values("entry_dt"),
        on="entry_dt", direction="nearest"
    )
    trd["roc10_abs"] = trd["roc10"].abs()
    trd["slope_abs"] = trd["ma10_slope"].abs()

    total     = len(trd)
    base_wins = (trd["pnl_pct"] > 0).sum()
    base_ret  = trd["pnl_pct"].sum()

    def _stats(sub):
        n = len(sub)
        if n < 3:
            return None
        w  = (sub["pnl_pct"] > 0).sum()
        rt = sub["pnl_pct"].sum()
        gw = sub[sub["pnl_pct"] > 0]["pnl_pct"].sum()
        gl = abs(sub[sub["pnl_pct"] < 0]["pnl_pct"].sum())
        pf = gw / gl if gl > 0 else 99.0
        return n, w, rt, rt / n, pf

    cfg        = BEST_THRESHOLDS.get(label, {})
    atr_vals   = [0.0] + cfg.get("atr7_pct",  [])
    roc_vals   = [0.0] + cfg.get("roc10_abs", [])
    slope_vals = [0.0] + cfg.get("slope_abs", [])

    print(f"\n  {label}  —  COMBINED  ATR7 + |roc10| + |ma10_slope|")
    print(f"  Baseline: {total} trades  win={base_wins/total*100:.1f}%  "
          f"ret={base_ret:+.2f}%")
    print()
    print(f"  {'ATR7>=':>6}  {'|roc10|>=':>9}  {'|slope|>=':>9}  "
          f"{'n':>4}  {'Win%':>6}  {'TotRet':>8}  {'AvgRet':>7}  {'PF':>5}  {'Drop':>5}")
    print(f"  {'-'*6}  {'-'*9}  {'-'*9}  "
          f"{'-'*4}  {'-'*6}  {'-'*8}  {'-'*7}  {'-'*5}  {'-'*5}")

    best_score = 0.0
    best_row   = None

    for atr_t in atr_vals:
        for roc_t in roc_vals:
            for slp_t in slope_vals:
                if atr_t == 0 and roc_t == 0 and slp_t == 0:
                    continue
                mask = (
                    (trd["atr7_pct"]  >= atr_t) &
                    (trd["roc10_abs"] >= roc_t) &
                    (trd["slope_abs"] >= slp_t)
                )
                sub = trd[mask]
                s = _stats(sub)
                if s is None:
                    continue
                n, w, rt, avg, pf = s
                drop_pct = (total - n) / total * 100
                score = pf * (w / n)
                flag = " <<"  if (pf > 2.0 and w/n > 0.55) else ""
                if score > best_score:
                    best_score = score
                    best_row = (atr_t, roc_t, slp_t, n, w/n*100, rt, avg, pf, drop_pct)
                print(f"  {atr_t:>6.2f}  {roc_t:>9.2f}  {slp_t:>9.2f}  "
                      f"{n:>4}  {w/n*100:>6.1f}%  {rt:>+8.2f}%  "
                      f"{avg:>+7.3f}%  {pf:>5.2f}  {drop_pct:>4.0f}%{flag}")

    if best_row:
        a, r, s, n, wr, rt, avg, pf, dp = best_row
        print()
        print(f"  *** LABAKAIS: atr7>={a:.2f}  |roc10|>={r:.2f}  |slope|>={s:.2f} ***")
        print(f"      {n} trades  win={wr:.1f}%  ret={rt:+.2f}%  avg={avg:+.3f}%  "
              f"PF={pf:.2f}  izfiltreti={dp:.0f}%")


# ── Feature importance comparison across configs ──────────────────────────────
all_fimp: dict = {}

for tf, mdl, nbar, label in CONFIGS:
    # Use _live_ files = out-of-sample Mar21-28 backtest
    rpt_path  = base / f"SIRENUSDT_{tf}_{mdl}_{nbar}bar_live_report.json"
    trd_path  = base / f"SIRENUSDT_{tf}_{mdl}_{nbar}bar_live_trades.csv"
    fimp_path = base / f"SIRENUSDT_{tf}_{mdl}_{nbar}bar_feature_importance.csv"

    rpt  = json.loads(rpt_path.read_text())
    trd  = pd.read_csv(trd_path)
    fimp_df = pd.read_csv(fimp_path)
    fimp = fimp_df.set_index("feature")["importance"]
    all_fimp[label] = fimp

    trd["win"] = (trd["pnl_pct"] > 0).astype(int)
    wins  = trd[trd["win"] == 1]
    loses = trd[trd["win"] == 0]

    print()
    print("=" * 62)
    print(f"  {label}   (out-of-sample Mar21-28)")
    print("=" * 62)
    print(f"  trades={rpt['trades']:4d}  win={rpt['win_rate_pct']:5.1f}%  "
          f"ret={rpt['total_return_pct']:+7.1f}%  "
          f"PF={rpt['profit_factor']:4.2f}  dd={rpt['max_drawdown_pct']:5.1f}%")
    print(f"  avg_win={rpt['avg_win_pct']:+.3f}%  avg_loss={rpt['avg_loss_pct']:+.3f}%")

    # Avg probability: wins vs losses
    print(f"  avg prob  wins={wins['prob'].mean():.3f}  losses={loses['prob'].mean():.3f}")

    # Breakdown by direction
    for d in ["long", "short"]:
        sub = trd[trd["direction"] == d]
        if len(sub) == 0:
            continue
        w = sub[sub["win"] == 1]
        print(f"  [{d.upper():5}] n={len(sub):4d}  win={len(w)/len(sub)*100:5.1f}%  "
              f"ret={sub['pnl_pct'].sum():+.2f}%  avg={sub['pnl_pct'].mean():+.3f}%")

    # Top 15 feature importances
    print()
    print("  Top 15 features:")
    top = fimp.nlargest(15)
    mx  = top.max()
    for feat, imp in top.items():
        bar_w = int(imp / mx * 20)
        print(f"    {feat:<32} {imp:.4f}  {'#' * bar_w}")

    # ── Feature threshold scans for this config ───────────────────────────
    try:
        candles = load_candles_with_atr("SIRENUSDT", tf, live=True)
        threshold_scan(trd, candles, label, col="atr7_pct",   signed=False)
        threshold_scan(trd, candles, label, col="roc10",      signed=True)
        threshold_scan(trd, candles, label, col="ma10_slope", signed=True)
        combined_scan(trd, candles, label)
    except Exception as e:
        print(f"  [scan skipped: {e}]")

print()
print("=" * 62)
print("  KOPEJIE FEATURES  (top-15 visos 3 configs)")
print("=" * 62)

labels   = list(all_fimp.keys())
top15    = [set(all_fimp[l].nlargest(15).index) for l in labels]
common   = top15[0] & top15[1] & top15[2]
in_two   = (top15[0] | top15[1] | top15[2]) - common

print(f"\n  Visu 3 top-15: {len(common)} features")
for f in sorted(common, key=lambda x: -sum(all_fimp[l].get(x, 0) for l in labels)):
    vals = "  ".join(f"{l}:{all_fimp[l].get(f, 0):.4f}" for l in labels)
    print(f"    {f:<28}  {vals}")

print(f"\n  2 no 3 top-15:")
pairs = sorted(
    [(f, sum(1 for s in top15 if f in s)) for f in in_two],
    key=lambda x: -x[1]
)
for f, cnt in pairs:
    if cnt == 2:
        flags = "  ".join("TOP" if f in s else "---" for s in top15)
        print(f"    {f:<28}  {flags}  [{labels[0]} / {labels[1]} / {labels[2]}]")
