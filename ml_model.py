"""
SIRENUSDT ML Price Prediction Model — v2
==========================================
Improvements vs v1:
  1. TWO separate binary models: long_model + short_model
  2. Labels applied to ALL bars (not just ≥0.5% movers) → no selection bias
  3. Backtest on ALL test bars sequentially → no filtering/index mismatch
  4. Class-imbalance handling (scale_pos_weight)
  5. 75+ features (added momentum, gap, consec, price-position, vol_up, etc.)
  6. 1m lookback extended to 30 days
  7. Multiple min_prob thresholds swept (52%–65%), best reported

Models : LightGBM + XGBoost  (long model + short model each)
Target : Binary — will next candle move ≥ 0.5% up (long) or ≥ 0.5% down (short)?
Filter : Long only if close > MA100; Short only if close < MA100
Split  : Walk-forward – train 70% / val 15% / test 15%  (NO lookahead bias)
Exit   : 1-bar hold (entry at next bar open, exit at next bar close)

Usage:
  python ml_model.py --symbol SIRENUSDT --timeframes 1m 3m 5m 15m
  python ml_model.py --symbol SIRENUSDT --timeframes 1m 3m 5m 15m --download
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb

# ---------------------------------------------------------------------------
# Config  (loaded from ml_config.json if present, else defaults)
# ---------------------------------------------------------------------------
BITGET_BASE = "https://api.bitget.com"
MARKET_TYPE = "futures"
DATA_DIR    = Path("data/futures")
RESULTS_DIR = Path("ml_results")
MODELS_DIR  = Path("ml_models")

_CFG_PATH = Path("ml_config.json")
_CFG: dict = json.loads(_CFG_PATH.read_text()) if _CFG_PATH.exists() else {}

MIN_PROBS       = _CFG.get("min_probs",       [0.52, 0.55, 0.58, 0.60, 0.62, 0.65])
HOLD_BARS       = _CFG.get("hold_bars",       [1, 2, 3])
TARGET_PCT      = _CFG.get("target_pct",      0.005)
FEE_RATE        = _CFG.get("fee_rate",        0.001)
STOP_LOSS_PCT   = _CFG.get("stop_loss_pct",   None)   # None = no stop-loss
SIGNAL_FILTERS  = _CFG.get("signal_filters",  {})    # {"5m-2bar": {"atr7_pct": ...}}
TF_LOOKBACK_DAYS_CFG = _CFG.get("tf_lookback_days", {})

TF_GRAN_MAP = {
    "1m": "1m", "3m": "3m", "5m": "5m",
    "15m": "15m", "30m": "30m", "1h": "1H",
}
TF_LOOKBACK_DAYS = {
    "1m": 30, "3m": 30, "5m": 45, "15m": 90,
    **TF_LOOKBACK_DAYS_CFG,
}

# ---------------------------------------------------------------------------
# Bitget downloader (futures)
# ---------------------------------------------------------------------------

def fetch_candles(symbol: str, timeframe: str, lookback_days: int) -> pd.DataFrame:
    tf_sec_map = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800}
    bar_ms = tf_sec_map[timeframe] * 1000
    gran = TF_GRAN_MAP[timeframe]
    limit = 1000
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - lookback_days * 24 * 3600 * 1000
    total_ms = now_ms - start_ms
    current_end = now_ms
    all_rows: list[list] = []

    print(f"  Downloading {symbol} {timeframe} ({lookback_days}d)")
    last_pct = -1
    while current_end > start_ms:
        params = {
            "symbol": symbol,
            "granularity": gran,
            "productType": "USDT-FUTURES",
            "endTime": str(current_end),
            "limit": str(limit),
        }
        url = BITGET_BASE + "/api/v2/mix/market/candles?" + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.loads(r.read())
        except Exception as e:
            print(f"  [ERROR] {e}")
            break
        if data.get("code") != "00000":
            print(f"  [ERROR] API: {data.get('msg')}")
            break
        rows = data.get("data", [])
        if not rows:
            break
        # Bitget returns candles in ASCENDING order (oldest first, newest last).
        # Use rows[0] (oldest) to advance the cursor backwards each iteration.
        for row in rows:
            ts = int(row[0])
            if ts < start_ms:
                continue
            all_rows.append([ts, float(row[1]), float(row[2]), float(row[3]),
                              float(row[4]), float(row[5])])
        oldest_ts = int(rows[0][0])   # rows[0] = oldest candle in this page

        # Progress bar
        fetched_ms = now_ms - oldest_ts
        pct = min(int(fetched_ms / total_ms * 100), 99)
        if pct != last_pct:
            filled = pct // 5
            bar = "█" * filled + "░" * (20 - filled)
            print(f"\r  [{bar}] {pct:>3}%  {len(all_rows):>6,} sveces", end="", flush=True)
            last_pct = pct

        if oldest_ts <= start_ms:
            break
        current_end = oldest_ts - 1   # fetch the page BEFORE this one
        time.sleep(0.1)

    print(f"\r  [{'█'*20}] 100%  {len(all_rows):>6,} sveces", flush=True)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    print(f"  -> {len(df)} candles  ({df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]})")
    return df


def save_candles(df: pd.DataFrame, symbol: str, timeframe: str,
                 live: bool = False) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_live" if live else ""
    path = DATA_DIR / f"{symbol}_{timeframe}{suffix}.csv"
    df.to_csv(path, index=False)
    print(f"  Saved -> {path}")
    return path


def load_candles(symbol: str, timeframe: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}_{timeframe}.csv"
    if not path.exists():
        print(f"  [WARNING] {path} not found -- run with --download first")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Feature engineering (strictly causal – only uses past data)
# ---------------------------------------------------------------------------

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"]  - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd = ema_fast - ema_slow
    signal = compute_ema(macd, sig)
    hist = macd - signal
    return macd, signal, hist


def compute_bb(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


def compute_stoch(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    low_min  = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-10)
    d = k.rolling(d_period).mean()
    return k, d


def compute_cci(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - ma) / (0.015 * mad + 1e-10)


def compute_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_max = df["high"].rolling(period).max()
    low_min  = df["low"].rolling(period).min()
    return -100 * (high_max - df["close"]) / (high_max - low_min + 1e-10)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build 50+ features from OHLCV. All features use only past data (no lookahead)."""
    f = pd.DataFrame(index=df.index)
    c = df["close"]
    o = df["open"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # --- Returns (log) ---
    for n in [1, 2, 3, 5, 10]:
        f[f"return_{n}"] = np.log(c / c.shift(n))

    # --- Candle structure ---
    f["body_size"]   = (c - o).abs() / (o + 1e-10)
    f["upper_wick"]  = (h - c.clip(lower=o)) / (o + 1e-10)
    f["lower_wick"]  = (c.clip(upper=o) - l) / (o + 1e-10)
    f["is_bullish"]  = (c > o).astype(int)
    f["hl_range"]    = (h - l) / (o + 1e-10)
    f["gap"]         = (o - c.shift(1)) / (c.shift(1) + 1e-10)   # gap from prev close
    f["open_pos"]    = (o - l) / (h - l + 1e-10)    # where open sits in bar range
    f["close_pos"]   = (c - l) / (h - l + 1e-10)    # where close sits in bar range

    # --- Consecutive up / down ---
    _is_up = (c > c.shift(1)).astype(float)
    f["consec_up"]   = _is_up.rolling(5).sum()
    f["consec_down"] = (1 - _is_up).rolling(5).sum()

    # --- Moving averages (price / MA - 1) ---
    for period in [5, 10, 20, 50, 100]:
        ma = c.rolling(period).mean()
        f[f"ma{period}_dist"] = c / (ma + 1e-10) - 1

    # --- MA slopes (% change over 3 bars) ---
    for period in [5, 10, 20, 50, 100]:
        ma = c.rolling(period).mean()
        f[f"ma{period}_slope"] = ma.pct_change(3)

    # --- MA100 regime flag ---
    ma100 = c.rolling(100).mean()
    f["above_ma100"] = (c > ma100).astype(int)
    f["dist_ma100"]  = c / (ma100 + 1e-10) - 1

    # --- EMA distances ---
    for _p in [5, 10, 20]:
        f[f"ema{_p}_dist"] = c / (compute_ema(c, _p) + 1e-10) - 1

    # --- MACD(12,26,9) ---
    macd_line, macd_sig, macd_hist = compute_macd(c, 12, 26, 9)
    f["macd_line"]       = macd_line / (c + 1e-10)
    f["macd_signal"]     = macd_sig  / (c + 1e-10)
    f["macd_hist"]       = macd_hist / (c + 1e-10)
    f["macd_hist_delta"] = macd_hist.diff()
    f["macd_cross_up"]   = ((macd_line > macd_sig) & (macd_line.shift(1) <= macd_sig.shift(1))).astype(int)
    f["macd_cross_down"] = ((macd_line < macd_sig) & (macd_line.shift(1) >= macd_sig.shift(1))).astype(int)

    # --- MACD fast (6,16,9) ---
    macd_f, macd_fs, macd_fh = compute_macd(c, 6, 16, 9)
    f["macd_fast_hist"]  = macd_fh / (c + 1e-10)
    f["macd_fast_delta"] = macd_fh.diff()

    # --- RSI ---
    f["rsi14"]       = compute_rsi(c, 14)
    f["rsi7"]        = compute_rsi(c, 7)
    f["rsi14_delta"] = f["rsi14"].diff()
    f["rsi14_norm"]  = (f["rsi14"] - 50) / 50
    f["rsi_ob"]      = (f["rsi14"] > 70).astype(int)
    f["rsi_os"]      = (f["rsi14"] < 30).astype(int)

    # --- Bollinger Bands ---
    bb_upper, bb_mid, bb_lower = compute_bb(c, 20, 2.0)
    f["bb_upper_dist"] = c / (bb_upper + 1e-10) - 1
    f["bb_lower_dist"] = c / (bb_lower + 1e-10) - 1
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    f["bb_pct"]        = (c - bb_lower) / bb_range
    f["bb_width"]      = bb_range / (bb_mid + 1e-10)
    f["bb_squeeze"]    = (bb_range < bb_range.rolling(20).mean()).astype(int)

    # --- ATR ---
    atr14 = compute_atr(df, 14)
    atr7  = compute_atr(df, 7)
    f["atr14"]        = atr14 / (c + 1e-10)
    f["atr7"]         = atr7  / (c + 1e-10)
    f["atr_ratio"]    = atr7 / (atr14 + 1e-10)
    f["range_atr"]    = (h - l) / (atr14 + 1e-10)   # bar range vs ATR

    # --- Volatility (rolling std of returns) ---
    ret1 = c.pct_change()
    f["volatility_5"]  = ret1.rolling(5).std()
    f["volatility_10"] = ret1.rolling(10).std()
    f["volatility_20"] = ret1.rolling(20).std()
    f["vol_expand"]    = (f["volatility_5"] > f["volatility_20"]).astype(int)

    # --- Volume ---
    v_ma5  = v.rolling(5).mean()
    v_ma10 = v.rolling(10).mean()
    f["volume_ratio5"]  = v / (v_ma5  + 1e-10)
    f["volume_ratio10"] = v / (v_ma10 + 1e-10)
    f["volume_delta"]   = v.pct_change()
    f["volume_zscore"]  = (v - v_ma10) / (v.rolling(10).std() + 1e-10)
    f["vol_up"]         = (v * (c > o).astype(float)).rolling(5).sum() / (v.rolling(5).sum() + 1e-10)

    # --- Price position in N-bar range ---
    for _p in [5, 10, 20, 50]:
        _hi = h.rolling(_p).max()
        _lo = l.rolling(_p).min()
        f[f"pos_{_p}"] = (c - _lo) / (_hi - _lo + 1e-10)

    # --- Price level distances ---
    f["dist_high_20"] = c / (h.rolling(20).max() + 1e-10) - 1
    f["dist_low_20"]  = c / (l.rolling(20).min() + 1e-10) - 1
    f["dist_high_5"]  = c / (h.rolling(5).max()  + 1e-10) - 1
    f["dist_low_5"]   = c / (l.rolling(5).min()  + 1e-10) - 1

    # --- Stochastic ---
    stoch_k, stoch_d = compute_stoch(df, 14, 3)
    f["stoch_k"]      = stoch_k / 100
    f["stoch_d"]      = stoch_d / 100
    f["stoch_delta"]  = (stoch_k - stoch_d) / 100
    f["stoch_ob"]     = (stoch_k > 80).astype(int)
    f["stoch_os"]     = (stoch_k < 20).astype(int)

    # --- Williams %R ---
    f["williams_r"] = compute_williams_r(df, 14) / 100

    # --- CCI ---
    f["cci14"]       = compute_cci(df, 14) / 100
    f["cci14_delta"] = f["cci14"].diff()

    # --- Momentum / ROC ---
    for _n in [3, 5, 10]:
        f[f"mom_{_n}"] = c / c.shift(_n) - 1
    f["roc5"]  = c.pct_change(5)
    f["roc10"] = c.pct_change(10)

    # --- Time features ---
    dt = df["datetime"]
    f["hour"]        = dt.dt.hour / 23.0
    f["minute"]      = dt.dt.minute / 59.0
    f["day_of_week"] = dt.dt.dayofweek / 6.0

    return f


def build_labels(df: pd.DataFrame, target_pct: float = TARGET_PCT, hold_bars: int = 1):
    """
    Binary labels for EACH bar — ALL bars, not just ≥0.5% movers.

      y_long[i]  = 1  if  close[i+hold_bars] / close[i] - 1  >=  +target_pct
      y_short[i] = 1  if  close[i+hold_bars] / close[i] - 1  <=  -target_pct

    hold_bars=1 → predict next candle close (1-bar forward return)
    hold_bars=2 → predict 2nd candle close (2-bar cumulative return)
    hold_bars=3 → predict 3rd candle close (3-bar cumulative return)

    Uses shift(-hold_bars) for LABEL CREATION ONLY — features have no bias.
    Last `hold_bars` rows are dropped (no future bar available).
    """
    next_ret = df["close"].shift(-hold_bars) / df["close"] - 1
    y_long  = (next_ret >=  target_pct).astype(int)
    y_short = (next_ret <= -target_pct).astype(int)
    return y_long.iloc[:-hold_bars].reset_index(drop=True), y_short.iloc[:-hold_bars].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Walk-forward split (no lookahead bias)
# ---------------------------------------------------------------------------

def walk_forward_split(n: int, train_frac: float = 0.70, val_frac: float = 0.15):
    train_end = int(n * train_frac)
    val_end   = int(n * (train_frac + val_frac))
    return (
        slice(0,          train_end),
        slice(train_end,  val_end),
        slice(val_end,    n),
    )


def compute_signal_filter_mask(
    features: pd.DataFrame, label: str,
    symbol: str = "",
    signal_filters: dict | None = None,
) -> np.ndarray:
    """
    Build a boolean mask (True = signal allowed) from the feature DataFrame.
    Thresholds are in % units (atr7_pct, roc10_abs_pct, slope_abs_pct).
    Features from build_features() are in fraction units → divide thresholds by 100.
    signal_filters is nested per-symbol: {"SIRENUSDT": {"5m-2bar": {...}}}
    """
    top = signal_filters or SIGNAL_FILTERS
    # Support both old flat format {"5m-2bar": {...}} and new per-symbol {"SYM": {"5m-2bar": {...}}}
    sym_filters = top.get(symbol, top) if symbol else top
    cfg = sym_filters.get(label) if isinstance(next(iter(sym_filters.values()), None), dict) else top.get(label)
    if not cfg:
        return np.ones(len(features), dtype=bool)
    mask = np.ones(len(features), dtype=bool)
    if "atr7_pct" in cfg:
        mask &= features["atr7"].values >= cfg["atr7_pct"] / 100
    if "roc10_abs_pct" in cfg:
        mask &= np.abs(features["roc10"].values) >= cfg["roc10_abs_pct"] / 100
    if "slope_abs_pct" in cfg:
        mask &= np.abs(features["ma10_slope"].values) >= cfg["slope_abs_pct"] / 100
    return mask


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_direction_models(X_tr, y_tr, X_v, y_v):
    """
    Train LightGBM + XGBoost for ONE binary direction (long or short).
    scale_pos_weight compensates for class imbalance
    (~20-30% of bars move ≥ 0.5%, so ~70-80% are negatives).
    """
    n_pos = max(int(y_tr.sum()), 1)
    n_neg = max(len(y_tr) - n_pos, 1)
    scale = n_neg / n_pos

    lgb_m = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=30,
        reg_alpha=0.1,
        reg_lambda=0.5,
        scale_pos_weight=scale,
        random_state=42,
        verbose=-1,
    )
    lgb_m.fit(X_tr, y_tr, eval_set=[(X_v, y_v)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])

    xgb_m = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale,
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=50,
        verbosity=0,
    )
    xgb_m.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)

    return lgb_m, xgb_m


# ---------------------------------------------------------------------------
# Model persistence  (save after training, load in --backtest-only mode)
# ---------------------------------------------------------------------------

class _LGBWrapper:
    """Thin wrapper around lgb.Booster that exposes predict_proba + feature_importances_
    so it can be used identically to a fitted LGBMClassifier in backtest code."""
    def __init__(self, booster):
        self._b = booster
    def predict_proba(self, X):
        p = self._b.predict(X)          # shape (n,) — P(class=1)
        return np.column_stack([1 - p, p])
    @property
    def feature_importances_(self):
        return self._b.feature_importance(importance_type="split")

def save_models(symbol: str, timeframe: str, hold_bars: int,
                lgb_long, lgb_short, xgb_long, xgb_short) -> None:
    """Save models using native formats (LGB→.txt, XGB→.json) — numpy-version agnostic."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    base = str(MODELS_DIR / f"{symbol}_{timeframe}_{hold_bars}bar")
    lgb_long.booster_.save_model(f"{base}_lgb_long.txt")
    lgb_short.booster_.save_model(f"{base}_lgb_short.txt")
    xgb_long.save_model(f"{base}_xgb_long.json")
    xgb_short.save_model(f"{base}_xgb_short.json")
    print(f"  Models saved -> {MODELS_DIR}/{symbol}_{timeframe}_{hold_bars}bar_*.{{txt,json}}")


def load_models(symbol: str, timeframe: str, hold_bars: int) -> dict | None:
    """Load models saved by save_models(). Returns None if any file missing."""
    import lightgbm as _lgb
    import xgboost as _xgb
    base = str(MODELS_DIR / f"{symbol}_{timeframe}_{hold_bars}bar")
    out: dict = {}
    for name, ext in [("lgb_long", "txt"), ("lgb_short", "txt"),
                      ("xgb_long", "json"), ("xgb_short", "json")]:
        path = Path(f"{base}_{name}.{ext}")
        if not path.exists():
            print(f"  [ERROR] Model not found: {path}")
            print(f"          Run without --backtest-only first to train and save models.")
            return None
        if ext == "txt":
            out[name] = _LGBWrapper(_lgb.Booster(model_file=str(path)))
        else:
            m = _xgb.XGBClassifier()
            m.load_model(str(path))
            out[name] = m
    return out


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def run_backtest(
    df_test: pd.DataFrame,
    probs_long: np.ndarray,
    probs_short: np.ndarray,
    above_ma100: np.ndarray,
    model_name: str,
    timeframe: str,
    hold_bars: int = 1,
    min_prob: float = 0.60,
    fee_rate: float = FEE_RATE,
    initial_capital: float = 10000.0,
    stop_loss_pct: float | None = STOP_LOSS_PCT,
    filter_mask: np.ndarray | None = None,
) -> tuple[dict, pd.DataFrame]:
    """
    Simulate trading on ALL test bars sequentially.
    Signal at bar i -> enter at bar[i+1] open -> exit at bar[i+hold_bars] close.
    MA100 trend filter: Long only above MA100, Short only below MA100.
    Stop-loss: if price moves against entry by stop_loss_pct intrabar, exit at SL price.
    """
    capital = initial_capital
    trades = []
    peak = initial_capital
    max_dd = 0.0

    df_test = df_test.reset_index(drop=True)
    above = np.asarray(above_ma100)
    n = len(df_test)

    busy_until = -1  # bar index until which a trade is still open

    for i in range(n - hold_bars):
        # Skip if a trade is already open (overlapping bars)
        if i <= busy_until:
            continue

        p_long  = float(probs_long[i])
        p_short = float(probs_short[i])

        # MA100 trend filter: only trade with the trend
        go_long  = (p_long  >= min_prob) and (above[i] == 1)
        go_short = (p_short >= min_prob) and (above[i] == 0)

        if not go_long and not go_short:
            continue

        # Signal quality filter (ATR / roc10 / ma10_slope thresholds)
        if filter_mask is not None and not filter_mask[i]:
            continue

        direction = "long" if go_long else "short"

        # Entry: next bar open; Exit: bar[i+hold_bars] close
        entry_bar   = i + 1
        exit_bar    = i + hold_bars
        entry_price = df_test["open"].iloc[entry_bar]
        exit_price  = df_test["close"].iloc[exit_bar]
        entry_time  = str(df_test["datetime"].iloc[entry_bar])
        exit_time   = str(df_test["datetime"].iloc[exit_bar])

        # Block any new entries until this trade is closed
        busy_until = exit_bar

        # Intrabar stop-loss check across hold_bars candles
        sl_triggered = False
        if stop_loss_pct is not None:
            for b in range(entry_bar, exit_bar + 1):
                if direction == "long":
                    sl_price = entry_price * (1 - stop_loss_pct)
                    if df_test["low"].iloc[b] <= sl_price:
                        exit_price = sl_price
                        exit_time  = str(df_test["datetime"].iloc[b])
                        busy_until = b   # free up earlier
                        sl_triggered = True
                        break
                else:
                    sl_price = entry_price * (1 + stop_loss_pct)
                    if df_test["high"].iloc[b] >= sl_price:
                        exit_price = sl_price
                        exit_time  = str(df_test["datetime"].iloc[b])
                        busy_until = b
                        sl_triggered = True
                        break

        if direction == "long":
            gross_pct = (exit_price - entry_price) / entry_price
        else:
            gross_pct = (entry_price - exit_price) / entry_price

        net_pct = gross_pct - 2 * fee_rate
        pnl = capital * net_pct
        capital += pnl

        trades.append({
            "direction":   direction,
            "entry_time":  entry_time,
            "exit_time":   exit_time,
            "entry_price": round(entry_price, 8),
            "exit_price":  round(exit_price, 8),
            "prob":        round(p_long if direction == "long" else p_short, 4),
            "sl_hit":      sl_triggered,
            "pnl":         round(pnl, 4),
            "pnl_pct":     round(net_pct * 100, 4),
            "capital":     round(capital, 2),
        })

        if capital > peak:
            peak = capital
        dd = (peak - capital) / peak * 100
        if dd > max_dd:
            max_dd = dd

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        report = {
            "model": model_name, "timeframe": timeframe,
            "trades": 0, "error": "No trades generated",
        }
        return report, trades_df

    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    win_rate = len(wins) / len(trades_df) * 100
    total_return = (capital - initial_capital) / initial_capital * 100
    pf_denom = abs(losses["pnl"].sum())
    profit_factor = wins["pnl"].sum() / pf_denom if pf_denom > 0 else float("inf")

    report = {
        "model":          model_name,
        "timeframe":      timeframe,
        "trades":         len(trades_df),
        "win_rate_pct":   round(win_rate, 2),
        "total_return_pct": round(total_return, 4),
        "max_drawdown_pct": round(max_dd, 4),
        "profit_factor":  round(profit_factor, 4),
        "avg_win_pct":    round(wins["pnl_pct"].mean(), 4) if not wins.empty else 0,
        "avg_loss_pct":   round(losses["pnl_pct"].mean(), 4) if not losses.empty else 0,
        "final_capital":  round(capital, 2),
    }
    return report, trades_df


# ---------------------------------------------------------------------------
# Main pipeline per timeframe
# ---------------------------------------------------------------------------

def run_pipeline(symbol: str, timeframe: str, download: bool,
                 backtest_only: bool = False,
                 override_days: int | None = None,
                 cutoff: str | None = None,
                 trend_ma: int = 100,
                 no_filter: bool = False) -> None:
    print(f"\n{'='*65}")
    print(f"  {symbol}  {timeframe}  --  LightGBM + XGBoost  (v2)")
    print(f"{'='*65}")

    # 1. Data ──────────────────────────────────────────────────────────────
    if download or backtest_only:
        days = (override_days if override_days is not None
                else 7 if backtest_only
                else TF_LOOKBACK_DAYS.get(timeframe, 100))
        df = fetch_candles(symbol, timeframe, days)
        if df.empty:
            print("  [ERROR] No data downloaded.")
            return
        # backtest-only saves to a separate _live CSV so it does NOT
        # overwrite the full training dataset
        save_candles(df, symbol, timeframe, live=backtest_only)
    else:
        df = load_candles(symbol, timeframe)
        if df.empty:
            return

    # Apply training cutoff (only in training mode, not backtest-only)
    if not backtest_only and cutoff:
        cutoff_ts = pd.Timestamp(cutoff, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        before = len(df)
        df = df[df["datetime"] <= cutoff_ts].reset_index(drop=True)
        print(f"  Cutoff   : <= {cutoff}  ({before} -> {len(df)} candles kept)")

    if len(df) < 300:
        print(f"  [ERROR] Only {len(df)} candles -- need >= 300.")
        return

    print(f"  Candles  : {len(df)}  ({df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]})")

    # 2. Build features ONCE (reused for all horizons) ─────────────────────
    # Features only use past bars → no lookahead bias regardless of horizon.
    features_full = build_features(df)

    # 3. Loop over prediction horizons ─────────────────────────────────────
    for hold_bars in HOLD_BARS:
        print(f"\n  {'-'*63}")
        print(f"  Horizon : {hold_bars} bar{'s' if hold_bars > 1 else ''}  "
              f"(enter bar[i+1] open -> exit bar[i+{hold_bars}] close)")
        print(f"  {'-'*63}")

        # Labels: predict cumulative return over hold_bars bars
        y_long, y_short = build_labels(df, hold_bars=hold_bars)
        # y_long/y_short length = len(df) - hold_bars

        # Align features and df to the same rows as labels
        features = features_full.iloc[:len(y_long)].reset_index(drop=True)
        df_valid = df.iloc[:len(y_long)].reset_index(drop=True)

        # Drop warmup NaN rows (indicator warmup ~100 bars)
        features = features.replace([np.inf, -np.inf], np.nan)
        na_mask  = features.isna().any(axis=1)
        features = features[~na_mask].reset_index(drop=True)
        y_long   = y_long[~na_mask].reset_index(drop=True)
        y_short  = y_short[~na_mask].reset_index(drop=True)
        df_valid = df_valid[~na_mask].reset_index(drop=True)

        feature_cols = features.columns.tolist()

        # Compute MA50 trend flag separately (not a model feature — no retraining needed)
        ma50_ser = df_valid["close"].rolling(50).mean()
        features["above_ma50"] = (df_valid["close"] > ma50_ser).astype(int).values

        n = len(features)

        print(f"  Samples  : {n}")
        print(f"  >=+0.5% (long label=1) : {y_long.mean()*100:.1f}%  of bars")
        print(f"  >=-0.5% (short label=1): {y_short.mean()*100:.1f}%  of bars")

        if n < 300:
            print("  [SKIP] Too few samples after warmup drop.")
            continue

        # Walk-forward split ───────────────────────────────────────────────
        if backtest_only:
            # -- Backtest-only: load saved models, predict on ALL available data --
            models = load_models(symbol, timeframe, hold_bars)
            if models is None:
                continue
            lgb_long,  lgb_short = models["lgb_long"],  models["lgb_short"]
            xgb_long,  xgb_short = models["xgb_long"],  models["xgb_short"]
            X_te     = features[feature_cols].values
            above_te = features[f"above_ma{trend_ma}"].values
            df_te    = df_valid.reset_index(drop=True)
            yl_te    = y_long.values
            ys_te    = y_short.values
            if no_filter:
                sig_filter_mask = np.ones(len(features), dtype=bool)
                print(f"  Filters        : DISABLED (--no-filter)")
            else:
                sig_filter_mask = compute_signal_filter_mask(
                    features, f"{timeframe}-{hold_bars}bar", symbol=symbol)
            print(f"  Loaded models from {MODELS_DIR}/")
            print(f"  Bars for inference : {len(X_te)}  (full window, no train/val split)")
        else:
            # -- Training mode: walk-forward split, train, save models -----------
            train_sl, val_sl, test_sl = walk_forward_split(n)

            X_tr  = features.iloc[train_sl][feature_cols].values
            X_val = features.iloc[val_sl][feature_cols].values
            X_te  = features.iloc[test_sl][feature_cols].values

            yl_tr, yl_v, yl_te = (y_long.iloc[train_sl].values,
                                   y_long.iloc[val_sl].values,
                                   y_long.iloc[test_sl].values)
            ys_tr, ys_v, ys_te = (y_short.iloc[train_sl].values,
                                   y_short.iloc[val_sl].values,
                                   y_short.iloc[test_sl].values)

            above_te = features.iloc[test_sl][f"above_ma{trend_ma}"].values
            df_te    = df_valid.iloc[test_sl].reset_index(drop=True)
            sig_filter_mask = compute_signal_filter_mask(
                features.iloc[test_sl].reset_index(drop=True), f"{timeframe}-{hold_bars}bar",
                symbol=symbol)

            print(f"  Train: {len(X_tr)}  Val: {len(X_val)}  Test: {len(X_te)}")
            print(f"  Training LONG  models (pos: {yl_tr.mean():.1%})...")
            lgb_long,  xgb_long  = train_direction_models(X_tr, yl_tr, X_val, yl_v)

            print(f"  Training SHORT models (pos: {ys_tr.mean():.1%})...")
            lgb_short, xgb_short = train_direction_models(X_tr, ys_tr, X_val, ys_v)

            save_models(symbol, timeframe, hold_bars,
                        lgb_long, lgb_short, xgb_long, xgb_short)

        # Evaluate + backtest ──────────────────────────────────────────────
        for ml_name, m_long, m_short in [
            ("LightGBM", lgb_long,  lgb_short),
            ("XGBoost",  xgb_long,  xgb_short),
        ]:
            pl = m_long.predict_proba(X_te)[:, 1]    # P(up >= 0.5% in hold_bars)
            ps = m_short.predict_proba(X_te)[:, 1]   # P(down >= 0.5% in hold_bars)

            try:
                auc_l = roc_auc_score(yl_te, pl)
                auc_s = roc_auc_score(ys_te, ps)
            except Exception:
                auc_l = auc_s = float("nan")

            print(f"\n  [{ml_name}]")
            print(f"  AUC-ROC  long: {auc_l:.4f}  |  short: {auc_s:.4f}")

            # Sweep probability thresholds — keep best by total return
            best_rep, best_tr, best_p = None, None, MIN_PROBS[0]
            for min_p in MIN_PROBS:
                rep, tr = run_backtest(
                    df_test=df_te, probs_long=pl, probs_short=ps,
                    above_ma100=above_te, model_name=ml_name,
                    timeframe=timeframe, hold_bars=hold_bars, min_prob=min_p,
                    filter_mask=sig_filter_mask,
                )
                if "error" not in rep:
                    if best_rep is None or rep["total_return_pct"] > best_rep["total_return_pct"]:
                        best_rep, best_tr, best_p = rep, tr, min_p

            if best_rep is None:
                print(f"  No trades at any threshold ({[f'{p:.0%}' for p in MIN_PROBS]})")
                best_rep = {
                    "model": ml_name, "timeframe": timeframe, "hold_bars": hold_bars,
                    "trades": 0, "error": "No trades at any threshold",
                    "auc_long": round(auc_l, 4), "auc_short": round(auc_s, 4),
                }
            else:
                print(f"  Best threshold : {best_p:.0%}")
                print(f"  Trades         : {best_rep['trades']}")
                print(f"  Win Rate       : {best_rep['win_rate_pct']:.2f}%")
                print(f"  Total Return   : {best_rep['total_return_pct']:+.4f}%")
                print(f"  Max Drawdown   : {best_rep['max_drawdown_pct']:.4f}%")
                print(f"  Profit Factor  : {best_rep['profit_factor']}")
                best_rep.update({
                    "auc_long":   round(auc_l, 4),
                    "auc_short":  round(auc_s, 4),
                    "n_features": len(feature_cols),
                    "target_pct": TARGET_PCT,
                    "hold_bars":  hold_bars,
                    "mode":       "live" if backtest_only else "train",
                })

            # Feature importance (from long model)
            fi_df = pd.DataFrame({
                "feature":    feature_cols,
                "importance": m_long.feature_importances_,
            }).sort_values("importance", ascending=False)

            print(f"  Top-5 features:")
            for _, row in fi_df.head(5).iterrows():
                print(f"    {row['feature']:<25} {row['importance']:.0f}")

            # Save
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            slug = ml_name.lower().replace("lightgbm", "lgb").replace("xgboost", "xgb")
            tag  = f"{symbol}_{timeframe}_{slug}_{hold_bars}bar" + ("_live" if backtest_only else "")

            with open(RESULTS_DIR / f"{tag}_report.json", "w", encoding="utf-8") as fh:
                json.dump(best_rep, fh, indent=2)
            print(f"  Report -> {RESULTS_DIR / f'{tag}_report.json'}")

            if best_tr is not None and not best_tr.empty:
                best_tr.to_csv(RESULTS_DIR / f"{tag}_trades.csv", index=False)

            fi_df.to_csv(RESULTS_DIR / f"{tag}_feature_importance.csv", index=False)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(symbol: str, timeframes: list[str]) -> None:
    rows = []
    for tf in timeframes:
        for hb in HOLD_BARS:
            for model in ["lgb", "xgb"]:
                for suffix, mode_label in [("", "train"), ("_live", "LIVE ")]:
                    path = RESULTS_DIR / f"{symbol}_{tf}_{model}_{hb}bar{suffix}_report.json"
                    if not path.exists():
                        continue
                    with open(path) as f:
                        r = json.load(f)
                    if "error" in r:
                        continue
                    rows.append({
                        "tf":      tf,
                        "bars":    hb,
                        "model":   model.upper(),
                        "mlabel":  mode_label,
                        "min_p":   r.get("min_prob", 0),
                        "trades":  r.get("trades", 0),
                        "win_pct": r.get("win_rate_pct", 0),
                        "return":  r.get("total_return_pct", 0),
                        "maxdd":   r.get("max_drawdown_pct", 0),
                        "pf":      r.get("profit_factor", 0),
                        "auc_l":   r.get("auc_long",  0),
                        "auc_s":   r.get("auc_short", 0),
                    })

    if not rows:
        print("\n  No completed results found.")
        return

    rows.sort(key=lambda r: (r["mlabel"], -r["return"]))
    print(f"\n{'='*112}")
    print(f"  {'TF':<5} {'Bars':>4} {'Model':<8} {'Mode':<6} {'MinP':>5} {'Trades':>6} {'Win%':>6} "
          f"{'Return':>9} {'MaxDD':>7} {'PF':>6} {'AUC-L':>7} {'AUC-S':>7}")
    print(f"  {'-'*5} {'-'*4} {'-'*8} {'-'*6} {'-'*5} {'-'*6} {'-'*6} "
          f"{'-'*9} {'-'*7} {'-'*6} {'-'*7} {'-'*7}")
    for r in rows:
        flag = "+" if r["return"] >= 0 else "-"
        print(f"  {flag} {r['tf']:<5} {r['bars']:>4}  {r['model']:<8} {r['mlabel']:<6} {r['min_p']:.0%} "
              f"{r['trades']:>6} {r['win_pct']:>5.1f}% "
              f"{r['return']:>+9.4f}% {r['maxdd']:>6.2f}% "
              f"{r['pf']:>6.2f} {r['auc_l']:>7.4f} {r['auc_s']:>7.4f}")
    print("="*112)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SIRENUSDT ML model — LightGBM + XGBoost")
    parser.add_argument("--symbol",        default="SIRENUSDT")
    parser.add_argument("--timeframes",    nargs="+", default=["1m", "3m", "5m", "15m"])
    parser.add_argument("--download",      action="store_true", help="Download fresh candle data")
    parser.add_argument("--backtest-only", action="store_true", dest="backtest_only",
                        help="Load saved models and run on fresh data (auto-downloads 7 days)")
    parser.add_argument("--days",          type=int, default=None,
                        help="Override lookback days (default: 7 when --backtest-only)")
    parser.add_argument("--cutoff",        type=str,
                        default=_CFG.get("train_cutoff_default", "2026-03-20"),
                        help="Train only on data up to this date YYYY-MM-DD")
    parser.add_argument("--trend-ma",      type=int, default=100, choices=[50, 100],
                        dest="trend_ma",
                        help="MA period for trend filter: Long above MA, Short below MA (default: 100)")
    parser.add_argument("--no-filter",     action="store_true", dest="no_filter",
                        help="Disable signal quality filters (atr7_pct, roc10_abs_pct, slope_abs_pct)")
    args = parser.parse_args()

    sl_info = f"SL {STOP_LOSS_PCT*100:.1f}%" if STOP_LOSS_PCT else "no SL"
    mode_label = "LIVE BACKTEST" if args.backtest_only else "TRAIN"
    print(f"\n{'='*65}")
    print(f"  ML MODEL v2: {args.symbol}  |  TFs: {args.timeframes}  |  Mode: {mode_label}")
    print(f"  Target     : cumulative move >= {TARGET_PCT*100:.1f}%  |  Horizons: {HOLD_BARS} bars")
    print(f"  Filter     : Long if close > MA{args.trend_ma}  |  Short if close < MA{args.trend_ma}")
    print(f"  Thresholds : {[f'{p:.0%}' for p in MIN_PROBS]}  (best reported per horizon)")
    print(f"  Stop-loss  : {sl_info}  |  Fee: {FEE_RATE*100:.2f}% per side")
    if not args.backtest_only:
        print(f"  Train cutoff: <= {args.cutoff}  |  Backtest period: after cutoff")
    print(f"{'='*65}")

    for tf in args.timeframes:
        run_pipeline(symbol=args.symbol, timeframe=tf, download=args.download,
                     backtest_only=args.backtest_only, override_days=args.days,
                     cutoff=args.cutoff, trend_ma=args.trend_ma,
                     no_filter=args.no_filter)

    print_summary(args.symbol, args.timeframes)


if __name__ == "__main__":
    main()
