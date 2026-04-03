"""
features.py
===========
Technical indicator computation and multi-timeframe feature engineering.

All indicators are implemented manually (no TA library required) so the
only dependency is numpy + pandas.

Public API:
    build_features_single(df, prefix)   → feature DataFrame for one TF
    merge_timeframes(df_5m, df_15m)     → merged 5m+15m feature DataFrame
    make_labels(df_5m, horizons, thr)   → binary label DataFrame
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Low-level indicator helpers
# ---------------------------------------------------------------------------

def _log_returns(close: pd.Series) -> pd.Series:
    """Natural log return: ln(p_t / p_{t-1})."""
    return np.log(close / close.shift(1))


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average (Wilder-style, adjust=False)."""
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder RSI.  Returns values in [0, 100].
    Uses EWM with alpha = 1/period (equivalent to Wilder smoothing).
    """
    delta     = close.diff()
    gain      = delta.clip(lower=0.0)
    loss      = -delta.clip(upper=0.0)
    avg_gain  = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss  = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs        = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    h, l, c   = df["high"], df["low"], df["close"]
    prev_c    = c.shift(1)
    tr        = pd.concat(
        [h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _bollinger_pct_b(close: pd.Series, window: int = 20,
                     n_std: float = 2.0) -> pd.Series:
    """
    Bollinger %B: (price - lower) / (upper - lower).
    Values < 0 → below band, > 1 → above band.
    """
    ma    = close.rolling(window, min_periods=window).mean()
    sigma = close.rolling(window, min_periods=window).std()
    upper = ma + n_std * sigma
    lower = ma - n_std * sigma
    width = (upper - lower).replace(0.0, np.nan)
    return (close - lower) / width


def _volume_oscillator(vol: pd.Series,
                        fast: int = 5, slow: int = 20) -> pd.Series:
    """
    Volume Oscillator = (fast_EMA − slow_EMA) / slow_EMA × 100.
    Positive: short-term volume above average → rising interest.
    """
    fast_ema = vol.ewm(span=fast, adjust=False).mean()
    slow_ema = vol.ewm(span=slow, adjust=False).mean()
    return (fast_ema - slow_ema) / slow_ema.replace(0.0, np.nan) * 100.0


def _stoch_rsi(rsi_series: pd.Series, period: int = 14) -> pd.Series:
    """
    Stochastic RSI: (RSI - min_RSI) / (max_RSI - min_RSI) over period.
    Ranges [0, 1].
    """
    min_rsi = rsi_series.rolling(period, min_periods=period).min()
    max_rsi = rsi_series.rolling(period, min_periods=period).max()
    denom   = (max_rsi - min_rsi).replace(0.0, np.nan)
    return (rsi_series - min_rsi) / denom


# ---------------------------------------------------------------------------
# Single-timeframe feature builder
# ---------------------------------------------------------------------------

def build_features_single(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """
    Compute all features for one OHLCV DataFrame.

    Parameters
    ----------
    df      : OHLCV DataFrame (columns: open, high, low, close, volume)
    prefix  : column prefix, e.g. "m15_" for 15-minute features

    Returns
    -------
    DataFrame with shape (len(df), n_features).
    First ~20 rows will contain NaN (rolling warm-up) — handled downstream.
    """
    p   = prefix
    out = pd.DataFrame(index=df.index)

    close = df["close"]
    vol   = df["volume"]

    # ── Price & returns ───────────────────────────────────────────────────────
    out[f"{p}log_ret"]      = _log_returns(close)
    out[f"{p}hl_range_pct"] = (df["high"] - df["low"]) / close          # normalised H-L span
    out[f"{p}co_pct"]       = (close - df["open"])     / df["open"]      # candle body %

    # ── EMA trend features ────────────────────────────────────────────────────
    ema9    = _ema(close, 9)
    ema21   = _ema(close, 21)
    ema50   = _ema(close, 50)
    ema200  = _ema(close, 200)

    out[f"{p}ema9_dist"]    = (close - ema9)   / close    # distance from fast EMA
    out[f"{p}ema21_dist"]   = (close - ema21)  / close
    out[f"{p}ema50_dist"]   = (close - ema50)  / close
    out[f"{p}ema200_dist"]  = (close - ema200) / close    # long-term trend distance
    out[f"{p}ema_cross_fs"] = (ema9  - ema21)  / close    # fast − slow cross signal

    # ── Momentum ─────────────────────────────────────────────────────────────
    rsi14 = _rsi(close, 14)
    out[f"{p}rsi14"]        = rsi14 / 100.0               # scale to [0, 1]
    out[f"{p}stoch_rsi"]    = _stoch_rsi(rsi14, 14)       # [0, 1]
    out[f"{p}roc5"]         = close.pct_change(5)
    out[f"{p}roc10"]        = close.pct_change(10)
    out[f"{p}roc20"]        = close.pct_change(20)

    # ── Volatility ────────────────────────────────────────────────────────────
    out[f"{p}atr14_pct"]    = _atr(df, 14) / close        # ATR normalised by price
    out[f"{p}bb_pct_b"]     = _bollinger_pct_b(close, 20, 2.0)
    out[f"{p}vol_5_std"]    = close.rolling(5).std()  / close
    out[f"{p}vol_20_std"]   = close.rolling(20).std() / close

    # ── Volume ────────────────────────────────────────────────────────────────
    vol_ma20 = vol.rolling(20, min_periods=1).mean()
    out[f"{p}vol_ratio"]    = vol / vol_ma20.replace(0, np.nan)   # relative volume
    out[f"{p}vol_osc"]      = _volume_oscillator(vol, 5, 20) / 100.0

    return out


# ---------------------------------------------------------------------------
# Multi-timeframe merge  (5m base + 15m overlay)
# ---------------------------------------------------------------------------

def merge_timeframes(df_5m: pd.DataFrame, df_15m: pd.DataFrame) -> pd.DataFrame:
    """
    Merge 5m features with 15m features — NO look-ahead bias.

    Strategy: reindex 15m features onto the 5m timeline using forward-fill,
    so each 5m candle only sees the LAST COMPLETED 15m bar.

    Returns
    -------
    DataFrame aligned to df_5m's index (RangeIndex).
    """
    feats_5m  = build_features_single(df_5m,  prefix="")
    feats_15m = build_features_single(df_15m, prefix="m15_")

    # Attach datetimes for merge
    feats_5m_dt  = feats_5m.copy()
    feats_15m_dt = feats_15m.copy()
    feats_5m_dt["__dt__"]  = df_5m["datetime"].values
    feats_15m_dt["__dt__"] = df_15m["datetime"].values

    feats_15m_dt = feats_15m_dt.set_index("__dt__")

    # Forward-fill 15m onto 5m timestamps (no future leakage)
    feats_15m_reindexed = (
        feats_15m_dt
        .reindex(feats_5m_dt["__dt__"], method="ffill")
        .reset_index(drop=True)
    )

    merged = pd.concat(
        [feats_5m_dt.drop(columns="__dt__"), feats_15m_reindexed],
        axis=1,
    )
    merged.index = df_5m.index   # keep original RangeIndex
    return merged


# ---------------------------------------------------------------------------
# Label creation
# ---------------------------------------------------------------------------

def make_labels(
    df: pd.DataFrame,
    horizons: list[int] = (1, 2, 3),
    threshold: float = 0.005,
) -> pd.DataFrame:
    """
    Binary classification labels.

    label_h = 1  if  close[t+h] / close[t] - 1  >=  threshold
              0  otherwise

    The last `h` rows are set to NaN (no future available).

    Parameters
    ----------
    df        : OHLCV DataFrame (must contain 'close')
    horizons  : list of look-ahead bar counts
    threshold : minimum upward move to count as positive (default 0.5%)

    Returns
    -------
    DataFrame with columns 'label_1', 'label_2', 'label_3'
    """
    labels: dict[str, pd.Series] = {}
    for h in horizons:
        future_ret = df["close"].shift(-h) / df["close"] - 1.0
        lbl = (future_ret >= threshold).astype(float)
        lbl.iloc[-h:] = np.nan          # mask tail — no future data
        labels[f"label_{h}"] = lbl
    return pd.DataFrame(labels, index=df.index)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_fetch import fetch_ohlcv
    from pathlib import Path

    df5  = fetch_ohlcv("SIRENUSDT", "5m",  days=30)
    df15 = fetch_ohlcv("SIRENUSDT", "15m", days=30)

    feats  = merge_timeframes(df5, df15)
    labels = make_labels(df5)

    print(f"\n  Features  : {feats.shape}")
    print(f"  NaN rows  : {feats.isna().any(axis=1).sum()}")
    print(f"  Columns   : {list(feats.columns)}")
    print(f"\n  Label +1 rate (1-bar): {labels['label_1'].mean():.1%}")
    print(f"  Label +1 rate (2-bar): {labels['label_2'].mean():.1%}")
    print(f"  Label +1 rate (3-bar): {labels['label_3'].mean():.1%}")
