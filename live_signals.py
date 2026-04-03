"""
LIVE SIGNAL SIMULATOR — XGB + LSTM konfigurācijas paralēli
===========================================================
  XGB (SIRENUSDT / ONTUSDT / STOUSDT / NOMUSDT):
    1. SIRENUSDT  5m  2-bar  XGB  (min_prob=0.55)
    2. SIRENUSDT  5m  3-bar  XGB  (min_prob=0.65)
    3. SIRENUSDT  15m 2-bar  XGB  (min_prob=0.55)
    4. NOMUSDT    15m 3-bar  rule-based
    5. ONTUSDT    15m 3-bar  rule-based
    6. ONTUSDT    15m 2-bar  XGB  (min_prob=0.62)
    7. STOUSDT    15m 1-bar  XGB  (min_prob=0.58)
  LSTM (SIRENUSDT, 6 varianti = 3 horizons × 2 exit veidi):
    8.  SIRENUSDT  5m  1-bar  LSTM  (horizon exit)
    9.  SIRENUSDT  5m  1-bar  LSTM  (TP/SL exit)
    10. SIRENUSDT  5m  2-bar  LSTM  (horizon exit)
    11. SIRENUSDT  5m  2-bar  LSTM  (TP/SL exit)
    12. SIRENUSDT  5m  3-bar  LSTM  (horizon exit)
    13. SIRENUSDT  5m  3-bar  LSTM  (TP/SL exit)

Izmantošana:
  python live_signals.py
  python live_signals.py --symbol SIRENUSDT
  python live_signals.py --no-wait
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import socketserver
import sys
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb_lib
import xgboost as xgb_lib

sys.path.insert(0, str(Path(__file__).parent))
from ml_model import (
    build_features,
    _LGBWrapper, MODELS_DIR, TF_GRAN_MAP,
    FEE_RATE,
)

# ── LSTM pipeline imports (optional — skip gracefully if TF not available) ──
LSTM_DIR = Path(__file__).parent / "lstm"
sys.path.insert(0, str(LSTM_DIR))
try:
    import tensorflow as _tf
    import joblib as _joblib
    from lstm.features import merge_timeframes as _lstm_merge_tfs, make_labels as _lstm_make_labels
    from lstm.dataset  import build_sequences as _lstm_build_seq, scale_X_with_fitted as _lstm_scale
    from lstm.dataset  import WINDOW as LSTM_WINDOW
    LSTM_AVAILABLE = True
except Exception as _lstm_exc:
    LSTM_AVAILABLE = False
    print(f"  [LSTM] Nav pieejams: {_lstm_exc}")

# ---------------------------------------------------------------------------
# Konfigurācijas — visas 3 darbojas paralēli
# ---------------------------------------------------------------------------
LIVE_CONFIGS: list[dict] = [
    {"timeframe": "5m",  "horizon": 2, "min_prob": 0.55, "label": "5m-2bar",
     "filters": {"atr7_pct": 4.14, "slope_abs_pct": 1.16}},
    {"timeframe": "5m",  "horizon": 3, "min_prob": 0.65, "label": "5m-3bar",
     "filters": {"atr7_pct": 3.80, "roc10_abs_pct": 2.49, "slope_abs_pct": 1.26}},
    {"timeframe": "15m", "horizon": 2, "min_prob": 0.55, "label": "15m-2bar",
     "filters": {"roc10_abs_pct": 6.81}},
    # NOMUSDT — rule-based (bez ML modela): ATR7>=0.99% + |ma10_slope|>=0.78%, virziens no MA100
    {"timeframe": "15m", "horizon": 3, "min_prob": 0.0,  "label": "15m-3bar",
     "symbol": "NOMUSDT", "rule_based": True,
     "filters": {"atr7_pct": 0.99, "slope_abs_pct": 0.78}},
    # ONTUSDT — rule-based: ATR7>=1.65% + |ma10_slope|>=1.11%, virziens no MA100
    {"timeframe": "15m", "horizon": 3, "min_prob": 0.0,  "label": "15m-3bar",
     "symbol": "ONTUSDT", "rule_based": True,
     "filters": {"atr7_pct": 1.65, "slope_abs_pct": 1.11}},
    # ONTUSDT — ML XGB 15m-2bar (min_prob=0.62), ATR>=0.79% filter → PF=3.03 live
    {"timeframe": "15m", "horizon": 2, "min_prob": 0.62, "label": "15m-2bar",
     "symbol": "ONTUSDT",
     "filters": {"atr7_pct": 0.79}},
    # STOUSDT — ML XGB 15m-1bar (min_prob=0.58), |roc10|>=2.0% filter → PF=2.62 live
    {"timeframe": "15m", "horizon": 1, "min_prob": 0.58, "label": "15m-1bar",
     "symbol": "STOUSDT",
     "filters": {"roc10_abs_pct": 2.00}},
]

TF_BAR_SECONDS: dict[str, int] = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
}
WARMUP_BARS     = 120
INITIAL_CAPITAL = 10_000.0
COL_W           = 22   # display column width per config
JSON_STATE_PATH  = Path("live_state.json")
HTTP_PORT        = 8080

# ── LSTM live configs (6 varianti = 3 horizons × 2 exit veidi) ──────────────
# exit_mode: "horizon" — iziet pēc horizon svecēm
#            "tpsl"    — iziet pie TP(+0.5%) / SL(-0.3%), fallback → horizon
LSTM_LIVE_CONFIGS: list[dict] = [
    {"horizon": 1, "exit_mode": "horizon", "min_prob": 0.60, "label": "LSTM-1bar-hrz"},
    {"horizon": 1, "exit_mode": "tpsl",    "min_prob": 0.60, "label": "LSTM-1bar-tpsl"},
    {"horizon": 2, "exit_mode": "horizon", "min_prob": 0.60, "label": "LSTM-2bar-hrz"},
    {"horizon": 2, "exit_mode": "tpsl",    "min_prob": 0.60, "label": "LSTM-2bar-tpsl"},
    {"horizon": 3, "exit_mode": "horizon", "min_prob": 0.60, "label": "LSTM-3bar-hrz"},
    {"horizon": 3, "exit_mode": "tpsl",    "min_prob": 0.60, "label": "LSTM-3bar-tpsl"},
]

# TP/SL konstantes (atbilst lstm/backtest.py)
LSTM_TAKE_PROFIT =  0.0050   # +0.50%
LSTM_STOP_LOSS   = -0.0030   # -0.30%
LSTM_MODELS_DIR  = LSTM_DIR / "saved_models"


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
def load_models(symbol: str, timeframe: str, hold_bars: int) -> dict | None:
    base = str(MODELS_DIR / f"{symbol}_{timeframe}_{hold_bars}bar")
    out: dict = {}
    for name, ext in [("lgb_long", "txt"), ("lgb_short", "txt"),
                      ("xgb_long", "json"), ("xgb_short", "json")]:
        path = Path(f"{base}_{name}.{ext}")
        if not path.exists():
            print(f"  [ERROR] Model not found: {path}")
            print(f"  Run: python ml_model.py --symbol {symbol} "
                  f"--timeframes {timeframe} --download --cutoff 2026-03-20")
            return None
        if ext == "txt":
            out[name] = _LGBWrapper(lgb_lib.Booster(model_file=str(path)))
        else:
            m = xgb_lib.XGBClassifier()
            m.load_model(str(path))
            out[name] = m
    return out


# ---------------------------------------------------------------------------
# Fetch latest N candles — single fast API call
# ---------------------------------------------------------------------------
def fetch_latest(symbol: str, timeframe: str, n_bars: int = 300) -> pd.DataFrame:
    gran   = TF_GRAN_MAP[timeframe]
    now_ms = int(time.time() * 1000)
    params = {
        "symbol": symbol, "granularity": gran,
        "productType": "USDT-FUTURES",
        "endTime": str(now_ms), "limit": str(min(n_bars, 1000)),
    }
    url = ("https://api.bitget.com/api/v2/mix/market/candles?"
           + urllib.parse.urlencode(params))
    with urllib.request.urlopen(url, timeout=15) as r:
        data = json.loads(r.read())
    if data.get("code") != "00000":
        raise RuntimeError(f"API error: {data.get('msg')}")
    rows = data.get("data", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(
        [[int(r[0]), float(r[1]), float(r[2]), float(r[3]),
          float(r[4]), float(r[5])] for r in rows],
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df = (df.drop_duplicates("timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True))
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


# ---------------------------------------------------------------------------
# Position tracker
# ---------------------------------------------------------------------------
class Position:
    def __init__(self, direction: str, entry_price: float,
                 entry_time: str, horizon: int):
        self.direction   = direction
        self.entry_price = entry_price
        self.entry_time  = entry_time
        self.horizon     = horizon
        self.bars_held   = 0

    def unrealized_pct(self, current_price: float) -> float:
        if self.direction == "long":
            gross = (current_price - self.entry_price) / self.entry_price
        else:
            gross = (self.entry_price - current_price) / self.entry_price
        return (gross - 2 * FEE_RATE) * 100


# ---------------------------------------------------------------------------
# Per-config simulation state
# ---------------------------------------------------------------------------
class SimState:
    def __init__(self, cfg: dict, symbol: str, models: dict | None):
        self.label      = cfg["label"]
        self.timeframe  = cfg["timeframe"]
        self.horizon    = cfg["horizon"]
        self.min_prob   = cfg["min_prob"]
        self.symbol     = cfg.get("symbol", symbol)   # per-config symbol override
        self.rule_based = cfg.get("rule_based", False)
        self.models     = models

        self.capital    = INITIAL_CAPITAL
        self.position: Position | None = None
        self.closed_trades: list[dict] = []
        self.last_candle_ts = 0

        # display state (updated in process())
        self.p_long      = 0.0
        self.p_short     = 0.0
        self.go_long     = False
        self.go_short    = False
        self.above_ma100 = 0
        self.last_close  = 0.0
        self.last_bar_dt = "—"
        self.trade_log   = ""
        self.status      = "loading"   # "ok" | "wait" | "err:<msg>"

        # filter thresholds (from config) and last computed values
        self.filters     = cfg.get("filters", {})
        self.filt_atr    = 0.0   # last atr7 in %
        self.filt_roc    = 0.0   # last |roc10| in %
        self.filt_slope  = 0.0   # last |ma10_slope| in %
        self.filt_pass   = True  # whether all filters passed

    def process(self, df: pd.DataFrame) -> None:
        """Recalculate signals and manage position given fresh candle data."""
        if df.empty or len(df) < WARMUP_BARS:
            self.status = "wait"
            return

        # iloc[-1] is still forming; use iloc[-2] as last closed bar
        closed_df   = df.iloc[:-1].reset_index(drop=True)
        current_bar = closed_df.iloc[-1]
        current_ts  = int(current_bar["timestamp"])

        self.last_close  = float(current_bar["close"])
        self.last_bar_dt = str(current_bar["datetime"])[:16].replace("T", " ")

        # ── Features ──────────────────────────────────────────────────────
        features = build_features(closed_df)
        features = features.replace([np.inf, -np.inf], np.nan)
        mask     = features.isna().any(axis=1)
        feat_cl  = features[~mask].reset_index(drop=True)
        if feat_cl.empty:
            return

        last_feat        = feat_cl.iloc[[-1]][feat_cl.columns.tolist()].values
        self.above_ma100 = int(feat_cl["above_ma100"].iloc[-1])

        # ── Signal quality filters (ATR / roc10 / ma10_slope) ─────────────────────
        self.filt_atr   = float(feat_cl["atr7"].iloc[-1]) * 100
        self.filt_roc   = abs(float(feat_cl["roc10"].iloc[-1])) * 100
        slope_raw       = float(feat_cl["ma10_slope"].iloc[-1]) * 100
        self.filt_slope = abs(slope_raw)
        flt = self.filters
        atr_ok   = self.filt_atr   >= flt.get("atr7_pct",      0.0)
        roc_ok   = self.filt_roc   >= flt.get("roc10_abs_pct", 0.0)
        slope_ok = self.filt_slope >= flt.get("slope_abs_pct", 0.0)
        self.filt_pass = atr_ok and roc_ok and slope_ok

        # ── Inference ─────────────────────────────────────────────────────
        if self.rule_based:
            # Pure rule signal: ATR + |slope| threshold, virziens no slope + MA100
            self.go_long  = self.filt_pass and (slope_raw > 0) and (self.above_ma100 == 1)
            self.go_short = self.filt_pass and (slope_raw < 0) and (self.above_ma100 == 0)
            # Normalizets ATR kā pseudo-prob displeja joslai (2x threshold = pilna josla)
            norm = min(self.filt_atr / max(flt.get("atr7_pct", 1.0) * 2, 0.01), 1.0)
            self.p_long  = norm if self.go_long  else (norm * 0.3 if self.filt_pass else 0.0)
            self.p_short = norm if self.go_short else (norm * 0.3 if self.filt_pass else 0.0)
        else:
            assert self.models is not None
            self.p_long  = float(self.models["xgb_long"].predict_proba(last_feat)[0, 1])
            self.p_short = float(self.models["xgb_short"].predict_proba(last_feat)[0, 1])
            self.go_long  = (self.p_long  >= self.min_prob) and (self.above_ma100 == 1) and self.filt_pass
            self.go_short = (self.p_short >= self.min_prob) and (self.above_ma100 == 0) and self.filt_pass

        self.status    = "ok"
        self.trade_log = ""
        new_bar = (current_ts != self.last_candle_ts)

        # ── Position management (only on new closed bar) ───────────────────
        if new_bar:
            self.last_candle_ts = current_ts

            if self.position is not None:
                self.position.bars_held += 1
                if self.position.bars_held >= self.position.horizon:
                    close_price = float(current_bar["open"])
                    pnl_pct     = self.position.unrealized_pct(close_price)
                    self.capital += self.capital * pnl_pct / 100
                    flag = "W" if pnl_pct > 0 else "L"
                    self.closed_trades.append({
                        "flag":      flag,
                        "direction": self.position.direction,
                        "exit":      self.last_bar_dt,
                        "pnl_pct":   round(pnl_pct, 3),
                        "capital":   round(self.capital, 2),
                    })
                    self.trade_log = (f"CLOSED {self.position.direction.upper()} "
                                      f"{pnl_pct:+.2f}%  cap:{self.capital:.0f}")
                    self.position = None

            if self.position is None and (self.go_long or self.go_short):
                direction   = "long" if self.go_long else "short"
                entry_price = float(current_bar["close"])
                self.position = Position(direction, entry_price,
                                         self.last_bar_dt, self.horizon)
                self.trade_log += (f" OPEN {direction.upper()} "
                                   f"@ {entry_price:.6f}")

    def unrealized_now(self, live_price: float) -> float | None:
        if self.position is None:
            return None
        return self.position.unrealized_pct(live_price)


# ---------------------------------------------------------------------------
# LSTM per-config simulation state
# ---------------------------------------------------------------------------
class LstmSimState:
    """
    Live simulation for one LSTM horizon + exit_mode variant.

    Two exit modes:
      "horizon" — iziet exacti pēc `horizon` 5m svecēm
      "tpsl"    — iziet pie TP(+0.50%) vai SL(-0.30%); fallback → horizon
    """

    def __init__(self, cfg: dict, symbol: str):
        self.label     = cfg["label"]
        self.horizon   = cfg["horizon"]
        self.exit_mode = cfg["exit_mode"]   # "horizon" | "tpsl"
        self.min_prob  = cfg["min_prob"]
        self.symbol    = symbol
        self.timeframe = "5m"   # LSTM vienmēr 5m bāze

        # ── Modeļi ───────────────────────────────────────────────────────────
        self.model  = None    # tf.keras.Model
        self.scaler = None    # sklearn MinMaxScaler
        self.loaded = False

        # ── Simulācijas stāvoklis ────────────────────────────────────────────
        self.capital     = INITIAL_CAPITAL
        self.position: Position | None = None
        self.closed_trades: list[dict] = []
        self.last_candle_ts = 0

        # ── Displeja stāvoklis ───────────────────────────────────────────────
        self.p_long      = 0.0
        self.go_long     = False
        self.above_ma100 = 0
        self.last_close  = 0.0
        self.last_bar_dt = "—"
        self.trade_log   = ""
        self.status      = "loading"

        # tpsl exit — seko atvērtajai pozīcijai bar-by-bar
        self._pending_exit: str | None = None   # "tp" | "sl" | None

    def load_models(self) -> bool:
        """Ielādē LSTM .keras modeli un scaler. Atgriež True ja veiksmīgi."""
        if not LSTM_AVAILABLE:
            self.status = "err:no_tensorflow"
            return False
        h = self.horizon
        model_path  = LSTM_MODELS_DIR / f"lstm_{self.symbol}_{h}bar.keras"
        scaler_path = LSTM_MODELS_DIR / f"scaler_{self.symbol}_{h}bar.pkl"
        for p in [model_path, scaler_path]:
            if not p.exists():
                self.status = f"err:nav_{p.name}"
                return False
        try:
            self.model  = _tf.keras.models.load_model(str(model_path))
            self.scaler = _joblib.load(scaler_path)
            self.loaded = True
            self.status = "ok"
            return True
        except Exception as exc:
            self.status = f"err:{exc}"
            return False

    def process(self, df_5m: pd.DataFrame, df_15m: pd.DataFrame) -> None:
        """
        Aprēķina LSTM signālu un pārvalda pozīciju.

        df_5m / df_15m — pilni OHLCV DataFrame (>LSTM_WINDOW+50 sveces).
        """
        if not self.loaded:
            self.status = "err:not_loaded"
            return
        if df_5m.empty or len(df_5m) < LSTM_WINDOW + 60:
            self.status = "wait"
            return

        # Pēdējā slēgtā svece
        closed_5m   = df_5m.iloc[:-1].reset_index(drop=True)
        current_bar = closed_5m.iloc[-1]
        current_ts  = int(current_bar["timestamp"])

        self.last_close  = float(current_bar["close"])
        self.last_bar_dt = str(current_bar["datetime"])[:16].replace("T", " ")

        # ── MA100 trends ──────────────────────────────────────────────────────
        closes = closed_5m["close"].values
        if len(closes) >= 100:
            ma100 = closes[-100:].mean()
            self.above_ma100 = int(self.last_close > ma100)

        # ── LSTM features + secence ───────────────────────────────────────────
        try:
            feat_df = _lstm_merge_tfs(
                closed_5m,
                df_15m.iloc[:-1].reset_index(drop=True),
            )
            # Pēdējās LSTM_WINDOW sveces — viena secence
            feat_clean = feat_df.replace([np.inf, -np.inf], np.nan).dropna()
            if len(feat_clean) < LSTM_WINDOW:
                self.status = "wait"
                return

            seq = feat_clean.values[-LSTM_WINDOW:].astype(np.float32)
            seq_scaled = _lstm_scale(
                seq[np.newaxis, :, :],   # shape (1, WINDOW, features)
                self.scaler,
            )
            prob = float(self.model.predict(seq_scaled, verbose=0)[0, 0])
        except Exception as exc:
            self.status = f"err:{exc}"
            return

        self.p_long  = prob
        self.go_long = (prob >= self.min_prob) and (self.above_ma100 == 1)
        self.status  = "ok"
        self.trade_log = ""
        new_bar = (current_ts != self.last_candle_ts)

        if not new_bar:
            return
        self.last_candle_ts = current_ts

        # ── Pozīcijas pārvaldība ──────────────────────────────────────────────
        if self.position is not None:
            self.position.bars_held += 1
            pos = self.position
            close_price = float(current_bar["close"])
            high_price  = float(current_bar["high"])
            low_price   = float(current_bar["low"])
            closed = False
            pnl_pct: float | None = None

            if self.exit_mode == "tpsl":
                # Pārbauda TP/SL intra-bar
                tp_price = pos.entry_price * (1.0 + LSTM_TAKE_PROFIT)
                sl_price = pos.entry_price * (1.0 + LSTM_STOP_LOSS)
                if high_price >= tp_price:
                    gross   = LSTM_TAKE_PROFIT
                    pnl_pct = (gross - 2 * FEE_RATE) * 100
                    closed  = True
                    self._pending_exit = "tp"
                elif low_price <= sl_price:
                    gross   = LSTM_STOP_LOSS
                    pnl_pct = (gross - 2 * FEE_RATE) * 100
                    closed  = True
                    self._pending_exit = "sl"
                elif pos.bars_held >= pos.horizon:
                    gross   = (close_price - pos.entry_price) / pos.entry_price
                    pnl_pct = (gross - 2 * FEE_RATE) * 100
                    closed  = True
                    self._pending_exit = "hrz"
            else:
                # Horizon exit
                if pos.bars_held >= pos.horizon:
                    gross   = (close_price - pos.entry_price) / pos.entry_price
                    pnl_pct = (gross - 2 * FEE_RATE) * 100
                    closed  = True

            if closed and pnl_pct is not None:
                self.capital += self.capital * pnl_pct / 100
                flag = "W" if pnl_pct > 0 else "L"
                exit_tag = f"[{self._pending_exit}]" if self._pending_exit else ""
                self.closed_trades.append({
                    "flag":      flag,
                    "direction": pos.direction,
                    "exit":      self.last_bar_dt,
                    "pnl_pct":   round(pnl_pct, 3),
                    "capital":   round(self.capital, 2),
                    "exit_type": self._pending_exit or "hrz",
                })
                self.trade_log = (
                    f"CLOSED {pos.direction.upper()} {exit_tag}"
                    f"{pnl_pct:+.2f}%  cap:{self.capital:.0f}"
                )
                self.position = None
                self._pending_exit = None

        if self.position is None and self.go_long:
            entry_price = float(current_bar["close"])
            self.position = Position("long", entry_price,
                                     self.last_bar_dt, self.horizon)
            self.trade_log += f" OPEN LONG @ {entry_price:.6f}"

    def unrealized_now(self, live_price: float) -> float | None:
        if self.position is None:
            return None
        return self.position.unrealized_pct(live_price)

    # Saskaņota saskarne ar SimState (display helpers lieto šos)
    @property
    def rule_based(self) -> bool:
        return False

    @property
    def p_short(self) -> float:
        return 0.0

    @property
    def go_short(self) -> bool:
        return False

    @property
    def filters(self) -> dict:
        return {}

    @property
    def filt_atr(self) -> float:
        return 0.0

    @property
    def filt_roc(self) -> float:
        return 0.0

    @property
    def filt_slope(self) -> float:
        return 0.0

    @property
    def filt_pass(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Console display helpers
# ---------------------------------------------------------------------------
def _pbar(p: float, w: int = 10) -> str:
    filled = min(int(p * w), w)
    return "[" + "#" * filled + "." * (w - filled) + "]"


def _fpnl(v: float) -> str:
    return f"{'+' if v >= 0 else ''}{v:.2f}%"


def _col(text: str, width: int = COL_W) -> str:
    """Left-align text in a fixed-width column."""
    return text[:width].ljust(width)


def display_all(symbol: str, sims: list,
                data_cache: dict, no_wait: bool) -> None:
    now = datetime.now(timezone.utc)
    os.system("cls" if os.name == "nt" else "clear")

    n   = len(sims)
    sep = "  " + ("-" * COL_W + "  ") * n
    wide = "=" * (COL_W * n + 2 * n + 2)

    def row(*cols: str) -> str:
        return "  " + "  ".join(_col(c) for c in cols)

    # ── Header ─────────────────────────────────────────────────────────────
    all_syms = " + ".join(dict.fromkeys(s.symbol for s in sims))
    print(wide)
    print(f"  LIVE SIGNALS  {all_syms}  —  {n} konfigurācijas paralēli")
    print(f"  {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(wide)

    # ── Config labels ──────────────────────────────────────────────────────
    def _cfg_hdr(s) -> str:
        if isinstance(s, LstmSimState):
            return f"  {s.label} (>={s.min_prob:.0%})"
        if s.rule_based:
            return f"  {s.symbol} {s.label} (rule)"
        return f"  {s.label}  (>={s.min_prob:.0%})"
    print(row(*[_cfg_hdr(s) for s in sims]))
    print(sep)

    # Live prices from the forming candle (iloc[-1] of each TF data)
    live_prices = {
        key: (float(df.iloc[-1]["close"]) if not df.empty else 0.0)
        for key, df in data_cache.items()
    }

    # ── Price & trend ──────────────────────────────────────────────────────
    print(row(*[f"  Close : {s.last_close:.6f}" for s in sims]))
    print(row(*[f"  Live  : {live_prices.get((s.symbol, s.timeframe), 0):.6f}" for s in sims]))
    print(row(*[f"  Trend : {'ABOVE' if s.above_ma100 else 'BELOW'} MA100" for s in sims]))
    print(sep)

    # ── Probabilities ──────────────────────────────────────────────────────
    def prob_line(label: str, p_attr: str, sig_attr: str) -> str:
        parts = []
        for s in sims:
            p   = getattr(s, p_attr)
            sig = " <<" if getattr(s, sig_attr) else "   "
            parts.append(f"  {label} {p:5.1%} {_pbar(p, 8)}{sig}")
        return row(*parts)

    print(prob_line("P(L):", "p_long",  "go_long"))
    print(prob_line("P(S):", "p_short", "go_short"))

    # ── Signal filter status ─────────────────────────────────────────────
    def filt_status(s: SimState) -> str:
        parts = []
        flt = s.filters
        if "atr7_pct" in flt:
            ch = "+" if s.filt_atr >= flt["atr7_pct"] else "-"
            parts.append(f"A{ch}{s.filt_atr:.1f}")
        if "roc10_abs_pct" in flt:
            ch = "+" if s.filt_roc >= flt["roc10_abs_pct"] else "-"
            parts.append(f"R{ch}{s.filt_roc:.1f}")
        if "slope_abs_pct" in flt:
            ch = "+" if s.filt_slope >= flt["slope_abs_pct"] else "-"
            parts.append(f"S{ch}{s.filt_slope:.1f}")
        if not parts:
            return "  (no filter)"
        status = " [OK]" if s.filt_pass else " [--]"
        return "  " + " ".join(parts) + status

    print(row(*[filt_status(s) for s in sims]))
    print(sep)

    # ── Open positions ─────────────────────────────────────────────────────
    for s in sims:
        lp  = live_prices.get(s.timeframe, 0.0)
        unr = s.unrealized_now(lp)
        if s.position is not None:
            pos = s.position
            print(f"  [{s.label}]  {pos.direction.upper():5s}  "
                  f"bar {pos.bars_held}/{pos.horizon}  "
                  f"entry {pos.entry_price:.6f}  "
                  f"unreal {_fpnl(unr) if unr is not None else '--':>8}")
        else:
            print(f"  [{s.label}]  No open position")
    print(sep)

    # ── Session stats ──────────────────────────────────────────────────────
    def stat1(s: SimState) -> str:
        ret = (s.capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        return f"  Cap:{s.capital:>9.2f}  {_fpnl(ret):>8}"

    def stat2(s: SimState) -> str:
        total = len(s.closed_trades)
        wins  = sum(1 for t in s.closed_trades if t["pnl_pct"] > 0)
        wr    = wins / total * 100 if total else 0.0
        return f"  Trades:{total:3d}   WR:{wr:4.0f}%  "

    print(row(*[stat1(s) for s in sims]))
    print(row(*[stat2(s) for s in sims]))
    print(sep)

    # ── Trade events this cycle ────────────────────────────────────────────
    if any(s.trade_log for s in sims):
        print()
        for s in sims:
            if s.trade_log:
                print(f"  [{s.label}] {s.trade_log}")

    # ── Last 3 closed trades per config ────────────────────────────────────
    if any(s.closed_trades for s in sims):
        print()
        print("  --- Pedejie darijumi ---")
        for s in sims:
            last  = s.closed_trades[-3:]
            parts = "  ".join(
                f"[{t['flag']}]{t['direction'][:1].upper()}{t['pnl_pct']:+.2f}%"
                for t in last
            )
            print(f"  {s.label:9s}: {parts if parts else '--'}")

    # ── Next bar timers ────────────────────────────────────────────────────
    print()
    timers: list[str] = []
    seen_keys: set[tuple] = set()
    for s in sims:
        key = (s.symbol, s.timeframe)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        tf = s.timeframe
        bar_sec = TF_BAR_SECONDS.get(tf, 900)
        df = data_cache.get(key)
        lbl = f"{s.symbol}/{tf}" if s.symbol != symbol else tf
        if df is not None and len(df) >= 2:
            last_ts     = int(df.iloc[-2]["timestamp"])
            bar_ms      = bar_sec * 1000
            next_bar_ms = (last_ts // bar_ms + 1) * bar_ms + bar_ms
            wait_sec    = max(0.0, (next_bar_ms - time.time() * 1000) / 1000)
            timers.append(f"{lbl}: {int(wait_sec // 60):02d}:{int(wait_sec % 60):02d}")
        else:
            timers.append(f"{lbl}: --:--")
    mode = "--no-wait (5s)" if no_wait else "15s refresh"
    print(f"  Nakama svece  —  {',   '.join(timers)}  [{mode}]")
    print(wide)


# ---------------------------------------------------------------------------
# HTTP server for web dashboard
# ---------------------------------------------------------------------------
def _start_http_server(port: int = HTTP_PORT) -> None:
    """Serve CWD over HTTP so live_dashboard.html can fetch live_state.json."""
    class _Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):          # suppress access logs
            pass
    class _Server(socketserver.TCPServer):
        allow_reuse_address = True
    try:
        with _Server(("", port), _Handler) as httpd:
            httpd.serve_forever()
    except OSError:
        pass   # port already in use — ignore


# ---------------------------------------------------------------------------
# JSON state writer — called every refresh cycle
# ---------------------------------------------------------------------------
def write_state_json(symbol: str, sims: list,
                     data_cache: dict, no_wait: bool) -> None:
    now = datetime.now(timezone.utc)
    live_prices = {
        key: float(df.iloc[-1]["close"]) if not df.empty else 0.0
        for key, df in data_cache.items()
    }

    configs_out: list[dict] = []
    for s in sims:
        lp  = live_prices.get((s.symbol, s.timeframe), 0.0)
        unr = s.unrealized_now(lp)

        # Next-bar countdown
        bar_sec = TF_BAR_SECONDS.get(s.timeframe, 900)
        df_c    = data_cache.get((s.symbol, s.timeframe))
        next_bar_secs: int | None = None
        if df_c is not None and len(df_c) >= 2:
            last_ts     = int(df_c.iloc[-2]["timestamp"])
            bar_ms      = bar_sec * 1000
            next_bar_ms = (last_ts // bar_ms + 1) * bar_ms + bar_ms
            next_bar_secs = max(0, int((next_bar_ms - time.time() * 1000) / 1000))

        pos_data: dict | None = None
        if s.position is not None:
            pos = s.position
            pos_data = {
                "direction":     pos.direction,
                "entry_price":   pos.entry_price,
                "entry_time":    pos.entry_time,
                "bars_held":     pos.bars_held,
                "horizon":       pos.horizon,
                "unrealized_pct": round(unr, 3) if unr is not None else None,
            }

        total   = len(s.closed_trades)
        wins    = sum(1 for t in s.closed_trades if t["pnl_pct"] > 0)
        ret_pct = (s.capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

        configs_out.append({
            "label":         s.label,
            "symbol":        s.symbol,
            "timeframe":     s.timeframe,
            "horizon":       s.horizon,
            "min_prob":      s.min_prob,
            "rule_based":    s.rule_based,
            "model_type":    "lstm" if isinstance(s, LstmSimState) else ("rule" if s.rule_based else "xgb"),
            "exit_mode":     getattr(s, "exit_mode", "horizon"),
            "status":        s.status,
            "last_close":    round(s.last_close, 8),
            "live_price":    round(lp, 8),
            "above_ma100":   s.above_ma100,
            "p_long":        round(s.p_long,  4),
            "p_short":       round(s.p_short, 4),
            "go_long":       s.go_long,
            "go_short":      s.go_short,
            "filters":       s.filters,
            "filt_atr":      round(s.filt_atr,   3),
            "filt_roc":      round(s.filt_roc,   3),
            "filt_slope":    round(s.filt_slope, 3),
            "filt_pass":     s.filt_pass,
            "position":      pos_data,
            "capital":       round(s.capital, 2),
            "return_pct":    round(ret_pct, 3),
            "trades_total":  total,
            "trades_wins":   wins,
            "win_rate":      round(wins / total * 100, 1) if total else 0.0,
            "recent_trades": s.closed_trades[-5:],
            "next_bar_secs": next_bar_secs,
        })

    state = {
        "updated_utc": now.strftime("%Y-%m-%d %H:%M:%S"),
        "updated_ts":  int(now.timestamp()),
        "symbol":      symbol,
        "no_wait":     no_wait,
        "configs":     configs_out,
    }
    try:
        JSON_STATE_PATH.write_text(json.dumps(state, indent=2))
    except Exception as exc:
        print(f"  [JSON write error] {exc}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run(symbol: str, no_wait: bool) -> None:
    print(f"\n  Ieladam XGB konfiguracijas ...")
    sims: list[SimState] = []
    for cfg in LIVE_CONFIGS:
        cfg_symbol = cfg.get("symbol", symbol)
        if cfg.get("rule_based", False):
            sims.append(SimState(cfg, symbol, None))
            print(f"  OK   {cfg['label']}  ({cfg_symbol} {cfg['timeframe']} "
                  f"{cfg['horizon']}-bar  RULE-BASED  "
                  f"ATR>={cfg['filters'].get('atr7_pct',0):.2f}%  "
                  f"Slp>={cfg['filters'].get('slope_abs_pct',0):.2f}%)")
        else:
            models = load_models(cfg_symbol, cfg["timeframe"], cfg["horizon"])
            if models is None:
                print(f"  [SKIP] {cfg['label']} — modeLi nav atrasti")
                continue
            sims.append(SimState(cfg, symbol, models))
            print(f"  OK   {cfg['label']}  ({cfg_symbol} {cfg['timeframe']} {cfg['horizon']}-bar "
                  f"XGB  min_prob={cfg['min_prob']:.0%})")

    # ── LSTM simulācijas ────────────────────────────────────────────────────
    lstm_sims: list[LstmSimState] = []
    if LSTM_AVAILABLE:
        print(f"\n  Ieladam LSTM konfiguracijas ({len(LSTM_LIVE_CONFIGS)} varianti)...")
        for cfg in LSTM_LIVE_CONFIGS:
            ls = LstmSimState(cfg, symbol)
            ok = ls.load_models()
            if ok:
                lstm_sims.append(ls)
                print(f"  OK   {cfg['label']}  ({symbol} 5m {cfg['horizon']}-bar  "
                      f"LSTM  exit={cfg['exit_mode']}  min_prob={cfg['min_prob']:.0%})")
            else:
                print(f"  [SKIP] {cfg['label']} — {ls.status}")
    else:
        print(f"\n  [LSTM] TensorFlow nav pieejams — LSTM simulācijas izlaistas")

    all_sims: list[SimState | LstmSimState] = sims + lstm_sims  # type: ignore[list-item]

    if not all_sims:
        print("  Nav neviena modeLa. Izejam.")
        return

    # Unikālie (symbol, timeframe) fetch pāri — XGB
    unique_sym_tfs = list(dict.fromkeys((s.symbol, s.timeframe) for s in sims))
    # LSTM vienmēr vajag SIRENUSDT/5m un SIRENUSDT/15m
    lstm_sym_tfs: list[tuple[str, str]] = []
    if lstm_sims:
        for tf in ("5m", "15m"):
            key = (symbol, tf)
            if key not in unique_sym_tfs:
                lstm_sym_tfs.append(key)

    all_fetch_pairs = list(dict.fromkeys(unique_sym_tfs + lstm_sym_tfs))
    tf_summary = ", ".join(f"{sym}/{tf}" for sym, tf in all_fetch_pairs)
    print(f"\n  Feeds           : {tf_summary}")
    print(f"  XGB konfigur.   : {len(sims)}")
    print(f"  LSTM konfigur.  : {len(lstm_sims)}")
    print(f"  Fee rate        : {FEE_RATE * 100:.2f}% per puse")
    print(f"  Sakotnejais kap : {INITIAL_CAPITAL:,.0f} USDT (simulacija)")
    print(f"\n  Ctrl+C lai apstaatos ...\n")
    threading.Thread(target=_start_http_server, daemon=True).start()
    print(f"  Dashboard       : http://localhost:{HTTP_PORT}/live_dashboard.html")
    time.sleep(2)

    data_cache: dict[tuple, pd.DataFrame] = {}

    while True:
        # Fetch fresh data for each unique (symbol, timeframe) pair
        for sym, tf in all_fetch_pairs:
            key = (sym, tf)
            n_bars = 300 if (sym == symbol and tf == "5m" and lstm_sims) else 250
            try:
                data_cache[key] = fetch_latest(sym, tf, n_bars=n_bars)
            except Exception as exc:
                data_cache.setdefault(key, pd.DataFrame())
                print(f"  [fetch error {sym}/{tf}] {exc}")

        # Process XGB configs
        for s in sims:
            df = data_cache.get((s.symbol, s.timeframe), pd.DataFrame())
            try:
                s.process(df)
            except Exception as exc:
                s.status = f"err:{exc}"

        # Process LSTM configs (vajag gan 5m, gan 15m)
        df_5m_lstm  = data_cache.get((symbol, "5m"),  pd.DataFrame())
        df_15m_lstm = data_cache.get((symbol, "15m"), pd.DataFrame())
        for ls in lstm_sims:
            try:
                ls.process(df_5m_lstm, df_15m_lstm)
            except Exception as exc:
                ls.status = f"err:{exc}"

        # Render unified display
        display_all(symbol, all_sims, data_cache, no_wait)
        write_state_json(symbol, all_sims, data_cache, no_wait)

        time.sleep(5 if no_wait else 15)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live signal simulator — XGB + LSTM konfigurācijas paralēli"
    )
    parser.add_argument("--symbol",  default="SIRENUSDT",
                        help="Tirdzniecibas paris (default: SIRENUSDT)")
    parser.add_argument("--no-wait", action="store_true",
                        help="Refresh every 5s (testesanai, bez gaidisanas uz jaunu sveci)")
    args = parser.parse_args()

    try:
        run(symbol=args.symbol, no_wait=args.no_wait)
    except KeyboardInterrupt:
        print("\n\n  Apturets.")


if __name__ == "__main__":
    main()
