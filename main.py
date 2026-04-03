"""
Bitget MACD + EMA Regime Trading Bot
======================================
Strategy: macd_ema
  - Indicators : MACD(fast, slow, signal) + EMA20 + EMA50 + EMA200
  - Regime filter:
      Bullish : EMA20 > EMA50 > EMA200  →  only LONG entries
      Bearish : EMA20 < EMA50 < EMA200  →  only SHORT entries
      Neutral : no entries
  - Long entry  : bullish regime  AND  close > EMA50  AND  MACD crosses ABOVE signal
  - Short entry : bearish regime  AND  close < EMA50  AND  MACD crosses BELOW signal
  - Long exit   : MACD line crosses BELOW signal line
  - Short exit  : MACD line crosses ABOVE signal line
  - No look-ahead bias: signal evaluated on closed bar, entry at next bar open

Usage:
  python main.py --download          # download OHLCV data
  python main.py --backtest          # run backtest with config settings
  python main.py --download --backtest
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.request
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Candle:
    timestamp: int        # Unix ms
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def dt(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp / 1000, tz=timezone.utc)

    def __str__(self) -> str:
        return f"{self.dt.strftime('%Y-%m-%d %H:%M')} UTC  O={self.open}  H={self.high}  L={self.low}  C={self.close}  V={self.volume}"


@dataclass
class MACDPoint:
    macd: float       # fast EMA - slow EMA
    signal: float     # EMA of macd line
    histogram: float  # macd - signal


@dataclass
class PendingOrder:
    direction: str        # "long" | "short"
    entry_price: float    # next bar open
    signal_bar_index: int


@dataclass
class Position:
    direction: str
    entry_price: float
    entry_index: int
    size: float           # units of base asset


@dataclass
class Trade:
    direction: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    exit_reason: str


# ---------------------------------------------------------------------------
# Indicator calculations
# ---------------------------------------------------------------------------

def calculate_ema(values: list[float], length: int) -> list[float]:
    """Standard EMA. Seeds from the first value (no NaN warmup)."""
    if not values:
        return []
    mult = 2.0 / (length + 1)
    result: list[float] = []
    ema = values[0]
    for v in values:
        ema = v * mult + ema * (1.0 - mult)
        result.append(ema)
    return result


def calculate_macd(
    closes: list[float],
    fast: int,
    slow: int,
    signal_len: int,
) -> list[MACDPoint]:
    """
    MACD calculation.
      macd      = EMA(fast) - EMA(slow)
      signal    = EMA(macd, signal_len)
      histogram = macd - signal
    Returns a MACDPoint for every input close (same length as closes).
    """
    if not closes:
        return []

    fast_ema = calculate_ema(closes, fast)
    slow_ema = calculate_ema(closes, slow)

    macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]
    signal_line = calculate_ema(macd_line, signal_len)

    result: list[MACDPoint] = []
    for m, s in zip(macd_line, signal_line):
        result.append(MACDPoint(macd=m, signal=s, histogram=m - s))
    return result


# ---------------------------------------------------------------------------
# Crossover helpers
# ---------------------------------------------------------------------------

def crossed_above(prev_l: float, prev_r: float, cur_l: float, cur_r: float) -> bool:
    """True when left line crosses from below to above right line."""
    return prev_l <= prev_r and cur_l > cur_r


def crossed_below(prev_l: float, prev_r: float, cur_l: float, cur_r: float) -> bool:
    """True when left line crosses from above to below right line."""
    return prev_l >= prev_r and cur_l < cur_r


# ---------------------------------------------------------------------------
# CSV / file helpers
# ---------------------------------------------------------------------------

def candles_csv_path(output_dir: str, market_type: str, symbol: str, timeframe: str) -> Path:
    return Path(output_dir) / market_type / f"{symbol}_{timeframe}.csv"


def save_candles_to_csv(candles: list[Candle], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "datetime_utc", "open", "high", "low", "close", "volume"])
        for c in candles:
            dt_str = datetime.fromtimestamp(c.timestamp / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            writer.writerow([c.timestamp, dt_str, c.open, c.high, c.low, c.close, c.volume])


def load_candles_from_csv(path: Path) -> list[Candle]:
    candles: list[Candle] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            candles.append(Candle(
                timestamp=int(row["timestamp"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            ))
    return candles


# ---------------------------------------------------------------------------
# Bitget API download
# ---------------------------------------------------------------------------

TIMEFRAME_MS: dict[str, int] = {
    "1m":  60_000,
    "3m":  180_000,
    "5m":  300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h":  3_600_000,
    "4h":  14_400_000,
    "1d":  86_400_000,
}


def fetch_candles_bitget(
    base_url: str,
    market_type: str,
    symbol: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
) -> list[Candle]:
    """
    Download OHLCV candles from Bitget REST API (paginated).
    Supports both spot and futures market types.
    """
    # Bitget API endpoint depends on market type
    if market_type == "spot":
        endpoint = "/api/v2/spot/market/candles"
        symbol_param = "symbol"
        granularity_param = "granularity"
        granularity_value = timeframe  # e.g. "15min" for spot
        # Bitget spot uses different granularity names
        tf_map = {
            "1m": "1min", "3m": "3min", "5m": "5min",
            "15m": "15min", "30m": "30min",
            "1h": "1H", "4h": "4H", "1d": "1D",
        }
        granularity_value = tf_map.get(timeframe, timeframe)
    else:
        endpoint = "/api/v2/mix/market/candles"
        symbol_param = "symbol"
        granularity_param = "granularity"
        tf_map = {
            "1m": "1m", "3m": "3m", "5m": "5m",
            "15m": "15m", "30m": "30m",
            "1h": "1H", "4h": "4H", "1d": "1D",
        }
        granularity_value = tf_map.get(timeframe, timeframe)

    bar_ms = TIMEFRAME_MS.get(timeframe, 900_000)
    all_candles: list[Candle] = []
    # Paginate backwards: start from end_ms and walk back to start_ms
    current_end = end_ms

    while current_end > start_ms:
        params: dict[str, str] = {
            symbol_param: symbol,
            granularity_param: granularity_value,
            "endTime": str(current_end),
            "limit": str(limit),
        }
        if market_type != "spot":
            params["productType"] = "USDT-FUTURES"

        url = base_url + endpoint + "?" + urllib.parse.urlencode(params)

        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            print(f"  [ERROR] HTTP request failed: {exc}")
            break

        if data.get("code") != "00000":
            print(f"  [ERROR] API error: {data.get('msg', data)}")
            break

        rows = data.get("data", [])
        if not rows:
            break

        batch: list[Candle] = []
        for row in rows:
            ts = int(row[0])
            if ts < start_ms:
                continue
            batch.append(Candle(
                timestamp=ts,
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
            ))

        # Sort ascending
        batch.sort(key=lambda c: c.timestamp)

        all_candles.extend(batch)

        # Find the earliest timestamp in the full raw batch (before start_ms filter)
        raw_timestamps = [int(row[0]) for row in rows]
        earliest_ts = min(raw_timestamps)

        if batch:
            print(f"  Downloaded {len(batch)} candles, range: {batch[0].dt.strftime('%Y-%m-%d %H:%M')} → {batch[-1].dt.strftime('%Y-%m-%d %H:%M')} UTC")

        # Move window back: next request ends just before earliest candle in this batch
        new_end = earliest_ts - 1
        if new_end >= current_end:
            break  # no progress
        current_end = new_end

        if earliest_ts <= start_ms:
            break  # we've covered the full range

        if len(rows) < limit:
            break  # API returned fewer than limit → no more history

        time.sleep(0.2)  # be polite to API

    # Deduplicate and sort
    seen: set[int] = set()
    unique: list[Candle] = []
    for c in sorted(all_candles, key=lambda x: x.timestamp):
        if c.timestamp not in seen:
            seen.add(c.timestamp)
            unique.append(c)

    return unique


def run_download(config: dict) -> None:
    """Download all configured symbols and timeframes."""
    base_url = config["bitget"]["base_url"]
    market_type = config["bitget"]["market_type"]
    output_dir = config["download"]["output_directory"]
    limit = int(config["download"]["limit_per_request"])
    lookback_days_map: dict[str, int] = config["download"]["lookback_days"]

    now_ms = int(time.time() * 1000)

    for market in config["markets"]:
        symbol = market["symbol"]
        timeframes = market["timeframes"]
        sym_market_type = market.get("market_type", market_type)

        for tf in timeframes:
            lookback_days = lookback_days_map.get(tf, 100)
            start_ms = now_ms - lookback_days * 24 * 3600 * 1000

            print(f"\nDownloading {symbol} {tf}  ({lookback_days} days)...")
            candles = fetch_candles_bitget(
                base_url=base_url,
                market_type=sym_market_type,
                symbol=symbol,
                timeframe=tf,
                start_ms=start_ms,
                end_ms=now_ms,
                limit=limit,
            )

            if not candles:
                print(f"  [WARNING] No data returned for {symbol} {tf}")
                continue

            path = candles_csv_path(output_dir, sym_market_type, symbol, tf)
            save_candles_to_csv(candles, path)
            print(f"  Saved {len(candles)} candles → {path}")


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def close_position(
    position: Position,
    exit_candle: Candle,
    exit_price: float,
    exit_reason: str,
    fee_rate: float,
) -> tuple[Trade, float]:
    """Close an open position and return (Trade, cash_after)."""
    entry_value = position.entry_price * position.size
    exit_value = exit_price * position.size

    if position.direction == "long":
        gross_pnl = exit_value - entry_value
    else:
        gross_pnl = entry_value - exit_value

    fees = (entry_value + exit_value) * fee_rate
    net_pnl = gross_pnl - fees
    pnl_pct = net_pnl / entry_value * 100

    cash = entry_value + net_pnl  # return initial capital + profit/loss

    trade = Trade(
        direction=position.direction,
        entry_time=datetime.fromtimestamp(
            exit_candle.timestamp / 1000, tz=timezone.utc  # will be overwritten
        ).strftime("%Y-%m-%d %H:%M"),
        exit_time=exit_candle.dt.strftime("%Y-%m-%d %H:%M"),
        entry_price=position.entry_price,
        exit_price=exit_price,
        size=position.size,
        pnl=round(net_pnl, 4),
        pnl_pct=round(pnl_pct, 4),
        exit_reason=exit_reason,
    )
    return trade, cash


def backtest_macd_ema_strategy(
    candles: list[Candle],
    timeframe: str,
    initial_capital: float,
    fee_rate: float,
    macd_fast: int,
    macd_slow: int,
    macd_signal: int,
    ema_fast_length: int,   # EMA20  – regime fast line
    ema_mid_length: int,    # EMA50  – price filter
    ema_slow_length: int,   # EMA200 – regime slow line
    exit_pct: float = 0.01,      # take-profit: exit when price moves this % from entry
    stop_loss_pct: float = 0.01,  # stop-loss:   exit when price moves this % against entry
    short_entry_mode: str = "bounce",  # "bounce" | "crossover" | "combined"
    ema200_slope_period: int = 20,     # bars to measure EMA200 slope over
    ema200_slope_min_pct: float = -3.0,  # min allowed slope % (e.g. -3 = not falling >3%)
) -> tuple[dict, list[Trade]]:
    """
    Backtest the MACD + EMA regime strategy.

    Regime filter (evaluated every bar):
      Bullish : EMA20 > EMA50 > EMA200  →  only LONG entries allowed
      Bearish : EMA20 < EMA50 < EMA200  →  only SHORT entries allowed
      Neutral : no new entries

    Entry rules:
      Long  : bullish regime  AND  close > EMA50  AND  MACD crosses above signal
      Short : bearish regime  AND  close < EMA50  AND  MACD crosses below signal

    Exit rules (intrabar, filled at exact level):
      Take-profit long  : price >= entry * (1 + exit_pct)
      Take-profit short : price <= entry * (1 - exit_pct)
      Stop-loss   long  : price <= entry * (1 - stop_loss_pct)
      Stop-loss   short : price >= entry * (1 + stop_loss_pct)
      Tie (same bar)    : stop-loss wins (conservative)
      Fallback          : end of data – close at last candle close

    No look-ahead bias:
      Entry : signal on closed bar[i], executed at bar[i+1] open.
      Exit  : target/stop hit intrabar – exit at exact price.
    """
    min_bars = max(macd_slow + macd_signal, ema_slow_length) + 2
    if len(candles) < min_bars:
        return {"error": f"Not enough candles for indicators (need {min_bars}, got {len(candles)})"}, []

    closes = [c.close for c in candles]
    macd_points = calculate_macd(closes, macd_fast, macd_slow, macd_signal)
    ema_fast_values = calculate_ema(closes, ema_fast_length)
    ema_mid_values  = calculate_ema(closes, ema_mid_length)
    ema_slow_values = calculate_ema(closes, ema_slow_length)

    tf_sec = tf_to_seconds(timeframe)  # used to compute candle close time

    cash = initial_capital
    position: Optional[Position] = None
    pending_order: Optional[PendingOrder] = None
    trades: list[Trade] = []

    equity_curve: list[float] = [initial_capital]
    peak_equity = initial_capital
    max_drawdown_pct = 0.0

    # Wait for slowest indicator to warm up before trading
    start_index = max(macd_slow + macd_signal, ema_slow_length)

    for i in range(start_index, len(candles)):
        candle = candles[i]
        macd_now = macd_points[i]
        macd_prev = macd_points[i - 1]
        ema_fast_now = ema_fast_values[i]
        ema_mid_now  = ema_mid_values[i]
        ema_slow_now = ema_slow_values[i]

        # ---- Step 1: Execute pending order at this bar's open ----
        if pending_order is not None:
            entry_price = candle.open
            entry_value = entry_price  # price per unit
            # Use all cash to buy
            size = (cash * (1.0 - fee_rate)) / entry_price
            cash = 0.0

            position = Position(
                direction=pending_order.direction,
                entry_price=entry_price,
                entry_index=i,
                size=size,
            )
            pending_order = None

        # ---- Step 2: Indicator values on this closed bar ----

        # Market regime (EMA20 > EMA50 > EMA200 = bullish, inverse = bearish)
        bullish_regime = ema_fast_now > ema_mid_now > ema_slow_now
        bearish_regime = ema_fast_now < ema_mid_now < ema_slow_now

        # EMA200 slope filter: skip signals if EMA200 is falling too steeply
        slope_ok = True
        if i >= ema200_slope_period:
            ema_slow_prev = ema_slow_values[i - ema200_slope_period]
            slope_pct = (ema_slow_now - ema_slow_prev) / ema_slow_prev * 100
            slope_ok = slope_pct >= ema200_slope_min_pct

        # MACD histogram direction
        histogram_rising = macd_now.histogram > macd_prev.histogram
        histogram_falling = macd_now.histogram < macd_prev.histogram

        # Price bounce off EMA20 or EMA50:
        #   Long  bounce: candle low touched EMA and close is back above it
        #   Short bounce: candle high touched EMA and close is back below it
        bounce_up_ema20   = candle.low  <= ema_fast_now and candle.close > ema_fast_now
        bounce_up_ema50   = candle.low  <= ema_mid_now  and candle.close > ema_mid_now
        bounce_down_ema20 = candle.high >= ema_fast_now and candle.close < ema_fast_now
        bounce_down_ema50 = candle.high >= ema_mid_now  and candle.close < ema_mid_now

        # Variant A: crossover down (prev close >= EMA, current close < EMA)
        prev_ema_fast    = ema_fast_values[i - 1]
        prev_ema_mid     = ema_mid_values[i - 1]
        prev_close       = candles[i - 1].close
        cross_down_ema20 = prev_close >= prev_ema_fast and candle.close < ema_fast_now
        cross_down_ema50 = prev_close >= prev_ema_mid  and candle.close < ema_mid_now

        # Entry signals
        long_signal = (slope_ok
                       and bullish_regime
                       and (bounce_up_ema20 or bounce_up_ema50)
                       and histogram_rising)

        if short_entry_mode == "crossover":
            short_signal = (slope_ok
                            and bearish_regime
                            and (cross_down_ema20 or cross_down_ema50)
                            and histogram_falling)
        elif short_entry_mode == "combined":
            short_signal = (slope_ok
                            and bearish_regime
                            and (bounce_down_ema20 or bounce_down_ema50
                                 or cross_down_ema20 or cross_down_ema50)
                            and histogram_falling)
        else:  # "bounce" (default)
            short_signal = (slope_ok
                            and bearish_regime
                            and (bounce_down_ema20 or bounce_down_ema50)
                            and histogram_falling)

        # ---- Step 3: Exit logic – take-profit and stop-loss (intrabar) ----
        # Both levels are checked against the candle's high and low.
        # If both are hit in the same bar, stop-loss wins (conservative).
        if position is not None:
            tp_long  = position.entry_price * (1.0 + exit_pct)
            sl_long  = position.entry_price * (1.0 - stop_loss_pct)
            tp_short = position.entry_price * (1.0 - exit_pct)
            sl_short = position.entry_price * (1.0 + stop_loss_pct)

            exit_now = False
            exit_reason = ""
            exit_fill_price = 0.0

            if position.direction == "long":
                tp_hit = candle.high >= tp_long
                sl_hit = candle.low  <= sl_long
                if sl_hit:          # SL wins on tie
                    exit_now = True
                    exit_reason = f"stop_loss_{stop_loss_pct*100:.2f}pct"
                    exit_fill_price = sl_long
                elif tp_hit:
                    exit_now = True
                    exit_reason = f"take_profit_{exit_pct*100:.2f}pct"
                    exit_fill_price = tp_long
            else:  # short
                tp_hit = candle.low  <= tp_short
                sl_hit = candle.high >= sl_short
                if sl_hit:          # SL wins on tie
                    exit_now = True
                    exit_reason = f"stop_loss_{stop_loss_pct*100:.2f}pct"
                    exit_fill_price = sl_short
                elif tp_hit:
                    exit_now = True
                    exit_reason = f"take_profit_{exit_pct*100:.2f}pct"
                    exit_fill_price = tp_short

            if exit_now:
                trade, new_cash = close_position(
                    position=position,
                    exit_candle=candle,
                    exit_price=exit_fill_price,
                    exit_reason=exit_reason,
                    fee_rate=fee_rate,
                )
                trade.entry_time = candles[position.entry_index].dt.strftime("%Y-%m-%d %H:%M")
                trade.exit_time = (candle.dt + timedelta(seconds=tf_sec)).strftime("%Y-%m-%d %H:%M")
                trades.append(trade)
                cash = new_cash
                position = None

        # ---- Step 4: Entry logic (signal on this bar, execute next open) ----
        if position is None and pending_order is None:
            if long_signal:
                pending_order = PendingOrder(
                    direction="long",
                    entry_price=0.0,  # will use next bar open
                    signal_bar_index=i,
                )
            elif short_signal:
                pending_order = PendingOrder(
                    direction="short",
                    entry_price=0.0,
                    signal_bar_index=i,
                )

        # ---- Equity tracking ----
        if position is not None:
            # Mark-to-market
            pos_value = position.size * candle.close
            if position.direction == "short":
                entry_val = position.size * position.entry_price
                unrealized = entry_val - pos_value
                equity = entry_val + unrealized - (entry_val + pos_value) * fee_rate
            else:
                equity = pos_value
        else:
            equity = cash

        equity_curve.append(equity)
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0.0
        if dd > max_drawdown_pct:
            max_drawdown_pct = dd

    # Close any remaining open position at last candle close
    if position is not None:
        last_candle = candles[-1]
        trade, cash = close_position(
            position=position,
            exit_candle=last_candle,
            exit_price=last_candle.close,
            exit_reason="end_of_data",
            fee_rate=fee_rate,
        )
        trade.entry_time = candles[position.entry_index].dt.strftime("%Y-%m-%d %H:%M")
        trade.exit_time = (last_candle.dt + timedelta(seconds=tf_sec)).strftime("%Y-%m-%d %H:%M")
        trades.append(trade)
        position = None

    # ---- Summary statistics ----
    final_equity = cash
    total_return_pct = (final_equity - initial_capital) / initial_capital * 100

    winning_trades = [t for t in trades if t.pnl > 0]
    losing_trades = [t for t in trades if t.pnl <= 0]
    win_rate = len(winning_trades) / len(trades) * 100 if trades else 0.0

    avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0.0
    avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0.0
    profit_factor = (
        sum(t.pnl for t in winning_trades) / abs(sum(t.pnl for t in losing_trades))
        if losing_trades and sum(t.pnl for t in losing_trades) != 0
        else float("inf")
    )

    report = {
        "strategy": "macd_ema",
        "timeframe": timeframe,
        "indicators": {
            "macd_fast": macd_fast,
            "macd_slow": macd_slow,
            "macd_signal": macd_signal,
            "ema_fast": ema_fast_length,
            "ema_mid": ema_mid_length,
            "ema_slow": ema_slow_length,
        },
        "rules": {
            "bullish_regime": f"EMA{ema_fast_length} > EMA{ema_mid_length} > EMA{ema_slow_length}",
            "bearish_regime": f"EMA{ema_fast_length} < EMA{ema_mid_length} < EMA{ema_slow_length}",
            "long_entry": f"bullish regime (EMA{ema_fast_length}>EMA{ema_mid_length}>EMA{ema_slow_length}) AND price bounces up off EMA{ema_fast_length} or EMA{ema_mid_length} AND MACD histogram rising",
            "short_entry": f"bearish regime (EMA{ema_fast_length}<EMA{ema_mid_length}<EMA{ema_slow_length}) AND price bounces down off EMA{ema_fast_length} or EMA{ema_mid_length} AND MACD histogram falling",
            "long_exit": f"TP: +{exit_pct*100:.2f}% | SL: -{stop_loss_pct*100:.2f}% (intrabar, SL wins on tie)",
            "short_exit": f"TP: -{exit_pct*100:.2f}% | SL: +{stop_loss_pct*100:.2f}% (intrabar, SL wins on tie)",
            "entry_execution": "next bar open after signal",
            "exit_execution": "intrabar at exact TP/SL price",
            "exit_pct": exit_pct,
            "stop_loss_pct": stop_loss_pct,
        },
        "backtest": {
            "initial_capital": initial_capital,
            "final_equity": round(final_equity, 4),
            "total_return_pct": round(total_return_pct, 4),
            "closed_trades": len(trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate_pct": round(win_rate, 4),
            "avg_win": round(avg_win, 4),
            "avg_loss": round(avg_loss, 4),
            "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else "inf",
            "max_drawdown_pct": round(max_drawdown_pct, 4),
            "fee_rate": fee_rate,
        },
    }

    return report, trades


# ---------------------------------------------------------------------------
# Save backtest results
# ---------------------------------------------------------------------------

def save_backtest_results(
    report: dict,
    trades: list[Trade],
    output_dir: str,
    market_type: str,
    symbol: str,
    timeframe: str,
    strategy: str,
) -> None:
    base_dir = Path(output_dir) / market_type
    base_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{symbol}_{timeframe}_{strategy}"

    # JSON report
    report_path = base_dir / f"{stem}_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Report  → {report_path}")

    # CSV trades
    trades_path = base_dir / f"{stem}_trades.csv"
    with open(trades_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "direction", "entry_time", "exit_time",
            "entry_price", "exit_price", "size",
            "pnl", "pnl_pct", "exit_reason",
        ])
        for t in trades:
            writer.writerow([
                t.direction, t.entry_time, t.exit_time,
                t.entry_price, t.exit_price, round(t.size, 6),
                t.pnl, t.pnl_pct, t.exit_reason,
            ])
    print(f"  Trades  → {trades_path}")


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------

def save_config(config: dict, path: str) -> None:
    """Write config dict back to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


# ---------------------------------------------------------------------------
# Per-symbol TP/SL optimizer
# ---------------------------------------------------------------------------

def find_best_tp_sl_for_symbol(
    candles_by_tf: dict[str, list[Candle]],
    macd_fast: int,
    macd_slow: int,
    macd_signal: int,
    ema_fast_length: int,
    ema_mid_length: int,
    ema_slow_length: int,
    initial_capital: float,
    fee_rate: float,
    ema200_slope_period: int = 20,
    ema200_slope_min_pct: float = -3.0,
) -> tuple[float, float, float]:
    """
    Grid search 1-10% x 1-10% across all timeframes for a symbol.
    Scores each (tp, sl) by average total_return across timeframes.
    Returns (best_tp_fraction, best_sl_fraction, best_score_pct).
    """
    tp_range = [x / 100 for x in range(1, 11)]
    sl_range = [x / 100 for x in range(1, 11)]
    total = len(tp_range) * len(sl_range)
    done = 0
    best_score = float("-inf")
    best_tp, best_sl = tp_range[-1], sl_range[-1]

    for tp in tp_range:
        for sl in sl_range:
            returns: list[float] = []
            total_trades = 0
            for tf, candles in candles_by_tf.items():
                report, _ = backtest_macd_ema_strategy(
                    candles=candles,
                    timeframe=tf,
                    initial_capital=initial_capital,
                    fee_rate=fee_rate,
                    macd_fast=macd_fast,
                    macd_slow=macd_slow,
                    macd_signal=macd_signal,
                    ema_fast_length=ema_fast_length,
                    ema_mid_length=ema_mid_length,
                    ema_slow_length=ema_slow_length,
                    exit_pct=tp,
                    stop_loss_pct=sl,
                    ema200_slope_period=ema200_slope_period,
                    ema200_slope_min_pct=ema200_slope_min_pct,
                )
                if "error" not in report:
                    returns.append(report["backtest"]["total_return_pct"])
                    total_trades += report["backtest"]["closed_trades"]
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{total} combinations...", end="\r")
            if returns and total_trades >= 3:
                score = sum(returns) / len(returns)
                if score > best_score:
                    best_score = score
                    best_tp, best_sl = tp, sl

    print(f"  {done}/{total} combinations done.   ")
    return best_tp, best_sl, best_score


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def validate_config(config: dict) -> list[str]:
    """Return a list of validation error messages (empty = valid)."""
    errors: list[str] = []

    required_keys = ["bitget", "markets", "download", "strategy"]
    for k in required_keys:
        if k not in config:
            errors.append(f"Missing top-level key: '{k}'")

    if "strategy" in config:
        s = config["strategy"]
        if s.get("active") != "macd_ema":
            errors.append(f"strategy.active must be 'macd_ema', got: {s.get('active')!r}")

        if "macd" in s:
            macd = s["macd"]
            fast = int(macd.get("fast_length", 0))
            slow = int(macd.get("slow_length", 0))
            if fast >= slow:
                errors.append(f"macd.fast_length ({fast}) must be < slow_length ({slow})")

        if "ema" in s:
            ema = s["ema"]
            ef = int(ema.get("fast_length", 0))
            em = int(ema.get("mid_length", 0))
            es = int(ema.get("slow_length", 0))
            if not (ef < em < es):
                errors.append(f"ema lengths must satisfy fast < mid < slow, got {ef} / {em} / {es}")

    return errors


def run_backtest_from_config(config: dict, config_path: str = "config.json") -> None:
    """Run backtest for all symbols.

    Per-symbol TP/SL: if a market entry has 'exit_pct' + 'stop_loss_pct',
    those values are used. Otherwise the optimizer runs automatically and
    the best values are saved back to config.json for future runs.
    """
    errors = validate_config(config)
    if errors:
        print("[ERROR] Config validation failed:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    strategy = config["strategy"]
    market_type = config["bitget"]["market_type"]
    output_dir = config["strategy"]["backtest"]["output_directory"]

    macd_fast = int(strategy["macd"]["fast_length"])
    macd_slow = int(strategy["macd"]["slow_length"])
    macd_signal = int(strategy["macd"]["signal_length"])
    ema_fast_length = int(strategy["ema"]["fast_length"])
    ema_mid_length  = int(strategy["ema"]["mid_length"])
    ema_slow_length = int(strategy["ema"]["slow_length"])
    initial_capital = float(strategy["backtest"]["initial_capital"])
    fee_rate = float(strategy["backtest"]["fee_rate"])
    short_entry_mode = strategy.get("short_entry_mode", "bounce")
    min_avg_return = float(strategy["backtest"].get("min_avg_return_pct", 5.0))
    scanner_cfg = strategy.get("scanner", {})
    ema200_slope_period  = int(scanner_cfg.get("ema200_slope_period", 20))
    ema200_slope_min_pct = float(scanner_cfg.get("ema200_slope_min_pct", -3.0))

    COMPARE_MODES = ["bounce", "crossover", "combined"]
    config_changed = False
    summary_rows: list[dict] = []

    for market in config["markets"]:
        symbol = market["symbol"]
        timeframes = market["timeframes"]
        sym_market_type = market.get("market_type", market_type)

        # Load all available candles for this symbol
        candles_by_tf: dict[str, list[Candle]] = {}
        for tf in timeframes:
            path = candles_csv_path(
                config["download"]["output_directory"], sym_market_type, symbol, tf
            )
            if path.exists():
                candles_by_tf[tf] = load_candles_from_csv(path)
            else:
                print(f"  [WARNING] {path} not found (run --download first)")

        if not candles_by_tf:
            print(f"[WARNING] No data for {symbol}, skipping.")
            continue

        # Skip symbols explicitly disabled (enabled=False), but re-optimize them
        # each run to check if market conditions have improved
        sym_enabled = market.get("enabled", True)

        # Determine TP/SL: use per-symbol if set AND symbol is enabled,
        # else auto-optimize (also re-optimizes disabled symbols)
        sym_tp = market.get("exit_pct") if sym_enabled else None
        sym_sl = market.get("stop_loss_pct") if sym_enabled else None

        if sym_tp is None or sym_sl is None:
            tfs_info = ", ".join(
                f"{tf}({len(c)})"
                for tf, c in candles_by_tf.items()
            )
            print(f"\n{'='*60}")
            print(f"Auto-optimizing {symbol}  [{tfs_info} candles]")
            print(f"{'='*60}")
            sym_tp, sym_sl, best_score = find_best_tp_sl_for_symbol(
                candles_by_tf=candles_by_tf,
                macd_fast=macd_fast,
                macd_slow=macd_slow,
                macd_signal=macd_signal,
                ema_fast_length=ema_fast_length,
                ema_mid_length=ema_mid_length,
                ema_slow_length=ema_slow_length,
                initial_capital=initial_capital,
                fee_rate=fee_rate,
                ema200_slope_period=ema200_slope_period,
                ema200_slope_min_pct=ema200_slope_min_pct,
            )
            market["exit_pct"] = round(sym_tp, 4)
            market["stop_loss_pct"] = round(sym_sl, 4)
            config_changed = True

            if best_score < min_avg_return:
                market["enabled"] = False
                print(f"  \u26a0 Best score {best_score:+.1f}% < threshold {min_avg_return:.0f}%  "
                      f"\u2192 {symbol} DISABLED (will re-check next run)")
                continue  # skip backtesting this symbol
            else:
                market["enabled"] = True
                print(f"  \u2713 Best: TP={sym_tp*100:.0f}%  SL={sym_sl*100:.0f}%  "
                      f"avg return {best_score:+.1f}%  \u2192 saved to config")
        else:
            sym_tp = float(sym_tp)
            sym_sl = float(sym_sl)

        if not market.get("enabled", True):
            continue  # safety guard

        # Backtest each timeframe with the symbol's TP/SL
        for timeframe, candles in candles_by_tf.items():
            date_range = (f"{candles[0].dt.strftime('%Y-%m-%d')} \u2192 "
                          f"{candles[-1].dt.strftime('%Y-%m-%d')}")

            if short_entry_mode == "compare":
                print(f"\n{symbol} {timeframe}  "
                      f"MACD({macd_fast},{macd_slow},{macd_signal})  "
                      f"EMA{ema_fast_length}/{ema_mid_length}/{ema_slow_length}  "
                      f"TP={sym_tp*100:.0f}%  SL={sym_sl*100:.0f}%  "
                      f"[{len(candles)} candles  {date_range}]")
                print(f"  {'Mode':<12} | {'Trades':>6} | {'Win%':>7} | {'Return':>9} | {'MaxDD':>7} | {'PF':>6}")
                print(f"  {'-'*12}-+-{'-'*6}-+-{'-'*7}-+-{'-'*9}-+-{'-'*7}-+-{'-'*6}")
                for mode in COMPARE_MODES:
                    report, _ = backtest_macd_ema_strategy(
                        candles=candles,
                        timeframe=timeframe,
                        initial_capital=initial_capital,
                        fee_rate=fee_rate,
                        macd_fast=macd_fast,
                        macd_slow=macd_slow,
                        macd_signal=macd_signal,
                        ema_fast_length=ema_fast_length,
                        ema_mid_length=ema_mid_length,
                        ema_slow_length=ema_slow_length,
                        exit_pct=sym_tp,
                        stop_loss_pct=sym_sl,
                        short_entry_mode=mode,
                        ema200_slope_period=ema200_slope_period,
                        ema200_slope_min_pct=ema200_slope_min_pct,
                    )
                    if "error" in report:
                        print(f"  {mode:<12} | ERROR: {report['error']}")
                        continue
                    s = report["backtest"]
                    print(f"  {mode:<12} | {s['closed_trades']:>6} | {s['win_rate_pct']:>6.1f}% | "
                          f"{s['total_return_pct']:>+8.2f}% | {s['max_drawdown_pct']:>6.2f}% | "
                          f"{s['profit_factor']:>6.2f}")
            else:
                print(f"\nBacktesting {symbol} {timeframe}  "
                      f"MACD({macd_fast},{macd_slow},{macd_signal})  "
                      f"EMA{ema_fast_length}/{ema_mid_length}/{ema_slow_length}  "
                      f"TP={sym_tp*100:.0f}%  SL={sym_sl*100:.0f}%")
                print(f"  Loaded {len(candles)} candles  ({date_range})")

                report, trades = backtest_macd_ema_strategy(
                    candles=candles,
                    timeframe=timeframe,
                    initial_capital=initial_capital,
                    fee_rate=fee_rate,
                    macd_fast=macd_fast,
                    macd_slow=macd_slow,
                    macd_signal=macd_signal,
                    ema_fast_length=ema_fast_length,
                    ema_mid_length=ema_mid_length,
                    ema_slow_length=ema_slow_length,
                    exit_pct=sym_tp,
                    stop_loss_pct=sym_sl,
                    short_entry_mode=short_entry_mode,
                    ema200_slope_period=ema200_slope_period,
                    ema200_slope_min_pct=ema200_slope_min_pct,
                )

                if "error" in report:
                    print(f"  [ERROR] {report['error']}")
                    continue

                stats = report["backtest"]
                print(f"  Trades        : {stats['closed_trades']}")
                print(f"  Win Rate      : {stats['win_rate_pct']:.2f}%")
                print(f"  Total Return  : {stats['total_return_pct']:.4f}%")
                print(f"  Max Drawdown  : {stats['max_drawdown_pct']:.4f}%")
                print(f"  Profit Factor : {stats['profit_factor']}")
                print(f"  Avg Win       : {stats['avg_win']:.4f}")
                print(f"  Avg Loss      : {stats['avg_loss']:.4f}")

                summary_rows.append({
                    "symbol": symbol,
                    "tf": timeframe,
                    "tp": sym_tp,
                    "sl": sym_sl,
                    "trades": stats["closed_trades"],
                    "win_pct": stats["win_rate_pct"],
                    "return": stats["total_return_pct"],
                    "maxdd": stats["max_drawdown_pct"],
                    "pf": stats["profit_factor"],
                })

                save_backtest_results(
                    report=report,
                    trades=trades,
                    output_dir=output_dir,
                    market_type=sym_market_type,
                    symbol=symbol,
                    timeframe=timeframe,
                    strategy="macd_ema",
                )

    if config_changed:
        save_config(config, config_path)
        print(f"\nConfig updated with per-symbol TP/SL \u2192 {config_path}")

    # ---- Summary table ----
    if summary_rows:
        summary_rows.sort(key=lambda r: r["return"], reverse=True)
        hdr = f"\n{'='*75}"
        print(hdr)
        print(f"  {'Symbol':<12} {'TF':<5} {'TP':>4} {'SL':>4} | "
              f"{'Trades':>6} {'Win%':>6} {'Return':>9} {'MaxDD':>7} {'PF':>6}")
        print(f"  {'-'*12} {'-'*5} {'-'*4} {'-'*4}-+-"
              f"{'-'*6}-{'-'*6}-{'-'*9}-{'-'*7}-{'-'*6}")
        for r in summary_rows:
            flag = "✓" if r["return"] >= 0 else "✗"
            print(f"  {flag} {r['symbol']:<11} {r['tf']:<5} "
                  f"{r['tp']*100:>3.0f}% {r['sl']*100:>3.0f}% | "
                  f"{r['trades']:>6} {r['win_pct']:>5.1f}% "
                  f"{r['return']:>+9.2f}% {r['maxdd']:>6.2f}% {r['pf']:>6.2f}")
        print('='*75)


# ---------------------------------------------------------------------------
# Parameter optimization (TP % x SL % grid search)
# ---------------------------------------------------------------------------

def run_optimize(config: dict) -> None:
    """
    Grid search over all TP/SL combinations (1%..10% x 1%..10%).
    Runs every combination for every symbol+timeframe and prints
    a sorted top-20 results table per timeframe.
    Saves a full CSV results file per timeframe.
    """
    strategy = config["strategy"]
    market_type = config["bitget"]["market_type"]
    output_dir = config["strategy"]["backtest"]["output_directory"]

    macd_fast = int(strategy["macd"]["fast_length"])
    macd_slow = int(strategy["macd"]["slow_length"])
    macd_signal = int(strategy["macd"]["signal_length"])
    ema_fast_length = int(strategy["ema"]["fast_length"])
    ema_mid_length  = int(strategy["ema"]["mid_length"])
    ema_slow_length = int(strategy["ema"]["slow_length"])
    initial_capital = float(strategy["backtest"]["initial_capital"])
    fee_rate = float(strategy["backtest"]["fee_rate"])

    tp_range = [x / 100 for x in range(1, 11)]   # 1%..10%
    sl_range = [x / 100 for x in range(1, 11)]   # 1%..10%
    total_combinations = len(tp_range) * len(sl_range)

    for market in config["markets"]:
        symbol = market["symbol"]
        timeframes = market["timeframes"]
        sym_market_type = market.get("market_type", market_type)

        for timeframe in timeframes:
            path = candles_csv_path(
                config["download"]["output_directory"], sym_market_type, symbol, timeframe
            )
            if not path.exists():
                print(f"[WARNING] Data file not found: {path}  (run --download first)")
                continue

            candles = load_candles_from_csv(path)
            print(f"\n{'='*60}")
            print(f"Optimizing {symbol} {timeframe}  "
                  f"({len(candles)} candles, {total_combinations} combinations)")
            print(f"{'='*60}")

            results: list[dict] = []
            done = 0

            for tp in tp_range:
                for sl in sl_range:
                    report, _ = backtest_macd_ema_strategy(
                        candles=candles,
                        timeframe=timeframe,
                        initial_capital=initial_capital,
                        fee_rate=fee_rate,
                        macd_fast=macd_fast,
                        macd_slow=macd_slow,
                        macd_signal=macd_signal,
                        ema_fast_length=ema_fast_length,
                        ema_mid_length=ema_mid_length,
                        ema_slow_length=ema_slow_length,
                        exit_pct=tp,
                        stop_loss_pct=sl,
                    )
                    done += 1
                    if "error" in report:
                        continue

                    s = report["backtest"]
                    pf = s["profit_factor"]
                    results.append({
                        "tp_pct": tp * 100,
                        "sl_pct": sl * 100,
                        "trades": s["closed_trades"],
                        "win_rate": s["win_rate_pct"],
                        "total_return": s["total_return_pct"],
                        "max_dd": s["max_drawdown_pct"],
                        "profit_factor": pf if pf != "inf" else 9999.0,
                    })

                    # Progress every 10 combos
                    if done % 10 == 0:
                        print(f"  {done}/{total_combinations} done...", end="\r")

            print(f"  {done}/{total_combinations} done.   ")

            if not results:
                print("  No results.")
                continue

            # Sort by total_return descending
            results.sort(key=lambda r: r["total_return"], reverse=True)

            # Print top 20
            header = f"{'TP%':>5} {'SL%':>5} {'Trades':>7} {'WinRate':>8} {'Return%':>9} {'MaxDD%':>8} {'PF':>7}"
            print(f"\nTop 20 by Total Return ({symbol} {timeframe}):")
            print(header)
            print("-" * len(header))
            for r in results[:20]:
                pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] < 9999 else "inf"
                print(
                    f"{r['tp_pct']:>5.0f} "
                    f"{r['sl_pct']:>5.0f} "
                    f"{r['trades']:>7} "
                    f"{r['win_rate']:>8.2f} "
                    f"{r['total_return']:>9.2f} "
                    f"{r['max_dd']:>8.2f} "
                    f"{pf_str:>7}"
                )

            # Save full results CSV
            base_dir = Path(output_dir) / sym_market_type
            base_dir.mkdir(parents=True, exist_ok=True)
            csv_path = base_dir / f"{symbol}_{timeframe}_optimize.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "tp_pct", "sl_pct", "trades", "win_rate",
                    "total_return", "max_dd", "profit_factor"
                ])
                writer.writeheader()
                writer.writerows(results)
            print(f"\n  Full results → {csv_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def tf_to_seconds(tf: str) -> int:
    mapping = {
        "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
        "1h": 3600, "4h": 14400, "1d": 86400,
    }
    return mapping.get(tf, 900)


def detect_signal(
    candles: list[Candle],
    macd_fast: int,
    macd_slow: int,
    macd_signal: int,
    ema_fast_length: int,
    ema_mid_length: int,
    ema_slow_length: int,
    short_entry_mode: str = "bounce",
    ema200_slope_period: int = 20,
    ema200_slope_min_pct: float = -3.0,
) -> Optional[str]:
    """
    Check the last bar in `candles` for an entry signal.
    Returns "long", "short", or None.
    Caller should pass candles with the signal bar as the LAST element
    (i.e. strip any still-forming bar before calling).
    """
    min_bars = max(macd_slow + macd_signal, ema_slow_length) + ema200_slope_period + 2
    if len(candles) < min_bars:
        return None

    closes = [c.close for c in candles]
    macd_points    = calculate_macd(closes, macd_fast, macd_slow, macd_signal)
    ema_fast_vals  = calculate_ema(closes, ema_fast_length)
    ema_mid_vals   = calculate_ema(closes, ema_mid_length)
    ema_slow_vals  = calculate_ema(closes, ema_slow_length)

    i = len(candles) - 1
    candle        = candles[i]
    macd_now      = macd_points[i]
    macd_prev     = macd_points[i - 1]
    ema_fast_now  = ema_fast_vals[i]
    ema_mid_now   = ema_mid_vals[i]
    ema_slow_now  = ema_slow_vals[i]

    # Slope filter
    slope_ok = True
    if i >= ema200_slope_period:
        ema_slow_prev = ema_slow_vals[i - ema200_slope_period]
        slope_pct = (ema_slow_now - ema_slow_prev) / ema_slow_prev * 100
        slope_ok = slope_pct >= ema200_slope_min_pct

    bullish_regime = ema_fast_now > ema_mid_now > ema_slow_now
    bearish_regime = ema_fast_now < ema_mid_now < ema_slow_now

    histogram_rising  = macd_now.histogram > macd_prev.histogram
    histogram_falling = macd_now.histogram < macd_prev.histogram

    bounce_up_ema20   = candle.low  <= ema_fast_now and candle.close > ema_fast_now
    bounce_up_ema50   = candle.low  <= ema_mid_now  and candle.close > ema_mid_now
    bounce_down_ema20 = candle.high >= ema_fast_now and candle.close < ema_fast_now
    bounce_down_ema50 = candle.high >= ema_mid_now  and candle.close < ema_mid_now

    prev_ema_fast    = ema_fast_vals[i - 1]
    prev_ema_mid     = ema_mid_vals[i - 1]
    prev_close       = candles[i - 1].close
    cross_down_ema20 = prev_close >= prev_ema_fast and candle.close < ema_fast_now
    cross_down_ema50 = prev_close >= prev_ema_mid  and candle.close < ema_mid_now

    long_signal = (slope_ok and bullish_regime
                   and (bounce_up_ema20 or bounce_up_ema50)
                   and histogram_rising)

    if short_entry_mode == "crossover":
        short_signal = (slope_ok and bearish_regime
                        and (cross_down_ema20 or cross_down_ema50)
                        and histogram_falling)
    elif short_entry_mode == "combined":
        short_signal = (slope_ok and bearish_regime
                        and (bounce_down_ema20 or bounce_down_ema50
                             or cross_down_ema20 or cross_down_ema50)
                        and histogram_falling)
    else:
        short_signal = (slope_ok and bearish_regime
                        and (bounce_down_ema20 or bounce_down_ema50)
                        and histogram_falling)

    if long_signal:
        return "long"
    if short_signal:
        return "short"
    return None


# ---------------------------------------------------------------------------
# Market scanner
# ---------------------------------------------------------------------------

def run_scanner(config: dict) -> None:
    """
    Market scanner: on each new candle close, fetches fresh data for all
    enabled symbol/TF pairs and checks for entry signals.

    Timing:
      - Waits until the next candle boundary of the smallest active TF.
      - Adds a short buffer (buffer_seconds) after the boundary so the
        candle is fully closed and available via the API.
      - Checks each TF only when its own boundary has just passed.

    Signal bar: candles[-2] (second-to-last), because candles[-1] may
    still be forming at fetch time.
    """
    errors = validate_config(config)
    if errors:
        for e in errors:
            print(f"  [ERROR] {e}")
        sys.exit(1)

    strategy    = config["strategy"]
    market_type = config["bitget"]["market_type"]
    base_url    = config["bitget"]["base_url"]

    macd_fast       = int(strategy["macd"]["fast_length"])
    macd_slow       = int(strategy["macd"]["slow_length"])
    macd_signal_len = int(strategy["macd"]["signal_length"])
    ema_fast_length = int(strategy["ema"]["fast_length"])
    ema_mid_length  = int(strategy["ema"]["mid_length"])
    ema_slow_length = int(strategy["ema"]["slow_length"])
    short_entry_mode = strategy.get("short_entry_mode", "bounce")

    scanner_cfg          = strategy.get("scanner", {})
    ema200_slope_period  = int(scanner_cfg.get("ema200_slope_period", 20))
    ema200_slope_min_pct = float(scanner_cfg.get("ema200_slope_min_pct", -3.0))
    buffer_seconds       = int(scanner_cfg.get("buffer_seconds", 10))
    fetch_candles_count  = int(scanner_cfg.get("fetch_candles", 300))

    # Collect active pairs
    pairs: list[tuple[str, str, str]] = []
    for market in config["markets"]:
        if not market.get("enabled", True):
            continue
        sym_mt = market.get("market_type", market_type)
        for tf in market["timeframes"]:
            pairs.append((market["symbol"], tf, sym_mt))

    if not pairs:
        print("[SCANNER] No active symbol/timeframe pairs. Check config.")
        return

    min_bars = max(macd_slow + macd_signal_len, ema_slow_length) + ema200_slope_period + 5
    fetch_n  = max(fetch_candles_count, min_bars + 10)

    all_tfs    = sorted({tf for _, tf, _ in pairs}, key=tf_to_seconds)
    min_tf_sec = tf_to_seconds(all_tfs[0])

    print(f"\n{'='*60}")
    print(f"  MARKET SCANNER  —  {len(pairs)} active pairs")
    print(f"  Pairs    : {', '.join(f'{s} {tf}' for s, tf, _ in pairs)}")
    print(f"  Mode     : {short_entry_mode}  |  "
          f"EMA200 slope ≥ {ema200_slope_min_pct:.0f}% / {ema200_slope_period} bars")
    print(f"  Timing   : checks every {min_tf_sec//60}m candle close + {buffer_seconds}s buffer")
    print(f"  Press Ctrl+C to stop.")
    print(f"{'='*60}\n")

    while True:
        now_ms       = int(time.time() * 1000)
        period_ms    = min_tf_sec * 1000
        next_close   = ((now_ms // period_ms) + 1) * period_ms
        sleep_sec    = (next_close - now_ms) / 1000 + buffer_seconds

        next_dt = datetime.fromtimestamp(next_close / 1000, tz=timezone.utc)
        print(f"[{datetime.fromtimestamp(now_ms/1000, tz=timezone.utc).strftime('%H:%M:%S')} UTC]  "
              f"Next check at {next_dt.strftime('%H:%M:%S')} UTC  "
              f"(sleep {sleep_sec:.0f}s)")

        time.sleep(max(1.0, sleep_sec))

        scan_dt  = datetime.fromtimestamp(time.time(), tz=timezone.utc)
        now_ms2  = int(time.time() * 1000)
        print(f"\n{'─'*60}")
        print(f"  {scan_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"{'─'*60}")

        signals_found = 0

        for symbol, tf, sym_mt in pairs:
            tf_sec     = tf_to_seconds(tf)
            tf_period  = tf_sec * 1000
            # Only scan this TF if its candle just closed (within last 2× buffer)
            ms_into_bar = now_ms2 % tf_period
            if ms_into_bar > (buffer_seconds * 2 + 5) * 1000:
                continue  # this TF closes later

            end_ms   = now_ms2
            start_ms = end_ms - fetch_n * tf_sec * 1000

            try:
                candles = fetch_candles_bitget(
                    base_url=base_url,
                    market_type=sym_mt,
                    symbol=symbol,
                    timeframe=tf,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    limit=min(1000, fetch_n),
                )
            except Exception as exc:
                print(f"  [ERR] {symbol} {tf}: {exc}")
                continue

            if len(candles) < min_bars + 1:
                print(f"  [WARN] {symbol} {tf}: {len(candles)} candles < {min_bars+1} needed")
                continue

            # Determine the last FULLY CLOSED candle by comparing its timestamp
            # to the expected last closed boundary.
            # Expected last closed bar starts at: floor(now / tf_ms) * tf_ms - tf_ms
            tf_ms = tf_sec * 1000
            last_closed_start = (now_ms2 // tf_ms) * tf_ms - tf_ms
            last_ts = candles[-1].timestamp

            if last_ts >= last_closed_start + tf_ms:
                # candles[-1] is still-forming (its start >= current boundary)
                signal_candles = candles[:-1]
            elif last_ts == last_closed_start:
                # candles[-1] is exactly the just-closed bar — API hasn't started new one yet
                signal_candles = candles
            else:
                # candles[-1] is older than expected — data lag, use as-is
                signal_candles = candles

            direction = detect_signal(
                candles=signal_candles,
                macd_fast=macd_fast,
                macd_slow=macd_slow,
                macd_signal=macd_signal_len,
                ema_fast_length=ema_fast_length,
                ema_mid_length=ema_mid_length,
                ema_slow_length=ema_slow_length,
                short_entry_mode=short_entry_mode,
                ema200_slope_period=ema200_slope_period,
                ema200_slope_min_pct=ema200_slope_min_pct,
            )

            if direction:
                signals_found += 1
                arrow      = "\u2191 LONG " if direction == "long" else "\u2193 SHORT"
                last_bar   = signal_candles[-1]
                bar_time   = last_bar.dt.strftime("%Y-%m-%d %H:%M")
                entry_est  = candles[-1].open  # next bar open as estimated entry
                print(f"  \U0001f514 {arrow}  {symbol:<12} {tf:<4}  "
                      f"signal bar close: {last_bar.close:.6g}  "
                      f"({bar_time} UTC)  "
                      f"est. entry: {entry_est:.6g}")

        if signals_found == 0:
            print("  — no signals this cycle —")
        print()



def load_config(path: str = "config.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bitget MACD+EMA Trading Bot")
    parser.add_argument("--download", action="store_true", help="Download OHLCV data")
    parser.add_argument("--backtest", action="store_true", help="Run backtest")
    parser.add_argument("--optimize", action="store_true", help="Grid search TP/SL 1-10%%")
    parser.add_argument("--scan",     action="store_true", help="Run live market scanner")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    args = parser.parse_args()

    if not args.download and not args.backtest and not args.optimize and not args.scan:
        parser.print_help()
        sys.exit(0)

    config = load_config(args.config)

    if args.download:
        print("=== DOWNLOAD ===")
        run_download(config)

    if args.backtest:
        print("\n=== BACKTEST ===")
        run_backtest_from_config(config, args.config)

    if args.optimize:
        print("\n=== OPTIMIZE ===")
        run_optimize(config)

    if args.scan:
        run_scanner(config)


if __name__ == "__main__":
    main()
