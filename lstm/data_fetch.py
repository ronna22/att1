"""
data_fetch.py
=============
Fetch historical OHLCV candle data from Bitget USDT-Futures API.

Features:
  - Automatic pagination (handles >1000 candles)
  - CSV cache (skip re-download unless force_download=True)
  - Configurable symbol, timeframe, look-back window

Usage:
  python data_fetch.py                 # downloads 5m + 15m SIRENUSDT
  python data_fetch.py --force         # re-download even if cached
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SYMBOL       = "SIRENUSDT"
PRODUCT_TYPE = "USDT-FUTURES"
BASE_URL     = "https://api.bitget.com/api/v2/mix/market/candles"

# Milliseconds per bar — used for pagination step
BAR_MS: dict[str, int] = {
    "1m":  60_000,
    "3m":  180_000,
    "5m":  300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1H":  3_600_000,
}

# Default look-back per timeframe
TF_LOOKBACK_DAYS: dict[str, int] = {
    "5m":  90,   # 3 mēneši
    "15m": 90,
}

# Data cache directory (relative to this file)
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Low-level API call
# ---------------------------------------------------------------------------
def _fetch_page(symbol: str, granularity: str,
                end_ms: int, limit: int = 1000) -> list[list]:
    """
    Fetch one page of candles ending at end_ms.
    Bitget returns up to 1000 rows per request.
    """
    params = {
        "symbol":      symbol,
        "granularity": granularity,
        "productType": PRODUCT_TYPE,
        "endTime":     str(end_ms),
        "limit":       str(limit),
    }
    url = BASE_URL + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            data = json.loads(r.read())
    except Exception as exc:
        raise ConnectionError(f"API request failed: {exc}") from exc

    if data.get("code") != "00000":
        raise RuntimeError(f"Bitget API error [{data.get('code')}]: {data.get('msg')}")

    return data.get("data", [])


# ---------------------------------------------------------------------------
# Public downloader
# ---------------------------------------------------------------------------
def fetch_ohlcv(
    symbol: str = SYMBOL,
    timeframe: str = "5m",
    days: int = 120,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Download OHLCV candles for *symbol* / *timeframe*, covering the last
    *days* calendar days.

    Returns a DataFrame with columns:
        timestamp (ms int) | open | high | low | close | volume | datetime (UTC)

    Results are cached as CSV — set force_download=True to re-fetch.
    """
    cache_path = DATA_DIR / f"{symbol}_{timeframe}.csv"

    # ── Return from cache if available ───────────────────────────────────────
    if cache_path.exists() and not force_download:
        df = pd.read_csv(cache_path)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        print(
            f"  [cache]  {symbol} {timeframe}: {len(df):>6,} candles  "
            f"({df['datetime'].iloc[0].date()} → {df['datetime'].iloc[-1].date()})"
        )
        return df

    # ── Paginate backwards from now until start_ms ────────────────────────
    bar_ms      = BAR_MS[timeframe]
    now_ms      = int(time.time() * 1000)
    start_ms    = now_ms - int(days * 24 * 3600 * 1000)
    total_ms    = now_ms - start_ms          # total window size for % calc
    page_end_ms = now_ms
    all_rows: list[list] = []

    print(f"  Downloading {symbol} {timeframe} ({days} days)", flush=True)

    last_pct = -1
    while True:
        rows = _fetch_page(symbol, timeframe, page_end_ms)
        if not rows:
            break

        all_rows.extend(rows)
        oldest_ts = int(rows[0][0])        # API returns ascending (oldest first)

        # Progress: how far back we've fetched relative to full window
        fetched_ms = now_ms - oldest_ts
        pct = min(int(fetched_ms / total_ms * 100), 99)
        if pct != last_pct:
            bar_filled = pct // 5                          # 20 segments
            bar = "█" * bar_filled + "░" * (20 - bar_filled)
            print(f"\r  [{bar}] {pct:>3}%  {len(all_rows):>6,} sveces", end="", flush=True)
            last_pct = pct

        if oldest_ts <= start_ms:
            break                           # have enough data

        page_end_ms = oldest_ts - bar_ms   # step one bar further back
        time.sleep(0.12)                   # ~8 req/s — stay within rate limit

    bar = "█" * 20
    print(f"\r  [{bar}] 100%  {len(all_rows):>6,} sveces", flush=True)

    if not all_rows:
        raise ValueError(f"No data returned for {symbol} {timeframe}")

    # ── Build DataFrame ───────────────────────────────────────────────────────
    df = pd.DataFrame(
        [
            [int(r[0]), float(r[1]), float(r[2]),
             float(r[3]), float(r[4]), float(r[5])]
            for r in all_rows
        ],
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )

    df = (
        df.drop_duplicates("timestamp")
          .sort_values("timestamp")
          .reset_index(drop=True)
    )

    # Trim to requested window
    df = df[df["timestamp"] >= start_ms].reset_index(drop=True)

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    # ── Cache to disk ─────────────────────────────────────────────────────────
    df.to_csv(cache_path, index=False)
    print(
        f"  Saved: {cache_path.name}  "
        f"({len(df):,} candles, "
        f"{df['datetime'].iloc[0].date()} → {df['datetime'].iloc[-1].date()})"
    )
    return df


# ---------------------------------------------------------------------------
# CLI — download both timeframes
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Download Bitget OHLCV data")
    parser.add_argument("--symbol",    default=SYMBOL)
    parser.add_argument("--days",      type=int, default=120)
    parser.add_argument("--force",     action="store_true",
                        help="Re-download even if cache exists")
    args = parser.parse_args()

    print(f"\n  {'='*50}")
    print(f"  DATA FETCH  —  {args.symbol}")
    print(f"  {'='*50}")
    for tf in TF_LOOKBACK_DAYS:
        fetch_ohlcv(args.symbol, tf, days=args.days, force_download=args.force)


if __name__ == "__main__":
    main()
