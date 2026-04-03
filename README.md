# Bitget Trading Bot

Automatizēts tirdzniecības bots Bitget biržai ar diviem neatkarīgiem stratēģiju moduļiem:

1. **MACD + EMA reģīma stratēģija** (`main.py`) — klasiska tehniskās analīzes stratēģija
2. **ML cenas prognozēšanas modelis** (`ml_model.py`) — LightGBM + XGBoost mašīnmācīšanās

---

## 1. MACD + EMA stratēģija (`main.py`)

### Stratēģijas loģika

**Indikatori:**
- MACD(6, 16, 9): `fast_ema - slow_ema = macd`; `EMA(macd, 9) = signal`; `histogram = macd - signal`
- EMA20, EMA50, EMA200 — reģīma filtrs un atsitiena ieeja

**Reģīma filtrs:**
- Bullish: `EMA20 > EMA50 > EMA200` → tikai LONG darījumi
- Bearish: `EMA20 < EMA50 < EMA200` → tikai SHORT darījumi
- Neitrāls: nav jaunu pozīciju

**EMA200 slīpuma filtrs:**
- `slope_pct = (EMA200[i] - EMA200[i-20]) / EMA200[i-20] * 100`
- Ja `slope_pct < -3.0%` → signāli bloķēti

**Ieeja (bounce mode):**
- Long: bullish_regime AND (low ≤ EMA20 AND close > EMA20) OR (low ≤ EMA50 AND close > EMA50) AND histogram_rising
- Short: bearish_regime AND (high ≥ EMA20 AND close < EMA20) OR (high ≥ EMA50 AND close < EMA50) AND histogram_falling
- Signāls uz aizvērtās sveces → ieeja nākamās sveces OPEN

**Izeja (intrabar):**
- Take-profit: long → high ≥ entry × (1 + tp%); short → low ≤ entry × (1 - tp%)
- Stop-loss: long → low ≤ entry × (1 - sl%); short → high ≥ entry × (1 + sl%)
- Vienādā svecē: SL uzvar (konservatīvi)

### CLI komandas

```bash
python main.py --download           # lejupielādē OHLCV datus
python main.py --backtest           # backtest no esošajiem datiem
python main.py --download --backtest
python main.py --optimize           # pilns TP/SL grid search
python main.py --scan               # live tirgus skeneris
```

### Per-simbola market_type

Simboliem kas eksistē tikai futures tirgū (piem., `4USDT`, `SIRENUSDT`), pievieno `"market_type": "futures"` simbola ierakstam config.json. Dati tiek saglabāti `data/futures/`, backtest rezultāti `backtests/futures/`.

### Auto-optimizācija

Simboliem bez `exit_pct`/`stop_loss_pct` tiek automātiski veikts grid search 1–10% × 1–10% (100 kombinācijas). Ja labākais vidējais `total_return` < `min_avg_return_pct` (5%), simbols tiek atspējots (`"enabled": false`) un pārbaudīts nākamajā izpildē.

### Pašreizējie backtest rezultāti (2026-02-21 → 2026-03-24)

| Simbols | TF | TP | SL | Darījumi | Win% | Atdeve | MaxDD | PF |
|---|---|---|---|---|---|---|---|---|
| SIRENUSDT | 15m | 5% | 10% | 61 | 82.0% | +199.92% | 25.40% | 1.99 |
| LIGHTUSDT | 30m | 10% | 6% | 19 | 63.2% | +81.35% | 15.86% | 2.30 |
| BTWUSDT | 15m | 10% | 10% | 30 | 63.3% | +75.48% | 36.98% | 1.33 |
| 4USDT | 15m | 10% | 5% | 20 | 55.0% | +56.36% | 25.90% | 2.33 |
| LIGHTUSDT | 15m | 10% | 6% | 20 | 55.0% | +54.02% | 18.94% | 1.72 |
| HYPEUSDT | 30m | 7% | 10% | 9 | 88.9% | +45.83% | 12.57% | 3.93 |
| TAOUSDT | 30m | 7% | 10% | 10 | 80.0% | +35.07% | 24.72% | 2.20 |
| 4USDT | 30m | 10% | 5% | 15 | 53.3% | +33.05% | 10.65% | 1.91 |
| BRUSDT | 15m | 8% | 7% | 18 | 61.1% | +32.95% | 21.57% | 1.56 |
| XRPUSDT | — | — | — | — | — | DISABLED | — | — |
| SUIUSDT | — | — | — | — | — | DISABLED | — | — |
| LINKUSDT | — | — | — | — | — | DISABLED | — | — |

### Skenera darbība (--scan)

- Gaida katras TF sveces aizvēršanos + buffer_seconds (10s)
- Ielādē pēdējās fetch_candles (300) sveces
- Timestamp verifikācija: salīdzina `candles[-1].timestamp` ar `last_closed_start`
- Izsauc `detect_signal()` uz pēdējās aizvērtās sveces
- Izvada signālu ar simbolu, TF un virzienu

---

## 2. ML cenas prognozēšanas modelis (`ml_model.py`)

### Mērķis

Prognozēt vai SIRENUSDT cena **nākamajā svecē mainīsies par ≥ 0.5%** (uz augšu vai uz leju).

### Timeframes

`1m`, `3m`, `5m`, `15m`

### Modeļi

- **LightGBM** (`lgb.LGBMClassifier`)
- **XGBoost** (`xgb.XGBClassifier`)
- **Divi modeļi uz TF:** `long_model` (P(up ≥ 0.5%)) + `short_model` (P(down ≥ 0.5%))

### Features (75+)

**Cenu atgriešanās:** `return_1..10`, `return_20` — log returns  
**Sveces struktūra:** `body_size`, `upper_wick`, `lower_wick`, `is_bullish`, `gap`, `open_pos`, `close_pos`  
**Konsekutīvas sveces:** `consec_up`, `consec_down` (5 svecēs)  
**Slīdošie vidējie:** `ma5/10/20/50/100` relatīvi + slope, `ema5/10/20_dist`  
**MA100 reģīms:** `above_ma100`, `dist_ma100`  
**MACD:** `macd_line`, `macd_signal`, `macd_hist`, `macd_hist_delta`, `macd_cross_up/down`  
**MACD ātrais (6,16,9):** `macd_fast_hist`, `macd_fast_delta`  
**RSI:** `rsi14`, `rsi7`, `rsi14_delta`, `rsi_ob`, `rsi_os`  
**Bollinger Bands:** distances, `bb_width`, `bb_pct`, `bb_squeeze`  
**Volatilitāte/ATR:** `atr14`, `atr7`, `range_atr`, `volatility_5/10/20`, `vol_expand`  
**Apjoms:** `volume_ratio5/10`, `volume_delta`, `volume_zscore`, `vol_up`  
**Cenu pozīcija:** `pos_5/10/20/50` (kur cena atrodas N-sveces diapazonā)  
**Cenu līmeņi:** `dist_high/low_20/5`  
**Stochastic:** `stoch_k`, `stoch_d`, `stoch_ob`, `stoch_os`  
**Williams %R, CCI:** `williams_r`, `cci14`  
**Moments/ROC:** `mom_3/5/10`, `roc5`, `roc10`  
**Laiks:** `hour`, `minute`, `day_of_week`

### Modeļu apmācība — v2 (bez lookahead bias, bez selekcijas bias)

- **Labels VisĀM svecēm:** `y_long[i]=1` ja `close[i+1] ≥ close[i]*1.005`, citādi 0  
  `y_short[i]=1` ja `close[i+1] ≤ close[i]*0.995`, citādi 0  
  *(v1 kļūda: filtrēja tikai sveces ar ≥0.5% kustību, veidojot nevienmērīgu datu kopu)*
- **Klases disbalanss:** `scale_pos_weight = n_neg/n_pos` abos modeļos
- Apmācība: pirmie 70% datu  
- Validācija: 15%  
- Tests: pēdējie 15% (VISI testa perioda bāri, ne filtrēti)
- MA100 filtrs: long tikai ja close > MA100, short tikai ja close < MA100
- **Slieksnis:** pārbaudīti [52%, 55%, 58%, 60%, 62%, 65%], ziņots labākais

### CLI komanda

```bash
python ml_model.py --symbol SIRENUSDT --timeframes 1m 3m 5m 15m
python ml_model.py --symbol SIRENUSDT --timeframes 1m 3m 5m 15m --download
```

---

## Projekta struktūra

```
jkd/
├── main.py                    # MACD+EMA stratēģija
├── ml_model.py                # LightGBM + XGBoost ML modelis
├── config.json                # Konfigurācija
├── requirements.txt           # Python atkarības
├── data/
│   ├── spot/
│   │   └── SYMBOL_TF.csv
│   └── futures/
│       └── SYMBOL_TF.csv
├── backtests/
│   ├── spot/
│   └── futures/
└── ml_results/
    └── SIRENUSDT_TF_MODEL_report.json
```

---

## Instalācija

```bash
python -m venv .venv
.venv\Scripts\activate
pip install lightgbm xgboost scikit-learn pandas numpy
```

**Python:** 3.10+  
**Bitget API:** publiski galapunkti, bez autentifikācijas vajadzības

