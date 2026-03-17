#!/usr/bin/env python3
"""
Advanced H4 Multi-Strategy Backtester with Parameter Optimization
==================================================================
Generates realistic 4-hour OHLCV bars, tests 7 advanced strategies with
full parameter grid search across BTCUSDT + Magnificent 7 stocks.
Ranks by composite score (Sharpe + Sortino + Calmar weighted).

Strategies:
  1. Dual EMA + ADX Trend Filter
  2. RSI Divergence with EMA Trend
  3. MACD Histogram Reversal + Volume Spike
  4. Keltner Channel Breakout + Momentum
  5. Ichimoku Cloud Breakout (adapted)
  6. VWAP Reversion + RSI Confluence
  7. SuperTrend + Stochastic RSI Filter

Author: Claude | Date: 2026-03-17
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tabulate import tabulate
from itertools import product
import json, os, sys

# ─── Configuration ───────────────────────────────────────────────────────────

TICKERS = ["BTCUSDT", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

END_DATE   = datetime(2026, 3, 17)
START_DATE = END_DATE - timedelta(days=183)
INITIAL_CAPITAL = 100_000.0
COMMISSION_PCT  = 0.0008      # 0.08% per trade
SLIPPAGE_PCT    = 0.0003      # 0.03% slippage
RISK_FREE_RATE  = 0.05
BARS_PER_DAY    = 6           # H4 = 6 bars per 24h session

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_results_h4")

# Asset profiles: (start_price, ann_vol, ann_drift, mean_revert, intraday_seasonality)
ASSET_PROFILES = {
    "BTCUSDT": (63500,  0.65,  0.40, 0.010, 1.2),
    "AAPL":    (233,    0.28,  0.12, 0.030, 0.8),
    "MSFT":    (430,    0.27,  0.15, 0.030, 0.8),
    "GOOGL":   (165,    0.30,  0.10, 0.020, 0.9),
    "AMZN":    (192,    0.32,  0.18, 0.020, 0.9),
    "NVDA":    (121,    0.55,  0.35, 0.015, 1.1),
    "META":    (565,    0.35,  0.20, 0.020, 0.9),
    "TSLA":    (248,    0.55,  0.05, 0.010, 1.1),
}


# ─── Technical Indicators ───────────────────────────────────────────────────

def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def sma(s, p):
    return s.rolling(p).mean()

def rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0.0)
    l = -d.where(d < 0, 0.0)
    ag = g.ewm(com=p-1, min_periods=p).mean()
    al = l.ewm(com=p-1, min_periods=p).mean()
    return 100 - 100 / (1 + ag / al)

def stoch_rsi(s, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3):
    r = rsi(s, rsi_period)
    rsi_min = r.rolling(stoch_period).min()
    rsi_max = r.rolling(stoch_period).max()
    stoch_k = 100 * (r - rsi_min) / (rsi_max - rsi_min + 1e-10)
    k = stoch_k.rolling(k_smooth).mean()
    d = k.rolling(d_smooth).mean()
    return k, d

def macd_calc(s, fast=12, slow=26, sig=9):
    ml = ema(s, fast) - ema(s, slow)
    sl = ema(ml, sig)
    return ml, sl, ml - sl

def atr(h, l, c, p=14):
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()

def adx(h, l, c, p=14):
    """Average Directional Index."""
    up = h.diff()
    dn = -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    _atr = atr(h, l, c, p)
    pdi = 100 * pd.Series(pdm, index=h.index).ewm(span=p, adjust=False).mean() / (_atr + 1e-10)
    ndi = 100 * pd.Series(ndm, index=h.index).ewm(span=p, adjust=False).mean() / (_atr + 1e-10)
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi + 1e-10)
    _adx = dx.ewm(span=p, adjust=False).mean()
    return _adx, pdi, ndi

def keltner_channels(c, h, l, ema_period=20, atr_period=14, mult=1.5):
    mid = ema(c, ema_period)
    _atr = atr(h, l, c, atr_period)
    return mid + mult * _atr, mid, mid - mult * _atr

def supertrend(h, l, c, period=10, multiplier=3.0):
    _atr = atr(h, l, c, period)
    hl2 = (h + l) / 2
    upper = hl2 + multiplier * _atr
    lower = hl2 - multiplier * _atr

    st = pd.Series(0.0, index=c.index)
    direction = pd.Series(1, index=c.index)

    for i in range(1, len(c)):
        if c.iloc[i] > upper.iloc[i-1]:
            direction.iloc[i] = 1
        elif c.iloc[i] < lower.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]

        if direction.iloc[i] == 1:
            st.iloc[i] = max(lower.iloc[i], st.iloc[i-1]) if direction.iloc[i-1] == 1 else lower.iloc[i]
        else:
            st.iloc[i] = min(upper.iloc[i], st.iloc[i-1]) if direction.iloc[i-1] == -1 else upper.iloc[i]

    return st, direction

def vwap(c, v):
    cum_vol = v.cumsum()
    cum_pv = (c * v).cumsum()
    return cum_pv / (cum_vol + 1e-10)

def rolling_vwap(c, h, l, v, period=20):
    """Rolling VWAP over N bars."""
    typical = (c + h + l) / 3
    pv = typical * v
    return pv.rolling(period).sum() / (v.rolling(period).sum() + 1e-10)


# ─── H4 Data Generation ─────────────────────────────────────────────────────

def generate_h4_ohlcv(label, n_days=130, seed=None):
    """Generate realistic H4 OHLCV with intraday patterns and vol clustering."""
    profile = ASSET_PROFILES[label]
    start_price, ann_vol, ann_drift, mr_strength, intraday_factor = profile

    if seed is None:
        seed = hash(label) % (2**31)
    rng = np.random.RandomState(seed)

    n_bars = n_days * BARS_PER_DAY
    dt = 1 / (252 * BARS_PER_DAY)
    h4_vol = ann_vol * np.sqrt(dt)
    h4_drift = ann_drift * dt

    # Intraday volume/volatility patterns (U-shape for stocks, flatter for crypto)
    if label == "BTCUSDT":
        session_vol_mult = np.array([1.1, 0.9, 0.8, 1.0, 1.2, 1.3])  # 24h crypto
    else:
        session_vol_mult = np.array([1.4, 1.0, 0.7, 0.8, 1.1, 1.5])  # U-shape stocks

    closes = np.zeros(n_bars)
    closes[0] = start_price
    vol_state = h4_vol

    for i in range(1, n_bars):
        session_idx = i % BARS_PER_DAY
        sv = session_vol_mult[session_idx] * intraday_factor

        # GARCH-like clustering
        shock = rng.normal(0, 0.12)
        vol_state = 0.92 * vol_state + 0.08 * h4_vol * (1 + shock)
        vol_state = max(vol_state, h4_vol * 0.25)

        # Mean reversion to trend
        day_idx = i // BARS_PER_DAY
        trend_price = start_price * np.exp(ann_drift * day_idx / 252)
        mr_pull = mr_strength * (np.log(trend_price) - np.log(closes[i-1])) / BARS_PER_DAY

        # Jump component (rare large moves)
        jump = 0
        if rng.random() < 0.02:  # 2% chance per bar
            jump = rng.normal(0, h4_vol * 3)

        z = rng.normal()
        log_ret = (h4_drift + mr_pull - 0.5 * (vol_state * sv)**2) + vol_state * sv * z + jump
        closes[i] = closes[i-1] * np.exp(log_ret)

    # Build OHLV
    dates = pd.date_range(start=START_DATE, periods=n_bars, freq="4h")
    opens = np.zeros(n_bars)
    highs = np.zeros(n_bars)
    lows = np.zeros(n_bars)
    volumes = np.zeros(n_bars)

    opens[0] = closes[0] * (1 + rng.normal(0, 0.001))
    base_vol = 8_000_000 if label != "BTCUSDT" else 5_000_000_000

    for i in range(n_bars):
        if i > 0:
            opens[i] = closes[i-1] * (1 + rng.normal(0, h4_vol * 0.15))

        session_idx = i % BARS_PER_DAY
        intra_range = abs(rng.normal(0, h4_vol * 1.2)) * session_vol_mult[session_idx]
        highs[i] = max(opens[i], closes[i]) + abs(closes[i]) * intra_range * 0.4
        lows[i] = min(opens[i], closes[i]) - abs(closes[i]) * intra_range * 0.4
        volumes[i] = base_vol * session_vol_mult[session_idx] * (1 + abs(rng.normal(0, 0.4)))

    return pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows, "Close": closes,
        "Volume": volumes.astype(int)
    }, index=dates[:n_bars])


# ─── Strategy Implementations ───────────────────────────────────────────────

def strat_dual_ema_adx(df, fast=9, slow=21, adx_thresh=25):
    """S1: Dual EMA crossover only when ADX > threshold (trending market)."""
    d = df.copy()
    ef = ema(d["Close"], fast)
    es = ema(d["Close"], slow)
    _adx, pdi, ndi = adx(d["High"], d["Low"], d["Close"])

    sig = np.zeros(len(d))
    for i in range(1, len(d)):
        if _adx.iloc[i] > adx_thresh:
            if ef.iloc[i] > es.iloc[i] and pdi.iloc[i] > ndi.iloc[i]:
                sig[i] = 1
            elif ef.iloc[i] < es.iloc[i] and ndi.iloc[i] > pdi.iloc[i]:
                sig[i] = -1
            else:
                sig[i] = sig[i-1] if abs(sig[i-1]) > 0 else 0
        else:
            sig[i] = 0  # flat in ranging market
    d["signal"] = sig
    return d


def strat_rsi_ema_divergence(df, rsi_period=14, ema_period=50, oversold=35, overbought=65):
    """S2: RSI reversal zones with EMA trend filter. Only long above EMA, short below."""
    d = df.copy()
    r = rsi(d["Close"], rsi_period)
    e = ema(d["Close"], ema_period)

    sig = np.zeros(len(d))
    pos = 0
    for i in range(2, len(d)):
        above_ema = d["Close"].iloc[i] > e.iloc[i]
        below_ema = d["Close"].iloc[i] < e.iloc[i]

        # RSI bouncing from oversold + above EMA = long
        if r.iloc[i] > oversold and r.iloc[i-1] <= oversold and above_ema:
            pos = 1
        # RSI dropping from overbought + below EMA = short
        elif r.iloc[i] < overbought and r.iloc[i-1] >= overbought and below_ema:
            pos = -1
        # Exit long if RSI overbought or price drops below EMA
        elif pos == 1 and (r.iloc[i] > overbought or below_ema):
            pos = 0
        # Exit short if RSI oversold or price rises above EMA
        elif pos == -1 and (r.iloc[i] < oversold or above_ema):
            pos = 0
        sig[i] = pos
    d["signal"] = sig
    return d


def strat_macd_histogram_reversal(df, fast=12, slow=26, signal=9, vol_mult=1.5):
    """S3: MACD histogram reversal with volume spike confirmation."""
    d = df.copy()
    ml, sl, hist = macd_calc(d["Close"], fast, slow, signal)
    vol_sma = sma(d["Volume"].astype(float), 20)

    sig = np.zeros(len(d))
    pos = 0
    for i in range(2, len(d)):
        hist_rising = hist.iloc[i] > hist.iloc[i-1]
        hist_falling = hist.iloc[i] < hist.iloc[i-1]
        vol_spike = d["Volume"].iloc[i] > vol_sma.iloc[i] * vol_mult if not np.isnan(vol_sma.iloc[i]) else False

        # Histogram turns up from negative = buy signal
        if hist.iloc[i] < 0 and hist_rising and hist.iloc[i-1] < hist.iloc[i-2] and vol_spike:
            pos = 1
        elif hist.iloc[i] > 0 and hist_falling and hist.iloc[i-1] > hist.iloc[i-2] and vol_spike:
            pos = -1
        # Trend continuation: histogram positive and rising
        elif pos == 1 and hist.iloc[i] > 0 and ml.iloc[i] > sl.iloc[i]:
            pos = 1
        elif pos == -1 and hist.iloc[i] < 0 and ml.iloc[i] < sl.iloc[i]:
            pos = -1
        # Exit on crossover
        elif pos == 1 and ml.iloc[i] < sl.iloc[i]:
            pos = 0
        elif pos == -1 and ml.iloc[i] > sl.iloc[i]:
            pos = 0
        sig[i] = pos
    d["signal"] = sig
    return d


def strat_keltner_momentum(df, ema_period=20, atr_period=14, mult=2.0, mom_period=10):
    """S4: Keltner Channel breakout with momentum confirmation."""
    d = df.copy()
    ku, km, kl = keltner_channels(d["Close"], d["High"], d["Low"], ema_period, atr_period, mult)
    momentum = d["Close"] / d["Close"].shift(mom_period) - 1

    sig = np.zeros(len(d))
    pos = 0
    for i in range(1, len(d)):
        if np.isnan(ku.iloc[i]) or np.isnan(momentum.iloc[i]):
            sig[i] = pos
            continue

        # Breakout above upper Keltner + positive momentum
        if d["Close"].iloc[i] > ku.iloc[i] and momentum.iloc[i] > 0:
            pos = 1
        elif d["Close"].iloc[i] < kl.iloc[i] and momentum.iloc[i] < 0:
            pos = -1
        # Take profit at mid band
        elif pos == 1 and d["Close"].iloc[i] < km.iloc[i]:
            pos = 0
        elif pos == -1 and d["Close"].iloc[i] > km.iloc[i]:
            pos = 0
        sig[i] = pos
    d["signal"] = sig
    return d


def strat_ichimoku(df, tenkan=9, kijun=26, senkou_b=52):
    """S5: Simplified Ichimoku Cloud breakout adapted for H4."""
    d = df.copy()
    # Tenkan-sen (conversion)
    t_high = d["High"].rolling(tenkan).max()
    t_low = d["Low"].rolling(tenkan).min()
    tenkan_sen = (t_high + t_low) / 2

    # Kijun-sen (base)
    k_high = d["High"].rolling(kijun).max()
    k_low = d["Low"].rolling(kijun).min()
    kijun_sen = (k_high + k_low) / 2

    # Senkou Span A & B (cloud)
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    sb_high = d["High"].rolling(senkou_b).max()
    sb_low = d["Low"].rolling(senkou_b).min()
    senkou_b_line = ((sb_high + sb_low) / 2).shift(kijun)

    cloud_top = pd.concat([senkou_a, senkou_b_line], axis=1).max(axis=1)
    cloud_bot = pd.concat([senkou_a, senkou_b_line], axis=1).min(axis=1)

    sig = np.zeros(len(d))
    pos = 0
    for i in range(1, len(d)):
        if np.isnan(cloud_top.iloc[i]):
            sig[i] = pos
            continue

        above_cloud = d["Close"].iloc[i] > cloud_top.iloc[i]
        below_cloud = d["Close"].iloc[i] < cloud_bot.iloc[i]
        tk_cross_up = tenkan_sen.iloc[i] > kijun_sen.iloc[i]
        tk_cross_dn = tenkan_sen.iloc[i] < kijun_sen.iloc[i]

        if above_cloud and tk_cross_up:
            pos = 1
        elif below_cloud and tk_cross_dn:
            pos = -1
        elif pos == 1 and below_cloud:
            pos = 0
        elif pos == -1 and above_cloud:
            pos = 0
        sig[i] = pos
    d["signal"] = sig
    return d


def strat_vwap_rsi(df, vwap_period=30, rsi_period=14, rsi_low=40, rsi_high=60):
    """S6: Rolling VWAP reversion + RSI confluence. Buy below VWAP when RSI oversold."""
    d = df.copy()
    rv = rolling_vwap(d["Close"], d["High"], d["Low"], d["Volume"].astype(float), vwap_period)
    r = rsi(d["Close"], rsi_period)
    _atr = atr(d["High"], d["Low"], d["Close"])

    sig = np.zeros(len(d))
    pos = 0
    for i in range(2, len(d)):
        if np.isnan(rv.iloc[i]) or np.isnan(_atr.iloc[i]):
            sig[i] = pos
            continue

        dist_from_vwap = (d["Close"].iloc[i] - rv.iloc[i]) / (_atr.iloc[i] + 1e-10)

        # Price below VWAP by >1 ATR and RSI oversold = long reversion
        if dist_from_vwap < -1.0 and r.iloc[i] < rsi_low and r.iloc[i] > r.iloc[i-1]:
            pos = 1
        elif dist_from_vwap > 1.0 and r.iloc[i] > rsi_high and r.iloc[i] < r.iloc[i-1]:
            pos = -1
        # Take profit at VWAP
        elif pos == 1 and d["Close"].iloc[i] > rv.iloc[i]:
            pos = 0
        elif pos == -1 and d["Close"].iloc[i] < rv.iloc[i]:
            pos = 0
        # Stop loss at 2 ATR
        elif pos != 0:
            pass  # managed by engine
        sig[i] = pos
    d["signal"] = sig
    return d


def strat_supertrend_stochrsi(df, st_period=10, st_mult=3.0, stoch_k_thresh=20):
    """S7: SuperTrend for trend + Stochastic RSI for timing entries."""
    d = df.copy()
    st, direction = supertrend(d["High"], d["Low"], d["Close"], st_period, st_mult)
    k, dd = stoch_rsi(d["Close"])

    sig = np.zeros(len(d))
    pos = 0
    for i in range(2, len(d)):
        if np.isnan(k.iloc[i]):
            sig[i] = pos
            continue

        uptrend = direction.iloc[i] == 1
        downtrend = direction.iloc[i] == -1

        # Long: uptrend + stoch RSI crossing up from oversold
        if uptrend and k.iloc[i] > stoch_k_thresh and k.iloc[i-1] <= stoch_k_thresh:
            pos = 1
        elif downtrend and k.iloc[i] < (100 - stoch_k_thresh) and k.iloc[i-1] >= (100 - stoch_k_thresh):
            pos = -1
        # Exit on trend reversal
        elif pos == 1 and downtrend:
            pos = 0
        elif pos == -1 and uptrend:
            pos = 0
        sig[i] = pos
    d["signal"] = sig
    return d


# ─── Strategy Registry with Parameter Grids ─────────────────────────────────

STRATEGIES = {
    "Dual EMA + ADX": {
        "func": strat_dual_ema_adx,
        "params": [
            {"fast": 8, "slow": 21, "adx_thresh": 20},
            {"fast": 9, "slow": 26, "adx_thresh": 25},
            {"fast": 12, "slow": 34, "adx_thresh": 22},
            {"fast": 5, "slow": 13, "adx_thresh": 20},
        ]
    },
    "RSI + EMA Trend": {
        "func": strat_rsi_ema_divergence,
        "params": [
            {"rsi_period": 14, "ema_period": 50, "oversold": 40, "overbought": 60},
            {"rsi_period": 10, "ema_period": 34, "oversold": 38, "overbought": 62},
            {"rsi_period": 14, "ema_period": 21, "oversold": 42, "overbought": 58},
            {"rsi_period": 7, "ema_period": 50, "oversold": 35, "overbought": 65},
        ]
    },
    "MACD Hist Reversal": {
        "func": strat_macd_histogram_reversal,
        "params": [
            {"fast": 12, "slow": 26, "signal": 9, "vol_mult": 1.3},
            {"fast": 8, "slow": 21, "signal": 5, "vol_mult": 1.5},
            {"fast": 12, "slow": 26, "signal": 9, "vol_mult": 1.0},
            {"fast": 5, "slow": 13, "signal": 4, "vol_mult": 1.2},
        ]
    },
    "Keltner Breakout": {
        "func": strat_keltner_momentum,
        "params": [
            {"ema_period": 20, "atr_period": 14, "mult": 2.0, "mom_period": 10},
            {"ema_period": 15, "atr_period": 10, "mult": 1.5, "mom_period": 8},
            {"ema_period": 20, "atr_period": 14, "mult": 2.5, "mom_period": 14},
            {"ema_period": 10, "atr_period": 7, "mult": 1.8, "mom_period": 6},
        ]
    },
    "Ichimoku Cloud": {
        "func": strat_ichimoku,
        "params": [
            {"tenkan": 9, "kijun": 26, "senkou_b": 52},
            {"tenkan": 7, "kijun": 22, "senkou_b": 44},
            {"tenkan": 12, "kijun": 30, "senkou_b": 60},
            {"tenkan": 6, "kijun": 18, "senkou_b": 36},
        ]
    },
    "VWAP Reversion + RSI": {
        "func": strat_vwap_rsi,
        "params": [
            {"vwap_period": 30, "rsi_period": 14, "rsi_low": 40, "rsi_high": 60},
            {"vwap_period": 20, "rsi_period": 10, "rsi_low": 35, "rsi_high": 65},
            {"vwap_period": 40, "rsi_period": 14, "rsi_low": 38, "rsi_high": 62},
            {"vwap_period": 24, "rsi_period": 7, "rsi_low": 30, "rsi_high": 70},
        ]
    },
    "SuperTrend + StochRSI": {
        "func": strat_supertrend_stochrsi,
        "params": [
            {"st_period": 10, "st_mult": 3.0, "stoch_k_thresh": 20},
            {"st_period": 7, "st_mult": 2.5, "stoch_k_thresh": 25},
            {"st_period": 14, "st_mult": 3.5, "stoch_k_thresh": 15},
            {"st_period": 10, "st_mult": 2.0, "stoch_k_thresh": 20},
        ]
    },
}


# ─── Backtesting Engine (clean percentage-based with ATR SL/TP) ─────────────

def run_backtest(df, signals, initial_capital=INITIAL_CAPITAL,
                 commission=COMMISSION_PCT, slippage=SLIPPAGE_PCT,
                 atr_sl_mult=2.0, atr_tp_mult=3.0):
    """
    Clean backtest engine. Tracks equity directly via percentage returns.
    No share-tracking bugs. ATR-based stop loss and take profit.
    """
    prices = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    sigs = signals.values
    n = len(prices)

    _atr_series = atr(df["High"], df["Low"], df["Close"], 14).values
    cost = commission + slippage  # total friction per trade

    equity = np.zeros(n)
    equity[0] = initial_capital
    current_equity = initial_capital

    pos = 0          # +1 long, -1 short, 0 flat
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    trades = []

    for i in range(1, n):
        sig = int(sigs[i])
        cur_atr = _atr_series[i] if not np.isnan(_atr_series[i]) else 0
        price = prices[i]

        # ── Check SL/TP hits on existing position ──
        if pos == 1:
            if lows[i] <= sl_price:
                ret = (sl_price / entry_price - 1) - cost
                current_equity *= (1 + ret)
                trades.append({"pnl_pct": ret, "dir": "L"})
                pos = 0
            elif highs[i] >= tp_price:
                ret = (tp_price / entry_price - 1) - cost
                current_equity *= (1 + ret)
                trades.append({"pnl_pct": ret, "dir": "L"})
                pos = 0
        elif pos == -1:
            if highs[i] >= sl_price:
                ret = (entry_price / sl_price - 1) - cost
                current_equity *= (1 + ret)
                trades.append({"pnl_pct": ret, "dir": "S"})
                pos = 0
            elif lows[i] <= tp_price:
                ret = (entry_price / tp_price - 1) - cost
                current_equity *= (1 + ret)
                trades.append({"pnl_pct": ret, "dir": "S"})
                pos = 0

        # ── Signal change ──
        if sig != pos:
            # Close existing position at current price
            if pos == 1:
                ret = (price / entry_price - 1) - cost
                current_equity *= (1 + ret)
                trades.append({"pnl_pct": ret, "dir": "L"})
            elif pos == -1:
                ret = (entry_price / price - 1) - cost
                current_equity *= (1 + ret)
                trades.append({"pnl_pct": ret, "dir": "S"})

            # Open new position
            if sig == 1 and cur_atr > 0:
                pos = 1
                entry_price = price
                sl_price = price - atr_sl_mult * cur_atr
                tp_price = price + atr_tp_mult * cur_atr
            elif sig == -1 and cur_atr > 0:
                pos = -1
                entry_price = price
                sl_price = price + atr_sl_mult * cur_atr
                tp_price = price - atr_tp_mult * cur_atr
            else:
                pos = 0

        # Mark to market
        if pos == 1:
            mtm = current_equity * (price / entry_price)
        elif pos == -1:
            mtm = current_equity * (2 - price / entry_price)  # short P&L
            mtm = max(mtm, 0)
        else:
            mtm = current_equity
        equity[i] = mtm

    equity_series = pd.Series(equity, index=df.index)
    returns = equity_series.pct_change().dropna()
    returns = returns.replace([np.inf, -np.inf], 0)

    total_return = (equity[-1] / initial_capital - 1) * 100

    # Sharpe
    if len(returns) > 1 and returns.std() > 0:
        ann_factor = np.sqrt(252 * BARS_PER_DAY)
        sharpe = (returns.mean() - RISK_FREE_RATE / (252 * BARS_PER_DAY)) / returns.std() * ann_factor
    else:
        sharpe = 0.0

    # Max drawdown
    peak = equity_series.cummax()
    dd = (equity_series - peak) / (peak + 1e-10)
    max_dd = dd.min() * 100

    # Sortino
    downside = returns[returns < 0]
    if len(downside) > 1 and downside.std() > 0:
        sortino = (returns.mean() - RISK_FREE_RATE / (252 * BARS_PER_DAY)) / downside.std() * np.sqrt(252 * BARS_PER_DAY)
    else:
        sortino = 0.0

    # Calmar (annualized return / max dd)
    ann_return = total_return / 100 * (365 / 183)
    calmar = ann_return / abs(max_dd / 100) if max_dd != 0 else 0

    # Trade stats
    n_trades = len(trades)
    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0
    avg_win = np.mean([t["pnl_pct"] for t in wins]) * 100 if wins else 0
    avg_loss = np.mean([t["pnl_pct"] for t in losses]) * 100 if losses else 0
    profit_factor = (sum(t["pnl_pct"] for t in wins) / abs(sum(t["pnl_pct"] for t in losses))
                     if losses and sum(t["pnl_pct"] for t in losses) != 0 else
                     (999 if wins else 0))

    # Composite score: weighted blend
    composite = 0.4 * sharpe + 0.3 * sortino + 0.2 * calmar + 0.1 * (win_rate / 100 * 3)

    return {
        "total_return_pct": round(total_return, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd_pct": round(max_dd, 2),
        "calmar": round(calmar, 3),
        "n_trades": n_trades,
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "final_equity": round(equity[-1], 2),
        "composite": round(composite, 3),
        "equity_curve": equity_series,
    }


# ─── Visualization ──────────────────────────────────────────────────────────

def plot_top_results(results, top_n=8):
    """Plot equity curves for top N results."""
    fig, axes = plt.subplots(min(top_n, 4), 2, figsize=(18, 5 * min(top_n, 4)), tight_layout=True)
    axes = axes.flatten() if top_n > 2 else [axes] if top_n == 1 else axes.flatten()

    for idx, (ax, r) in enumerate(zip(axes[:top_n], results[:top_n])):
        eq = r["equity_curve"]
        ax.plot(eq.index, eq.values, linewidth=1.2, color="#1565C0")
        ax.axhline(INITIAL_CAPITAL, color="gray", ls="--", alpha=0.4)
        ax.fill_between(eq.index, INITIAL_CAPITAL, eq.values,
                        where=eq.values >= INITIAL_CAPITAL, alpha=0.12, color="#4CAF50")
        ax.fill_between(eq.index, INITIAL_CAPITAL, eq.values,
                        where=eq.values < INITIAL_CAPITAL, alpha=0.12, color="#F44336")
        params_str = str(r.get("params", ""))[:40]
        ax.set_title(f"#{idx+1} {r['asset']} | {r['strategy']}\n"
                     f"Ret:{r['total_return_pct']:+.1f}% Sharpe:{r['sharpe']:.2f} "
                     f"DD:{r['max_dd_pct']:.1f}% WR:{r['win_rate']:.0f}% "
                     f"PF:{r['profit_factor']:.1f} [{params_str}]",
                     fontsize=9, fontweight="bold")
        ax.set_ylabel("Equity ($)")
        ax.grid(True, alpha=0.2)

    for ax in axes[top_n:]:
        ax.set_visible(False)

    fig.suptitle("TOP H4 STRATEGIES — Ranked by Composite Score (Sharpe·Sortino·Calmar·WR)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.savefig(os.path.join(OUTPUT_DIR, "top_h4_equity_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_heatmap(results):
    """Heatmap of best composite score per asset × strategy."""
    assets = list(set(r["asset"] for r in results))
    strats = list(set(r["strategy"] for r in results))
    assets.sort()
    strats.sort()

    matrix = np.full((len(assets), len(strats)), np.nan)
    for r in results:
        ai = assets.index(r["asset"])
        si = strats.index(r["strategy"])
        if np.isnan(matrix[ai, si]) or r["composite"] > matrix[ai, si]:
            matrix[ai, si] = r["composite"]

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-2, vmax=3)
    ax.set_xticks(range(len(strats)))
    ax.set_xticklabels(strats, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(assets)))
    ax.set_yticklabels(assets, fontsize=10)

    for i in range(len(assets)):
        for j in range(len(strats)):
            v = matrix[i, j]
            if not np.isnan(v):
                c = "white" if abs(v) > 1.5 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, fontweight="bold", color=c)

    plt.colorbar(im, ax=ax, label="Composite Score")
    ax.set_title("H4 COMPOSITE SCORE HEATMAP (Best Params per Cell)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "h4_composite_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("  ADVANCED H4 BACKTESTER — Parameter Optimization")
    print(f"  Period: {START_DATE.date()} to {END_DATE.date()} (4-Hour Bars)")
    print(f"  Capital: ${INITIAL_CAPITAL:,.0f} | Commission: {COMMISSION_PCT*100:.2f}% | Slippage: {SLIPPAGE_PCT*100:.2f}%")
    print(f"  Assets: {len(TICKERS)} | Strategies: {len(STRATEGIES)} | Param sets: {sum(len(s['params']) for s in STRATEGIES.values())}")
    total_tests = len(TICKERS) * sum(len(s["params"]) for s in STRATEGIES.values())
    print(f"  Total backtests: {total_tests}")
    print("=" * 78)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Generate H4 data
    print("\n[1/4] Generating H4 data...")
    data = {}
    for label in TICKERS:
        df = generate_h4_ohlcv(label)
        data[label] = df
        print(f"  {label:8s}: {len(df)} bars, ${df['Close'].iloc[0]:>10,.2f} → ${df['Close'].iloc[-1]:>10,.2f} "
              f"(range: ${df['Low'].min():,.2f} - ${df['High'].max():,.2f})")

    # 2. Run all backtests with parameter optimization
    print(f"\n[2/4] Running {total_tests} backtests with parameter optimization...")
    all_results = []
    count = 0

    for asset, df in data.items():
        for strat_name, strat_config in STRATEGIES.items():
            best_for_combo = None
            for params in strat_config["params"]:
                count += 1
                try:
                    strat_df = strat_config["func"](df, **params)
                    result = run_backtest(df, strat_df["signal"])
                    result["asset"] = asset
                    result["strategy"] = strat_name
                    result["params"] = params

                    if best_for_combo is None or result["composite"] > best_for_combo["composite"]:
                        best_for_combo = result
                except Exception as e:
                    pass

            if best_for_combo:
                all_results.append(best_for_combo)
                m = "+" if best_for_combo["total_return_pct"] > 0 else "-"
                print(f"  [{m}] {asset:8s} | {strat_name:24s} | "
                      f"Ret:{best_for_combo['total_return_pct']:+8.2f}% | "
                      f"Sharpe:{best_for_combo['sharpe']:6.2f} | "
                      f"DD:{best_for_combo['max_dd_pct']:7.2f}% | "
                      f"WR:{best_for_combo['win_rate']:5.1f}% | "
                      f"PF:{best_for_combo['profit_factor']:5.2f} | "
                      f"Score:{best_for_combo['composite']:5.2f}")

    # 3. Sort by composite score
    all_results.sort(key=lambda x: x["composite"], reverse=True)

    print(f"\n[3/4] Generating report...")

    # Full table
    rows = []
    for i, r in enumerate(all_results, 1):
        rows.append([i, r["asset"], r["strategy"], r["total_return_pct"], r["sharpe"],
                      r["sortino"], r["max_dd_pct"], r["calmar"], r["n_trades"],
                      r["win_rate"], r["profit_factor"], r["final_equity"], r["composite"],
                      str(r["params"])])

    headers = ["#", "Asset", "Strategy", "Ret%", "Sharpe", "Sortino", "MaxDD%",
               "Calmar", "Trades", "WR%", "PF", "Equity", "Score", "Best Params"]

    print("\n" + "=" * 78)
    print("  ALL RESULTS — Ranked by Composite Score")
    print("=" * 78)
    print(tabulate(rows, headers=headers, tablefmt="grid",
                   floatfmt=(".0f", "", "", ".2f", ".3f", ".3f", ".2f", ".3f", ".0f", ".1f", ".2f", ",.2f", ".3f", "")))

    # Top 10
    print("\n" + "=" * 78)
    print("  TOP 10 BEST H4 SETUPS")
    print("=" * 78)
    print(tabulate(rows[:10], headers=headers, tablefmt="grid",
                   floatfmt=(".0f", "", "", ".2f", ".3f", ".3f", ".2f", ".3f", ".0f", ".1f", ".2f", ",.2f", ".3f", "")))

    # Best setup detail
    best = all_results[0]
    print("\n" + "*" * 78)
    print(f"  BEST H4 SETUP: {best['asset']} × {best['strategy']}")
    print(f"  {'─' * 60}")
    print(f"  Parameters:     {best['params']}")
    print(f"  Total Return:   {best['total_return_pct']:+.2f}%")
    print(f"  Sharpe Ratio:   {best['sharpe']:.3f}")
    print(f"  Sortino Ratio:  {best['sortino']:.3f}")
    print(f"  Max Drawdown:   {best['max_dd_pct']:.2f}%")
    print(f"  Calmar Ratio:   {best['calmar']:.3f}")
    print(f"  Win Rate:       {best['win_rate']:.1f}%")
    print(f"  Profit Factor:  {best['profit_factor']:.2f}")
    print(f"  Total Trades:   {best['n_trades']}")
    print(f"  Final Equity:   ${best['final_equity']:,.2f}")
    print(f"  Composite:      {best['composite']:.3f}")
    print("*" * 78)

    # Second and third for comparison
    if len(all_results) >= 3:
        print(f"\n  RUNNER-UP: {all_results[1]['asset']} × {all_results[1]['strategy']}")
        print(f"  Params: {all_results[1]['params']}")
        print(f"  Ret: {all_results[1]['total_return_pct']:+.2f}% | Sharpe: {all_results[1]['sharpe']:.3f} | Score: {all_results[1]['composite']:.3f}")
        print(f"\n  3RD PLACE: {all_results[2]['asset']} × {all_results[2]['strategy']}")
        print(f"  Params: {all_results[2]['params']}")
        print(f"  Ret: {all_results[2]['total_return_pct']:+.2f}% | Sharpe: {all_results[2]['sharpe']:.3f} | Score: {all_results[2]['composite']:.3f}")

    # 4. Charts
    print(f"\n[4/4] Saving charts...")
    plot_top_results(all_results, top_n=8)
    plot_heatmap(all_results)

    # Save CSV
    df_out = pd.DataFrame(rows, columns=headers)
    csv_path = os.path.join(OUTPUT_DIR, "h4_full_results.csv")
    df_out.to_csv(csv_path, index=False)

    # Save best setup JSON
    best_json = {k: v for k, v in best.items() if k != "equity_curve"}
    best_json["timeframe"] = "H4"
    best_json["period"] = f"{START_DATE.date()} to {END_DATE.date()}"
    best_json["params"] = str(best["params"])
    json_path = os.path.join(OUTPUT_DIR, "best_h4_setup.json")
    with open(json_path, "w") as f:
        json.dump(best_json, f, indent=2, default=str)

    print(f"\n  Saved: {csv_path}")
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'top_h4_equity_curves.png')}")
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'h4_composite_heatmap.png')}")
    print(f"  Saved: {json_path}")
    print("\n" + "=" * 78)
    print("  H4 BACKTESTING COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
