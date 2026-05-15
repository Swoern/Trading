"""
indicators.py — Technische indicatoren berekend zonder externe bibliotheken.

Elke functie heeft een uitleg zodat je begrijpt wat hij doet.
"""
import pandas as pd
import numpy as np


# ── Trendlijnen ──────────────────────────────────────────────────────────────

def sma(series: pd.Series, period: int) -> pd.Series:
    """
    Simple Moving Average (SMA) — gewoon gemiddelde over N candles.
    Gebruik: SMA20 en SMA50 om de trendrichting te bepalen.
    Als close > SMA20 > SMA50 → uptrend.
    """
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """
    Exponential Moving Average (EMA) — recentere candles wegen zwaarder.
    Reageert sneller op prijsveranderingen dan SMA.
    """
    return series.ewm(span=period, adjust=False).mean()


# ── Volatiliteit ─────────────────────────────────────────────────────────────

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range (ATR) — maat voor hoe veel de markt beweegt.
    Grotere ATR = wildere markt. Gebruik voor stop-loss bepaling.
    Vuistregel: stop-loss = 1.5 × ATR onder entry.
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    return tr.rolling(window=period, min_periods=period).mean()


# ── Momentum ─────────────────────────────────────────────────────────────────

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI) — geeft een waarde van 0 tot 100.
    > 70 : mogelijk overbought (te duur, kans op daling)
    < 30 : mogelijk oversold  (te goedkoop, kans op stijging)
    50   : neutraal
    """
    delta = series.diff()
    winst = delta.clip(lower=0).rolling(window=period, min_periods=period).mean()
    verlies = (-delta.clip(upper=0)).rolling(window=period, min_periods=period).mean()
    rs = winst / verlies.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ── Support & Resistance ──────────────────────────────────────────────────────

def support_resistance(df: pd.DataFrame, lookback: int = 20):
    """
    Simpele support en resistance op basis van rolling min/max.
    Support    = laagste punt van de laatste N candles (vloer, kopers stappen in)
    Resistance = hoogste punt van de laatste N candles (plafond, verkopers stappen in)
    """
    support    = df["low"].rolling(window=lookback, min_periods=lookback).min()
    resistance = df["high"].rolling(window=lookback, min_periods=lookback).max()
    return support, resistance


# ── Volume ────────────────────────────────────────────────────────────────────

def volume_ma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Gemiddeld handelsvolume over N perioden. Hogere volumepieken = sterkere beweging."""
    return df["volume"].rolling(window=period, min_periods=period).mean()


# ── Alles in één keer toevoegen ───────────────────────────────────────────────

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Voeg alle indicatoren toe aan het DataFrame en geef het terug.
    Originele data wordt NIET aangepast (copy).
    """
    df = df.copy()

    df["sma20"]      = sma(df["close"], 20)
    df["sma50"]      = sma(df["close"], 50)
    df["ema20"]      = ema(df["close"], 20)
    df["atr14"]      = atr(df, 14)
    df["rsi14"]      = rsi(df["close"], 14)
    df["vol_ma20"]   = volume_ma(df, 20)
    df["support"]    = df["low"].rolling(20, min_periods=20).min()
    df["resistance"] = df["high"].rolling(20, min_periods=20).max()
    df["atr_pct"]    = df["atr14"] / df["close"] * 100   # ATR als % van de prijs

    return df
