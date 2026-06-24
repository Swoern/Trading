"""
features.py - Extra feature-engineering voor TradeAI.

Deze laag bouwt bovenop de bestaande technische indicatoren. De strategie mag
voorzichtig blijven; het leergeheugen krijgt juist rijkere context om patronen
te herkennen zoals momentum, volatiliteit, OBV en tijdsvensters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators import add_all_indicators, ema, sma


def _safe_log_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce").replace(0, np.nan)
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    return np.log(numerator / denominator)


def _obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()


def _session_bucket(hour: int) -> str:
    if 0 <= hour < 7:
        return "azie"
    if 7 <= hour < 13:
        return "europa_ochtend"
    if 13 <= hour < 18:
        return "europa_us_overlap"
    return "us_avond"


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Voeg extra model/learning-features toe.

    Belangrijk: op intraday crypto-data betekent "yesterday_*" hier de vorige
    candle. Bij daily data is dat letterlijk gisteren.
    """
    df = add_all_indicators(df)

    # Moving averages uit de screenshot, naast de bestaande sma20/sma50/ema20.
    df["ma10"] = sma(df["close"], 10)
    df["ma20"] = df["sma20"]
    df["ma30"] = sma(df["close"], 30)
    df["ema10"] = ema(df["close"], 10)
    df["ema30"] = ema(df["close"], 30)

    # Previous/yesterday features.
    df["yesterday_close"] = df["close"].shift(1)
    df["yesterday_open"] = df["open"].shift(1)
    df["yesterday_high"] = df["high"].shift(1)
    df["yesterday_low"] = df["low"].shift(1)
    df["yesterday_volume"] = df["volume"].shift(1)
    df["yesterday_close_logr"] = _safe_log_ratio(df["close"], df["close"].shift(1))
    df["yesterday_open_logr"] = _safe_log_ratio(df["open"], df["open"].shift(1))
    df["yesterday_high_logr"] = _safe_log_ratio(df["high"], df["high"].shift(1))
    df["yesterday_low_logr"] = _safe_log_ratio(df["low"], df["low"].shift(1))
    df["yesterday_volume_logr"] = _safe_log_ratio(df["volume"], df["volume"].shift(1))

    # Return, gap, momentum en volatiliteit.
    df["return_1"] = df["close"].pct_change()
    df["log_return_1"] = df["yesterday_close_logr"]
    df["gap_pct"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1).replace(0, np.nan) * 100
    for period in (3, 10, 20):
        df[f"momentum_{period}"] = df["close"].pct_change(periods=period)
    for period in (10, 20, 30):
        df[f"volatility_{period}"] = df["log_return_1"].rolling(window=period, min_periods=period).std()

    # Mean-reversion / flow.
    close_ma20 = df["close"].rolling(window=20, min_periods=20).mean()
    close_std20 = df["close"].rolling(window=20, min_periods=20).std()
    df["zscore_20"] = (df["close"] - close_ma20) / close_std20.replace(0, np.nan)
    df["obv"] = _obv(df)
    df["obv_slope"] = df["obv"].diff(5)

    # Skew en intraday/tijdfeatures.
    df["skew_20"] = df["log_return_1"].rolling(window=20, min_periods=20).skew()
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["day_of_week"] = ts.dt.dayofweek
    df["day_of_month"] = ts.dt.day
    df["month_number"] = ts.dt.month
    df["hour_utc"] = ts.dt.hour
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["session_bucket"] = df["hour_utc"].apply(lambda h: _session_bucket(int(h)) if pd.notna(h) else "onbekend")

    # Sentiment placeholders: gevuld wanneer externe bronnen later beschikbaar zijn.
    if "fear_greed" not in df.columns:
        df["fear_greed"] = np.nan
    df["fear_greed_change"] = pd.to_numeric(df["fear_greed"], errors="coerce").diff()

    return df
