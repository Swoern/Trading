"""
market_regime.py - Bepaal eerst het marktregime, daarna pas de strategie.

Dit voorkomt dat de bot een losse candle of indicator volgt terwijl het grotere
plaatje eigenlijk niet bij de strategie past.
"""
from __future__ import annotations

import math

import pandas as pd

from src.features import add_all_features


TREND_LONG_REGIMES = frozenset({"trend_up", "breakout_up"})
TREND_SHORT_REGIMES = frozenset({"trend_down", "breakout_down", "panic"})
RANGE_REGIMES = frozenset({"range", "squeeze", "reversal"})

STRATEGY_REGIME_ALLOWLIST = {
    "trend_long": TREND_LONG_REGIMES,
    "pullback": TREND_LONG_REGIMES | frozenset({"reversal"}),
    "breakout": frozenset({"breakout_up", "squeeze", "trend_up"}),
    "trend_short": TREND_SHORT_REGIMES,
    "pullback_short": TREND_SHORT_REGIMES | frozenset({"reversal"}),
    "breakdown": frozenset({"breakout_down", "panic", "trend_down"}),
    "macd_bull_div": frozenset({"range", "squeeze", "reversal", "panic"}),
    "macd_bear_div": frozenset({"range", "squeeze", "reversal", "euphoria"}),
    "range_long": frozenset({"range", "squeeze"}),
    "range_short": frozenset({"range", "squeeze"}),
}


def _num(value, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def classify_market_regime(
    df: pd.DataFrame,
    htf_trend: str = "onduidelijk",
    fear_greed: float | None = None,
) -> dict:
    """
    Geef een compact regime-object terug.

    score: 0-100, hoger betekent dat het regime duidelijker/sterker is.
    label: trend_up, trend_down, range, squeeze, breakout_up, breakout_down,
           panic, euphoria, reversal of unknown.
    """
    if df is None or len(df) < 60:
        return {
            "label": "unknown",
            "score": 0,
            "direction": "neutral",
            "reason": "Te weinig candles voor regime-analyse.",
        }

    try:
        data = add_all_features(df)
        last = data.iloc[-1]
        prev = data.iloc[-2]
    except Exception as exc:
        return {
            "label": "unknown",
            "score": 0,
            "direction": "neutral",
            "reason": f"Regime-analyse mislukt: {exc}",
        }

    close = _num(last.get("close"))
    sma20 = _num(last.get("sma20"))
    sma50 = _num(last.get("sma50"))
    adx = _num(last.get("adx14"), 0.0)
    atr_pct = _num(last.get("atr_pct"), 0.0)
    bb_width = _num(last.get("bb_width"), 0.0)
    bb_pct = _num(last.get("bb_pct"), 0.5)
    momentum = _num(last.get("momentum_10"), 0.0)
    volatility = _num(last.get("volatility_20"), 0.0)
    obv_slope = _num(last.get("obv_slope"), 0.0)
    volume = _num(last.get("volume"), 0.0)
    vol_ma = _num(last.get("vol_ma20"), 0.0)
    resistance = _num(prev.get("resistance"))
    support = _num(prev.get("support"))
    fg = _num(fear_greed, 50.0)

    trend_up = close > sma20 > sma50 if not any(math.isnan(v) for v in (close, sma20, sma50)) else False
    trend_down = close < sma20 < sma50 if not any(math.isnan(v) for v in (close, sma20, sma50)) else False
    high_volume = vol_ma > 0 and volume >= 1.25 * vol_ma
    squeeze = bb_width > 0 and bb_width < 0.018 and adx < 25
    high_volatility = atr_pct >= 1.4 or volatility >= 0.012
    momentum_up = momentum > 0.006 and obv_slope >= 0
    momentum_down = momentum < -0.006 and obv_slope <= 0
    breakout_up = (
        not math.isnan(resistance)
        and close > resistance
        and high_volume
        and momentum_up
        and bb_pct >= 0.75
    )
    breakout_down = (
        not math.isnan(support)
        and close < support
        and high_volume
        and momentum_down
        and bb_pct <= 0.25
    )

    points: list[str] = []
    label = "range"
    direction = "neutral"
    score = 45

    if high_volatility and fg <= 25 and momentum_down:
        label = "panic"
        direction = "down"
        score = 85
        points.append("hoge volatiliteit + extreme fear + neerwaarts momentum")
    elif high_volatility and fg >= 75 and momentum_up:
        label = "euphoria"
        direction = "up"
        score = 80
        points.append("hoge volatiliteit + greed + opwaarts momentum")
    elif breakout_up:
        label = "breakout_up"
        direction = "up"
        score = 80
        points.append("breakout boven resistance met volume en momentum")
    elif breakout_down:
        label = "breakout_down"
        direction = "down"
        score = 80
        points.append("breakdown onder support met volume en momentum")
    elif trend_up and adx >= 25 and htf_trend == "uptrend":
        label = "trend_up"
        direction = "up"
        score = min(90, 55 + int(adx))
        points.append("5m en 1u trend omhoog")
    elif trend_down and adx >= 25 and htf_trend == "downtrend":
        label = "trend_down"
        direction = "down"
        score = min(90, 55 + int(adx))
        points.append("5m en 1u trend omlaag")
    elif squeeze:
        label = "squeeze"
        score = 65
        points.append("lage bandbreedte en zwakke ADX")
    elif adx < 22 or htf_trend == "onduidelijk":
        label = "range"
        score = 55
        points.append("geen duidelijke trend")
    else:
        label = "reversal"
        direction = "up" if momentum_up else "down" if momentum_down else "neutral"
        score = 60
        points.append("trend/momentum spreken elkaar tegen")

    if high_volatility:
        points.append("volatiliteit hoog")
    if high_volume:
        points.append("volume boven gemiddeld")

    return {
        "label": label,
        "score": int(max(0, min(100, score))),
        "direction": direction,
        "adx": round(adx, 2),
        "atr_pct": round(atr_pct, 3),
        "bb_width": round(bb_width, 4),
        "momentum_10": round(momentum, 5),
        "fear_greed": fg,
        "reason": "; ".join(points) or "Regime neutraal.",
    }


def strategy_allowed_in_regime(strategy: str, regime: dict) -> tuple[bool, str]:
    label = (regime or {}).get("label", "unknown")
    if label == "unknown":
        return False, "Regime onbekend; geen nieuwe trade."
    allowed = STRATEGY_REGIME_ALLOWLIST.get(strategy)
    if not allowed:
        return False, f"Strategie {strategy} heeft geen regime-regel."
    if label not in allowed:
        return False, f"Strategie {strategy} past niet bij regime {label}."
    return True, f"Strategie {strategy} past bij regime {label}."
