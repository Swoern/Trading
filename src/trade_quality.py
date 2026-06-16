"""trade_quality.py - Kwaliteit van een setup meten vóór entry."""
from __future__ import annotations

import math

import pandas as pd

from src.features import add_all_features


def _num(value, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def score_trade_quality(
    df: pd.DataFrame,
    signal: dict,
    liquidation_zones: list | None = None,
    market_context: dict | None = None,
) -> dict:
    score = 100
    reasons: list[str] = []
    liquidation_zones = liquidation_zones or []
    market_context = market_context or {}

    try:
        data = add_all_features(df)
        last = data.iloc[-1]
    except Exception as exc:
        return {"score": 40, "reasons": [f"kwaliteit niet berekend: {exc}"], "block": False}

    entry = _num(signal.get("entry"))
    sl = _num(signal.get("stop_loss"))
    tp = _num(signal.get("take_profit"))
    direction = signal.get("richting", "long")
    atr = _num(last.get("atr14"), 0.0)
    close = _num(last.get("close"))
    support = _num(last.get("support"))
    resistance = _num(last.get("resistance"))
    bb_pct = _num(last.get("bb_pct"), 0.5)
    risk = abs(entry - sl)
    reward = abs(tp - entry)

    if entry <= 0 or risk <= 0:
        return {"score": 0, "reasons": ["ongeldige entry/SL"], "block": True}

    if atr > 0:
        tp_atr = reward / atr
        sl_atr = risk / atr
        if tp_atr > 5.0:
            score -= 15
            reasons.append(f"TP ver weg ({tp_atr:.1f} ATR)")
        if sl_atr < 0.7:
            score -= 15
            reasons.append(f"SL krap ({sl_atr:.1f} ATR)")
        if abs(close - entry) > 1.8 * atr:
            score -= 10
            reasons.append("entry ver van huidige candle-close")

    if direction == "long":
        if not math.isnan(resistance) and tp > resistance and abs(tp - resistance) / entry < 0.004:
            score -= 10
            reasons.append("TP ligt vlak achter resistance")
        if bb_pct > 0.95:
            score -= 10
            reasons.append("long dicht bij Bollinger-bovenband")
    else:
        if not math.isnan(support) and tp < support and abs(tp - support) / entry < 0.004:
            score -= 10
            reasons.append("TP ligt vlak achter support")
        if bb_pct < 0.05:
            score -= 10
            reasons.append("short dicht bij Bollinger-onderband")

    spread_pct = _num((market_context.get("spread") or {}).get("spread_pct"), 0.0)
    if spread_pct > 0.08:
        score -= 20
        reasons.append(f"spread hoog ({spread_pct:.3f}%)")

    orderbook = market_context.get("orderbook") or {}
    imbalance = _num(orderbook.get("imbalance"), 0.0)
    if direction == "long" and imbalance < -0.18:
        score -= 12
        reasons.append(f"orderbook verkoopdruk ({imbalance:.2f})")
    if direction == "short" and imbalance > 0.18:
        score -= 12
        reasons.append(f"orderbook koopdruk ({imbalance:.2f})")

    derivatives = market_context.get("derivatives") or {}
    funding = _num(derivatives.get("funding_rate"), 0.0)
    oi_change = _num(derivatives.get("open_interest_change"), 0.0)
    long_short = _num(derivatives.get("long_short_ratio"), 1.0)
    if direction == "long" and funding > 0.0008:
        score -= 8
        reasons.append("funding hoog voor long")
    if direction == "short" and funding < -0.0008:
        score -= 8
        reasons.append("funding negatief voor short")
    if abs(oi_change) > 0 and oi_change < -3.0:
        score -= 8
        reasons.append(f"open interest daalt ({oi_change:.1f}%)")
    if direction == "long" and long_short > 2.0:
        score -= 6
        reasons.append("long/short ratio crowded long")
    if direction == "short" and long_short < 0.5:
        score -= 6
        reasons.append("long/short ratio crowded short")

    for zone in liquidation_zones[:10]:
        price = _num(zone.get("price"))
        if math.isnan(price):
            continue
        distance_pct = abs(price - sl) / entry * 100
        if distance_pct < 0.12:
            score -= 15
            reasons.append("SL ligt dicht bij liquidatiecluster")
            break

    score = max(0, min(100, score))
    return {
        "score": score,
        "reasons": reasons or ["setupkwaliteit ok"],
        "block": score < 55,
    }
