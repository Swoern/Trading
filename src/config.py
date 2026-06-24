"""
config.py — centrale, env-overschrijfbare afstel-knoppen voor de bot.

Alle defaults zijn gelijk aan het huidige gedrag, dus zonder env-variabelen
verandert er niets. Eén plek om de bot bij te stellen i.p.v. constanten verspreid
door auto_trader.py.

Voorbeelden:
    export TRADEAI_MIN_RR=1.5
    export TRADEAI_STRATEGIES="trend_long,trend_short,macd_bear_div"   # alleen deze
    export TRADEAI_CANDLE_SECONDS=300
"""
from __future__ import annotations

import os


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _i(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return int(default)


# ── Discipline / risico ─────────────────────────────────────────────────────────
RISICO_PCT = _f("TRADEAI_RISICO_PCT", 0.01)            # 1% risico per trade
MAX_DAGVERLIES = _f("TRADEAI_MAX_DAGVERLIES", 0.20)    # harde kill-switch
SOFT_DAGVERLIES = _f("TRADEAI_SOFT_DAGVERLIES", 0.15)  # zachte rem
MAX_OPEN_TRADES = _i("TRADEAI_MAX_OPEN_TRADES", 5)
CANDLE_SECONDS = _i("TRADEAI_CANDLE_SECONDS", 300)     # 300 = 5-min candles
MIN_RR_START = _f("TRADEAI_MIN_RR", 2.0)
MIN_CONFIDENCE_SCORE = _i("TRADEAI_MIN_CONFIDENCE", 75)

# ── Strategie-whitelist ─────────────────────────────────────────────────────────
ALL_STRATEGIES = (
    "trend_long", "pullback", "breakout", "trend_short", "pullback_short",
    "breakdown", "macd_bull_div", "macd_bear_div", "range_long", "range_short",
)


def get_enabled_strategies() -> frozenset:
    """Welke strategieën actief zijn (env TRADEAI_STRATEGIES, default = alle)."""
    raw = os.environ.get("TRADEAI_STRATEGIES", "").strip()
    if not raw:
        return frozenset(ALL_STRATEGIES)
    gekozen = frozenset(s.strip() for s in raw.split(",") if s.strip())
    # Alleen geldige namen; lege selectie valt terug op alle (veilig).
    geldig = gekozen & frozenset(ALL_STRATEGIES)
    return geldig or frozenset(ALL_STRATEGIES)


ENABLED_STRATEGIES = get_enabled_strategies()
