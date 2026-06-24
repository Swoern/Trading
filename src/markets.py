"""
markets.py — centrale, configureerbare definitie van de te verhandelen assets.

BTC-USD is het anker van de 'BTC-first' hiërarchie. ETH/SOL en eventuele extra
majors zijn satellieten die alleen handelen als BTC gezond is. De lijst is
instelbaar via de env-variabele TRADEAI_MARKETS (komma-gescheiden), bv:

    TRADEAI_MARKETS="BTC-USD,ETH-USD,SOL-USD"

Nieuwe (niet-core) majors krijgen automatisch voorzichtige satelliet-parameters,
zodat je een munt kunt toevoegen zonder de risico-instellingen handmatig te tunen.
"""
from __future__ import annotations

import os

ANCHOR_ASSET = "BTC-USD"

# Standaard: de drie kernmarkten + een paar zeer liquide majors (gratis data op
# Binance + Coinbase). Bewust géén illiquide small-caps: die verdunnen de
# per-asset learning en hebben hogere spreads/slippage.
DEFAULT_MARKETS = [
    "BTC-USD", "ETH-USD", "SOL-USD",
    "XRP-USD", "DOGE-USD", "ADA-USD",
]

# Core-assets hebben handmatig getunede parameters in auto_trader.ASSET_PARAMS.
CORE_ASSETS = ("BTC-USD", "ETH-USD", "SOL-USD")

# Voorzichtige defaults voor extra majors (gemodelleerd naar SOL: kleinere inzet,
# iets ruimere stop, kortere tijd-stop, hogere drempels).
SATELLITE_ASSET_PARAMS = {
    "min_confidence": 76,
    "risk_multiplier": 0.60,
    "min_backtest_trades": 1,
    "min_regime_score": 60,
    "size_cap": 0.20,
    "sl_multiplier": 1.20,
    "walk_forward_min_trades": 1,
    "walk_forward_min_winrate": 42.0,
    "max_candles_open": 12,
}

# Satellieten gedragen zich als waarnemers onder de BTC-first check.
SATELLITE_MARKT_MODUS = "observe_or_explore"

# Per-strategie cap voor satellieten (klein, zoals SOL).
SATELLITE_STRATEGY_PARAMS = {
    "*": {"size_cap": 0.20, "min_regime_score": 60, "sl_multiplier": 1.20},
    "breakout": {"require_btc_mode": "active"},
    "breakdown": {"require_btc_mode": "active"},
}


def get_markets() -> list[str]:
    """Geef de actieve marktlijst terug (env-override of default). Anker staat voorop."""
    raw = os.environ.get("TRADEAI_MARKETS", "").strip()
    if raw:
        markets = []
        for part in raw.split(","):
            sym = part.strip().upper()
            if sym and sym not in markets:
                markets.append(sym)
    else:
        markets = list(DEFAULT_MARKETS)

    if not markets:
        markets = [ANCHOR_ASSET]
    if ANCHOR_ASSET not in markets:
        markets.insert(0, ANCHOR_ASSET)
    return markets


def is_satellite(asset: str) -> bool:
    return asset not in CORE_ASSETS
