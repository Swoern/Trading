"""
liquidations.py — Coinglass liquidatiedata ophalen (optioneel).

Configuratie: voeg je Coinglass API-key toe aan telegram_config.json:
    {
        "token": "...",
        "chat_id": "...",
        "coinglass_key": "jouw-api-key-hier"
    }

Gratis account aanmaken: https://www.coinglass.com/
Na registratie → API → Free tier geeft toegang tot liquidatiezones.

Zonder API-key werkt de liquidatiestrategie puur op price-action detectie.
"""
import requests
from pathlib import Path
from src.secure_config import load_tradeai_config

_CONFIG_PAD = Path(__file__).parent.parent / "telegram_config.json"

# Coinbase → Binance symbool mapping (Coinglass gebruikt Binance-symbolen)
_SYMBOOL_MAP = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
    "SOL-USD": "SOLUSDT",
}


def _api_key() -> str:
    return load_tradeai_config(_CONFIG_PAD).get("coinglass_key", "").strip()


def haal_liquidatie_zones(asset: str, uren: int = 4) -> list[dict]:
    """
    Haal liquidatiezones op via de Coinglass API.

    Geeft een lijst terug van dicts met:
        {"price": float, "amount": float, "side": "long" | "short"}

    Als er geen API-key is, geeft het een lege lijst terug.
    De liquidatiestrategie valt dan terug op pure price-action detectie.
    """
    sleutel = _api_key()
    if not sleutel:
        return []

    symbool = _SYMBOOL_MAP.get(asset, "")
    if not symbool:
        return []

    try:
        r = requests.get(
            "https://open-api.coinglass.com/public/v2/liquidation_map",
            params={
                "symbol":     symbool,
                "timeType":   0,           # 0 = afgelopen N uur
                "timeLength": min(uren, 24),
            },
            headers={"coinglassSecret": sleutel},
            timeout=10,
        )
        if not r.ok:
            return []

        data = r.json().get("data", {})
        zones = []

        # Long liquidaties (prijs daalt naar deze niveaus)
        for item in data.get("longLiquidationMap", []):
            zones.append({
                "price":  float(item.get("price", 0)),
                "amount": float(item.get("amount", 0)),
                "side":   "long",
            })

        # Short liquidaties (prijs stijgt naar deze niveaus)
        for item in data.get("shortLiquidationMap", []):
            zones.append({
                "price":  float(item.get("price", 0)),
                "amount": float(item.get("amount", 0)),
                "side":   "short",
            })

        # Sorteer op grootte (grootste zones zijn het meest relevant)
        return sorted(zones, key=lambda z: z["amount"], reverse=True)[:20]

    except Exception:
        return []


def heeft_coinglass_key() -> bool:
    """Geeft True als er een Coinglass API-key geconfigureerd is."""
    return bool(_api_key())
