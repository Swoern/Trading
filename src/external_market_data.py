"""
external_market_data.py - Optionele externe marktcontext.

Alle functies falen zacht: als een API-key ontbreekt of een endpoint niet werkt,
geeft de bot neutrale waarden terug en handelt hij niet blind op incomplete data.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime, timezone

import requests

from src.secure_config import load_tradeai_config


_CACHE: dict[str, tuple[float, dict]] = {}
_SYMBOL_MAP = {
    "BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT", "SOL-USD": "SOLUSDT",
    "XRP-USD": "XRPUSDT", "DOGE-USD": "DOGEUSDT", "ADA-USD": "ADAUSDT",
}
_COINGLASS_BASE = "https://open-api-v4.coinglass.com"


def _cached(key: str, max_age: int = 300) -> dict | None:
    item = _CACHE.get(key)
    if not item:
        return None
    ts, value = item
    if time.time() - ts <= max_age:
        return value
    return None


def _store(key: str, value: dict) -> dict:
    _CACHE[key] = (time.time(), value)
    return value


def _coinglass_get(path: str, params: dict, max_age: int = 300) -> dict:
    cfg = load_tradeai_config()
    api_key = cfg.get("coinglass_key", "").strip()
    if not api_key:
        return {"ok": False, "reason": "coinglass_key ontbreekt"}

    cache_key = f"coinglass:{path}:{sorted(params.items())}"
    cached = _cached(cache_key, max_age=max_age)
    if cached is not None:
        return cached

    headers = {"CG-API-KEY": api_key, "accept": "application/json"}
    try:
        response = requests.get(
            f"{_COINGLASS_BASE}{path}",
            params=params,
            headers=headers,
            timeout=10,
        )
        if response.status_code in (401, 403):
            # Oudere Coinglass endpoints gebruiken soms coinglassSecret.
            response = requests.get(
                f"{_COINGLASS_BASE}{path}",
                params=params,
                headers={"coinglassSecret": api_key, "accept": "application/json"},
                timeout=10,
            )
        response.raise_for_status()
        payload = response.json()
        return _store(cache_key, {"ok": True, "data": payload.get("data", payload)})
    except Exception as exc:
        return _store(cache_key, {"ok": False, "reason": str(exc)})


def fetch_orderbook_spread(asset: str) -> dict:
    """Haal beste bid/ask en spread op via Coinbase public orderbook."""
    cache_key = f"spread:{asset}"
    cached = _cached(cache_key, max_age=20)
    if cached is not None:
        return cached
    try:
        response = requests.get(
            f"https://api.exchange.coinbase.com/products/{asset}/book",
            params={"level": 1},
            headers={"User-Agent": "TradeAI-Coach/1.0"},
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()
        bids = data.get("bids") or []
        asks = data.get("asks") or []
        if not bids or not asks:
            return _store(cache_key, {"ok": False, "spread_pct": 0.0, "reason": "lege orderbook"})
        bid = float(bids[0][0])
        ask = float(asks[0][0])
        mid = (bid + ask) / 2
        spread_pct = (ask - bid) / mid * 100 if mid > 0 else 0.0
        return _store(cache_key, {
            "ok": True,
            "bid": bid,
            "ask": ask,
            "spread_pct": round(spread_pct, 4),
        })
    except Exception as exc:
        return _store(cache_key, {"ok": False, "spread_pct": 0.0, "reason": str(exc)})


def fetch_orderbook_depth(asset: str, levels: int = 25) -> dict:
    """Meet orderbook-druk via Coinbase level 2: bid/ask depth en imbalance."""
    cache_key = f"depth:{asset}:{levels}"
    cached = _cached(cache_key, max_age=20)
    if cached is not None:
        return cached
    try:
        response = requests.get(
            f"https://api.exchange.coinbase.com/products/{asset}/book",
            params={"level": 2},
            headers={"User-Agent": "TradeAI-Coach/1.0"},
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()
        bids = data.get("bids", [])[:levels]
        asks = data.get("asks", [])[:levels]
        bid_depth = sum(float(price) * float(size) for price, size, *_ in bids)
        ask_depth = sum(float(price) * float(size) for price, size, *_ in asks)
        total = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / total if total > 0 else 0.0
        pressure = "buy" if imbalance > 0.12 else ("sell" if imbalance < -0.12 else "neutral")
        return _store(cache_key, {
            "ok": True,
            "levels": levels,
            "bid_depth": round(bid_depth, 2),
            "ask_depth": round(ask_depth, 2),
            "imbalance": round(imbalance, 4),
            "pressure": pressure,
        })
    except Exception as exc:
        return _store(cache_key, {"ok": False, "imbalance": 0.0, "pressure": "neutral", "reason": str(exc)})


def fetch_btc_dominance() -> dict:
    """Gratis CoinGecko global endpoint voor BTC dominance / totale markttrend."""
    cached = _cached("coingecko:global", max_age=1800)
    if cached is not None:
        return cached
    try:
        response = requests.get("https://api.coingecko.com/api/v3/global", timeout=10)
        response.raise_for_status()
        data = response.json().get("data", {})
        dominance = float(data.get("market_cap_percentage", {}).get("btc", 0.0))
        change = float(data.get("market_cap_change_percentage_24h_usd", 0.0))
        return _store("coingecko:global", {
            "ok": True,
            "btc_dominance": round(dominance, 3),
            "market_cap_change_24h": round(change, 3),
        })
    except Exception as exc:
        return _store("coingecko:global", {"ok": False, "reason": str(exc)})


def fetch_coinglass_derivatives(asset: str) -> dict:
    """
    Haal funding, open interest en long/short context op als Coinglass v4-key werkt.
    Endpointnamen volgen de actuele v4 documentatie; als het account geen toegang
    heeft, blijft de uitkomst neutraal.
    """
    symbol = _SYMBOL_MAP.get(asset)
    if not symbol:
        return {"ok": False, "reason": "asset niet ondersteund"}

    funding = _coinglass_get(
        "/api/futures/funding-rate/history",
        {"exchange": "Binance", "symbol": symbol, "interval": "1h", "limit": 2},
        max_age=900,
    )
    oi = _coinglass_get(
        "/api/futures/open-interest/history",
        {"exchange": "Binance", "symbol": symbol, "interval": "1h", "limit": 2},
        max_age=900,
    )
    long_short = _coinglass_get(
        "/api/futures/global-long-short-account-ratio/history",
        {"symbol": symbol, "interval": "1h", "limit": 2},
        max_age=900,
    )

    result = {
        "ok": any(item.get("ok") for item in (funding, oi, long_short)),
        "funding_rate": None,
        "open_interest_change": None,
        "long_short_ratio": None,
        "warnings": [],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    for name, payload in (("funding", funding), ("open_interest", oi), ("long_short", long_short)):
        if not payload.get("ok"):
            result["warnings"].append(f"{name}: {payload.get('reason', 'niet beschikbaar')}")

    def _rows(payload: dict) -> list:
        data = payload.get("data")
        if isinstance(data, dict):
            rows = data.get("list") or data.get("data") or data.get("items")
        else:
            rows = data
        return rows if isinstance(rows, list) else []

    def nth_number(payload: dict, keys: tuple[str, ...], index: int = -1) -> float | None:
        rows = _rows(payload)
        if not rows or abs(index) > len(rows):
            return None
        row = rows[index]
        if not isinstance(row, dict):
            return None
        for key in keys:
            if key in row:
                try:
                    return float(row[key])
                except Exception:
                    pass
        return None

    _OI_KEYS = ("openInterest", "open_interest", "oi", "close")
    result["funding_rate"] = nth_number(funding, ("fundingRate", "funding_rate", "rate", "close"))
    oi_now = nth_number(oi, _OI_KEYS, index=-1)
    oi_prev = nth_number(oi, _OI_KEYS, index=-2)
    if oi_now is not None and oi_prev:
        result["open_interest_change"] = round((oi_now - oi_prev) / oi_prev * 100, 3)
    result["long_short_ratio"] = nth_number(long_short, ("longShortRatio", "long_short_ratio", "ratio"))
    return result


def fetch_market_context(asset: str) -> dict:
    def timed(fn, default: dict, timeout: float = 4.0) -> dict:
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(fn)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            return default | {"ok": False, "reason": "timeout"}
        except Exception as exc:
            return default | {"ok": False, "reason": str(exc)}
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    return {
        "spread": timed(lambda: fetch_orderbook_spread(asset), {"spread_pct": 0.0}),
        "orderbook": timed(lambda: fetch_orderbook_depth(asset), {"imbalance": 0.0, "pressure": "neutral"}),
        "dominance": timed(fetch_btc_dominance, {}),
        "derivatives": timed(lambda: fetch_coinglass_derivatives(asset), {}),
    }
