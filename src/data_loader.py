"""
data_loader.py — CSV-data laden en sample data aanmaken.

Verwacht formaat CSV:
    timestamp, open, high, low, close, volume

De timestamp mag ook 'date', 'datetime' of 'Date' heten.
"""
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from datetime import datetime, timedelta, timezone

from src.reliability import http_get_json


def load_csv(filepath: str) -> pd.DataFrame:
    """
    Laad OHLCV-data vanuit een CSV-bestand.
    Geeft een pandas DataFrame terug, gesorteerd op datum.
    """
    df = pd.read_csv(filepath)

    # Kolommen naar kleine letters
    df.columns = df.columns.str.strip().str.lower()

    # Zoek de timestamp-kolom
    timestamp_kandidaten = ["timestamp", "date", "datetime", "time"]
    ts_col = next((c for c in timestamp_kandidaten if c in df.columns), None)

    if ts_col is None:
        raise ValueError(
            "Geen timestamp-kolom gevonden. Zorg dat je CSV een kolom heeft "
            "met de naam: timestamp, date of datetime."
        )

    df["timestamp"] = pd.to_datetime(df[ts_col])
    if ts_col != "timestamp":
        df = df.drop(columns=[ts_col])

    # Controleer verplichte kolommen
    verplicht = ["open", "high", "low", "close", "volume"]
    ontbreekt = [c for c in verplicht if c not in df.columns]
    if ontbreekt:
        raise ValueError(
            f"Ontbrekende kolommen: {ontbreekt}. "
            f"Je CSV moet de volgende kolommen bevatten: {verplicht}"
        )

    # Zorg voor juiste types
    for col in verplicht:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=verplicht)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def generate_sample_data(filepath: str = "data/sample_btc.csv",
                          n_days: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Genereer nep BTC-achtige OHLCV-data voor het testen van het systeem.
    Dit is GEEN echte marktdata — alleen voor leren en oefenen.
    """
    np.random.seed(seed)

    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n_days)]

    # Random walk met kleine opwaartse drift
    prijs = 25000.0
    closes = [prijs]
    for _ in range(n_days - 1):
        # Voeg een lichte trend toe met een sinus-golf (voor realisme)
        dag = len(closes)
        trend_factor = 0.0005 + 0.001 * np.sin(dag / 60)
        dagelijkse_verandering = np.random.normal(trend_factor, 0.022)
        prijs = max(prijs * (1 + dagelijkse_verandering), 1000)
        closes.append(prijs)

    rows = []
    for datum, close in zip(dates, closes):
        ruis = close * 0.008
        open_ = close * (1 + np.random.uniform(-0.006, 0.006))
        high = max(open_, close) + abs(np.random.normal(0, ruis))
        low = min(open_, close) - abs(np.random.normal(0, ruis))
        volume = np.random.uniform(500, 8000) * (
            1 + abs(close - open_) / close * 8
        )
        rows.append({
            "timestamp": datum.strftime("%Y-%m-%d"),
            "open":   round(open_, 2),
            "high":   round(high, 2),
            "low":    round(low, 2),
            "close":  round(close, 2),
            "volume": round(volume, 2),
        })

    df = pd.DataFrame(rows)
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    return df


def get_available_csv_files(data_dir: str = "data") -> list[str]:
    """Geef een lijst van alle CSV-bestanden in de data-map."""
    pad = Path(data_dir)
    if not pad.exists():
        return []
    return sorted(str(f) for f in pad.glob("*.csv"))


_fg_cache: dict = {"value": None, "label": None, "ts": 0.0}


def fetch_fear_greed(max_age_seconds: int = 3600) -> dict:
    """
    Haal de crypto Fear & Greed Index op via de gratis alternative.me API.
    Resultaat wordt 1 uur gecached om API-verzoeken te beperken.

    Returns: {"value": int (0-100), "label": str}
      0-24  : Extreme Fear   — markt is overpanisch, mogelijke bodems
      25-44 : Fear
      45-55 : Neutral
      56-74 : Greed
      75-100: Extreme Greed  — markt is overerhit, mogelijke toppen
    """
    import time
    nu = time.time()
    if _fg_cache["value"] is not None and nu - _fg_cache["ts"] < max_age_seconds:
        return {"value": _fg_cache["value"], "label": _fg_cache["label"]}
    try:
        payload = http_get_json(
            "https://api.alternative.me/fng/?limit=1",
            timeout=8,
            headers={"User-Agent": "TradeAI-Coach/1.0"},
            retries=2,
        )
        data = payload["data"][0]
        _fg_cache["value"] = int(data["value"])
        _fg_cache["label"] = data["value_classification"]
        _fg_cache["ts"]    = nu
    except Exception:
        if _fg_cache["value"] is None:
            return {"value": 50, "label": "Neutral"}
    return {"value": _fg_cache["value"], "label": _fg_cache["label"]}


_BINANCE_INTERVAL = {60: "1m", 300: "5m", 900: "15m", 3600: "1h", 21600: "6h", 86400: "1d"}


def _coinbase_id_to_binance(product_id: str) -> str:
    """Convert 'BTC-USD' → 'BTCUSDT', 'ETH-BTC' stays 'ETHBTC'."""
    base, quote = product_id.upper().split("-")
    if quote == "USD":
        quote = "USDT"
    return base + quote


def _fetch_from_coinbase(product_id: str, granularity: int, limit: int) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(seconds=granularity * limit)
    data = http_get_json(
        f"https://api.exchange.coinbase.com/products/{product_id}/candles",
        params={"granularity": granularity, "start": start.isoformat(), "end": end.isoformat()},
        headers={"User-Agent": "TradeAI-Coach/1.0"},
        timeout=15,
    )
    if not data:
        raise ValueError(f"Geen live candles ontvangen voor {product_id}.")
    rows = []
    for candle in data:
        if len(candle) < 5:
            continue
        rows.append({
            "timestamp": pd.to_datetime(candle[0], unit="s", utc=True),
            "low": float(candle[1]),
            "high": float(candle[2]),
            "open": float(candle[3]),
            "close": float(candle[4]),
            "volume": float(candle[5]) if len(candle) > 5 else 0.0,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"Live data voor {product_id} kon niet worden verwerkt.")
    return df.sort_values("timestamp").reset_index(drop=True)


def _fetch_from_binance(product_id: str, granularity: int, limit: int) -> pd.DataFrame:
    interval = _BINANCE_INTERVAL.get(granularity)
    if interval is None:
        raise ValueError("Ongeldige timeframe voor Binance candles.")
    symbol = _coinbase_id_to_binance(product_id)
    data = http_get_json(
        "https://api.binance.com/api/v3/klines",
        params={"symbol": symbol, "interval": interval, "limit": min(limit, 1000)},
        headers={"User-Agent": "TradeAI-Coach/1.0"},
        timeout=15,
    )
    if not data:
        raise ValueError(f"Geen live candles ontvangen voor {symbol} via Binance.")
    rows = []
    for candle in data:
        rows.append({
            "timestamp": pd.to_datetime(candle[0], unit="ms", utc=True),
            "open": float(candle[1]),
            "high": float(candle[2]),
            "low": float(candle[3]),
            "close": float(candle[4]),
            "volume": float(candle[5]),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"Live data voor {symbol} kon niet worden verwerkt.")
    return df.sort_values("timestamp").reset_index(drop=True)


def _generate_synthetic_candles(
    product_id: str,
    granularity: int,
    limit: int,
) -> pd.DataFrame:
    """Genereer synthetische candles als alle externe API's onbereikbaar zijn."""
    seed = abs(hash(product_id)) % (2**31)
    rng = np.random.default_rng(seed)

    base_prices = {
        "BTC": 65000.0, "ETH": 3000.0, "SOL": 150.0,
        "XRP": 0.55, "BNB": 580.0, "ADA": 0.45, "DOGE": 0.16,
    }
    base = base_prices.get(product_id.split("-")[0].upper(), 100.0)

    end = datetime.now(timezone.utc)
    timestamps = [end - timedelta(seconds=granularity * (limit - i)) for i in range(limit)]

    price = base
    rows = []
    for ts in timestamps:
        change = rng.normal(0.0, 0.012)
        close = max(price * (1 + change), base * 0.1)
        open_ = price
        high = max(open_, close) * (1 + abs(rng.normal(0, 0.004)))
        low  = min(open_, close) * (1 - abs(rng.normal(0, 0.004)))
        volume = rng.uniform(200, 2000) * (1 + abs(change) * 20)
        rows.append({
            "timestamp": pd.Timestamp(ts),
            "open": round(open_, 4),
            "high": round(high, 4),
            "low": round(low, 4),
            "close": round(close, 4),
            "volume": round(volume, 2),
        })
        price = close

    df = pd.DataFrame(rows)
    df["_simulated"] = True
    return df.sort_values("timestamp").reset_index(drop=True)


def fetch_live_crypto_candles(
    product_id: str = "BTC-USD",
    granularity: int = 300,
    limit: int = 300,
) -> pd.DataFrame:
    """
    Haal actuele candles op. Probeert Binance, dan Coinbase, dan synthetische data
    als beide onbereikbaar zijn (bijv. geen internetverbinding op de server).
    """
    allowed_granularities = {60, 300, 900, 3600, 21600, 86400}
    if granularity not in allowed_granularities:
        raise ValueError("Ongeldige timeframe voor candles.")

    limit = max(55, min(int(limit), 300))

    errors = []

    try:
        return _fetch_from_binance(product_id, granularity, limit)
    except Exception as e:
        errors.append(f"Binance: {e}")

    try:
        return _fetch_from_coinbase(product_id, granularity, limit)
    except Exception as e:
        errors.append(f"Coinbase: {e}")

    df = _generate_synthetic_candles(product_id, granularity, limit)
    df["_simulated"] = True
    df["_errors"] = "; ".join(errors)
    return df
