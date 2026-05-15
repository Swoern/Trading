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


def fetch_live_crypto_candles(
    product_id: str = "BTC-USD",
    granularity: int = 300,
    limit: int = 300,
) -> pd.DataFrame:
    """
    Haal actuele candles op via de publieke Coinbase Exchange API.

    Dit gebruikt geen API-key en plaatst geen orders. Coinbase accepteert maximaal
    300 candles per request, daarom begrenzen we de limit bewust.
    """
    allowed_granularities = {60, 300, 900, 3600, 21600, 86400}
    if granularity not in allowed_granularities:
        raise ValueError("Ongeldige timeframe voor Coinbase candles.")

    limit = max(55, min(int(limit), 300))
    end = datetime.now(timezone.utc)
    start = end - timedelta(seconds=granularity * limit)

    response = requests.get(
        f"https://api.exchange.coinbase.com/products/{product_id}/candles",
        params={
            "granularity": granularity,
            "start": start.isoformat(),
            "end": end.isoformat(),
        },
        headers={"User-Agent": "TradeAI-Coach/1.0"},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()

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
