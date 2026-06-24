"""
backtest_report.py — eerlijke edge-meting per markt, strategie EN timeframe.

Haalt historie van Binance, draait run_backtest met de live-kosten/parameters, en
rapporteert per timeframe of er ná kosten edge is. Test de RAUWE strategieën
(zonder de live-gates), zodat je ziet of het signaal zelf waarde heeft.

Gebruik:  python tools/backtest_report.py
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests

from src.backtest import run_backtest
from src.trading_costs import DEFAULT_FEE_RATE, DEFAULT_SLIPPAGE_RATE, DEFAULT_SPREAD_RATE

MARKETS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD"]

# (interval, dagen historie) — hogere timeframes hebben betere signaal/kosten-ratio.
CONFIGS = [
    ("15m", 90),
    ("1h", 180),
    ("4h", 365),
]
_PER_DAY = {"5m": 288, "15m": 96, "1h": 24, "4h": 6}

MIN_RR = 2.0
RISICO_PCT = 1.0
MAX_CANDLES_OPEN = 18
INITIAL = 10_000.0
ROUND_TRIP = (DEFAULT_FEE_RATE + DEFAULT_SPREAD_RATE / 2 + DEFAULT_SLIPPAGE_RATE) * 2 * 100


def _binance_symbol(product_id: str) -> str:
    base, quote = product_id.split("-")
    return base + ("USDT" if quote == "USD" else quote)


def fetch_history(product_id: str, interval: str, total: int) -> pd.DataFrame:
    symbol = _binance_symbol(product_id)
    out: list = []
    end = None
    while len(out) < total:
        params = {"symbol": symbol, "interval": interval, "limit": 1000}
        if end:
            params["endTime"] = end
        r = requests.get(
            "https://api.binance.com/api/v3/klines",
            params=params, timeout=20, headers={"User-Agent": "TradeAI-Backtest/1.0"},
        )
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        out = data + out
        end = data[0][0] - 1
        if len(data) < 1000:
            break
        time.sleep(0.2)
    rows = [
        {
            "timestamp": pd.to_datetime(c[0], unit="ms", utc=True),
            "open": float(c[1]), "high": float(c[2]), "low": float(c[3]),
            "close": float(c[4]), "volume": float(c[5]),
        }
        for c in out[-total:]
    ]
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def _expectancy_r(winrate_frac: float) -> float:
    """Verwachting per trade in R bij 1:2 (positief = edge, zelfs zonder kosten)."""
    return winrate_frac * 2.0 - (1 - winrate_frac) * 1.0


def run_config(interval: str, days: int) -> dict:
    total = days * _PER_DAY[interval]
    print(f"\n{'='*70}\n### TIMEFRAME {interval}  ({days} dagen, ~{total} candles/markt)\n{'='*70}")
    combined: dict = defaultdict(lambda: {"n": 0, "w": 0, "pct": 0.0, "eur": 0.0})
    market_lines = []
    for market in MARKETS:
        try:
            df = fetch_history(market, interval, total)
            res = run_backtest(
                df, min_rr=MIN_RR, initial_capital=INITIAL, risico_per_trade_pct=RISICO_PCT,
                max_candles_open=MAX_CANDLES_OPEN, fee_rate=DEFAULT_FEE_RATE,
                spread_rate=DEFAULT_SPREAD_RATE, slippage_rate=DEFAULT_SLIPPAGE_RATE,
            )
        except Exception as exc:
            market_lines.append(f"  {market:<9} data/backtest mislukt: {exc}")
            continue
        rendement = (res["eind_kapitaal"] - INITIAL) / INITIAL * 100
        market_lines.append(
            f"  {market:<9} trades={res['total_trades']:>4}  "
            f"winrate={res['winrate']:>3.0f}%  netto={rendement:>+7.1f}%"
        )
        for t in res["trades"]:
            b = combined[t["strategie"]]
            b["n"] += 1
            b["w"] += 1 if t["win"] else 0
            b["pct"] += t["result_pct"]
            b["eur"] += t["result_euro"]

    print("\n".join(market_lines))
    print(f"\n  per strategie (alle markten samen, {interval}):")
    print(f"    {'strategie':<16}{'n':>5}{'winrate':>9}{'gem%':>8}{'EV(R)':>8}{'€':>11}")
    for s in sorted(combined, key=lambda k: combined[k]["eur"], reverse=True):
        d = combined[s]
        wr = d["w"] / d["n"] if d["n"] else 0
        gem = d["pct"] / d["n"] if d["n"] else 0
        print(f"    {s:<16}{d['n']:>5}{100*wr:>8.0f}%{gem:>8.2f}{_expectancy_r(wr):>8.2f}{d['eur']:>11.0f}")

    n = sum(d["n"] for d in combined.values())
    w = sum(d["w"] for d in combined.values())
    eur = sum(d["eur"] for d in combined.values())
    wr = w / n if n else 0
    print(f"\n  >> {interval} TOTAAL: {n} trades  winrate={100*wr:.0f}%  "
          f"EV={_expectancy_r(wr):+.2f}R/trade  netto=€{eur:,.0f}")
    return {"interval": interval, "trades": n, "winrate": wr, "eur": eur}


def main() -> None:
    print(f"EDGE-RAPPORT — min_rr={MIN_RR}, kosten {ROUND_TRIP:.2f}% round-trip, "
          f"rauwe strategieën (geen live-gates)")
    print("EV(R) = verwachting per trade in R bij 1:2; >0 = edge zelfs zonder kosten.")
    samenvatting = []
    for interval, days in CONFIGS:
        samenvatting.append(run_config(interval, days))

    print(f"\n{'='*70}\nSAMENVATTING per timeframe\n{'='*70}")
    print(f"  {'tf':<6}{'trades':>8}{'winrate':>9}{'netto €':>12}")
    for s in samenvatting:
        print(f"  {s['interval']:<6}{s['trades']:>8}{100*s['winrate']:>8.0f}%{s['eur']:>12,.0f}")


if __name__ == "__main__":
    main()
