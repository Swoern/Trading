"""
orderflow_signal.py — test of ORDER-FLOW (taker-buy imbalance) edge geeft.

De live orderbook-snapshot is niet gratis historisch beschikbaar. Maar Binance'
candles bevatten taker-buy-volume (agressieve market-buys) — een echte order-flow
maat die gratis én historisch is. We testen of extreme imbalance momentum- of
mean-reversion edge geeft, ná kosten, met significantie.

Gebruik:  python tools/orderflow_signal.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import requests  # noqa: E402

from stats import edge_stats  # noqa: E402

MARKETS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
           "XRP": "XRPUSDT", "DOGE": "DOGEUSDT", "ADA": "ADAUSDT"}
COST = 0.35 / 100
Z_WIN = 30
Z_THRESH = 1.0
CONFIGS = [("1h", 180), ("4h", 365)]
PER_DAY = {"1h": 24, "4h": 6}
HOLDS = [1, 3, 6]


def fetch_orderflow(symbol, interval, total):
    out, end = [], None
    while len(out) < total:
        p = {"symbol": symbol, "interval": interval, "limit": 1000}
        if end:
            p["endTime"] = end
        r = requests.get("https://api.binance.com/api/v3/klines", params=p, timeout=20,
                         headers={"User-Agent": "TradeAI/1.0"})
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        out = data + out
        end = data[0][0] - 1
        if len(data) < 1000:
            break
        time.sleep(0.2)
    rows = []
    for c in out[-total:]:
        vol = float(c[5])
        taker_buy = float(c[9])           # veld 9 = taker buy base volume
        rows.append({"time": pd.to_datetime(c[0], unit="ms", utc=True),
                     "close": float(c[4]),
                     "of_ratio": (taker_buy / vol) if vol > 0 else 0.5})
    df = pd.DataFrame(rows)
    df["z"] = (df["of_ratio"] - df["of_ratio"].rolling(Z_WIN).mean()) / df["of_ratio"].rolling(Z_WIN).std()
    return df


def collect(df, mode, hold):
    rets = []
    for i in range(Z_WIN, len(df) - hold):
        z = df["z"].iloc[i]
        if pd.isna(z) or abs(z) < Z_THRESH:
            continue
        # momentum: koop-druk -> long. reversion: koop-druk -> short.
        sign = 1 if z > 0 else -1
        richting = sign if mode == "momentum" else -sign
        entry, exit_ = df["close"].iloc[i], df["close"].iloc[i + hold]
        rets.append(richting * (exit_ - entry) / entry - COST)
    return rets


def main():
    print("ORDER-FLOW SIGNAAL — taker-buy imbalance (|z|>1.0), netto na 0,35% kosten, met t-test\n")
    for interval, days in CONFIGS:
        total = days * PER_DAY[interval]
        data = {}
        for name, sym in MARKETS.items():
            try:
                data[name] = fetch_orderflow(sym, interval, total)
            except Exception as exc:
                print(f"  {name}: mislukt — {exc}")
        print(f"=== {interval} ({days}d) ===")
        for mode in ("momentum", "reversion"):
            for hold in HOLDS:
                rets = []
                for df in data.values():
                    rets += collect(df, mode, hold)
                st = edge_stats(rets)
                vlag = "  <<< EDGE" if st.get("t_stat", 0) > 1.96 else ""
                print(f"  {mode:<10} hold {hold*PER_DAY[interval]//PER_DAY[interval]}c "
                      f"({hold}): n={st['n']:>4} WR={st['winrate']:>4}% "
                      f"exp={st['expectancy_pct']:+.3f}% t={st['t_stat']:+.2f} {st['verdict']}{vlag}")
        print()


if __name__ == "__main__":
    main()
