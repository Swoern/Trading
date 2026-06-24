"""
funding_signal.py — test of EXTREME funding-rates mean-reversion edge geven.

Gratis via Binance futures (geen key). Hypothese: zeer hoge funding = overvolle
longs -> short-kans; zeer lage/negatieve funding = overvolle shorts -> long-kans.
Meet de netto forward-return ná kosten op extreme-funding events.

Gebruik:  python tools/funding_signal.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import requests  # noqa: E402

MARKETS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
           "XRP": "XRPUSDT", "DOGE": "DOGEUSDT", "ADA": "ADAUSDT"}
COST = 0.35 / 100          # round-trip kosten
Z_WIN = 30                 # rolling window voor z-score van funding
Z_THRESH = 1.5             # 'extreem' = |z| > 1.5
HOLDS = [1, 3, 6]          # hold in 8h-perioden (8h, 24h, 48h)


def _get(url, params):
    r = requests.get(url, params=params, timeout=20, headers={"User-Agent": "TradeAI/1.0"})
    r.raise_for_status()
    return r.json()


def fetch_funding(symbol, total=1100):
    out, end = [], None
    while len(out) < total:
        p = {"symbol": symbol, "limit": 1000}
        if end:
            p["endTime"] = end
        data = _get("https://fapi.binance.com/fapi/v1/fundingRate", p)
        if not data:
            break
        out = data + out
        end = data[0]["fundingTime"] - 1
        if len(data) < 1000:
            break
        time.sleep(0.2)
    df = pd.DataFrame(out[-total:])
    df["time"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding"] = df["fundingRate"].astype(float)
    return df[["time", "funding"]]


def fetch_8h_closes(symbol, total=1100):
    out, end = [], None
    while len(out) < total:
        p = {"symbol": symbol, "interval": "8h", "limit": 1000}
        if end:
            p["endTime"] = end
        data = _get("https://fapi.binance.com/fapi/v1/klines", p)
        if not data:
            break
        out = data + out
        end = data[0][0] - 1
        if len(data) < 1000:
            break
        time.sleep(0.2)
    df = pd.DataFrame([{"time": pd.to_datetime(c[0], unit="ms", utc=True), "close": float(c[4])}
                       for c in out[-total:]])
    return df


def test_market(name, symbol) -> dict:
    fund = fetch_funding(symbol)
    price = fetch_8h_closes(symbol)
    df = pd.merge_asof(fund.sort_values("time"), price.sort_values("time"),
                       on="time", direction="nearest", tolerance=pd.Timedelta("4h")).dropna()
    df = df.reset_index(drop=True)
    df["z"] = (df["funding"] - df["funding"].rolling(Z_WIN).mean()) / df["funding"].rolling(Z_WIN).std()

    results = {h: [] for h in HOLDS}
    for i in range(Z_WIN, len(df)):
        z = df["z"].iloc[i]
        if pd.isna(z) or abs(z) < Z_THRESH:
            continue
        richting = -1 if z > 0 else 1  # hoge funding -> short (-1)
        for h in HOLDS:
            if i + h >= len(df):
                continue
            entry, exit_ = df["close"].iloc[i], df["close"].iloc[i + h]
            ret = richting * (exit_ - entry) / entry - COST
            results[h].append(ret)
    return {name: results}


def main() -> None:
    print("FUNDING-SIGNAAL — mean-reversion op extreme funding (|z|>1.5), netto na 0,35% kosten")
    print(f"hold in 8h-perioden: {HOLDS}\n")
    totaal = {h: [] for h in HOLDS}
    for name, sym in MARKETS.items():
        try:
            res = test_market(name, sym)[name]
        except Exception as exc:
            print(f"  {name}: mislukt — {exc}")
            continue
        line = f"  {name:<5}"
        for h in HOLDS:
            r = res[h]
            totaal[h] += r
            if r:
                wr = 100 * sum(1 for x in r if x > 0) / len(r)
                line += f"  h{h*8}h: n={len(r):>3} WR={wr:>3.0f}% gem={sum(r)/len(r)*100:+.3f}%"
        print(line)

    print(f"\n  {'TOTAAL':<5}")
    for h in HOLDS:
        r = totaal[h]
        if r:
            wr = 100 * sum(1 for x in r if x > 0) / len(r)
            avg = sum(r) / len(r) * 100
            vlag = "  <-- EDGE" if avg > 0 else ""
            print(f"     hold {h*8}h: n={len(r)}  WR={wr:.0f}%  netto gem%/trade={avg:+.3f}%{vlag}")


if __name__ == "__main__":
    main()
