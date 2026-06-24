"""
ml_meta.py — combineert ALLE gratis signalen in één ML-model en test op edge.

Features (causaal): TA-indicatoren (RSI/ADX/MACD/BB/ATR/momentum/volatiliteit),
order-flow (taker-buy imbalance + z) en funding (+ z). Een logistische regressie
(numpy) leert de richting van de forward-return. Walk-forward: traint op het
verleden, handelt op de ongeziene toekomst, alleen bij hoge zekerheid. Netto na
kosten, met t-test.

Beantwoordt: geeft de INTERACTIE van gratis signalen edge, waar elk los faalt?

Gebruik:  python tools/ml_meta.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

from src.indicators import add_all_indicators  # noqa: E402
from stats import edge_stats  # noqa: E402

MARKETS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
           "XRP": "XRPUSDT", "DOGE": "DOGEUSDT", "ADA": "ADAUSDT"}
INTERVAL, DAYS = "4h", 365
PER_DAY = 6
COST = 0.35 / 100
H = 3                  # forward-return horizon (bars)
TRAIN, TEST, STEP = 500, 150, 150
CONF = 0.0             # 0 = handel in elke voorspelde richting (meet volledige OOS-edge)
Z_WIN = 30
# Funding bewust niet als feature: die data (8h, beperkte historie) zou het sample
# tot ~60 dagen knippen. Funding had los al geen edge. Hier de volle-historie signalen.
FEATURES = ["rsi14", "adx14", "macd_hist", "bb_pct", "atr_pct",
            "mom3", "vol", "taker", "taker_z"]


def _klines(symbol, total):
    out, end = [], None
    while len(out) < total:
        p = {"symbol": symbol, "interval": INTERVAL, "limit": 1000}
        if end:
            p["endTime"] = end
        d = requests.get("https://api.binance.com/api/v3/klines", params=p, timeout=20,
                         headers={"User-Agent": "TradeAI/1.0"}).json()
        if not d:
            break
        out = d + out
        end = d[0][0] - 1
        if len(d) < 1000:
            break
        time.sleep(0.2)
    return pd.DataFrame([{"timestamp": pd.to_datetime(c[0], unit="ms", utc=True),
                          "open": float(c[1]), "high": float(c[2]), "low": float(c[3]),
                          "close": float(c[4]), "volume": float(c[5]),
                          "taker": (float(c[9]) / float(c[5])) if float(c[5]) > 0 else 0.5}
                         for c in out[-total:]])


def _funding(symbol):
    out, end = [], None
    while len(out) < 1100:
        p = {"symbol": symbol, "limit": 1000}
        if end:
            p["endTime"] = end
        d = requests.get("https://fapi.binance.com/fapi/v1/fundingRate", params=p, timeout=20,
                         headers={"User-Agent": "TradeAI/1.0"}).json()
        if not d:
            break
        out = d + out
        end = d[0]["fundingTime"] - 1
        if len(d) < 1000:
            break
        time.sleep(0.2)
    return pd.DataFrame([{"time": pd.to_datetime(r["fundingTime"], unit="ms", utc=True),
                          "funding": float(r["fundingRate"])} for r in out])


def build(symbol):
    k = _klines(symbol, DAYS * PER_DAY)
    df = add_all_indicators(k)
    df["mom3"] = df["close"].pct_change(3)
    df["vol"] = df["close"].pct_change().rolling(20).std()
    df["taker_z"] = (df["taker"] - df["taker"].rolling(Z_WIN).mean()) / df["taker"].rolling(Z_WIN).std()
    df["fwd_ret"] = df["close"].shift(-H) / df["close"] - 1
    df["y"] = (df["fwd_ret"] > 0).astype(float)
    return df.dropna(subset=FEATURES + ["fwd_ret"]).reset_index(drop=True)


def train_logreg(X, y, iters=600, lr=0.5, l2=0.2):
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        err = p - y
        w -= lr * (X.T @ err / n + l2 * w / n)
        b -= lr * err.mean()
    return w, b


def main():
    print(f"ML-META — alle gratis signalen gecombineerd ({INTERVAL}, {DAYS}d, H={H}, "
          f"walk-forward), netto na kosten, met t-test\n")
    print(f"features: {FEATURES}\n")
    trades = []
    for name, sym in MARKETS.items():
        try:
            df = build(sym)
        except Exception as exc:
            print(f"  {name}: mislukt — {exc}")
            continue
        start = 0
        while start + TRAIN + TEST <= len(df):
            tr = df.iloc[start:start + TRAIN]
            te = df.iloc[start + TRAIN:start + TRAIN + TEST]
            mu, sd = tr[FEATURES].mean(), tr[FEATURES].std().replace(0, 1)
            Xtr = ((tr[FEATURES] - mu) / sd).values
            Xte = ((te[FEATURES] - mu) / sd).values
            w, b = train_logreg(Xtr, tr["y"].values)
            prob = 1 / (1 + np.exp(-(Xte @ w + b)))
            for k in range(len(te)):
                if abs(prob[k] - 0.5) <= CONF:
                    continue
                d = 1 if prob[k] > 0.5 else -1
                trades.append(d * te["fwd_ret"].values[k] - COST)
            start += STEP

    st = edge_stats(trades)
    print(f"  Out-of-sample trades (hoge zekerheid): {st['n']}")
    if st.get("te_weinig_data") or st["n"] < 2:
        print("  te weinig trades voor een oordeel.")
        return
    print(f"  winrate={st['winrate']}%  expectancy={st['expectancy_pct']:+.4f}/trade  "
          f"PF={st['profit_factor']}  Sharpe={st['sharpe_per_trade']}")
    print(f"  t-stat={st['t_stat']}  ->  {st['verdict']}")
    print("\n  >> EDGE!" if st.get("t_stat", 0) > 1.96 else "\n  >> geen edge (consistent met de rest)")


if __name__ == "__main__":
    main()
