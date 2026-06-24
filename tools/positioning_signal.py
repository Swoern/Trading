"""
positioning_signal.py — test long/short-ratio en open-interest als signaal (gratis).

Binance gratis futures-data (geen key): global long/short account ratio en open
interest historie. Hypotheses: extreme long/short = contrair (massa zit fout);
OI-verandering = momentum/squeeze. Forward-return ná kosten, met t-test.

LET OP: deze endpoints geven maar ~30 dagen historie → klein sample, één regime.

Gebruik:  python tools/positioning_signal.py
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
PERIOD = "1h"
LIMIT = 500
Z_WIN = 30
Z_THRESH = 1.0
HOLDS = [1, 3, 6]
FAPI = "https://fapi.binance.com"


def _get(path, params):
    r = requests.get(FAPI + path, params=params, timeout=20, headers={"User-Agent": "TradeAI/1.0"})
    r.raise_for_status()
    return r.json()


def fetch_signals(symbol) -> pd.DataFrame:
    ls = _get("/futures/data/globalLongShortAccountRatio",
              {"symbol": symbol, "period": PERIOD, "limit": LIMIT})
    oi = _get("/futures/data/openInterestHist",
              {"symbol": symbol, "period": PERIOD, "limit": LIMIT})
    kl = _get("/fapi/v1/klines", {"symbol": symbol, "interval": PERIOD, "limit": LIMIT})
    df_ls = pd.DataFrame([{"time": pd.to_datetime(r["timestamp"], unit="ms", utc=True),
                           "ls_ratio": float(r["longShortRatio"])} for r in ls])
    df_oi = pd.DataFrame([{"time": pd.to_datetime(r["timestamp"], unit="ms", utc=True),
                           "oi": float(r["sumOpenInterest"])} for r in oi])
    df_pr = pd.DataFrame([{"time": pd.to_datetime(c[0], unit="ms", utc=True),
                           "close": float(c[4])} for c in kl])
    df = pd.merge_asof(df_pr.sort_values("time"), df_ls.sort_values("time"),
                       on="time", direction="nearest", tolerance=pd.Timedelta("1h"))
    df = pd.merge_asof(df.sort_values("time"), df_oi.sort_values("time"),
                       on="time", direction="nearest", tolerance=pd.Timedelta("1h")).dropna()
    df = df.reset_index(drop=True)
    df["ls_z"] = (df["ls_ratio"] - df["ls_ratio"].rolling(Z_WIN).mean()) / df["ls_ratio"].rolling(Z_WIN).std()
    oi_chg = df["oi"].pct_change()
    df["oi_z"] = (oi_chg - oi_chg.rolling(Z_WIN).mean()) / oi_chg.rolling(Z_WIN).std()
    return df


def collect(df, zcol, mode, hold):
    rets = []
    for i in range(Z_WIN, len(df) - hold):
        z = df[zcol].iloc[i]
        if pd.isna(z) or abs(z) < Z_THRESH:
            continue
        sign = 1 if z > 0 else -1
        richting = sign if mode == "momentum" else -sign
        entry, exit_ = df["close"].iloc[i], df["close"].iloc[i + hold]
        rets.append(richting * (exit_ - entry) / entry - COST)
    return rets


def main():
    print(f"POSITIONERING-SIGNALEN — long/short-ratio + open interest ({PERIOD}, ~30d), "
          f"netto na 0,35% kosten, met t-test\n")
    data = {}
    for name, sym in MARKETS.items():
        try:
            data[name] = fetch_signals(sym)
        except Exception as exc:
            print(f"  {name}: mislukt — {exc}")

    for zcol, label in (("ls_z", "LONG/SHORT-ratio"), ("oi_z", "OPEN INTEREST")):
        print(f"=== {label} ===")
        for mode in ("momentum", "reversion"):
            for hold in HOLDS:
                rets = []
                for df in data.values():
                    rets += collect(df, zcol, mode, hold)
                st = edge_stats(rets)
                vlag = "  <<< EDGE" if st.get("t_stat", 0) > 1.96 else ""
                print(f"  {mode:<10} hold {hold}: n={st['n']:>4} WR={st['winrate']:>4}% "
                      f"exp={st['expectancy_pct']:+.3f}% t={st['t_stat']:+.2f} {st['verdict']}{vlag}")
        print()


if __name__ == "__main__":
    main()
