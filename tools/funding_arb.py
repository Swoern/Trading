"""
funding_arb.py — schat de opbrengst van funding-arbitrage (marktneutraal).

Funding-arb: long spot + short perpetual (of omgekeerd) = delta-neutraal, prijs
maakt niet uit. Je zit op de kant die de funding ONTVANGT en oogst die elke 8u.
Gratis data (Binance funding-historie). Dit is een ECHTE, mechanische edge — geen
voorspelling.

Rapporteert de bruto jaarlijkse funding-oogst (bovengrens) + een conservatieve
netto-schatting na hedge-/uitvoeringskosten. LET OP: vereist kapitaal + spot- én
futures-account; bruto is een bovengrens (negeert basis-risico en herbalanceren).

Gebruik:  python tools/funding_arb.py
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
PER_YEAR = 3 * 365   # 3 funding-momenten per dag
# Conservatieve kosten: je flipt soms van kant; reken een kleine kost per dag aan
# hedge/rebalance. Ruwe aanname, geen exacte uitvoeringssimulatie.
EST_COST_PER_YEAR = 0.06   # ~6%/jaar aan uitvoerings-/hedge-frictie (conservatief)


def fetch_funding(symbol, total=1500):
    out, end = [], None
    while len(out) < total:
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
    return [float(r["fundingRate"]) for r in out[-total:]]


def main():
    print("FUNDING-ARBITRAGE (marktneutraal, gratis data) — geschatte jaarlijkse opbrengst\n")
    print(f"  {'markt':<6}{'gem funding/8u':>16}{'bruto/jaar':>12}{'netto/jaar*':>13}")
    bruto_tot = []
    for name, sym in MARKETS.items():
        try:
            f = fetch_funding(sym)
        except Exception as exc:
            print(f"  {name}: mislukt — {exc}")
            continue
        if not f:
            continue
        mean_abs = sum(abs(x) for x in f) / len(f)     # altijd op ontvangende kant
        bruto = mean_abs * PER_YEAR * 100               # %/jaar
        netto = bruto - EST_COST_PER_YEAR * 100
        bruto_tot.append(bruto)
        print(f"  {name:<6}{mean_abs*100:>14.4f}%{bruto:>11.1f}%{netto:>12.1f}%")

    if bruto_tot:
        gem = sum(bruto_tot) / len(bruto_tot)
        print(f"\n  Gemiddeld bruto: ~{gem:.1f}%/jaar  |  conservatief netto: "
              f"~{gem - EST_COST_PER_YEAR*100:.1f}%/jaar")
    print("\n  * netto = bruto minus ~6%/jaar geschatte hedge-/uitvoeringsfrictie (ruwe aanname).")
    print("  Vereist: kapitaal + spot- en futures-account. Bruto is een bovengrens")
    print("  (negeert basis-risico, herbalanceren, en momenten dat funding ongunstig staat).")


if __name__ == "__main__":
    main()
