"""
alpha_experiments.py — vergelijkt configuraties op edge (netto gem%/trade na kosten).

Bouwt op de snelle backtester. Test of regime-filter (mét-de-trend), strategie-
curatie en omgekeerde selectie (lage R/R) de edge over de nullijn tillen.

Let op: 'curated' en 'with-trend' zijn hypothesen uit eerdere analyse — de échte
out-of-sample test is forward paper trading. Dit is de snelle voorselectie.

Gebruik:  python tools/alpha_experiments.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest_report import fetch_history, _PER_DAY  # noqa: E402
from fast_backtest import fast_run_backtest  # noqa: E402

MARKETS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD"]
INTERVAL, DAYS = "4h", 365
MIN_RR = 1.5
HTF = 50  # ~8 dagen macro-trend op 4h
CURATED = {"trend_short", "trend_long", "macd_bear_div"}

VARIANTS = [
    ("baseline (alles, hoogste R/R)", {}),
    ("with-trend (alles)", {"htf_period": HTF}),
    ("with-trend + curated", {"htf_period": HTF, "allowed": CURATED}),
    ("with-trend + curated + lage R/R", {"htf_period": HTF, "allowed": CURATED, "select": "lowrr"}),
    ("curated + lage R/R (geen trend)", {"allowed": CURATED, "select": "lowrr"}),
]


def _summ(trades: list[dict]) -> tuple[int, float, float]:
    n = len(trades)
    if not n:
        return 0, 0.0, 0.0
    w = sum(1 for t in trades if t["win"])
    return n, 100 * w / n, sum(t["result_pct"] for t in trades) / n


def main() -> None:
    total = DAYS * _PER_DAY[INTERVAL]
    print(f"ALPHA-EXPERIMENTEN — {INTERVAL}, {DAYS}d, min_rr={MIN_RR}, kosten 0,35% round-trip")
    print("netto gem%/trade > 0 = edge na kosten\n")

    data = {}
    for m in MARKETS:
        try:
            data[m] = fetch_history(m, INTERVAL, total)
        except Exception as exc:
            print(f"  {m}: ophalen mislukt — {exc}")

    print(f"  {'variant':<34}{'trades':>8}{'winrate':>9}{'gem%/trade':>13}")
    for naam, kw in VARIANTS:
        alle: list[dict] = []
        for df in data.values():
            alle += fast_run_backtest(df, min_rr=MIN_RR, max_candles_open=18, **kw)["trades"]
        n, wr, avg = _summ(alle)
        vlag = " <-- EDGE" if avg > 0 else ""
        print(f"  {naam:<34}{n:>8}{wr:>8.0f}%{avg:>12.3f}%{vlag}")


if __name__ == "__main__":
    main()
