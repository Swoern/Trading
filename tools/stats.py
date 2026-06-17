"""
stats.py — statistische maatstaven voor edge-beoordeling.

Berekent uit een lijst per-trade returns (%): expectancy, winrate, profit factor,
Sharpe & Sortino (per trade), en een t-test of de gemiddelde return ≠ 0
(significantie). Zo beoordeel je een config hard i.p.v. op winrate-gevoel.
"""
from __future__ import annotations

import math


def edge_stats(returns_pct: list[float]) -> dict:
    n = len(returns_pct)
    if n < 2:
        return {"n": n, "te_weinig_data": True}
    mean = sum(returns_pct) / n
    var = sum((r - mean) ** 2 for r in returns_pct) / (n - 1)
    std = math.sqrt(var)
    wins = [r for r in returns_pct if r > 0]
    losses = [r for r in returns_pct if r <= 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    downside = [r for r in returns_pct if r < 0]
    dstd = math.sqrt(sum(r ** 2 for r in downside) / len(downside)) if downside else 0.0
    t_stat = (mean / (std / math.sqrt(n))) if std > 0 else 0.0
    return {
        "n": n,
        "winrate": round(100 * len(wins) / n, 1),
        "expectancy_pct": round(mean, 4),
        "std_pct": round(std, 4),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "sharpe_per_trade": round(mean / std, 3) if std > 0 else 0.0,
        "sortino_per_trade": round(mean / dstd, 3) if dstd > 0 else 0.0,
        "t_stat": round(t_stat, 2),
        "significant_95": abs(t_stat) > 1.96,
        "verdict": (
            "edge (significant +)" if t_stat > 1.96 else
            "verlies (significant -)" if t_stat < -1.96 else
            "geen significante edge"
        ),
    }


def print_stats(label: str, returns_pct: list[float]) -> None:
    s = edge_stats(returns_pct)
    if s.get("te_weinig_data"):
        print(f"  {label}: te weinig data (n={s['n']})")
        return
    print(f"  {label}")
    print(f"     n={s['n']}  winrate={s['winrate']}%  expectancy={s['expectancy_pct']:+.3f}%/trade")
    print(f"     profit factor={s['profit_factor']}  Sharpe/trade={s['sharpe_per_trade']}  "
          f"Sortino/trade={s['sortino_per_trade']}")
    print(f"     t-stat={s['t_stat']}  →  {s['verdict']}")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from backtest_report import fetch_history, _PER_DAY
    from fast_backtest import fast_run_backtest

    MARKETS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD"]
    CURATED = {"trend_short", "trend_long", "macd_bear_div"}
    total = 365 * _PER_DAY["4h"]
    print("RIGOUREUZE STATISTIEK — beste config (4h, curated, with-trend, min_rr=1.5)\n")
    rets: list[float] = []
    for m in MARKETS:
        try:
            df = fetch_history(m, "4h", total)
        except Exception:
            continue
        rets += [t["result_pct"] for t in fast_run_backtest(
            df, min_rr=1.5, max_candles_open=18, htf_period=50, allowed=CURATED)["trades"]]
    print_stats("Beste gratis config (alle markten)", rets)
