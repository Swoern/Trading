"""
monte_carlo.py — Monte Carlo robuustheidsanalyse op een set trades.

Gegeven de winst/verlies per trade: schud de volgorde 1000x en bouw telkens de
equity-curve. Zo zie je niet één toevallige uitkomst, maar de hele VERDELING van
mogelijke uitkomsten (eindrendement, max drawdown) met betrouwbaarheidsintervallen.

Belangrijk: dit MEET de robuustheid van een strategie — het maakt geen edge. Een
strategie zonder edge komt er ook met Monte Carlo als verliesgevend uit.
"""
from __future__ import annotations

import numpy as np


def monte_carlo(pnls_eur: list[float], start: float = 10_000.0,
                n_iter: int = 1000, seed: int = 42) -> dict:
    """
    pnls_eur: euro-resultaat per trade (compoundt op het huidige kapitaal).
    Geeft de verdeling van eindrendement en max drawdown over n_iter shuffles.
    """
    arr = np.asarray(pnls_eur, dtype=float)
    n = len(arr)
    if n < 2:
        return {"n_trades": n, "te_weinig_data": True}

    # Per-trade multiplier (compounding): risico geschaald op startkapitaal.
    mult = 1.0 + arr / start
    rng = np.random.default_rng(seed)

    finals = np.empty(n_iter)
    maxdds = np.empty(n_iter)
    for i in range(n_iter):
        path = start * np.cumprod(rng.permutation(mult))
        path = np.concatenate([[start], path])
        finals[i] = path[-1]
        peak = np.maximum.accumulate(path)
        maxdds[i] = ((path - peak) / peak).min() * 100  # negatief %

    rets = (finals / start - 1.0) * 100
    return {
        "n_trades": n,
        "iters": n_iter,
        "return_median_pct": round(float(np.median(rets)), 2),
        "return_ci95": [round(float(np.percentile(rets, 2.5)), 2),
                        round(float(np.percentile(rets, 97.5)), 2)],
        "return_ci99": [round(float(np.percentile(rets, 0.5)), 2),
                        round(float(np.percentile(rets, 99.5)), 2)],
        "prob_profit": round(float((finals > start).mean()), 3),
        "maxdd_median_pct": round(float(np.median(maxdds)), 2),
        "maxdd_worst_pct": round(float(np.percentile(maxdds, 5)), 2),  # 5e percentiel = slechtste 5%
        "calmar": round(float(np.median(rets) / abs(np.median(maxdds))), 2) if np.median(maxdds) != 0 else 0.0,
    }


def print_monte_carlo(label: str, pnls_eur: list[float], start: float = 10_000.0) -> None:
    mc = monte_carlo(pnls_eur, start=start)
    if mc.get("te_weinig_data"):
        print(f"  {label}: te weinig trades (n={mc['n_trades']})")
        return
    print(f"  {label}  ({mc['n_trades']} trades, {mc['iters']} simulaties)")
    print(f"     mediaan rendement: {mc['return_median_pct']:+.1f}%   "
          f"95%-band: [{mc['return_ci95'][0]:+.1f}%, {mc['return_ci95'][1]:+.1f}%]")
    print(f"     kans op winst: {mc['prob_profit']:.0%}   "
          f"mediaan max drawdown: {mc['maxdd_median_pct']:.1f}%   "
          f"slechtste 5%: {mc['maxdd_worst_pct']:.1f}%")
    print(f"     Calmar (rendement/drawdown): {mc['calmar']}")


if __name__ == "__main__":
    import sqlite3
    from pathlib import Path
    db = str(Path(__file__).resolve().parents[1] / "tradeai.db")
    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT result_euro FROM paper_trades WHERE source='auto' AND status='closed' "
        "AND result_euro IS NOT NULL"
    ).fetchall()
    conn.close()
    pnls = [r[0] for r in rows]
    print("MONTE CARLO — live bot-trades\n")
    print_monte_carlo("Live auto-trades", pnls)
    print("\n  (Meet robuustheid/spreiding van uitkomsten — geen edge-creatie.)")
