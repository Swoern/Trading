"""
governance.py — strategie-leaderboard + governance-lifecycle (AlphaFlow-stijl).

Meet per strategie de prestatie, kent een status toe (CANDIDATE → CHALLENGER →
ACTIVE → DEPRECATED → RETIRED) op basis van sample-grootte + significantie, en past
een confidence-decay toe op basis van hoe lang geleden de strategie nog handelde.

Dit is governance: het bestuurt WELKE strategieën mogen meedoen. Het meet en
beschermt — het creëert geen edge.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

# Drempels voor de lifecycle.
MIN_CHALLENGER = 5      # onder dit = nog CANDIDATE (te weinig data)
MIN_PROVEN = 30         # vanaf dit telt het oordeel echt mee
MIN_RETIRE = 40         # genoeg om significant-slecht hard af te keuren


def _stats(returns: list[float]) -> dict:
    n = len(returns)
    if n == 0:
        return {"n": 0, "winrate": 0.0, "profit_factor": 0.0, "expectancy": 0.0, "t_stat": 0.0}
    mean = sum(returns) / n
    wins = [r for r in returns if r > 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(r for r in returns if r <= 0))
    if n >= 2:
        std = math.sqrt(sum((r - mean) ** 2 for r in returns) / (n - 1))
        t = (mean / (std / math.sqrt(n))) if std > 0 else 0.0
    else:
        t = 0.0
    return {
        "n": n,
        "winrate": round(100 * len(wins) / n, 1),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0),
        "expectancy": round(mean, 4),
        "t_stat": round(t, 2),
    }


def confidence_decay(days_since_last: float | None) -> float:
    """Recency-factor (AlphaFlow): hoe ouder de laatste trade, hoe lager het vertrouwen."""
    if days_since_last is None:
        return 1.0
    if days_since_last <= 30:
        return 1.0
    if days_since_last <= 60:
        return 0.98
    if days_since_last <= 90:
        return 0.95
    return 0.90


def assign_status(n: int, expectancy: float, t_stat: float) -> str:
    if n < MIN_CHALLENGER:
        return "CANDIDATE"          # te weinig data om te oordelen
    if n >= MIN_RETIRE and t_stat < -1.96:
        return "RETIRED"            # significant slecht
    if n >= MIN_PROVEN:
        if expectancy > 0 and t_stat > 1.96:
            return "ACTIVE"         # bewezen positief
        if expectancy > 0:
            return "WATCHLIST"      # positief maar niet significant
        return "DEPRECATED"         # bewezen niet-positief
    return "CHALLENGER"             # genoeg om mee te testen, nog niet bewezen


def governance_score(st: dict, decay: float) -> float:
    """0-100 score: blend van winrate, profit factor, expectancy-teken en recency."""
    pf = min(st["profit_factor"], 3.0) if st["profit_factor"] != float("inf") else 3.0
    wr = st["winrate"] / 100
    exp_sign = 1.0 if st["expectancy"] > 0 else 0.0
    raw = 0.35 * (pf / 3.0) + 0.30 * wr + 0.20 * exp_sign + 0.15 * min(st["n"] / MIN_PROVEN, 1.0)
    return round(100 * raw * decay, 1)


def strategy_leaderboard(trades_by_strategy: dict[str, list[dict]],
                         now: datetime | None = None) -> list[dict]:
    """
    trades_by_strategy: {strategie: [{result_pct, closed_at}, ...]}.
    Geeft een gerangschikte leaderboard met status, decay en governance-score.
    """
    now = now or datetime.now(timezone.utc)
    board = []
    for strat, trades in trades_by_strategy.items():
        rets = [float(t["result_pct"]) for t in trades if t.get("result_pct") is not None]
        st = _stats(rets)
        # Dagen sinds laatste trade.
        days = None
        last_dates = [t.get("closed_at") for t in trades if t.get("closed_at")]
        if last_dates:
            try:
                latest = max(datetime.fromisoformat(str(d)[:19]).replace(tzinfo=timezone.utc)
                             for d in last_dates)
                days = (now - latest).total_seconds() / 86400
            except Exception:
                days = None
        decay = confidence_decay(days)
        total_pnl = round(sum(rets), 3)
        board.append({
            "strategy": strat,
            **st,
            "total_pnl_pct": total_pnl,
            "days_since_last": round(days, 1) if days is not None else None,
            "confidence_decay": decay,
            "status": assign_status(st["n"], st["expectancy"], st["t_stat"]),
            "governance_score": governance_score(st, decay),
        })
    board.sort(key=lambda r: r["governance_score"], reverse=True)
    # Contributie = aandeel in totale (absolute) P&L.
    tot_abs = sum(abs(r["total_pnl_pct"]) for r in board) or 1.0
    for r in board:
        r["contribution_pct"] = round(100 * abs(r["total_pnl_pct"]) / tot_abs, 1)
    return board


def build_leaderboard_from_db(db_path: str | None = None) -> list[dict]:
    import os
    import sqlite3
    from pathlib import Path
    path = db_path or os.environ.get("TRADEAI_DB_PATH") or str(
        Path(__file__).resolve().parents[1] / "tradeai.db")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT strategy, result_pct, closed_at FROM paper_trades "
        "WHERE source='auto' AND status='closed' AND result_pct IS NOT NULL"
    ).fetchall()
    conn.close()
    by_strat: dict[str, list[dict]] = {}
    for r in rows:
        by_strat.setdefault(r["strategy"] or "onbekend", []).append(
            {"result_pct": r["result_pct"], "closed_at": r["closed_at"]})
    return strategy_leaderboard(by_strat)


if __name__ == "__main__":
    print("STRATEGIE-LEADERBOARD (governance)\n")
    print(f"  {'strategie':<16}{'n':>4}{'WR':>6}{'PF':>6}{'exp':>9}{'status':>12}{'score':>7}")
    for r in build_leaderboard_from_db():
        print(f"  {r['strategy']:<16}{r['n']:>4}{r['winrate']:>5.0f}%{r['profit_factor']:>6}"
              f"{r['expectancy']:>+9.3f}{r['status']:>12}{r['governance_score']:>7}")
