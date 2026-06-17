"""
track_record.py — rigoureus track-record van de LIVE bot uit tradeai.db.

Leest de echte gesloten auto-trades en rapporteert met volledige statistiek
(expectancy, profit factor, Sharpe/Sortino, t-test) of de bot edge heeft —
totaal en per strategie. Draai dit elk moment om de werkelijke prestatie te meten.

Gebruik:  python tools/track_record.py [pad-naar-db]   (default tradeai.db)
"""
from __future__ import annotations

import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stats import edge_stats, print_stats  # noqa: E402

DB = sys.argv[1] if len(sys.argv) > 1 else "tradeai.db"


def main() -> None:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT strategy, result_pct, result_euro FROM paper_trades "
        "WHERE source='auto' AND status='closed' AND result_pct IS NOT NULL"
    ).fetchall()
    conn.close()

    if not rows:
        print("Geen gesloten auto-trades in de database.")
        return

    alle = [r["result_pct"] for r in rows]
    pnl = sum((r["result_euro"] or 0) for r in rows)
    print(f"LIVE TRACK-RECORD — {len(rows)} gesloten auto-trades, netto P&L €{pnl:.2f}\n")
    print_stats("TOTAAL", alle)

    per = defaultdict(list)
    for r in rows:
        per[r["strategy"] or "onbekend"].append(r["result_pct"])
    print("\n  Per strategie:")
    for s in sorted(per, key=lambda k: sum(per[k]), reverse=True):
        st = edge_stats(per[s])
        if st.get("te_weinig_data"):
            print(f"     {s:<16} n={st['n']} (te weinig voor statistiek)")
        else:
            print(f"     {s:<16} n={st['n']:>3}  WR={st['winrate']:>3.0f}%  "
                  f"exp={st['expectancy_pct']:+.3f}%  PF={st['profit_factor']}  {st['verdict']}")

    print("\n  Let op: een betrouwbaar oordeel vraagt ~100+ trades per strategie.")
    print("  Bij minder is dit indicatief, geen bewijs.")


if __name__ == "__main__":
    main()
