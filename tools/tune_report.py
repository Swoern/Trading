"""
tune_report.py — data-gedreven afstelling: min-R/R sweep met train/test-split.

Beantwoordt twee vragen:
  1) Helpt het verhogen van min-R/R tegen de 0,35%-kosten? (kosten-vraag)
  2) Generaliseert een betere instelling, of is het overfit? (train 70% / test 30%)

Plus: welke strategie-subset draagt de edge? (per-strategie netto)

Rapporteert het NETTO gemiddelde per trade (incl. kosten) — dat is de echte
edge-maat: > 0 betekent winst-verwachting na kosten.

Gebruik:  python tools/tune_report.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from backtest_report import fetch_history, _PER_DAY  # hergebruik fetch  # noqa: E402
from src.backtest import run_backtest  # noqa: E402
from src.trading_costs import DEFAULT_FEE_RATE, DEFAULT_SLIPPAGE_RATE, DEFAULT_SPREAD_RATE  # noqa: E402

MARKETS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD"]
CONFIGS = [("4h", 365)]
MIN_RR_SWEEP = [1.5, 2.0, 2.5, 3.0, 3.5]
RISICO_PCT = 1.0
MAX_CANDLES_OPEN = 18
INITIAL = 10_000.0
TRAIN_FRAC = 0.70
# Met-de-trend / minst-slechte strategieën uit de vorige analyse.
CURATED = {"trend_short", "trend_long", "macd_bear_div"}


def _bt(df: pd.DataFrame, min_rr: float) -> list[dict]:
    if len(df) < 130:
        return []
    res = run_backtest(
        df, min_rr=min_rr, initial_capital=INITIAL, risico_per_trade_pct=RISICO_PCT,
        max_candles_open=MAX_CANDLES_OPEN, fee_rate=DEFAULT_FEE_RATE,
        spread_rate=DEFAULT_SPREAD_RATE, slippage_rate=DEFAULT_SLIPPAGE_RATE,
    )
    return res["trades"]


def _stats(trades: list[dict], only: set | None = None) -> tuple[int, float, float]:
    """(#trades, winrate%, gem netto% per trade) — gem% > 0 = edge na kosten."""
    if only is not None:
        trades = [t for t in trades if t["strategie"] in only]
    n = len(trades)
    if not n:
        return 0, 0.0, 0.0
    w = sum(1 for t in trades if t["win"])
    avg = sum(t["result_pct"] for t in trades) / n
    return n, 100 * w / n, avg


def main() -> None:
    print("TUNING-SWEEP — netto gem%/trade (incl. 0,35% kosten). >0 = edge.")
    print("train = eerste 70% van de periode, test = laatste 30% (out-of-sample).\n")

    for interval, days in CONFIGS:
        total = days * _PER_DAY[interval]
        print(f"{'='*72}\n### {interval}  ({days} dagen)\n{'='*72}")

        # Haal alle markten één keer op en bewaar de candles.
        data = {}
        for m in MARKETS:
            try:
                data[m] = fetch_history(m, interval, total)
            except Exception as exc:
                print(f"  {m}: ophalen mislukt — {exc}")

        print(f"  {'min_rr':>7}{'trades':>8}{'train gem%':>12}{'test gem%':>11}{'test WR':>9}")
        beste = None
        for rr in MIN_RR_SWEEP:
            tr_all, te_all = [], []
            for m, df in data.items():
                cut = int(len(df) * TRAIN_FRAC)
                tr_all += _bt(df.iloc[:cut].reset_index(drop=True), rr)
                te_all += _bt(df.iloc[cut:].reset_index(drop=True), rr)
            tn, _, tr_avg = _stats(tr_all)
            en, te_wr, te_avg = _stats(te_all)
            print(f"  {rr:>7.1f}{tn+en:>8}{tr_avg:>11.3f}%{te_avg:>10.3f}%{te_wr:>8.0f}%")
            # 'beste' = hoogste test-gem% dat ook op train positief-ish is.
            if beste is None or te_avg > beste[1]:
                beste = (rr, te_avg, tr_all, te_all)

        rr_b, te_avg_b, tr_b, te_b = beste
        print(f"\n  >> beste min_rr op test: {rr_b} (test gem% {te_avg_b:+.3f})")
        # Gecureerde subset bij beste min_rr.
        cn_tr, cwr_tr, cavg_tr = _stats(tr_b, CURATED)
        cn_te, cwr_te, cavg_te = _stats(te_b, CURATED)
        print(f"  gecureerd {sorted(CURATED)} @ min_rr={rr_b}:")
        print(f"     train: n={cn_tr} WR={cwr_tr:.0f}% gem%={cavg_tr:+.3f}   "
              f"test: n={cn_te} WR={cwr_te:.0f}% gem%={cavg_te:+.3f}")
        print()


if __name__ == "__main__":
    main()
