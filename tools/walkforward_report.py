"""
walkforward_report.py — ZUIVERE walk-forward, geen selectie-bias.

Per rollend venster:
  1) bepaal op de TRAIN-candles welke strategieën positief netto presteren
     (selectie gebruikt ALLEEN het verleden);
  2) handel op de volgende, ongeziene TEST-candles ALLEEN die strategieën;
  3) accumuleer de test-resultaten.

Vergelijkt 'adaptief' (selectie op verleden) met 'baseline' (alle strategieën).
Als adaptief over alle out-of-sample vensters een positieve netto gem%/trade
houdt, is dat een echt edge-signaal — geen overfit.

Gebruik:  python tools/walkforward_report.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from backtest_report import fetch_history, _PER_DAY  # noqa: E402
from src.backtest import run_backtest  # noqa: E402
from src.trading_costs import DEFAULT_FEE_RATE, DEFAULT_SLIPPAGE_RATE, DEFAULT_SPREAD_RATE  # noqa: E402

MARKETS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD"]
INTERVAL, DAYS = "4h", 365
MIN_RR = 1.5            # beste uit de sweep
TRAIN, TEST, STEP = 400, 150, 150
MIN_TRAIN_TRADES = 5   # minimaal aantal train-trades om een strategie te vertrouwen
RISICO_PCT, MAX_OPEN, INITIAL = 1.0, 18, 10_000.0


def _bt(df: pd.DataFrame) -> list[dict]:
    if len(df) < 130:
        return []
    return run_backtest(
        df, min_rr=MIN_RR, initial_capital=INITIAL, risico_per_trade_pct=RISICO_PCT,
        max_candles_open=MAX_OPEN, fee_rate=DEFAULT_FEE_RATE,
        spread_rate=DEFAULT_SPREAD_RATE, slippage_rate=DEFAULT_SLIPPAGE_RATE,
    )["trades"]


def _select_strategies(train_trades: list[dict]) -> set:
    """Kies strategieën met positieve netto gem% en genoeg train-trades."""
    agg = defaultdict(lambda: {"n": 0, "pct": 0.0})
    for t in train_trades:
        agg[t["strategie"]]["n"] += 1
        agg[t["strategie"]]["pct"] += t["result_pct"]
    return {
        s for s, d in agg.items()
        if d["n"] >= MIN_TRAIN_TRADES and d["pct"] / d["n"] > 0
    }


def _summ(trades: list[dict]) -> str:
    n = len(trades)
    if not n:
        return "geen trades"
    w = sum(1 for t in trades if t["win"])
    avg = sum(t["result_pct"] for t in trades) / n
    return f"n={n:>4}  WR={100*w/n:>3.0f}%  netto gem%/trade={avg:+.3f}%  som%={sum(t['result_pct'] for t in trades):+.1f}"


def main() -> None:
    total = DAYS * _PER_DAY[INTERVAL]
    print(f"ZUIVERE WALK-FORWARD — {INTERVAL}, {DAYS}d, min_rr={MIN_RR}, "
          f"train={TRAIN}/test={TEST} candles, kosten 0,35% round-trip\n")

    adaptive, baseline = [], []
    sel_history = defaultdict(int)
    windows = 0
    for m in MARKETS:
        try:
            df = fetch_history(m, INTERVAL, total)
        except Exception as exc:
            print(f"  {m}: ophalen mislukt — {exc}")
            continue
        start = 0
        while start + TRAIN + TEST <= len(df):
            train_df = df.iloc[start:start + TRAIN].reset_index(drop=True)
            test_df = df.iloc[start + TRAIN:start + TRAIN + TEST].reset_index(drop=True)
            selected = _select_strategies(_bt(train_df))
            for s in selected:
                sel_history[s] += 1
            test_trades = _bt(test_df)
            baseline += test_trades
            adaptive += [t for t in test_trades if t["strategie"] in selected]
            windows += 1
            start += STEP

    print(f"Vensters getest: {windows}  (over {len(MARKETS)} markten)\n")
    print(f"  ADAPTIEF (selectie op verleden):  {_summ(adaptive)}")
    print(f"  BASELINE (alle strategieën):      {_summ(baseline)}")
    print()
    print("  Hoe vaak werd elke strategie door de train-selectie gekozen:")
    for s in sorted(sel_history, key=lambda k: sel_history[k], reverse=True):
        print(f"     {s:<16} {sel_history[s]}x")


if __name__ == "__main__":
    main()
