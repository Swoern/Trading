"""
fast_backtest.py — O(n) backtester die identieke resultaten geeft als src.backtest.run_backtest.

De trage versie herberekent elke candle ALLE indicatoren op de groeiende dataset
(O(n²)). Omdat add_all_indicators causaal is (support/resistance wordt met
shift(lookback) vertraagd), zijn de indicatorwaarden op rij i identiek of je nu op
de volledige df of op df[:i+1] rekent. Daarom: bereken één keer vooraf en scan met
precomputed=True op een venster van 60 rijen → O(n).

`verify()` bewijst de gelijkwaardigheid trade-voor-trade.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.indicators import add_all_indicators  # noqa: E402
from src.strategy import scan_alle_strategieen  # noqa: E402
from src.trading_costs import (  # noqa: E402
    DEFAULT_FEE_RATE, DEFAULT_SLIPPAGE_RATE, DEFAULT_SPREAD_RATE, apply_round_trip_costs,
)

WARMUP = 60


def fast_run_backtest(df: pd.DataFrame, min_rr: float = 2.0, initial_capital: float = 10_000.0,
                      risico_per_trade_pct: float = 1.0, max_candles_open: int = 40,
                      fee_rate: float = DEFAULT_FEE_RATE, spread_rate: float = DEFAULT_SPREAD_RATE,
                      slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
                      htf_period: int | None = None, allowed: set | None = None,
                      select: str = "rr") -> dict:
    """
    Extra (optionele) knoppen voor alpha-experimenten — standaard = identiek aan slow:
      htf_period : alleen mét de macro-trend traden (close vs SMA(htf_period)).
      allowed    : alleen deze strategieën toestaan.
      select     : 'rr' (hoogste R/R, default) of 'lowrr' (laagste R/R).
    """
    df = add_all_indicators(df).reset_index(drop=True)
    macro = df["close"].rolling(htf_period).mean() if htf_period else None
    trades: list[dict] = []
    i = WARMUP
    kapitaal = initial_capital
    n = len(df)

    while i < n - 2:
        venster = df.iloc[max(0, i - WARMUP + 1):i + 1]  # 60 rijen, indicatoren al aanwezig
        setups = scan_alle_strategieen(venster, min_rr=min_rr, precomputed=True)
        if allowed is not None:
            setups = [s for s in setups if s.get("strategie") in allowed]
        if macro is not None and not pd.isna(macro.iloc[i]):
            up = float(df.iloc[i]["close"]) > float(macro.iloc[i])
            setups = [s for s in setups if (s["richting"] == "long") == up]
        if not setups:
            i += 1
            continue

        key = (lambda s: s.get("risk_reward", 0)) if select == "rr" else (lambda s: -s.get("risk_reward", 0))
        signal = max(setups, key=key)
        richting = signal["richting"]
        entry_candle = df.iloc[i + 1]
        entry_prijs = float(entry_candle["open"])
        entry_datum = entry_candle["timestamp"]

        risico = abs(signal["entry"] - signal["stop_loss"])
        if richting == "long":
            sl_prijs = entry_prijs - risico
            tp_prijs = entry_prijs + min_rr * risico
        else:
            sl_prijs = entry_prijs + risico
            tp_prijs = entry_prijs - min_rr * risico

        euro_risico = kapitaal * (risico_per_trade_pct / 100)
        risico_per_eenheid = abs(entry_prijs - sl_prijs)
        positiegrootte = euro_risico / risico_per_eenheid if risico_per_eenheid > 0 else 0

        resultaat = exit_prijs = exit_datum = exit_reden = exit_index = None
        for j in range(i + 2, min(i + 2 + max_candles_open, n)):
            c = df.iloc[j]
            if richting == "long":
                if c["low"] <= sl_prijs:
                    exit_prijs, exit_reden, resultaat = sl_prijs, "sl", (sl_prijs - entry_prijs) / entry_prijs * 100
                elif c["high"] >= tp_prijs:
                    exit_prijs, exit_reden, resultaat = tp_prijs, "tp", (tp_prijs - entry_prijs) / entry_prijs * 100
            else:
                if c["high"] >= sl_prijs:
                    exit_prijs, exit_reden, resultaat = sl_prijs, "sl", (entry_prijs - sl_prijs) / entry_prijs * 100
                elif c["low"] <= tp_prijs:
                    exit_prijs, exit_reden, resultaat = tp_prijs, "tp", (entry_prijs - tp_prijs) / entry_prijs * 100
            if resultaat is not None:
                exit_datum, exit_index = c["timestamp"], j
                break

        if resultaat is not None:
            if richting == "long":
                bruto = positiegrootte * (exit_prijs - entry_prijs)
            else:
                bruto = positiegrootte * (entry_prijs - exit_prijs)
            notional = positiegrootte * entry_prijs
            winst_euro, kosten_euro = apply_round_trip_costs(
                bruto, notional, fee_rate=fee_rate, spread_rate=spread_rate, slippage_rate=slippage_rate)
            resultaat = (winst_euro / notional * 100) if notional > 0 else resultaat
            kapitaal += winst_euro
            trades.append({
                "entry_datum": entry_datum, "exit_datum": exit_datum, "richting": richting,
                "strategie": signal.get("strategie", "?"), "result_pct": round(resultaat, 3),
                "result_euro": round(winst_euro, 2), "costs_euro": round(kosten_euro, 2),
                "exit_reden": exit_reden, "win": resultaat > 0,
            })
            i = exit_index + 1
        else:
            i += 1

    if not trades:
        return {"trades": [], "total_trades": 0, "winrate": 0.0, "eind_kapitaal": round(kapitaal, 2)}
    wins = sum(1 for t in trades if t["win"])
    return {
        "trades": trades, "total_trades": len(trades), "winning_trades": wins,
        "losing_trades": len(trades) - wins, "winrate": round(wins / len(trades) * 100, 1),
        "eind_kapitaal": round(kapitaal, 2),
    }


def verify(df: pd.DataFrame, min_rr: float = 2.0) -> None:
    """Bewijs dat fast == slow trade-voor-trade op een sample."""
    from src.backtest import run_backtest
    slow = run_backtest(df.copy(), min_rr=min_rr, max_candles_open=18)["trades"]
    fast = fast_run_backtest(df.copy(), min_rr=min_rr, max_candles_open=18)["trades"]
    print(f"  slow trades={len(slow)}  fast trades={len(fast)}")
    if len(slow) != len(fast):
        print("  ❌ AANTAL VERSCHILT"); return
    mism = 0
    for a, b in zip(slow, fast):
        if (str(a["entry_datum"]) != str(b["entry_datum"]) or round(a["result_pct"], 2) != round(b["result_pct"], 2)
                or a["strategie"] != b["strategie"]):
            mism += 1
    print("  ✅ IDENTIEK" if mism == 0 else f"  ❌ {mism} trades verschillen")


if __name__ == "__main__":
    import time
    from backtest_report import fetch_history
    print("Verificatie fast == slow op 1500 candles 1h BTC:")
    df = fetch_history("BTC-USD", "1h", 1500)
    verify(df)
    t0 = time.time(); fast_run_backtest(df, max_candles_open=18); tf = time.time() - t0
    print(f"  fast tijd: {tf:.2f}s")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.backtest import run_backtest
    t0 = time.time(); run_backtest(df.copy(), max_candles_open=18); ts = time.time() - t0
    print(f"  slow tijd: {ts:.2f}s  → versnelling {ts/tf:.0f}x")
