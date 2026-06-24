"""
backtest.py — Simpele backtester die de strategie test op historische data.

Werkwijze (geen lookahead):
  1. Ga candle voor candle door de dataset
  2. Genereer een signaal op basis van candles TOT EN MET die candle
  3. Simuleer de trade: enter op de VOLGENDE open-prijs
  4. Controleer op volgende candles of SL of TP geraakt wordt
  5. Registreer het resultaat

⚠️  Backtestresultaten zijn GEEN garantie voor de toekomst.
"""
import pandas as pd
import numpy as np
from src.indicators import add_all_indicators
from src.strategy import scan_alle_strategieen
from src.trading_costs import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    DEFAULT_SPREAD_RATE,
    apply_round_trip_costs,
)


def run_backtest(df: pd.DataFrame,
                 min_rr: float = 2.0,
                 initial_capital: float = 10_000.0,
                 risico_per_trade_pct: float = 1.0,
                 max_candles_open: int = 40,
                 fee_rate: float = DEFAULT_FEE_RATE,
                 spread_rate: float = DEFAULT_SPREAD_RATE,
                 slippage_rate: float = DEFAULT_SLIPPAGE_RATE) -> dict:
    """
    Voer de backtest uit.

    Parameters
    ----------
    df                  : DataFrame met OHLCV-data
    min_rr              : Minimum risk/reward ratio (standaard 2.0 = 1:2)
    initial_capital     : Startkapitaal in euro's (alleen voor simulatie)
    risico_per_trade_pct: Hoeveel % van het kapitaal je riskeert per trade
    max_candles_open    : Maximaal aantal candles dat een trade openblijft

    Returns
    -------
    dict met metrics en een lijst van trades
    """
    df = add_all_indicators(df)
    WARMUP = 60   # Minimaal aantal candles voor scan_alle_strategieen

    trades    = []
    i         = WARMUP
    kapitaal  = initial_capital

    while i < len(df) - 2:
        subset = df.iloc[:i + 1].copy()
        setups = scan_alle_strategieen(subset, min_rr=min_rr)

        if not setups:
            i += 1
            continue

        # Kies de setup met de hoogste R/R als er meerdere zijn
        signal   = max(setups, key=lambda s: s.get("risk_reward", 0))
        richting = signal["richting"]
        sl_prijs = signal["stop_loss"]
        tp_prijs = signal["take_profit"]

        # Entry op de OPEN van de volgende candle (geen lookahead)
        entry_candle = df.iloc[i + 1]
        entry_prijs  = float(entry_candle["open"])
        entry_datum  = entry_candle["timestamp"]

        # Herbereken SL/TP relatief aan werkelijke entry (zelfde afstand)
        risico = abs(signal["entry"] - sl_prijs)
        if richting == "long":
            sl_prijs = entry_prijs - risico
            tp_prijs = entry_prijs + min_rr * risico
        else:
            sl_prijs = entry_prijs + risico
            tp_prijs = entry_prijs - min_rr * risico

        # Positiegrootte op basis van risico%
        euro_risico        = kapitaal * (risico_per_trade_pct / 100)
        risico_per_eenheid = abs(entry_prijs - sl_prijs)
        positiegrootte     = euro_risico / risico_per_eenheid if risico_per_eenheid > 0 else 0

        # Simuleer de trade op de volgende candles
        resultaat  = None
        exit_prijs = None
        exit_datum = None
        exit_reden = None
        exit_index = None

        zoek_einde = min(i + 2 + max_candles_open, len(df))

        for j in range(i + 2, zoek_einde):
            candle = df.iloc[j]

            if richting == "long":
                if candle["low"] <= sl_prijs:
                    exit_prijs = sl_prijs
                    exit_reden = "sl"
                    exit_datum = candle["timestamp"]
                    exit_index = j
                    resultaat  = (sl_prijs - entry_prijs) / entry_prijs * 100
                    break
                if candle["high"] >= tp_prijs:
                    exit_prijs = tp_prijs
                    exit_reden = "tp"
                    exit_datum = candle["timestamp"]
                    exit_index = j
                    resultaat  = (tp_prijs - entry_prijs) / entry_prijs * 100
                    break
            else:  # short
                if candle["high"] >= sl_prijs:
                    exit_prijs = sl_prijs
                    exit_reden = "sl"
                    exit_datum = candle["timestamp"]
                    exit_index = j
                    resultaat  = (entry_prijs - sl_prijs) / entry_prijs * 100
                    break
                if candle["low"] <= tp_prijs:
                    exit_prijs = tp_prijs
                    exit_reden = "tp"
                    exit_datum = candle["timestamp"]
                    exit_index = j
                    resultaat  = (entry_prijs - tp_prijs) / entry_prijs * 100
                    break

        if resultaat is not None:
            if richting == "long":
                bruto_winst_euro = positiegrootte * (exit_prijs - entry_prijs)
            else:
                bruto_winst_euro = positiegrootte * (entry_prijs - exit_prijs)
            notional = positiegrootte * entry_prijs
            winst_euro, kosten_euro = apply_round_trip_costs(
                bruto_winst_euro,
                notional,
                fee_rate=fee_rate,
                spread_rate=spread_rate,
                slippage_rate=slippage_rate,
            )
            resultaat = (winst_euro / notional * 100) if notional > 0 else resultaat
            kapitaal += winst_euro

            trades.append({
                "entry_datum":  entry_datum,
                "exit_datum":   exit_datum,
                "richting":     richting,
                "strategie":    signal.get("strategie", "?"),
                "entry_prijs":  round(entry_prijs, 4),
                "exit_prijs":   round(exit_prijs, 4),
                "stop_loss":    round(sl_prijs, 4),
                "take_profit":  round(tp_prijs, 4),
                "result_pct":   round(resultaat, 3),
                "result_euro":  round(winst_euro, 2),
                "gross_result_euro": round(bruto_winst_euro, 2),
                "costs_euro":   round(kosten_euro, 2),
                "exit_reden":   exit_reden,
                "win":          resultaat > 0,
            })
            i = exit_index + 1
        else:
            # Trade niet gesloten binnen max_candles → sla over, geen lookahead
            i += 1

    # ── Bereken statistieken ──────────────────────────────────────
    if not trades:
        return {
            "trades":          [],
            "total_trades":    0,
            "winning_trades":  0,
            "losing_trades":   0,
            "winrate":         0.0,
            "total_pnl":       0.0,
            "avg_win":         0.0,
            "avg_loss":        0.0,
            "max_drawdown":    0.0,
            "eind_kapitaal":   round(initial_capital, 2),
            "fee_rate":        fee_rate,
            "spread_rate":     spread_rate,
            "slippage_rate":   slippage_rate,
            "kosten_euro":     0.0,
            "boodschap":       (
                "Geen trades gevonden in deze periode. "
                "Probeer een langere dataset of een andere min. R/R."
            ),
        }

    df_t = pd.DataFrame(trades)
    winst   = df_t[df_t["win"]]
    verlies = df_t[~df_t["win"]]

    total_trades   = len(df_t)
    winning_trades = len(winst)
    losing_trades  = len(verlies)
    winrate        = winning_trades / total_trades * 100

    avg_win  = winst["result_pct"].mean()  if len(winst)   > 0 else 0.0
    avg_loss = verlies["result_pct"].mean() if len(verlies) > 0 else 0.0

    df_t["cumulatief_pnl"] = df_t["result_pct"].cumsum()
    total_pnl  = df_t["result_pct"].sum()
    running_max = df_t["cumulatief_pnl"].cummax()
    max_drawdown = (df_t["cumulatief_pnl"] - running_max).min()

    return {
        "trades":          trades,
        "trades_df":       df_t,
        "total_trades":    total_trades,
        "winning_trades":  winning_trades,
        "losing_trades":   losing_trades,
        "winrate":         round(winrate, 1),
        "total_pnl":       round(total_pnl, 2),
        "avg_win":         round(avg_win, 2),
        "avg_loss":        round(avg_loss, 2),
        "max_drawdown":    round(max_drawdown, 2),
        "eind_kapitaal":   round(kapitaal, 2),
        "fee_rate":        fee_rate,
        "spread_rate":     spread_rate,
        "slippage_rate":   slippage_rate,
        "kosten_euro":     round(float(df_t["costs_euro"].sum()), 2),
        "boodschap": (
            f"Backtest klaar: {total_trades} trades gevonden, "
            f"winrate {winrate:.0f}%, totaal P&L {total_pnl:.1f}%."
        ),
    }


def run_walk_forward_backtest(
    df: pd.DataFrame,
    min_rr: float = 2.0,
    train_size: int = 180,
    test_size: int = 80,
    min_train_trades: int = 2,
    min_train_winrate: float = 40.0,
    min_train_pnl: float = 0.0,
    **kwargs,
) -> dict:
    """Train op venster A, test alleen bewezen strategieën op venster B."""
    windows = []
    start = 0
    while start + train_size + test_size <= len(df):
        train_df = df.iloc[start:start + train_size].reset_index(drop=True)
        test_df = df.iloc[start + train_size:start + train_size + test_size].reset_index(drop=True)
        train = run_backtest(train_df, min_rr=min_rr, **kwargs)
        by_strategy: dict[str, dict] = {}
        for trade in train.get("trades", []):
            strategy = trade.get("strategie", "unknown")
            stats = by_strategy.setdefault(strategy, {"trades": 0, "wins": 0, "pnl": 0.0})
            stats["trades"] += 1
            stats["wins"] += 1 if trade.get("result_pct", 0) > 0 else 0
            stats["pnl"] += float(trade.get("result_pct") or 0)
        allowed = {
            strategy for strategy, stats in by_strategy.items()
            if stats["trades"] >= min_train_trades
            and stats["wins"] / stats["trades"] * 100 >= min_train_winrate
            and stats["pnl"] > min_train_pnl
        }
        test = run_backtest(test_df, min_rr=min_rr, **kwargs)
        filtered_trades = [t for t in test.get("trades", []) if t.get("strategie") in allowed]
        windows.append({
            "start": start,
            "allowed_strategies": sorted(allowed),
            "train": train,
            "test_trades": filtered_trades,
            "test_pnl": round(sum(float(t.get("result_pct") or 0) for t in filtered_trades), 3),
            "test_wins": sum(1 for t in filtered_trades if t.get("result_pct", 0) > 0),
        })
        start += test_size

    total_trades = sum(len(w["test_trades"]) for w in windows)
    total_wins = sum(w["test_wins"] for w in windows)
    total_pnl = sum(w["test_pnl"] for w in windows)
    strategy_stats: dict[str, dict] = {}
    for window in windows:
        for trade in window["test_trades"]:
            strategy = trade.get("strategie", "unknown")
            stats = strategy_stats.setdefault(strategy, {
                "strategy": strategy,
                "trades": 0,
                "wins": 0,
                "pnl": 0.0,
                "windows": 0,
            })
            stats["trades"] += 1
            stats["wins"] += 1 if trade.get("result_pct", 0) > 0 else 0
            stats["pnl"] += float(trade.get("result_pct") or 0)
    for strategy, stats in strategy_stats.items():
        stats["winrate"] = round(stats["wins"] / stats["trades"] * 100, 1) if stats["trades"] else 0.0
        stats["pnl"] = round(stats["pnl"], 3)
        stats["windows"] = sum(
            1 for window in windows
            if any(trade.get("strategie") == strategy for trade in window["test_trades"])
        )
    return {
        "windows": windows,
        "total_trades": total_trades,
        "winning_trades": total_wins,
        "winrate": round(total_wins / total_trades * 100, 1) if total_trades else 0.0,
        "total_pnl": round(total_pnl, 3),
        "strategy_stats": strategy_stats,
    }
