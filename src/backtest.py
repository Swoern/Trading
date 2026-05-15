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
from src.strategy import generate_signal


def run_backtest(df: pd.DataFrame,
                 min_rr: float = 2.0,
                 initial_capital: float = 10_000.0,
                 risico_per_trade_pct: float = 1.0,
                 max_candles_open: int = 40) -> dict:
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
    WARMUP = 55   # Minimaal aantal candles voor indicatoren

    trades    = []
    i         = WARMUP
    kaptiaal  = initial_capital

    while i < len(df) - 2:
        subset = df.iloc[: i + 1].copy()
        signal = generate_signal(subset, min_rr=min_rr)

        if signal["type"] != "setup" or signal["richting"] != "long":
            i += 1
            continue

        sl_prijs = signal["stop_loss"]
        tp_prijs = signal["take_profit"]

        # Entry op de OPEN van de volgende candle (geen lookahead)
        entry_candle = df.iloc[i + 1]
        entry_prijs  = entry_candle["open"]
        entry_datum  = entry_candle["timestamp"]

        # Herbereken SL/TP relatief aan werkelijke entry
        risico      = signal["entry"] - sl_prijs
        sl_prijs    = entry_prijs - risico
        tp_prijs    = entry_prijs + min_rr * risico

        # Positiegrootte op basis van risico%
        euro_risico       = kaptiaal * (risico_per_trade_pct / 100)
        risico_per_eenheid = entry_prijs - sl_prijs
        positiegrootte    = euro_risico / risico_per_eenheid if risico_per_eenheid > 0 else 0

        # Simuleer de trade op de volgende candles
        resultaat   = None
        exit_prijs  = None
        exit_datum  = None
        exit_reden  = None
        exit_index  = None

        zoek_einde = min(i + 2 + max_candles_open, len(df))

        for j in range(i + 2, zoek_einde):
            candle = df.iloc[j]

            # SL geraakt? (low van de candle raakt de stop-loss)
            if candle["low"] <= sl_prijs:
                exit_prijs = sl_prijs
                exit_reden = "sl"
                exit_datum = candle["timestamp"]
                exit_index = j
                resultaat  = (sl_prijs - entry_prijs) / entry_prijs * 100
                break

            # TP geraakt? (high van de candle raakt de take-profit)
            if candle["high"] >= tp_prijs:
                exit_prijs = tp_prijs
                exit_reden = "tp"
                exit_datum = candle["timestamp"]
                exit_index = j
                resultaat  = (tp_prijs - entry_prijs) / entry_prijs * 100
                break

        if resultaat is not None:
            winst_euro = positiegrootte * (exit_prijs - entry_prijs)
            kaptiaal  += winst_euro

            trades.append({
                "entry_datum":  entry_datum,
                "exit_datum":   exit_datum,
                "entry_prijs":  round(entry_prijs, 4),
                "exit_prijs":   round(exit_prijs, 4),
                "stop_loss":    round(sl_prijs, 4),
                "take_profit":  round(tp_prijs, 4),
                "result_pct":   round(resultaat, 3),
                "result_euro":  round(winst_euro, 2),
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
        "eind_kapitaal":   round(kaptiaal, 2),
        "boodschap": (
            f"Backtest klaar: {total_trades} trades gevonden, "
            f"winrate {winrate:.0f}%, totaal P&L {total_pnl:.1f}%."
        ),
    }
