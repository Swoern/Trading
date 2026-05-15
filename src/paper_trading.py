"""
paper_trading.py — Beheer van paper trades (nep-trades zonder echt geld).

Paper trading = je oefent alsof je echt handelt, maar je riskeert niets.
Het doel is om te leren van je beslissingen zonder financieel risico.

⚠️  Automatisch handelen met echt geld is NIET mogelijk in deze versie.
"""
from src.database import (
    add_paper_trade,
    close_paper_trade,
    get_paper_trades,
    init_database,
)
from src.strategy import generate_signal

MIN_RR = 1.5   # Minimum risk/reward voor een handmatige trade


def open_signal_trade(df, asset: str, reden: str = "",
                       kapitaal: float = 1000.0) -> dict:
    """
    Open automatisch een paper trade op basis van het huidig signaal.
    Werkt alleen als er een echte setup is.
    """
    init_database()
    signal = generate_signal(df)

    if signal["type"] != "setup":
        return {
            "success":   False,
            "boodschap": f"Geen setup gevonden. {signal['uitleg']}",
        }

    trade_id = add_paper_trade(
        asset       = asset,
        direction   = signal["richting"],
        entry_price = signal["entry"],
        stop_loss   = signal["stop_loss"],
        take_profit = signal["take_profit"],
        reason      = reden or "; ".join(signal["reden"]),
        capital     = kapitaal,
    )

    return {
        "success":   True,
        "trade_id":  trade_id,
        "signal":    signal,
        "boodschap": (
            f"Paper trade #{trade_id} geopend: "
            f"{signal['richting'].upper()} {asset} @ {signal['entry']}"
        ),
    }


def open_manual_trade(asset: str, direction: str,
                       entry_price: float, stop_loss: float,
                       take_profit: float, reden: str = "",
                       kapitaal: float = 1000.0) -> dict:
    """
    Open een handmatige paper trade met zelf gekozen levels.

    Controles:
      - Risk/Reward minimaal 1:MIN_RR
      - Stop-loss op juiste kant van entry
    """
    init_database()

    if direction == "long":
        risk   = entry_price - stop_loss
        reward = take_profit - entry_price
    else:
        risk   = stop_loss - entry_price
        reward = entry_price - take_profit

    if risk <= 0:
        return {
            "success":   False,
            "boodschap": (
                "Stop-loss staat op de verkeerde kant van de entry. "
                "Bij een long trade moet stop-loss ONDER de entry liggen."
            ),
        }

    rr = reward / risk

    if rr < MIN_RR:
        return {
            "success":   False,
            "boodschap": (
                f"Risk/Reward ({rr:.2f}) is te laag. "
                f"Minimum is 1:{MIN_RR}. Pas je levels aan."
            ),
        }

    trade_id = add_paper_trade(
        asset       = asset,
        direction   = direction,
        entry_price = entry_price,
        stop_loss   = stop_loss,
        take_profit = take_profit,
        reason      = reden,
        capital     = kapitaal,
    )

    return {
        "success":   True,
        "trade_id":  trade_id,
        "rr":        round(rr, 2),
        "boodschap": (
            f"Handmatige paper trade #{trade_id} geopend: "
            f"{direction.upper()} {asset} @ {entry_price} | "
            f"SL: {stop_loss} | TP: {take_profit} | R/R: 1:{rr:.1f}"
        ),
    }


def sluit_paper_trade(trade_id: int, exit_price: float,
                       exit_reden: str = "manual",
                       fout_analyse: str = "") -> dict:
    """
    Sluit een open paper trade en sla het resultaat op.

    exit_reden: 'manual', 'sl' (stop-loss geraakt) of 'tp' (take-profit geraakt)
    fout_analyse: wat ging er fout of wat leerde je hiervan?
    """
    gelukt = close_paper_trade(trade_id, exit_price, exit_reden, fout_analyse)

    if gelukt:
        return {
            "success":   True,
            "boodschap": f"Trade #{trade_id} succesvol gesloten @ {exit_price}.",
        }
    return {
        "success":   False,
        "boodschap": (
            f"Trade #{trade_id} kon niet worden gesloten. "
            f"Is hij al gesloten of bestaat hij niet?"
        ),
    }


def get_open_trades() -> list[dict]:
    """Geef alle open paper trades terug."""
    return get_paper_trades(status="open")


def get_closed_trades() -> list[dict]:
    """Geef alle gesloten paper trades terug."""
    return get_paper_trades(status="closed")


def get_all_trades() -> list[dict]:
    """Geef alle paper trades terug (open én gesloten)."""
    return get_paper_trades()
