"""
trading_costs.py - Conservatief kostenmodel voor paper trading en backtests.

Alle waarden zijn per kant van de trade, behalve spread: die wordt als volledige
bid/ask-spread opgegeven en per fill half meegerekend.
"""
from __future__ import annotations

import os


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# Standaard: 0,10% fee + 0,05% spread + 0,05% slippage per round-trip kantachtig.
DEFAULT_FEE_RATE = _env_float("TRADEAI_FEE_RATE", 0.0010)
DEFAULT_SPREAD_RATE = _env_float("TRADEAI_SPREAD_RATE", 0.0005)
DEFAULT_SLIPPAGE_RATE = _env_float("TRADEAI_SLIPPAGE_RATE", 0.0005)


def per_side_cost_rate(
    fee_rate: float = DEFAULT_FEE_RATE,
    spread_rate: float = DEFAULT_SPREAD_RATE,
    slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
) -> float:
    """Effectieve kosten per fill als fractie van notional."""
    return max(0.0, fee_rate) + max(0.0, spread_rate) / 2 + max(0.0, slippage_rate)


def round_trip_cost_euro(
    notional_euro: float,
    fee_rate: float = DEFAULT_FEE_RATE,
    spread_rate: float = DEFAULT_SPREAD_RATE,
    slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
) -> float:
    """Schat totale entry+exit kosten op basis van ingezette notional."""
    return max(0.0, float(notional_euro or 0.0)) * per_side_cost_rate(
        fee_rate, spread_rate, slippage_rate
    ) * 2


def apply_round_trip_costs(
    gross_result_euro: float,
    original_notional_euro: float,
    fee_rate: float = DEFAULT_FEE_RATE,
    spread_rate: float = DEFAULT_SPREAD_RATE,
    slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
) -> tuple[float, float]:
    """Geef (netto_resultaat, kosten) terug."""
    costs = round_trip_cost_euro(
        original_notional_euro,
        fee_rate=fee_rate,
        spread_rate=spread_rate,
        slippage_rate=slippage_rate,
    )
    return float(gross_result_euro or 0.0) - costs, costs

