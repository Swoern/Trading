"""setup_confidence.py - Eén totaalscore voor setup-beslissingen."""
from __future__ import annotations


def calculate_confidence(
    regime_ok: bool,
    htf_ok: bool,
    proof_details: dict,
    score_info: dict,
    quality: dict,
    market_context: dict | None = None,
    strategy_profile: dict | None = None,
    time_profile: dict | None = None,
) -> dict:
    market_context = market_context or {}
    strategy_profile = strategy_profile or {}
    time_profile = time_profile or {}
    points = 0
    reasons: list[str] = []

    if regime_ok:
        points += 25
        reasons.append("regime past +25")
    if htf_ok:
        points += 20
        reasons.append("HTF past +20")

    bt_pnl = float(proof_details.get("backtest_pnl") or 0)
    bt_wr = float(proof_details.get("backtest_winrate") or 0)
    if bt_pnl > 0 and bt_wr >= 40:
        points += 25
        reasons.append("backtest positief +25")

    wf_trades = int(proof_details.get("walk_forward_trades") or 0)
    wf_pnl = float(proof_details.get("walk_forward_pnl") or 0)
    wf_wr = float(proof_details.get("walk_forward_winrate") or 0)
    if proof_details.get("walk_forward_ok"):
        points += 10
        reasons.append("walk-forward positief +10")
    elif proof_details.get("walk_forward_caution"):
        points -= 10
        reasons.append("walk-forward zwak -10")
    elif wf_trades and (wf_pnl <= 0 or wf_wr < 40):
        points -= 5
        reasons.append("walk-forward matig -5")

    paper_score = float(score_info.get("score") or 0.5)
    paper_samples = int(score_info.get("sample_count") or 0)
    if paper_samples == 0:
        points += 10
        reasons.append("paper-score neutraal +10")
    elif paper_score >= 0.55:
        points += 20
        reasons.append("paper-score goed +20")
    elif paper_score >= 0.45:
        points += 10
        reasons.append("paper-score matig +10")

    quality_score = int(quality.get("score") or 0)
    if quality_score >= 80:
        points += 10
        reasons.append("kwaliteit goed +10")
    elif quality_score >= 65:
        points += 5
        reasons.append("kwaliteit voldoende +5")

    spread_pct = float((market_context.get("spread") or {}).get("spread_pct") or 0)
    if spread_pct <= 0.04:
        points += 5
        reasons.append("spread laag +5")

    orderbook = market_context.get("orderbook") or {}
    imbalance = float(orderbook.get("imbalance") or 0)
    if abs(imbalance) <= 0.12:
        points += 2
        reasons.append("orderbook neutraal +2")

    derivatives = market_context.get("derivatives") or {}
    oi_change = derivatives.get("open_interest_change")
    if isinstance(oi_change, (int, float)) and oi_change >= 0:
        points += 3
        reasons.append("open interest stabiel/omhoog +3")

    profile_adjustment = int(strategy_profile.get("confidence_adjustment") or 0)
    if profile_adjustment:
        points += profile_adjustment
        if profile_adjustment > 0:
            reasons.append(f"strategieprofiel +{profile_adjustment}")
        else:
            reasons.append(f"strategieprofiel {profile_adjustment}")

    time_adjustment = int(time_profile.get("confidence_adjustment") or 0)
    if time_adjustment:
        points += time_adjustment
        if time_adjustment > 0:
            reasons.append(f"tijdprofiel +{time_adjustment}")
        else:
            reasons.append(f"tijdprofiel {time_adjustment}")

    return {
        "score": max(0, min(100, points)),
        "reasons": reasons,
    }
