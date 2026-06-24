"""
bot_learning.py - Transparante leerlaag voor de paper trading bot.

Hoe het werkt:
  1. Elke trade-setup krijgt een context (buckets voor trend, RSI, volume, etc.)
  2. Na het sluiten van een trade wordt het resultaat opgeslagen als leer-sample.
  3. Bij een nieuwe setup wordt de score opgezocht op basis van vergelijkbare samples.
  4. Slechte contexten worden pas geblokkeerd na genoeg samples.
  5. Bootstrap vult het geheugen met historische simulaties zodat de bot niet blanco start.
"""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone, timedelta

import pandas as pd

from src.database import get_connection
from src.features import add_all_features
from src.market_regime import classify_market_regime
from src.strategy import scan_alle_strategieen

MIN_SAMPLES_SCORE  = 5
MIN_SAMPLES_BLOCK  = 15
BLOCK_THRESHOLD    = 0.25
BOOTSTRAP_MIN_LIVE = 8     # Bootstrap stopt eerder zodat echte trades sneller tellen

# Backwards-compatible names used by the existing unittest suite.
NEUTRAL_SCORE = 0.5
MIN_BLOCK_SAMPLES = MIN_SAMPLES_BLOCK
BLOCK_SCORE_THRESHOLD = BLOCK_THRESHOLD
BOOTSTRAP_MIN_SAMPLES_PER_ASSET = 20

REGIME_PROFILE_MIN_SAMPLES = 5
REGIME_PROFILE_BLOCK_SAMPLES = 12
REGIME_PROFILE_BLOCK_SCORE = 0.30
REGIME_PROFILE_CAUTION_SCORE = 0.45
REGIME_PROFILE_BOOST_SCORE = 0.65


def _is_db_locked(exc: Exception) -> bool:
    return isinstance(exc, sqlite3.OperationalError) and "locked" in str(exc).lower()


def _rsi_bucket(rsi: float) -> str:
    if pd.isna(rsi): return "unknown"
    if rsi < 30: return "oversold"
    if rsi < 45: return "low"
    if rsi < 60: return "neutral"
    if rsi < 70: return "elevated"
    return "overbought"


def _adx_bucket(adx: float) -> str:
    if pd.isna(adx): return "unknown"
    if adx < 20: return "weak"
    if adx < 35: return "moderate"
    return "strong"


def _rr_bucket(rr: float) -> str:
    if rr < 2.5: return "2.0-2.5"
    if rr < 3.0: return "2.5-3.0"
    return "3.0+"


def _fg_bucket(fg: float) -> str:
    if pd.isna(fg): return "unknown"
    if fg <= 20:    return "extreme_fear"
    if fg <= 44:    return "fear"
    if fg <= 55:    return "neutral"
    if fg <= 74:    return "greed"
    return "extreme_greed"


def _bb_bucket(bb_pct: float) -> str:
    if pd.isna(bb_pct): return "unknown"
    if bb_pct < 0.0:  return "below_band"
    if bb_pct < 0.25: return "lower"
    if bb_pct < 0.75: return "mid"
    if bb_pct < 1.0:  return "upper"
    return "above_band"


def _macd_bucket(macd_line: float, macd_signal: float) -> str:
    if pd.isna(macd_line) or pd.isna(macd_signal): return "unknown"
    if macd_line > 0 and macd_line > macd_signal:  return "bullish"
    if macd_line > 0 and macd_line <= macd_signal: return "bullish_fading"
    if macd_line <= 0 and macd_line > macd_signal: return "bearish_recovering"
    return "bearish"


def _zscore_bucket(zscore: float) -> str:
    if pd.isna(zscore): return "unknown"
    if zscore <= -2.0: return "extreme_oversold"
    if zscore <= -1.0: return "oversold"
    if zscore < 1.0: return "neutral"
    if zscore < 2.0: return "overbought"
    return "extreme_overbought"


def _momentum_bucket(momentum: float) -> str:
    if pd.isna(momentum): return "unknown"
    if momentum <= -0.03: return "strong_bearish"
    if momentum < -0.005: return "bearish"
    if momentum <= 0.005: return "flat"
    if momentum < 0.03: return "bullish"
    return "strong_bullish"


def _feature_volatility_bucket(volatility: float) -> str:
    if pd.isna(volatility): return "unknown"
    if volatility < 0.003: return "low"
    if volatility < 0.010: return "normal"
    if volatility < 0.020: return "high"
    return "extreme"


def _obv_bucket(obv_slope: float) -> str:
    if pd.isna(obv_slope): return "unknown"
    if obv_slope > 0: return "accumulation"
    if obv_slope < 0: return "distribution"
    return "neutral"


def _gap_bucket(gap_pct: float) -> str:
    if pd.isna(gap_pct): return "unknown"
    if gap_pct <= -1.0: return "gap_down_big"
    if gap_pct < -0.2: return "gap_down"
    if gap_pct <= 0.2: return "flat"
    if gap_pct < 1.0: return "gap_up"
    return "gap_up_big"


def _skew_bucket(skew: float) -> str:
    if pd.isna(skew): return "unknown"
    if skew <= -1.0: return "left_tail"
    if skew < -0.25: return "mild_left"
    if skew <= 0.25: return "balanced"
    if skew < 1.0: return "mild_right"
    return "right_tail"


def make_context_key(context: dict) -> str:
    return (
        f"{context.get('asset', '')}|{context.get('strategy', '')}|"
        f"{context.get('trend', '')}|{context.get('rsi_zone', '')}|"
        f"{context.get('volatility', '')}|{context.get('volume', '')}|"
        f"{context.get('adx_zone', '')}|{context.get('rr_zone', '')}|"
        f"{context.get('bb_zone', '')}|{context.get('macd_zone', '')}|"
        f"{context.get('htf_trend', '')}|{context.get('fear_greed_zone', '')}|"
        f"{context.get('momentum_zone', '')}|{context.get('feature_volatility_zone', '')}|"
        f"{context.get('zscore_zone', '')}|{context.get('obv_zone', '')}|"
        f"{context.get('gap_zone', '')}|{context.get('skew_zone', '')}|"
        f"{context.get('session_bucket', '')}|{context.get('weekend_zone', '')}|"
        f"{context.get('regime_label', '')}|{context.get('regime_direction', '')}"
    )


def _timeblock_utc() -> str:
    uur = datetime.now(timezone.utc).hour
    if uur < 6: return "00-06"
    if uur < 12: return "06-12"
    if uur < 18: return "12-18"
    return "18-24"


def build_market_context(
    asset: str,
    signal: dict,
    df: pd.DataFrame,
    analyse: dict | None = None,
    timeframe: str = "300s",
    htf_trend: str = "onduidelijk",
    fear_greed: float = 50.0,
) -> dict:
    df_ind = add_all_features(df)
    last   = df_ind.iloc[-1]

    trend      = (analyse or {}).get("trend", "onduidelijk")
    volatility = (analyse or {}).get("volatiliteit", "normaal")
    volume_st  = (analyse or {}).get("volume_status", "normaal")
    rsi_v      = float(last.get("rsi14", float("nan")))
    adx_v      = float(last.get("adx14", float("nan")))
    bb_pct_v   = float(last.get("bb_pct", float("nan")))
    macd_line_v   = float(last.get("macd_line", float("nan")))
    macd_signal_v = float(last.get("macd_signal", float("nan")))
    momentum_10_v = float(last.get("momentum_10", float("nan")))
    volatility_20_v = float(last.get("volatility_20", float("nan")))
    zscore_20_v = float(last.get("zscore_20", float("nan")))
    obv_slope_v = float(last.get("obv_slope", float("nan")))
    gap_pct_v = float(last.get("gap_pct", float("nan")))
    skew_20_v = float(last.get("skew_20", float("nan")))
    rr         = float(signal.get("risk_reward") or 2.0)
    strategy   = signal.get("strategie", "unknown")
    regime     = signal.get("_market_regime") or classify_market_regime(
        df, htf_trend=htf_trend, fear_greed=fear_greed
    )

    ctx = {
        "asset":           asset,
        "strategy":        strategy,
        "timeframe":       timeframe,
        "trend":           trend,
        "volatility":      volatility,
        "volume":          volume_st,
        "rsi_zone":        _rsi_bucket(rsi_v),
        "adx_zone":        _adx_bucket(adx_v),
        "rr_zone":         _rr_bucket(rr),
        "bb_zone":         _bb_bucket(bb_pct_v),
        "macd_zone":       _macd_bucket(macd_line_v, macd_signal_v),
        "htf_trend":       htf_trend,
        "fear_greed_zone": _fg_bucket(float(fear_greed)),
        "momentum_zone":   _momentum_bucket(momentum_10_v),
        "feature_volatility_zone": _feature_volatility_bucket(volatility_20_v),
        "zscore_zone":     _zscore_bucket(zscore_20_v),
        "obv_zone":        _obv_bucket(obv_slope_v),
        "gap_zone":        _gap_bucket(gap_pct_v),
        "skew_zone":       _skew_bucket(skew_20_v),
        "session_bucket":  last.get("session_bucket", "onbekend"),
        "weekend_zone":    "weekend" if int(last.get("is_weekend", 0) or 0) else "weekday",
        "regime_label":    regime.get("label", "unknown"),
        "regime_score":    regime.get("score", 0),
        "regime_direction": regime.get("direction", "neutral"),
        "timeblock":       _timeblock_utc(),
    }
    ctx["context_key"] = make_context_key(ctx)
    return ctx


def _scorecard_lookup(context_key: str) -> dict | None:
    conn = get_connection()
    row  = conn.execute(
        "SELECT * FROM bot_learning_scorecards WHERE context_key = ?", (context_key,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def _fallback_score(asset: str, strategy: str) -> dict | None:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM bot_learning_scorecards WHERE asset = ? AND strategy = ?",
        (asset, strategy),
    ).fetchall()
    conn.close()
    if not rows:
        return None
    totaal = sum(r["sample_count"] for r in rows)
    if totaal < MIN_SAMPLES_SCORE:
        return None
    gewogen = sum(r["weighted_score"] * r["sample_count"] for r in rows) / totaal
    return {"weighted_score": gewogen, "sample_count": totaal}


def score_setup(context: dict) -> dict:
    key      = context.get("context_key", "")
    asset    = context.get("asset", "")
    strategy = context.get("strategy", "")

    kaart = _scorecard_lookup(key)
    if kaart and kaart["sample_count"] >= MIN_SAMPLES_SCORE:
        score  = kaart["weighted_score"]
        n      = kaart["sample_count"]
        level  = "exact"
        reason = (
            f"Exacte context: {n} vergelijkbare trades, score {score:.0%} "
            f"({kaart.get('avg_result_pct', 0):+.1f}% gem. resultaat)."
        )
    else:
        fallback = _fallback_score(asset, strategy)
        if fallback:
            score  = fallback["weighted_score"]
            n      = fallback["sample_count"]
            level  = "fallback"
            reason = f"Brede context ({asset}+{strategy}): {n} trades, score {score:.0%}."
        else:
            score  = 0.5
            n      = kaart["sample_count"] if kaart else 0
            level  = "neutral"
            reason = f"Onvoldoende data ({n} samples, min {MIN_SAMPLES_SCORE} nodig). Neutraal."

    # Exacte context mag hard blokkeren. Brede fallback-data is nuttig als waarschuwing,
    # maar kan door bootstrap/virtuele samples te snel alle nieuwe trades dichtzetten.
    should_block = (
        level == "exact"
        and n >= MIN_SAMPLES_BLOCK
        and score < BLOCK_THRESHOLD
    )
    return {
        "score":        round(score, 4),
        "sample_count": n,
        "reason":       reason,
        "should_block": should_block,
        "level":        level,
    }


def get_strategy_regime_profile(asset: str, strategy: str, regime_label: str) -> dict:
    """
    Leerprofiel per asset + strategie + marktregime.

    Dit is expres een laag bovenop de exacte context-score. Exacte contexten kunnen
    te weinig samples hebben; dit profiel leert breder: werkt deze strategie op dit
    asset in dit regime gemiddeld wel of niet?
    """
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT source, context_json, outcome, result_pct, weight
        FROM bot_learning_samples
        WHERE asset = ? AND strategy = ?
        ORDER BY id DESC LIMIT 500
        """,
        (asset, strategy),
    ).fetchall()
    conn.close()

    total_weight = 0.0
    weighted_outcome = 0.0
    weighted_pct = 0.0
    samples = 0
    live_samples = 0
    for row in rows:
        try:
            context = json.loads(row["context_json"] or "{}")
        except Exception:
            context = {}
        if (context.get("regime_label") or "unknown") != regime_label:
            continue
        source = row["source"] or ""
        base_weight = float(row["weight"] if row["weight"] is not None else 1.0)
        if source == "bootstrap":
            base_weight = min(base_weight, 0.6)
        elif source == "skipped_virtual":
            base_weight = min(base_weight, 0.5)
        else:
            live_samples += 1
        total_weight += base_weight
        weighted_outcome += float(row["outcome"] or 0.0) * base_weight
        weighted_pct += float(row["result_pct"] or 0.0) * base_weight
        samples += 1

    if samples == 0 or total_weight <= 0:
        return {
            "samples": 0,
            "live_samples": 0,
            "score": 0.5,
            "avg_result_pct": 0.0,
            "action": "neutral",
            "confidence_adjustment": 0,
            "size_cap": None,
            "reason": f"Nog geen profieldata voor {strategy} op {asset} in regime {regime_label}.",
        }

    score = weighted_outcome / total_weight
    avg_pct = weighted_pct / total_weight
    action = "neutral"
    confidence_adjustment = 0
    size_cap = None
    reason = (
        f"Profiel {asset}/{strategy}/{regime_label}: "
        f"{samples} samples, score {score:.0%}, gem. {avg_pct:+.2f}%."
    )

    if samples >= REGIME_PROFILE_BLOCK_SAMPLES and score < REGIME_PROFILE_BLOCK_SCORE:
        action = "block"
        confidence_adjustment = -25
        size_cap = 0.0
        reason += " Slechte combinatie wordt geblokkeerd."
    elif samples >= REGIME_PROFILE_MIN_SAMPLES and score < REGIME_PROFILE_CAUTION_SCORE:
        action = "caution"
        confidence_adjustment = -10
        size_cap = 0.50
        reason += " Zwakke combinatie: alleen klein/voorzichtig."
    elif samples >= REGIME_PROFILE_MIN_SAMPLES and score >= REGIME_PROFILE_BOOST_SCORE and avg_pct > 0:
        action = "boost"
        confidence_adjustment = 10
        size_cap = None
        reason += " Sterke combinatie: confidence krijgt bonus."

    return {
        "samples": samples,
        "live_samples": live_samples,
        "score": round(score, 4),
        "avg_result_pct": round(avg_pct, 4),
        "action": action,
        "confidence_adjustment": confidence_adjustment,
        "size_cap": size_cap,
        "reason": reason,
    }


def get_time_strategy_profile(asset: str, strategy: str, context: dict) -> dict:
    """Leer of deze strategie op dit tijdblok/weekend historisch goed werkt."""
    timeblock = context.get("timeblock") or "unknown"
    session = context.get("session_bucket") or "unknown"
    weekend = context.get("weekend_zone") or "unknown"
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT source, context_json, outcome, result_pct, weight
        FROM bot_learning_samples
        WHERE asset = ? AND strategy = ?
        ORDER BY id DESC LIMIT 500
        """,
        (asset, strategy),
    ).fetchall()
    conn.close()

    total_weight = 0.0
    weighted_outcome = 0.0
    weighted_pct = 0.0
    samples = 0
    for row in rows:
        try:
            sample_context = json.loads(row["context_json"] or "{}")
        except Exception:
            sample_context = {}
        if sample_context.get("timeblock") != timeblock:
            continue
        if sample_context.get("weekend_zone") != weekend:
            continue
        if session != "unknown" and sample_context.get("session_bucket") not in (session, "unknown"):
            continue
        weight = float(row["weight"] if row["weight"] is not None else 1.0)
        if row["source"] == "bootstrap":
            weight = min(weight, 0.6)
        elif row["source"] == "skipped_virtual":
            weight = min(weight, 0.5)
        total_weight += weight
        weighted_outcome += float(row["outcome"] or 0.0) * weight
        weighted_pct += float(row["result_pct"] or 0.0) * weight
        samples += 1

    if samples == 0 or total_weight <= 0:
        return {
            "samples": 0,
            "score": 0.5,
            "avg_result_pct": 0.0,
            "action": "neutral",
            "confidence_adjustment": 0,
            "size_cap": None,
            "reason": f"Nog geen tijddata voor {asset}/{strategy} in {timeblock}/{weekend}.",
        }

    score = weighted_outcome / total_weight
    avg_pct = weighted_pct / total_weight
    action = "neutral"
    confidence_adjustment = 0
    size_cap = None
    reason = (
        f"Tijdprofiel {asset}/{strategy}/{timeblock}/{weekend}: "
        f"{samples} samples, score {score:.0%}, gem. {avg_pct:+.2f}%."
    )
    if samples >= 10 and score < 0.30 and avg_pct < 0:
        action = "block"
        confidence_adjustment = -20
        size_cap = 0.0
        reason += " Slecht tijdblok wordt geblokkeerd."
    elif samples >= 5 and score < 0.42:
        action = "caution"
        confidence_adjustment = -8
        size_cap = 0.50
        reason += " Zwak tijdblok: kleiner handelen."
    elif samples >= 5 and score >= 0.62 and avg_pct > 0:
        action = "boost"
        confidence_adjustment = 5
        reason += " Sterk tijdblok: lichte bonus."

    return {
        "samples": samples,
        "score": round(score, 4),
        "avg_result_pct": round(avg_pct, 4),
        "action": action,
        "confidence_adjustment": confidence_adjustment,
        "size_cap": size_cap,
        "timeblock": timeblock,
        "session_bucket": session,
        "weekend_zone": weekend,
        "reason": reason,
    }


def _update_scorecard(context: dict, outcome: float, result_pct: float):
    key  = context["context_key"]
    nu   = datetime.now(timezone.utc).isoformat()
    conn = get_connection()
    bestaand = conn.execute(
        "SELECT * FROM bot_learning_scorecards WHERE context_key = ?", (key,)
    ).fetchone()
    if bestaand:
        n     = bestaand["sample_count"] + 1
        score = (bestaand["weighted_score"] * bestaand["sample_count"] + outcome) / n
        avg   = (bestaand["avg_result_pct"] * bestaand["sample_count"] + result_pct) / n
        conn.execute(
            "UPDATE bot_learning_scorecards SET sample_count=?, weighted_score=?, avg_result_pct=?, updated_at=? WHERE context_key=?",
            (n, round(score, 6), round(avg, 4), nu, key),
        )
    else:
        conn.execute(
            "INSERT INTO bot_learning_scorecards (context_key, asset, strategy, context_json, sample_count, weighted_score, avg_result_pct, updated_at) VALUES (?, ?, ?, ?, 1, ?, ?, ?)",
            (key, context.get("asset", ""), context.get("strategy", ""), json.dumps(context), round(outcome, 6), round(result_pct, 4), nu),
        )
    conn.commit()
    conn.close()


def record_learning_sample(
    context: dict,
    source: str = "live",
    result_pct: float = 0.0,
    trade_id: int | None = None,
    refresh: bool = True,
) -> int:
    context = dict(context)
    context["context_key"] = context.get("context_key") or make_context_key(context)
    outcome = 1.0 if result_pct > 0 else 0.0
    now = datetime.now(timezone.utc).isoformat()

    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO bot_learning_samples "
        "(created_at, source, asset, strategy, timeframe, context_key, context_json, "
        "outcome, result_pct, result_euro, weight, trade_id, closed_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0.0, 1.0, ?, ?)",
        (
            now,
            source,
            context.get("asset", ""),
            context.get("strategy", "unknown"),
            context.get("timeframe", "300s"),
            context["context_key"],
            json.dumps(context),
            outcome,
            result_pct,
            trade_id,
            now if source == "live" else None,
        ),
    )
    sample_id = cur.lastrowid
    conn.commit()
    conn.close()

    if refresh:
        _update_scorecard(context, outcome, result_pct)
    return sample_id


def refresh_scorecards() -> int:
    conn = get_connection()
    rows = conn.execute(
        "SELECT context_json, outcome, result_pct FROM bot_learning_samples ORDER BY id"
    ).fetchall()
    conn.execute("DELETE FROM bot_learning_scorecards")
    conn.commit()
    conn.close()

    refreshed = 0
    for row in rows:
        try:
            context = json.loads(row["context_json"])
            context["context_key"] = context.get("context_key") or make_context_key(context)
            _update_scorecard(context, float(row["outcome"]), float(row["result_pct"] or 0))
            refreshed += 1
        except Exception:
            continue
    return refreshed


def update_learning_from_closed_trade(trade_id) -> dict:
    # Accepteer zowel een integer trade_id als een volledig trade-dict
    if isinstance(trade_id, dict):
        trade_id = trade_id.get("id")
    trade_id = int(trade_id)
    conn = get_connection()
    trade = conn.execute("SELECT * FROM paper_trades WHERE id = ?", (trade_id,)).fetchone()
    conn.close()

    if not trade:
        return {"ok": False, "reason": f"Trade #{trade_id} niet gevonden."}
    if trade["status"] != "closed":
        return {"ok": False, "reason": f"Trade #{trade_id} is nog open."}

    try:
        context = json.loads(trade["market_context"] or "{}")
    except Exception:
        context = {}

    if not context or "context_key" not in context:
        return {"ok": False, "reason": "Geen context opgeslagen voor deze trade."}

    result_pct = float(trade["result_pct"] or 0)
    outcome    = 1.0 if result_pct > 0 else 0.0

    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT OR IGNORE INTO bot_learning_samples (created_at, source, asset, strategy, timeframe, context_key, context_json, outcome, result_pct, result_euro, weight, trade_id, closed_at) VALUES (?, 'live', ?, ?, ?, ?, ?, ?, ?, ?, 1.0, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), context.get("asset", trade["asset"]),
             context.get("strategy", "unknown"), context.get("timeframe", "300s"),
             context["context_key"], json.dumps(context), outcome, result_pct,
             float(trade["result_euro"] or 0), trade_id, trade["closed_at"]),
        )
        inserted = cur.rowcount > 0
        if not inserted:
            conn.execute(
                """
                UPDATE bot_learning_samples
                SET context_json = ?, outcome = ?, result_pct = ?,
                    result_euro = ?, closed_at = ?
                WHERE source = 'live' AND trade_id = ?
                """,
                (json.dumps(context), outcome, result_pct,
                 float(trade["result_euro"] or 0), trade["closed_at"], trade_id),
            )
        conn.commit()
    finally:
        conn.close()

    if inserted:
        _update_scorecard(context, outcome, result_pct)
    else:
        refresh_scorecards()
    return {"ok": True, "outcome": outcome}


def record_exit_learning_from_closed_trade(trade_id) -> dict:
    """Sla apart op hoe de exit-methode presteerde voor deze trade."""
    if isinstance(trade_id, dict):
        trade_id = trade_id.get("id")
    trade_id = int(trade_id)
    conn = get_connection()
    trade = conn.execute("SELECT * FROM paper_trades WHERE id = ?", (trade_id,)).fetchone()
    conn.close()

    if not trade:
        return {"ok": False, "reason": f"Trade #{trade_id} niet gevonden."}
    if trade["status"] != "closed":
        return {"ok": False, "reason": f"Trade #{trade_id} is nog open."}

    try:
        context = json.loads(trade["market_context"] or "{}")
    except Exception:
        context = {}
    if not context:
        context = {
            "asset": trade["asset"],
            "strategy": trade["strategy"] or "unknown",
            "timeframe": "300s",
        }

    exit_reason = trade["exit_reason"] or "unknown"
    base_strategy = trade["strategy"] or context.get("strategy") or "unknown"
    exit_context = dict(context)
    exit_context["strategy"] = f"exit:{base_strategy}:{exit_reason}"
    exit_context["exit_reason"] = exit_reason
    exit_context["direction"] = trade["direction"]
    exit_context["context_key"] = make_context_key(exit_context) + f"|exit:{exit_reason}"

    result_pct = float(trade["result_pct"] or 0)
    outcome = 1.0 if result_pct > 0 else 0.0
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT OR IGNORE INTO bot_learning_samples "
            "(created_at, source, asset, strategy, timeframe, context_key, context_json, "
            "outcome, result_pct, result_euro, weight, trade_id, closed_at) "
            "VALUES (?, 'exit_review', ?, ?, ?, ?, ?, ?, ?, ?, 0.8, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                trade["asset"],
                exit_context["strategy"],
                exit_context.get("timeframe", "300s"),
                exit_context["context_key"],
                json.dumps(exit_context),
                outcome,
                result_pct,
                float(trade["result_euro"] or 0),
                trade_id,
                trade["closed_at"],
            ),
        )
        inserted = cur.rowcount > 0
        conn.commit()
    finally:
        conn.close()

    if inserted:
        _update_scorecard(exit_context, outcome, result_pct)
    else:
        refresh_scorecards()
    return {"ok": True, "outcome": outcome}


def get_exit_strategy_profile(asset: str, strategy: str, exit_reason: str) -> dict:
    """Leerprofiel voor slimme exits per asset/strategie/exit-type."""
    exit_strategy = f"exit:{strategy or 'unknown'}:{exit_reason or 'unknown'}"
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT outcome, result_pct, result_euro, weight
        FROM bot_learning_samples
        WHERE source = 'exit_review' AND asset = ? AND strategy = ?
        ORDER BY id DESC LIMIT 120
        """,
        (asset, exit_strategy),
    ).fetchall()
    conn.close()

    if not rows:
        return {
            "samples": 0,
            "score": 0.5,
            "avg_result_pct": 0.0,
            "avg_result_euro": 0.0,
            "action": "neutral",
            "reason": f"Nog geen exit-data voor {exit_strategy}.",
        }

    total_weight = sum(float(r["weight"] if r["weight"] is not None else 1.0) for r in rows)
    if total_weight <= 0:
        total_weight = float(len(rows))
    score = sum(float(r["outcome"] or 0.0) * float(r["weight"] if r["weight"] is not None else 1.0) for r in rows) / total_weight
    avg_pct = sum(float(r["result_pct"] or 0.0) * float(r["weight"] if r["weight"] is not None else 1.0) for r in rows) / total_weight
    avg_euro = sum(float(r["result_euro"] or 0.0) * float(r["weight"] if r["weight"] is not None else 1.0) for r in rows) / total_weight

    action = "neutral"
    if len(rows) >= 5 and score < 0.35 and avg_pct < 0:
        action = "avoid"
    elif len(rows) >= 5 and score >= 0.60 and avg_pct >= 0:
        action = "prefer"

    return {
        "samples": len(rows),
        "score": round(score, 4),
        "avg_result_pct": round(avg_pct, 4),
        "avg_result_euro": round(avg_euro, 2),
        "action": action,
        "reason": (
            f"Exitprofiel {exit_strategy}: {len(rows)} samples, "
            f"score {score:.0%}, gem. {avg_pct:+.2f}%."
        ),
    }


def generate_trade_review(trade: dict, exit_reason: str) -> str:
    """Maak een compacte automatische review die later ook voor leren bruikbaar is."""
    entry = float(trade.get("entry_price") or 0)
    sl = float(trade.get("stop_loss") or 0)
    tp = float(trade.get("take_profit") or 0)
    direction = trade.get("direction", "long")
    capital = float(trade.get("capital") or 0)
    risk_pct = abs(entry - sl) / entry * 100 if entry > 0 and sl > 0 else 0.0
    reward_pct = abs(tp - entry) / entry * 100 if entry > 0 and tp > 0 else 0.0
    rr = reward_pct / risk_pct if risk_pct > 0 else 0.0
    risk_euro = capital * risk_pct / 100 if capital > 0 else 0.0
    parts = [
        f"Automatische trade review: exit={exit_reason}, richting={direction}.",
        f"Initieel risico: {risk_pct:.2f}% (~EUR{risk_euro:.2f}); doelafstand: {reward_pct:.2f}%; R/R ~1:{rr:.2f}.",
    ]
    if exit_reason == "sl":
        parts.append("Review: stop-loss geraakt. Controleer of entry te vroeg was, SL te strak stond of regime tegen de trade draaide.")
    elif exit_reason == "tp":
        parts.append("Review: take-profit gehaald. Deze setup/exit-combinatie positief meewegen in het leergeheugen.")
    elif exit_reason == "time_stop":
        parts.append("Review: tijd-stop. De trade maakte te weinig progressie; dit leert of sneller uitstappen beter is dan wachten op SL/TP.")
    elif exit_reason == "regime_flip":
        parts.append("Review: regime draaide tegen de positie. Deze exit leert of vroeg sluiten schade beperkt.")
    else:
        parts.append("Review: handmatige/overige exit. Resultaat wordt wel meegenomen, maar minder specifiek dan SL/TP/regime/tijd-stop.")
    return " ".join(parts)


def _is_short(signal: dict) -> bool:
    return signal.get("richting") == "short"


def _result_pct_for_exit(signal: dict, exit_price: float) -> float:
    entry = signal["entry"]
    if _is_short(signal):
        return (entry - exit_price) / entry * 100
    return (exit_price - entry) / entry * 100


def _simulate_outcome(df: pd.DataFrame, entry_idx: int, signal: dict) -> float | None:
    sl = signal["stop_loss"]
    tp = signal["take_profit"]
    for _, candle in df.iloc[entry_idx + 1: entry_idx + 60].iterrows():
        if _is_short(signal):
            if candle["high"] >= sl:
                return 0.0
            if candle["low"] <= tp:
                return 1.0
        else:
            if candle["low"] <= sl:
                return 0.0
            if candle["high"] >= tp:
                return 1.0
    return None


def should_bootstrap(asset: str, timeframe: str = "300s") -> tuple[bool, str]:
    conn = get_connection()
    boot_samples = conn.execute(
        "SELECT COUNT(*) FROM bot_learning_samples WHERE source='bootstrap' AND asset=? AND timeframe=?",
        (asset, timeframe),
    ).fetchone()[0]
    last_boot = conn.execute(
        "SELECT MAX(run_at) FROM bot_bootstrap_runs WHERE asset=? AND timeframe=?",
        (asset, timeframe),
    ).fetchone()[0]
    conn.close()

    if not last_boot:
        return True, "Nog geen bootstrap uitgevoerd."

    try:
        last_dt = datetime.fromisoformat(last_boot)
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - last_dt
        if age > timedelta(days=30):
            return True, "Laatste bootstrap is ouder dan 30 dagen."
    except Exception:
        return True, "Bootstrap-tijd kon niet worden gelezen."

    if boot_samples < BOOTSTRAP_MIN_SAMPLES_PER_ASSET:
        return True, f"Te weinig bootstrap-samples ({boot_samples})."

    return False, "Recente bootstrap met genoeg samples aanwezig."


def ensure_bootstrap(
    asset: str,
    df: pd.DataFrame,
    timeframe: str = "300s",
    min_rr: float = 2.0,
) -> dict:
    conn = get_connection()
    live_count = conn.execute(
        "SELECT COUNT(*) FROM bot_learning_samples WHERE source='live' AND asset=?", (asset,)
    ).fetchone()[0]
    last_boot = conn.execute(
        "SELECT MAX(run_at) FROM bot_bootstrap_runs WHERE asset=? AND timeframe=?", (asset, timeframe)
    ).fetchone()[0]
    conn.close()

    if live_count >= BOOTSTRAP_MIN_LIVE:
        return {"ran": False, "samples": 0, "reason": f"Genoeg live data ({live_count})."}

    if last_boot:
        try:
            ts = datetime.fromisoformat(last_boot).replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) - ts < timedelta(hours=6):
                return {"ran": False, "samples": 0, "reason": "Recent al gebootstrapt."}
        except Exception:
            pass

    # Verzamel alle samples eerst in geheugen, schrijf daarna in één transactie
    samples = []
    df_ind_cache = add_all_features(df)

    for i in range(60, len(df) - 60, 10):  # stap 10 i.p.v. 5 = sneller
        try:
            setups = scan_alle_strategieen(df.iloc[:i], min_rr=min_rr)
        except Exception:
            continue
        for setup in setups:
            outcome = _simulate_outcome(df, i, setup)
            if outcome is None:
                continue
            exit_price = setup["take_profit"] if outcome == 1.0 else setup["stop_loss"]
            result_pct = _result_pct_for_exit(setup, exit_price)
            try:
                last      = df_ind_cache.iloc[i - 1]
                rsi_v     = float(last.get("rsi14", float("nan")))
                adx_v     = float(last.get("adx14", float("nan")))
                bb_pct_v  = float(last.get("bb_pct", float("nan")))
                macd_l_v  = float(last.get("macd_line", float("nan")))
                macd_s_v  = float(last.get("macd_signal", float("nan")))
                close_v   = float(last.get("close", float("nan")))
                sma20_v   = float(last.get("sma20", float("nan")))
                sma50_v   = float(last.get("sma50", float("nan")))
                momentum_10_v = float(last.get("momentum_10", float("nan")))
                volatility_20_v = float(last.get("volatility_20", float("nan")))
                zscore_20_v = float(last.get("zscore_20", float("nan")))
                obv_slope_v = float(last.get("obv_slope", float("nan")))
                gap_pct_v = float(last.get("gap_pct", float("nan")))
                skew_20_v = float(last.get("skew_20", float("nan")))
                session_v = last.get("session_bucket", "onbekend")
                weekend_v = int(last.get("is_weekend", 0) or 0)
            except Exception:
                rsi_v = adx_v = bb_pct_v = macd_l_v = macd_s_v = float("nan")
                close_v = sma20_v = sma50_v = float("nan")
                momentum_10_v = volatility_20_v = zscore_20_v = float("nan")
                obv_slope_v = gap_pct_v = skew_20_v = float("nan")
                session_v = "onbekend"
                weekend_v = 0

            # Bepaal de werkelijke trend op dit punt in de data
            if not (pd.isna(sma20_v) or pd.isna(sma50_v) or pd.isna(close_v)):
                if close_v > sma20_v > sma50_v:
                    trend_bs = "uptrend"
                elif close_v < sma20_v < sma50_v:
                    trend_bs = "downtrend"
                else:
                    trend_bs = "onduidelijk"
            else:
                trend_bs = "onduidelijk"

            ctx = {
                "asset":           asset,
                "strategy":        setup.get("strategie", "unknown"),
                "timeframe":       timeframe,
                "trend":           trend_bs,
                "volatility":      "normaal",
                "volume":          "normaal",
                "rsi_zone":        _rsi_bucket(rsi_v),
                "adx_zone":        _adx_bucket(adx_v),
                "rr_zone":         _rr_bucket(float(setup.get("risk_reward") or 2.0)),
                "bb_zone":         _bb_bucket(bb_pct_v),
                "macd_zone":       _macd_bucket(macd_l_v, macd_s_v),
                "htf_trend":       "onduidelijk",  # Niet beschikbaar tijdens bootstrap
                "fear_greed_zone": "neutral",       # Niet beschikbaar tijdens bootstrap
                "momentum_zone":   _momentum_bucket(momentum_10_v),
                "feature_volatility_zone": _feature_volatility_bucket(volatility_20_v),
                "zscore_zone":     _zscore_bucket(zscore_20_v),
                "obv_zone":        _obv_bucket(obv_slope_v),
                "gap_zone":        _gap_bucket(gap_pct_v),
                "skew_zone":       _skew_bucket(skew_20_v),
                "session_bucket":  session_v,
                "weekend_zone":    "weekend" if weekend_v else "weekday",
                "timeblock":       "12-18",
            }
            ctx["context_key"] = make_context_key(ctx)
            samples.append((ctx, outcome, round(result_pct, 4)))

    # Schrijf alles in een korte transactie. Bij SQLite kan web + bot tegelijk
    # schrijven; retry voorkomt dat een tijdelijke lock de bootstrap laat falen.
    aangemaakt = 0
    nu = datetime.now(timezone.utc).isoformat()
    for attempt in range(4):
        conn = get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            for ctx, outcome, result_pct in samples:
                conn.execute(
                    "INSERT OR IGNORE INTO bot_learning_samples "
                    "(created_at, source, asset, strategy, timeframe, context_key, context_json, "
                    "outcome, result_pct, result_euro, weight, trade_id, closed_at) "
                    "VALUES (?, 'bootstrap', ?, ?, ?, ?, ?, ?, ?, 0.0, 0.7, NULL, NULL)",
                    (nu, asset, ctx["strategy"], timeframe, ctx["context_key"],
                     json.dumps(ctx), outcome, result_pct),
                )
                if conn.execute("SELECT changes()").fetchone()[0] > 0:
                    aangemaakt += 1
            conn.execute(
                "INSERT INTO bot_bootstrap_runs "
                "(run_at, asset, timeframe, samples_created, live_closed_count, reason) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (nu, asset, timeframe, aangemaakt, live_count, f"live={live_count}"),
            )
            conn.commit()
            conn.close()
            break
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            conn.close()
            if not _is_db_locked(e):
                raise
            if attempt == 3:
                return {"ran": False, "samples": 0, "reason": "Database tijdelijk bezet; bootstrap overgeslagen."}
            aangemaakt = 0
            time.sleep(0.75 * (attempt + 1))

    # Scorecards bijwerken buiten de hoofdtransactie
    for ctx, outcome, result_pct in samples:
        try:
            _update_scorecard(ctx, outcome, result_pct)
        except Exception:
            pass
    return {"ran": True, "samples": aangemaakt, "reason": f"{aangemaakt} samples toegevoegd."}


def mark_api_health(asset: str, status: str, message: str = ""):
    try:
        conn = get_connection()
        conn.execute(
            "INSERT OR REPLACE INTO bot_api_health (asset, checked_at, status, message) VALUES (?, ?, ?, ?)",
            (asset, datetime.now(timezone.utc).isoformat(), status, message[:500]),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


# Speciale sleutel in bot_api_health die de algehele liveness van de bot bijhoudt.
HEARTBEAT_KEY = "_heartbeat"


def mark_heartbeat(message: str = "cyclus voltooid") -> None:
    """Registreer dat een bot-cyclus succesvol is afgerond (liveness-signaal)."""
    mark_api_health(HEARTBEAT_KEY, "alive", message)


def get_health_overview() -> dict:
    """
    Geef de actuele gezondheidsstatus terug voor dashboard/Telegram.

    Returns: {
      "heartbeat": {"checked_at", "age_seconds", "status", "message"} | None,
      "assets": [{"asset", "checked_at", "age_seconds", "status", "message"}, ...],
    }
    De heartbeat-rij wordt apart gehaald; overige rijen zijn per-asset databronnen.
    """
    overview: dict = {"heartbeat": None, "assets": []}
    try:
        conn = get_connection()
        rows = conn.execute(
            "SELECT asset, checked_at, status, message FROM bot_api_health ORDER BY asset"
        ).fetchall()
        conn.close()
    except Exception:
        return overview

    now = datetime.now(timezone.utc)

    def _age(checked_at: str) -> float | None:
        try:
            return (now - pd.to_datetime(checked_at, utc=True).to_pydatetime()).total_seconds()
        except Exception:
            return None

    for row in rows:
        entry = {
            "asset": row["asset"],
            "checked_at": row["checked_at"],
            "age_seconds": _age(row["checked_at"]),
            "status": row["status"],
            "message": row["message"],
        }
        if row["asset"] == HEARTBEAT_KEY:
            overview["heartbeat"] = entry
        else:
            overview["assets"].append(entry)
    return overview


def latest_candle_is_fresh(df: pd.DataFrame, max_age_seconds: int = 600) -> tuple[bool, str]:
    if df.empty:
        return False, "Geen candles ontvangen."
    try:
        laatste  = pd.to_datetime(df["timestamp"].iloc[-1], utc=True)
        leeftijd = (datetime.now(timezone.utc) - laatste).total_seconds()
        if leeftijd > max_age_seconds:
            return False, f"Laatste candle is {leeftijd:.0f}s oud (max {max_age_seconds}s)."
        return True, f"Vers ({leeftijd:.0f}s geleden)."
    except Exception as e:
        return False, f"Tijdstempel fout: {e}"


def record_decision(
    asset: str,
    strategy: str | None,
    action: str,
    reason: str,
    context: dict | None = None,
    score_info: dict | None = None,
    trade_id: int | None = None,
    details: dict | None = None,
):
    try:
        import sqlite3
        from src.database import DB_PATH

        conn = sqlite3.connect(DB_PATH, timeout=0.5)
        conn.execute("PRAGMA busy_timeout = 500")
        conn.execute(
            "INSERT INTO bot_learning_decisions (created_at, asset, strategy, action, score, sample_count, context_key, context_json, reason, details_json, trade_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(), asset, strategy, action,
                score_info.get("score") if score_info else None,
                score_info.get("sample_count", 0) if score_info else 0,
                (context or {}).get("context_key"),
                json.dumps(context or {}),
                reason[:1000],
                json.dumps(details or {}),
                trade_id,
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _simulate_setup_result_from_candles(df: pd.DataFrame, setup: dict) -> float | None:
    entry = float(setup.get("entry") or 0)
    sl = float(setup.get("stop_loss") or 0)
    tp = float(setup.get("take_profit") or 0)
    direction = setup.get("richting", "long")
    if entry <= 0 or sl <= 0 or tp <= 0:
        return None
    for _, candle in df.iterrows():
        high = float(candle["high"])
        low = float(candle["low"])
        if direction == "long":
            if low <= sl:
                return (sl - entry) / entry * 100
            if high >= tp:
                return (tp - entry) / entry * 100
        else:
            if high >= sl:
                return (entry - sl) / entry * 100
            if low <= tp:
                return (entry - tp) / entry * 100
    return None


def _simulate_missed_trade_scenarios(df: pd.DataFrame, setup: dict) -> list[dict]:
    """Bekijk gemiste setup met meerdere horizons en alternatieve SL/TP-afstanden."""
    scenarios: list[dict] = []
    entry = float(setup.get("entry") or 0)
    sl = float(setup.get("stop_loss") or 0)
    tp = float(setup.get("take_profit") or 0)
    direction = setup.get("richting", "long")
    if entry <= 0 or sl <= 0 or tp <= 0:
        return scenarios
    risk = abs(entry - sl)
    if risk <= 0:
        return scenarios

    variants = [
        ("origineel", sl, tp),
        ("tp_1_5r", entry - risk if direction == "long" else entry + risk,
         entry + 1.5 * risk if direction == "long" else entry - 1.5 * risk),
        ("tp_1r", entry - risk if direction == "long" else entry + risk,
         entry + risk if direction == "long" else entry - risk),
        ("sl_ruimer_1_25r", entry - 1.25 * risk if direction == "long" else entry + 1.25 * risk, tp),
        ("sl_krapper_0_75r", entry - 0.75 * risk if direction == "long" else entry + 0.75 * risk, tp),
    ]
    for horizon in (5, 10, 20, 40):
        future = df.head(horizon)
        if len(future) < 3:
            continue
        for name, scenario_sl, scenario_tp in variants:
            scenario_setup = dict(setup)
            scenario_setup["stop_loss"] = scenario_sl
            scenario_setup["take_profit"] = scenario_tp
            result_pct = _simulate_setup_result_from_candles(future, scenario_setup)
            if result_pct is None:
                close = float(future.iloc[-1]["close"])
                result_pct = (
                    (close - entry) / entry * 100
                    if direction == "long"
                    else (entry - close) / entry * 100
                )
                exit_type = "horizon_close"
            else:
                exit_type = "tp" if result_pct > 0 else "sl"
            scenarios.append({
                "horizon": horizon,
                "variant": name,
                "result_pct": round(result_pct, 4),
                "exit_type": exit_type,
            })
    return scenarios


def evaluate_skipped_setups(asset: str, df: pd.DataFrame, max_rows: int = 25) -> int:
    """
    False-positive geheugen: bekijk geweigerde setups achteraf.
    Als een geweigerde setup later TP/SL zou hebben geraakt, wordt dat als
    skipped_virtual sample opgeslagen. Zo leert de bot ook van gemiste kansen.
    """
    if df is None or df.empty or "timestamp" not in df.columns:
        return 0
    candles = df.copy()
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True, errors="coerce")
    # Sorteer chronologisch zodat .head(N) hieronder de eerstvolgende candles ná de skip
    # pakt, niet de oorspronkelijke (mogelijk ongesorteerde) rij-volgorde.
    candles = candles.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT id, created_at, strategy, context_json, details_json
        FROM bot_learning_decisions
        WHERE asset = ? AND action = 'skipped'
        ORDER BY id DESC LIMIT ?
        """,
        (asset, max_rows),
    ).fetchall()
    bijgewerkt = 0
    for row in rows:
        try:
            details = json.loads(row["details_json"] or "{}")
            if details.get("false_positive_checked"):
                continue
            setup = details.get("setup")
            if not setup:
                continue
            context = json.loads(row["context_json"] or "{}")
            created_at = pd.to_datetime(row["created_at"], utc=True, errors="coerce")
            future = candles[candles["timestamp"] > created_at].head(40)
            if len(future) < 3:
                continue
            scenarios = _simulate_missed_trade_scenarios(future, setup)
            if not scenarios:
                continue
            original = next((s for s in scenarios if s["variant"] == "origineel" and s["horizon"] == 40), scenarios[0])
            for scenario in scenarios:
                scenario_context = dict(context)
                scenario_context["missed_horizon"] = scenario["horizon"]
                scenario_context["missed_variant"] = scenario["variant"]
                scenario_context["missed_exit_type"] = scenario["exit_type"]
                scenario_context["context_key"] = (
                    (context.get("context_key") or make_context_key(context))
                    + f"|missed:{scenario['horizon']}:{scenario['variant']}"
                )
                record_learning_sample(
                    scenario_context,
                    source="skipped_virtual",
                    result_pct=float(scenario["result_pct"]),
                    trade_id=None,
                    refresh=True,
                )
            details["false_positive_checked"] = True
            details["virtual_result_pct"] = round(float(original["result_pct"]), 4)
            details["virtual_outcome"] = "win" if float(original["result_pct"]) > 0 else "loss"
            details["missed_trade_scenarios"] = scenarios[:20]
            conn.execute(
                "UPDATE bot_learning_decisions SET details_json = ? WHERE id = ?",
                (json.dumps(details), row["id"]),
            )
            bijgewerkt += 1
        except Exception:
            continue
    conn.commit()
    conn.close()
    return bijgewerkt


def get_learning_overview(limit: int = 5) -> dict:
    conn = get_connection()
    total_samples = conn.execute("SELECT COUNT(*) FROM bot_learning_samples").fetchone()[0]
    live_samples  = conn.execute("SELECT COUNT(*) FROM bot_learning_samples WHERE source='live'").fetchone()[0]
    boot_samples  = conn.execute("SELECT COUNT(*) FROM bot_learning_samples WHERE source='bootstrap'").fetchone()[0]
    total_cards   = conn.execute("SELECT COUNT(*) FROM bot_learning_scorecards").fetchone()[0]
    beste = conn.execute(
        "SELECT context_key, strategy, weighted_score, sample_count, avg_result_pct FROM bot_learning_scorecards WHERE sample_count >= ? ORDER BY weighted_score DESC LIMIT ?",
        (MIN_SAMPLES_SCORE, limit),
    ).fetchall()
    slechtste = conn.execute(
        "SELECT context_key, strategy, weighted_score, sample_count, avg_result_pct FROM bot_learning_scorecards WHERE sample_count >= ? ORDER BY weighted_score ASC LIMIT ?",
        (MIN_SAMPLES_SCORE, limit),
    ).fetchall()
    beslissingen = conn.execute(
        "SELECT created_at, asset, strategy, action, score, sample_count, reason FROM bot_learning_decisions ORDER BY created_at DESC LIMIT ?",
        (limit * 4,),
    ).fetchall()
    api_health = conn.execute(
        "SELECT asset, status, checked_at, message FROM bot_api_health ORDER BY checked_at DESC"
    ).fetchall()
    avg_score_row = conn.execute(
        "SELECT AVG(weighted_score) FROM bot_learning_scorecards WHERE sample_count >= ?",
        (MIN_SAMPLES_SCORE,),
    ).fetchone()
    alle_scorecards = conn.execute(
        "SELECT context_key, asset, strategy, weighted_score, sample_count, avg_result_pct "
        "FROM bot_learning_scorecards ORDER BY sample_count DESC"
    ).fetchall()
    conn.close()

    avg_score = float(avg_score_row[0]) if avg_score_row and avg_score_row[0] is not None else 0.5
    alle_kaarten = [dict(r) for r in alle_scorecards]
    assets     = sorted({r["asset"]    for r in alle_kaarten if r.get("asset")})
    strategies = sorted({r["strategy"] for r in alle_kaarten if r.get("strategy")})

    return {
        # Tellers
        "total_samples":     total_samples,
        "live_samples":      live_samples,
        "bootstrap_samples": boot_samples,
        "total_scorecards":  total_cards,
        "avg_score":         avg_score,
        # Scorecards
        "beste_condities":     [dict(r) for r in beste],
        "slechtste_condities": [dict(r) for r in slechtste],
        "scorecards":          alle_kaarten,
        "assets":              assets,
        "strategies":          strategies,
        # Beslissingen & API
        "recente_beslissingen": [dict(r) for r in beslissingen],
        "api_health":           [dict(r) for r in api_health],
    }


def get_regime_strategy_stats(limit: int = 20) -> list[dict]:
    """Winrate/P&L per strategie en regime op basis van gesloten paper trades."""
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT strategy, market_context, result_pct, result_euro
        FROM paper_trades
        WHERE source='auto' AND status='closed'
          AND market_context IS NOT NULL
        """
    ).fetchall()
    conn.close()
    stats: dict[tuple[str, str], dict] = {}
    for row in rows:
        try:
            context = json.loads(row["market_context"] or "{}")
        except Exception:
            context = {}
        strategy = row["strategy"] or context.get("strategy") or "unknown"
        regime = context.get("regime_label") or "unknown"
        key = (strategy, regime)
        item = stats.setdefault(key, {
            "strategy": strategy,
            "regime": regime,
            "trades": 0,
            "wins": 0,
            "pnl_euro": 0.0,
            "avg_pct": 0.0,
        })
        result_pct = float(row["result_pct"] or 0)
        item["trades"] += 1
        item["wins"] += 1 if result_pct > 0 else 0
        item["pnl_euro"] += float(row["result_euro"] or 0)
        item["avg_pct"] += result_pct
    result = []
    for item in stats.values():
        if item["trades"]:
            item["winrate"] = round(item["wins"] / item["trades"] * 100, 1)
            item["avg_pct"] = round(item["avg_pct"] / item["trades"], 4)
            item["pnl_euro"] = round(item["pnl_euro"], 2)
            result.append(item)
    return sorted(result, key=lambda x: (x["trades"], x["pnl_euro"]), reverse=True)[:limit]


def reset_learning(bevestig: bool = False, asset: str | None = None, reason: str = "") -> dict | int:
    conn = get_connection()

    if asset is not None:
        deleted = conn.execute(
            "DELETE FROM bot_learning_samples WHERE asset = ?", (asset,)
        ).rowcount
        conn.execute("DELETE FROM bot_learning_scorecards WHERE asset = ?", (asset,))
        conn.execute("DELETE FROM bot_learning_decisions WHERE asset = ?", (asset,))
        conn.execute("DELETE FROM bot_bootstrap_runs WHERE asset = ?", (asset,))
        conn.commit()
        conn.close()
        return deleted

    if not bevestig:
        conn.close()
        return {"ok": False, "reason": "Zet bevestig=True om het leergeheugen te wissen."}

    for tabel in ["bot_learning_samples", "bot_learning_scorecards", "bot_learning_decisions", "bot_bootstrap_runs"]:
        conn.execute(f"DELETE FROM {tabel}")
    conn.commit()
    conn.close()
    return {"ok": True, "reason": "Leergeheugen volledig gewist."}
