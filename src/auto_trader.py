"""
auto_trader.py — Automatische paper trading bot met discipline en leergedrag.

De bot handelt UITSLUITEND met nep-geld. Er worden NOOIT echte orders geplaatst.

Discipline-regels (behandelt het alsof het echt geld is):
  - Max 1% risico per trade van het gesimuleerde kapitaal
  - Ruimere paper-leerstand met 15% dagverlies-noodrem
  - Max 5 gelijktijdige open trades
  - Nooit twee trades in hetzelfde asset tegelijk
  - Korte cooling-periode na 3 opeenvolgende verliezen

Leergedrag:
  - Houdt per context bij welke condities leiden tot winst of verlies
  - Scoort nieuwe setups op basis van vergelijkbare paper-resultaten
  - Past minimum R/R aan op basis van recente prestaties
  - Onthoudt welke marktomstandigheden herhaaldelijk slecht uitpakten
"""
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime, timezone, timedelta

import pandas as pd

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

from src.data_loader import fetch_live_crypto_candles, fetch_fear_greed
from src.backtest import run_backtest, run_walk_forward_backtest
from src.external_market_data import fetch_market_context
from src.notifier import (
    melding_trade_geopend,
    melding_take_profit,
    melding_stop_loss,
    melding_verbindingsfout,
    melding_dagrapport,
    melding_drawdown_limiet,
)
from src.strategy import generate_signal, analyze_market, scan_alle_strategieen, STRATEGIE_NAMEN
from src.liquidations import haal_liquidatie_zones, heeft_coinglass_key
from src.indicators import add_all_indicators
from src.market_regime import classify_market_regime, strategy_allowed_in_regime
from src.news_filter import news_watch_snapshot, should_block_new_trades
from src.meta_label import meta_label_score
from src.llm_agents import (
    is_enabled as llm_is_enabled,
    run_panel as llm_run_panel,
    bounded_min_confidence_delta,
)
from src.markets import (
    get_markets,
    is_satellite,
    SATELLITE_ASSET_PARAMS,
    SATELLITE_STRATEGY_PARAMS,
    SATELLITE_MARKT_MODUS,
)
from src.config import (
    RISICO_PCT,
    MAX_DAGVERLIES,
    SOFT_DAGVERLIES,
    MAX_OPEN_TRADES,
    CANDLE_SECONDS,
    MIN_RR_START,
    MIN_CONFIDENCE_SCORE,
    ENABLED_STRATEGIES,
)
from src.setup_confidence import calculate_confidence
from src.trade_quality import score_trade_quality
from src.trading_costs import round_trip_cost_euro
from src.bot_learning import (
    build_market_context,
    ensure_bootstrap,
    latest_candle_is_fresh,
    mark_api_health,
    mark_heartbeat,
    record_decision,
    evaluate_skipped_setups,
    get_strategy_regime_profile,
    get_time_strategy_profile,
    get_exit_strategy_profile,
    generate_trade_review,
    record_exit_learning_from_closed_trade,
    score_setup,
    update_learning_from_closed_trade,
)
from src.database import (
    init_database,
    add_paper_trade,
    close_paper_trade,
    get_trade_by_id,
    get_auto_trades,
    bot_log as db_log,
    get_bot_learning,
    save_bot_learning,
    save_daily_summary,
    get_connection,
    update_trade_sl,
    update_trade_partial_tp,
    get_latest_llm_advice,
)

AMSTERDAM = ZoneInfo("Europe/Amsterdam")

# Cache voor hogere-timeframe trend (per asset, per cyclus ververst)
_htf_cache: dict[str, dict] = {}
_proof_cache: dict[tuple, dict] = {}

# Configureerbare marktlijst (env TRADEAI_MARKETS, default = core + majors).
MARKETS = get_markets()
# Assets die onderling sterk gecorreleerd zijn (BTC-richting trekt de rest mee).
GECORRELEERDE_ASSETS = frozenset(MARKETS)
BTC_MARKT: dict = {"mode": "unknown", "reason": "Nog niet bepaald."}

# ── Discipline-instellingen ────────────────────────────────────────────────────
# RISICO_PCT, MAX_DAGVERLIES, SOFT_DAGVERLIES, MAX_OPEN_TRADES, CANDLE_SECONDS,
# MIN_RR_START en MIN_CONFIDENCE_SCORE komen nu uit src/config.py (env-overschrijfbaar).
STARTKAPITAAL          = 10_000.0   # Gesimuleerd startkapitaal (euro)
COOLING_MINUTEN        = 30         # Langere cooling na 3 opeenvolgende verliezen
MIN_TRADES_PAUZEER     = 20         # Strategie pas pauzeren na minstens dit aantal trades
CANDLE_LIMIT           = 300        # 300 × 5min ≈ 25 uur aan data voor indicatoren
MIN_BACKTEST_TRADES    = 5          # Minimaal recente trades voor volledige strategie-gate
WF_GENERALIZATION_RATIO = 0.70      # Out-of-sample winrate moet ≥70% van in-sample zijn
MIN_BACKTEST_WINRATE   = 40.0       # Netto winrate na kosten
MIN_BACKTEST_PNL       = 0.0        # Netto P&L moet positief zijn
MIN_PAPER_SCORE        = 0.50       # Zodra genoeg samples bestaan moet paper-score positief zijn
HARD_BLOCK_PAPER_SCORE = 0.20       # Alleen extreem slechte paper-score blokkeert hard
EXPLORATION_CONFIDENCE_SCORE = 65   # Kleine test-trades bij twijfelachtige maar niet slechte setup
PANIC_LONG_CONFIDENCE_SCORE = 80     # Alleen sterke BTC long/reversal setups in extreme fear
MAX_PANIC_EXPLORATION_SIZE_MULTIPLIER = 0.10
RANGE_STRATEGIES       = frozenset({"range_long", "range_short"})
PANIC_LONG_STRATEGIES   = frozenset({"trend_long", "pullback", "breakout", "macd_bull_div", "range_long"})
MIN_EXPECTED_NET_REWARD_EURO = 1.50
MIN_REWARD_COST_RATIO = 1.30
MIN_AGENT_COMMITTEE_SCORE = 70
MIN_AGENT_PASSES = 4
MAX_CANDLES_OPEN       = 18         # Tijd-stop: na ~90 min zonder resultaat eruit
MAX_CONFIDENCE_SIZE_MULTIPLIER = 1.25
MAX_EXPLORATION_SIZE_MULTIPLIER = 0.25

# Fractional-Kelly sizing: schaal de inzet naar de werkelijke recente edge.
MIN_KELLY_TRADES = 15          # Minimaal aantal gesloten trades voor een betrouwbare Kelly
KELLY_FRACTION = 0.25          # Kwart-Kelly: groei behouden, drawdowns beperkt
KELLY_MIN_MULTIPLIER = 0.25    # Ondergrens van de inzet-schaal
MAX_PORTFOLIO_FRACTION = 0.25  # Eén positie nooit groter dan 25% van het saldo

# Confidence schaalt de normale risicogebaseerde positie beperkt op/af.
# De 1%-risicoregel blijft de basis; deze factor maakt goede setups iets groter,
# maar voorkomt dat "zekerheid" wordt vertaald naar roekeloos groot inzetten.
CONFIDENCE_SIZE_STEPS = (
    (93, 1.25, "zeer hoge confidence"),
    (85, 1.00, "hoge confidence"),
    (75, 0.50, "minimale confidence"),
)

ASSET_PARAMS = {
    "BTC-USD": {
        "min_confidence": 75,
        "risk_multiplier": 0.80,
        "min_backtest_trades": 1,
        "min_regime_score": 55,
        "walk_forward_min_trades": 1,
        "walk_forward_min_winrate": 40.0,
        "max_candles_open": 18,
    },
    "ETH-USD": {
        "min_confidence": 72,
        "risk_multiplier": 0.90,
        "min_backtest_trades": 1,
        "min_regime_score": 50,
        "confidence_offset": -2,
        "walk_forward_min_trades": 1,
        "walk_forward_min_winrate": 38.0,
        "max_candles_open": 18,
    },
    "SOL-USD": {
        "min_confidence": 75,
        "risk_multiplier": 0.70,
        "min_backtest_trades": 1,
        "min_regime_score": 60,
        "size_cap": 0.25,
        "sl_multiplier": 1.18,
        "walk_forward_min_trades": 1,
        "walk_forward_min_winrate": 42.0,
        "max_candles_open": 12,
    },
}

ASSET_STRATEGY_PARAMS = {
    "BTC-USD": {
        "trend_long": {"min_regime_score": 60},
        "trend_short": {"min_regime_score": 65, "require_htf": "downtrend"},
        "breakout": {"min_regime_score": 60},
        "breakdown": {"min_regime_score": 65, "require_htf": "downtrend"},
        "range_long": {"size_cap": 0.25},
        "range_short": {"size_cap": 0.25},
    },
    "ETH-USD": {
        "pullback": {"confidence_offset": -3},
        "pullback_short": {"confidence_offset": -3},
        "macd_bull_div": {"size_cap": 0.50},
        "macd_bear_div": {"size_cap": 0.50},
    },
    "SOL-USD": {
        "*": {"size_cap": 0.25, "min_regime_score": 60, "sl_multiplier": 1.18},
        "breakout": {"require_btc_mode": "active"},
        "breakdown": {"require_btc_mode": "active"},
    },
}

ALT_MARKT_MODUS = {
    "BTC-USD": "active",
    "ETH-USD": "btc_confirmed",
    "SOL-USD": "observe_or_explore",
}

# Vul voorzichtige satelliet-defaults aan voor extra (niet-core) majors uit de
# configureerbare marktlijst, zodat een toegevoegde munt meteen veilige
# parameters heeft zonder handmatig tunen.
for _markt in MARKETS:
    if is_satellite(_markt):
        ASSET_PARAMS.setdefault(_markt, dict(SATELLITE_ASSET_PARAMS))
        ASSET_STRATEGY_PARAMS.setdefault(_markt, dict(SATELLITE_STRATEGY_PARAMS))
        ALT_MARKT_MODUS.setdefault(_markt, SATELLITE_MARKT_MODUS)


def _fetch_fear_greed_safe(timeout: float = 5.0) -> dict:
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fetch_fear_greed)
    try:
        return future.result(timeout=timeout)
    except TimeoutError:
        return {"value": 50, "label": "Neutral"}
    except Exception:
        return {"value": 50, "label": "Neutral"}
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("auto_trader")


def _get_htf_trend(asset: str) -> str:
    """
    Haal de hogere-timeframe (1u) trend op voor een asset.
    Resultaat wordt per cyclus gecached om extra API-calls te beperken.
    Geeft 'uptrend', 'downtrend' of 'onduidelijk' terug.
    """
    if asset in _htf_cache:
        return _htf_cache[asset]
    trend = "onduidelijk"
    try:
        df_1h = fetch_live_crypto_candles(asset, 3600, 100)
        df_1h = add_all_indicators(df_1h)
        last  = df_1h.iloc[-1]
        if last["close"] > last["sma20"] and last["sma20"] > last["sma50"]:
            trend = "uptrend"
        elif last["close"] < last["sma20"] and last["sma20"] < last["sma50"]:
            trend = "downtrend"
    except Exception:
        pass
    _htf_cache[asset] = trend
    return trend


def _bepaal_btc_marktmodus(regime: dict, htf_trend: str, fear_greed: float, external_context: dict) -> dict:
    """BTC is de marktleider; deze modus bepaalt hoe voorzichtig ETH/SOL mogen zijn."""
    label = (regime or {}).get("label", "unknown")
    score = int((regime or {}).get("score") or 0)
    spread_pct = float(((external_context or {}).get("spread") or {}).get("spread_pct") or 0)

    if label in {"panic", "euphoria", "unknown"}:
        mode = "blocked"
        reason = f"BTC-regime {label}; alts tijdelijk observeren."
    elif fear_greed <= 20 or fear_greed >= 80:
        mode = "blocked"
        reason = f"BTC sentiment extreem ({fear_greed:.0f}); alts tijdelijk observeren."
    elif spread_pct > 0.08:
        mode = "blocked"
        reason = f"BTC spread te hoog ({spread_pct:.3f}%); alts tijdelijk observeren."
    elif htf_trend == "uptrend" and label in {"trend_up", "breakout_up", "range", "squeeze"}:
        mode = "active"
        reason = f"BTC gezond: {label}, HTF {htf_trend}, F&G {fear_greed:.0f}."
    elif htf_trend == "downtrend" and label in {"trend_down", "breakout_down"}:
        mode = "cautious"
        reason = f"BTC neerwaarts maar ordelijk: {label}; alleen kleine/short-gerichte alt setups."
    elif label in {"range", "squeeze", "reversal"} and score >= 50:
        mode = "cautious"
        reason = f"BTC neutraal: {label}; ETH alleen voorzichtig, SOL observeren."
    else:
        mode = "blocked"
        reason = f"BTC-context onduidelijk ({label}, HTF {htf_trend}); alts observeren."

    return {
        "mode": mode,
        "reason": reason,
        "regime": label,
        "score": score,
        "htf_trend": htf_trend,
        "fear_greed": fear_greed,
        "spread_pct": spread_pct,
    }


def _mag_asset_in_btc_first_modus(asset: str) -> tuple[bool, str]:
    """Pas BTC-first marktmodus toe voordat ETH/SOL mogen handelen."""
    if asset == "BTC-USD":
        return True, "BTC is hoofdmarkt."

    mode = BTC_MARKT.get("mode", "unknown")
    reason = BTC_MARKT.get("reason", "BTC-marktmodus onbekend.")

    if mode in {"unknown", "blocked"}:
        return False, f"BTC-first: {asset} observeert. {reason}"
    # 'observe_or_explore'-satellieten (SOL + extra majors) alleen bij actieve BTC-markt.
    if ALT_MARKT_MODUS.get(asset) == "observe_or_explore" and mode != "active":
        return False, f"BTC-first: {asset} alleen bij actieve BTC-markt. {reason}"
    return True, f"BTC-first ok: {reason}"


def _asset_strategy_rule(asset: str, strategy: str) -> dict:
    rules = dict(ASSET_STRATEGY_PARAMS.get(asset, {}).get("*", {}))
    rules.update(ASSET_STRATEGY_PARAMS.get(asset, {}).get(strategy, {}))
    return rules


def _pas_asset_strategy_params_toe(
    asset: str,
    setup: dict,
    regime: dict,
    htf_trend: str,
) -> tuple[dict | None, str, dict]:
    """Asset-specifieke strategieparameters zonder de strategiecode rommelig te maken."""
    strategy = setup.get("strategie", "?")
    rules = _asset_strategy_rule(asset, strategy)
    if not rules:
        return setup, "Geen asset-specifieke aanpassing.", {}

    min_regime_score = int(rules.get("min_regime_score") or ASSET_PARAMS.get(asset, {}).get("min_regime_score") or 0)
    regime_score = int((regime or {}).get("score") or 0)
    if min_regime_score and regime_score < min_regime_score:
        return (
            None,
            f"{asset}/{strategy}: regime-score {regime_score} te laag (min {min_regime_score}).",
            {"rules": rules, "regime": regime},
        )

    required_htf = rules.get("require_htf")
    if required_htf and htf_trend != required_htf:
        return (
            None,
            f"{asset}/{strategy}: HTF moet {required_htf} zijn, nu {htf_trend}.",
            {"rules": rules, "htf_trend": htf_trend},
        )

    required_btc_mode = rules.get("require_btc_mode")
    if required_btc_mode and BTC_MARKT.get("mode") != required_btc_mode:
        return (
            None,
            f"{asset}/{strategy}: BTC-modus moet {required_btc_mode} zijn, nu {BTC_MARKT.get('mode', 'unknown')}.",
            {"rules": rules, "btc_mode": BTC_MARKT},
        )

    adjusted = dict(setup)
    sl_multiplier = float(rules.get("sl_multiplier") or 1.0)
    if sl_multiplier > 1.0:
        entry = float(adjusted["entry"])
        sl = float(adjusted["stop_loss"])
        rr = float(adjusted.get("risk_reward") or MIN_RR_START)
        risk = abs(entry - sl) * sl_multiplier
        if adjusted.get("richting") == "short":
            adjusted["stop_loss"] = round(entry + risk, 6)
            adjusted["take_profit"] = round(entry - rr * risk, 6)
        else:
            adjusted["stop_loss"] = round(entry - risk, 6)
            adjusted["take_profit"] = round(entry + rr * risk, 6)
        adjusted["reden"] = list(adjusted.get("reden", [])) + [
            f"{asset}: SL/TP aangepast aan asset-volatiliteit ({sl_multiplier:.2f}x)"
        ]

    adjusted["_asset_strategy_rules"] = rules
    return adjusted, "Asset-specifieke strategieparameters toegepast.", {"rules": rules}


def _log(msg: str, level: str = "info"):
    getattr(LOG, level)(msg)
    try:
        db_log(msg, level)
    except Exception:
        pass


def _huidig_saldo() -> float:
    """Bereken huidig gesimuleerd saldo op basis van gesloten auto-trades."""
    gesloten = get_auto_trades(status="closed")
    totaal_pnl = sum(t.get("result_euro", 0) or 0 for t in gesloten)
    return STARTKAPITAAL + totaal_pnl


def _dagverlies_vandaag() -> float:
    """Bereken het totale verlies van gesloten auto-trades van vandaag (negatief getal)."""
    vandaag = datetime.now(AMSTERDAM).date().isoformat()
    gesloten = get_auto_trades(status="closed")
    return sum(
        t.get("result_euro", 0) or 0
        for t in gesloten
        if t.get("closed_at", "")[:10] == vandaag
    )


def _open_auto_trades() -> list[dict]:
    return get_auto_trades(status="open")


def _trade_context(trade: dict) -> dict:
    try:
        data = json.loads(trade.get("market_context") or "{}")
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _recente_auto_trades(n: int = 20) -> list[dict]:
    gesloten = get_auto_trades(status="closed")
    return gesloten[:n]


def _actieve_min_rr(leren: dict) -> float:
    """Gebruik de geleerde R/R, maar nooit lager dan de basisdrempel."""
    try:
        return max(float(leren.get("min_rr", MIN_RR_START) or MIN_RR_START), MIN_RR_START)
    except (TypeError, ValueError):
        return MIN_RR_START


def _opeenvolgende_verliezen() -> int:
    """Tel hoeveel opeenvolgende verliezen de bot heeft gehad."""
    gesloten = get_auto_trades(status="closed")
    teller = 0
    for t in gesloten:
        if (t.get("result_pct") or 0) < 0:
            teller += 1
        else:
            break
    return teller


# ── Leren ─────────────────────────────────────────────────────────────────────

def _laad_leren() -> dict:
    return get_bot_learning()


# ── Dashboard safety-overrides ─────────────────────────────────────────────────
# Het webdashboard schrijft instellingen naar bot_learning["dashboard_safety"].
# Zolang die key niet bestaat (of een control uitstaat) gebruikt de bot exact
# zijn eigen defaults — het gedrag verandert dus alleen als de gebruiker bewust
# iets opslaat in de UI.

def _safety_override() -> dict:
    try:
        data = _laad_leren().get("dashboard_safety")
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _override_actief(sleutel: str) -> bool:
    s = _safety_override()
    if not s:
        return False
    controls = s.get("controls") if isinstance(s.get("controls"), dict) else {}
    return bool(controls.get(sleutel, True))


def _override_waarde(veld: str):
    try:
        v = float(_safety_override().get(veld))
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def _effectief_dagverlies_limiet() -> float:
    if _override_actief("daily_loss_limit"):
        v = _override_waarde("dailyLossLimit")
        if v is not None:
            return v
    return STARTKAPITAAL * MAX_DAGVERLIES


def _llm_min_confidence_delta(asset: str) -> float:
    """
    Begrensde drempel-verschuiving uit het laatste LLM-advies (alleen als de laag
    aan staat én het advies vers is). Standaard 0.0 — geen effect.
    """
    if not llm_is_enabled():
        return 0.0
    try:
        advies = get_latest_llm_advice(asset)
        if not advies:
            return 0.0
        leeftijd = (
            datetime.now(timezone.utc)
            - pd.to_datetime(advies["created_at"], utc=True).to_pydatetime()
        ).total_seconds()
        if leeftijd > 2 * 3600:  # ouder dan 2 uur → negeren
            return 0.0
        return bounded_min_confidence_delta(advies)
    except Exception:
        return 0.0


def _effectieve_min_confidence(asset: str) -> float:
    if _override_actief("confidence_gate"):
        v = _override_waarde("minConfidence")
        if v is not None:
            return v * 100.0  # UI 0–1 → confidence-score 0–100
    basis = ASSET_PARAMS.get(asset, {}).get("min_confidence", MIN_CONFIDENCE_SCORE)
    # Optioneel LLM-advies kan de drempel begrensd bijsturen (standaard 0).
    return max(50.0, min(100.0, basis + _llm_min_confidence_delta(asset)))


def _effectieve_max_positie(saldo: float) -> float:
    basis = saldo / MAX_OPEN_TRADES
    if _override_actief("max_position_size"):
        v = _override_waarde("maxPositionSize")
        if v is not None:
            return min(basis, v)
    return basis


def _winrate_per_strategie() -> dict:
    """Bereken winrate per strategie op basis van alle gesloten auto-trades."""
    gesloten = get_auto_trades(status="closed")
    stats = {}
    for t in gesloten:
        strat = t.get("reason", "")
        # Strategie-naam staat in de reason als prefix "strategie:naam | ..."
        naam = "trend_long"
        if t.get("reason", "").startswith("strategie:"):
            naam = t["reason"].split("|")[0].replace("strategie:", "").strip()
        if naam not in stats:
            stats[naam] = {"wins": 0, "total": 0}
        stats[naam]["total"] += 1
        if (t.get("result_pct") or 0) > 0:
            stats[naam]["wins"] += 1
    return {
        k: round(v["wins"] / v["total"], 3) if v["total"] > 0 else 0.5
        for k, v in stats.items()
    }


def _update_leren(gesloten_trade: dict, exit_reden: str):
    """Pas leerparameters aan na het sluiten van een trade."""
    leren  = _laad_leren()
    recente = _recente_auto_trades(10)

    if len(recente) >= 5:
        winrate = sum(1 for t in recente if (t.get("result_pct") or 0) > 0) / len(recente)
        leren["winrate_recent"] = round(winrate, 3)

        huidige_rr = leren.get("min_rr", MIN_RR_START)
        if winrate < 0.35:
            nieuwe_rr = min(huidige_rr + 0.2, 2.5)
            _log(f"Leren: winrate laag ({winrate:.0%}) → min R/R stap omhoog naar {nieuwe_rr:.1f}", "warning")
        elif winrate < 0.45:
            nieuwe_rr = min(huidige_rr + 0.1, 2.2)
            _log(f"Leren: winrate matig ({winrate:.0%}) → min R/R stap omhoog naar {nieuwe_rr:.1f}", "warning")
        elif winrate > 0.60:
            nieuwe_rr = max(huidige_rr - 0.1, MIN_RR_START)
            _log(f"Leren: winrate goed ({winrate:.0%}) → min R/R stap omlaag naar {nieuwe_rr:.1f}")
        else:
            nieuwe_rr = huidige_rr
        leren["min_rr"] = round(nieuwe_rr, 1)
    else:
        leren["min_rr"] = leren.get("min_rr", MIN_RR_START)

    # Per-strategie winrate opslaan
    per_strat = _winrate_per_strategie()
    aanpassingen = leren.get("aanpassingen", {})
    aanpassingen["strategie_winrates"] = per_strat

    # Strategie tijdelijk pauzeren als winrate < 30% (min 5 trades)
    gesloten = get_auto_trades(status="closed")
    gepauzeerd = aanpassingen.get("gepauzeerde_strategieen", [])
    for naam, wr in per_strat.items():
        trades_strat = sum(
            1 for t in gesloten
            if t.get("reason", "").startswith(f"strategie:{naam}")
        )
        if trades_strat >= MIN_TRADES_PAUZEER and wr < 0.30 and naam not in gepauzeerd:
            gepauzeerd.append(naam)
            _log(
                f"Leren: strategie '{naam}' gepauzeerd "
                f"(winrate {wr:.0%} op {trades_strat} trades).", "warning"
            )
        elif naam in gepauzeerd and wr >= 0.40:
            gepauzeerd.remove(naam)
            _log(f"Leren: strategie '{naam}' hervat (winrate hersteld naar {wr:.0%}).")
    aanpassingen["gepauzeerde_strategieen"] = gepauzeerd

    # Cooling-periode na 3 opeenvolgende verliezen
    if _opeenvolgende_verliezen() >= 3:
        cooling = (datetime.now(timezone.utc) + timedelta(minutes=COOLING_MINUTEN)).isoformat()
        leren["cooling_tot"] = cooling
        _log(f"Leren: 3 opeenvolgende verliezen → {COOLING_MINUTEN} min pauze ingesteld.", "warning")

    save_bot_learning({
        "min_rr":          leren["min_rr"],
        "cooling_tot":     leren.get("cooling_tot"),
        "winrate_recent":  leren.get("winrate_recent", 0.5),
        "aanpassingen":    aanpassingen,
    })

    trade_id_raw = gesloten_trade.get("id")
    closed_trade = get_trade_by_id(int(trade_id_raw)) if trade_id_raw is not None else None
    if closed_trade:
        update_learning_from_closed_trade(int(closed_trade["id"]))
        record_exit_learning_from_closed_trade(int(closed_trade["id"]))
        score = closed_trade.get("learning_score")
        score_txt = f"score {score:.0%}" if isinstance(score, (int, float)) else "geen openingsscore"
        _log(f"Leren: trade #{closed_trade['id']} verwerkt in context-memory ({score_txt}).")


def _in_cooling() -> bool:
    leren = _laad_leren()
    cooling_tot = leren.get("cooling_tot")
    if not cooling_tot:
        return False
    try:
        tot = datetime.fromisoformat(cooling_tot)
        if tot.tzinfo is None:
            tot = tot.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) < tot
    except Exception:
        return False


# ── SL/TP controleren ─────────────────────────────────────────────────────────

def _trade_opened_at_utc(trade: dict) -> datetime | None:
    """Lees created_at uit de database en normaliseer naar UTC."""
    try:
        opened_at = datetime.fromisoformat(trade.get("created_at", ""))
        if opened_at.tzinfo is None:
            opened_at = opened_at.replace(tzinfo=AMSTERDAM)
        return opened_at.astimezone(timezone.utc)
    except Exception:
        return None


def _candles_na_open(df: pd.DataFrame, trade: dict) -> pd.DataFrame:
    """
    Gebruik alleen candles na het openen van de trade.
    Zo kunnen oude highs/lows uit de laatste 20 minuten geen verse trade sluiten.
    """
    opened_at = _trade_opened_at_utc(trade)
    if opened_at is None:
        return df
    result = df.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    return result[result["timestamp"] >= opened_at].reset_index(drop=True)


def _candles_open_sinds_entry(trade: dict, df_trade: pd.DataFrame) -> int:
    """
    Tel open-tijd in strategie-candles.

    SL/TP gebruikt 1-minuut candles voor precisie, maar de time-stop is bedoeld
    als 5-minuut strategie-candles. Gebruik daarom verstreken tijd in plaats van
    het aantal opgehaalde 1-minuut candles.
    """
    opened_at = _trade_opened_at_utc(trade)
    if opened_at is None or df_trade.empty or "timestamp" not in df_trade.columns:
        return max(0, len(df_trade) - 1)
    try:
        latest_ts = pd.to_datetime(df_trade["timestamp"].iloc[-1], utc=True)
        elapsed_seconds = max(0.0, (latest_ts.to_pydatetime() - opened_at).total_seconds())
        return int(elapsed_seconds // CANDLE_SECONDS)
    except Exception:
        return max(0, len(df_trade) - 1)


def _trade_progress_metrics(trade: dict, df_trade: pd.DataFrame) -> dict:
    """Meet hoe ver een open trade werkelijk kwam, uitgedrukt in R."""
    entry = float(trade.get("entry_price") or 0)
    sl = float(trade.get("stop_loss") or 0)
    tp = float(trade.get("take_profit") or 0)
    direction = trade.get("direction", "long")
    risk = abs(entry - sl)
    if entry <= 0 or risk <= 0 or df_trade.empty:
        return {
            "max_favorable_r": 0.0,
            "max_adverse_r": 0.0,
            "candles_to_1r": None,
            "near_tp": False,
            "current_r": 0.0,
            "candles_open": _candles_open_sinds_entry(trade, df_trade),
        }

    max_favorable = 0.0
    max_adverse = 0.0
    candles_to_1r = None
    first_hit = None

    for idx, candle in df_trade.reset_index(drop=True).iterrows():
        high = float(candle["high"])
        low = float(candle["low"])
        if direction == "short":
            favorable = max(0.0, entry - low)
            adverse = max(0.0, high - entry)
            hit_sl = high >= sl
            hit_tp = low <= tp
        else:
            favorable = max(0.0, high - entry)
            adverse = max(0.0, entry - low)
            hit_sl = low <= sl
            hit_tp = high >= tp

        max_favorable = max(max_favorable, favorable / risk)
        max_adverse = max(max_adverse, adverse / risk)
        if candles_to_1r is None and favorable >= risk:
            candles_to_1r = int(idx)
        if first_hit is None:
            if hit_sl:
                first_hit = "sl"
            elif hit_tp:
                first_hit = "tp"

    close = float(df_trade.iloc[-1]["close"])
    current_r = ((close - entry) / risk) if direction == "long" else ((entry - close) / risk)
    tp_distance_r = abs(tp - close) / risk if tp > 0 else 999.0
    return {
        "max_favorable_r": round(max_favorable, 4),
        "max_adverse_r": round(max_adverse, 4),
        "candles_to_1r": candles_to_1r,
        "first_hit": first_hit,
        "near_tp": tp_distance_r <= 0.25,
        "current_r": round(current_r, 4),
        "candles_open": _candles_open_sinds_entry(trade, df_trade),
    }


def _check_trade_anomalies(
    trade: dict,
    exit_price: float,
    exit_reason: str,
    replay: dict | None = None,
) -> list[str]:
    """Waakhond voor inconsistente sluitingen voordat ze stil in het leergeheugen belanden."""
    anomalies: list[str] = []
    entry = float(trade.get("entry_price") or 0)
    sl = float(trade.get("stop_loss") or 0)
    tp = float(trade.get("take_profit") or 0)
    direction = trade.get("direction", "long")
    exit_price = float(exit_price)
    replay = replay or {}

    tolerance = max(abs(entry) * 0.000001, 0.000001)
    if exit_reason == "sl" and abs(exit_price - sl) > tolerance:
        anomalies.append("sl_exit_price_mismatch")
    if exit_reason == "tp" and abs(exit_price - tp) > tolerance:
        anomalies.append("tp_exit_price_mismatch")
    if direction == "short" and exit_reason == "time_stop" and sl > 0 and exit_price > sl:
        anomalies.append("time_stop_worse_than_short_sl")
    if direction == "long" and exit_reason == "time_stop" and sl > 0 and exit_price < sl:
        anomalies.append("time_stop_worse_than_long_sl")

    if exit_reason == "time_stop":
        max_candles = int(ASSET_PARAMS.get(trade.get("asset"), {}).get("max_candles_open") or MAX_CANDLES_OPEN)
        candles_open = int(replay.get("candles_open") or 0)
        if candles_open and candles_open < max_candles:
            anomalies.append("time_stop_before_configured_candle_limit")

    if replay.get("first_hit") in {"sl", "tp"} and exit_reason not in {"sl", "tp"}:
        anomalies.append(f"hard_{replay['first_hit']}_seen_before_{exit_reason}")

    if anomalies:
        reason = f"Trade #{trade.get('id')}: anomalie gedetecteerd: {', '.join(anomalies)}"
        record_decision(
            asset=trade.get("asset", ""),
            strategy=trade.get("strategy"),
            action="trade_anomaly",
            reason=reason,
            context=_trade_context(trade),
            trade_id=trade.get("id"),
            details={"anomalies": anomalies, "replay": replay, "exit_price": exit_price, "exit_reason": exit_reason},
        )
        _log(reason, "warning")
    return anomalies


def _record_trade_replay(
    trade: dict,
    df_trade: pd.DataFrame | None,
    exit_price: float,
    exit_reason: str,
) -> dict:
    """Leg na elke exit vast of de trade echt progressie maakte."""
    replay = _trade_progress_metrics(trade, df_trade) if df_trade is not None else {}
    replay.update({
        "exit_price": round(float(exit_price), 6),
        "exit_reason": exit_reason,
    })
    record_decision(
        asset=trade.get("asset", ""),
        strategy=trade.get("strategy"),
        action="trade_replay",
        reason=(
            f"Replay trade #{trade.get('id')}: MFE {replay.get('max_favorable_r', 0):.2f}R, "
            f"MAE {replay.get('max_adverse_r', 0):.2f}R, exit={exit_reason}."
        ),
        context=_trade_context(trade),
        trade_id=trade.get("id"),
        details={"replay": replay},
    )
    _check_trade_anomalies(trade, exit_price, exit_reason, replay)
    return replay


def _trail_stop(asset: str, trade_id: int, richting: str, huidige_sl: float) -> float:
    """Beweeg de trailing stop alleen in de winstgevende richting."""
    try:
        df_trail = fetch_live_crypto_candles(asset, 60, 30)
        df_trail = add_all_indicators(df_trail)
        atr_val = float(df_trail.iloc[-1].get("atr14", float("nan")))
        if pd.isna(atr_val):
            return huidige_sl
        close = float(df_trail.iloc[-1]["close"])
        if richting == "long":
            trail_sl = round(close - 1.5 * atr_val, 6)
            if trail_sl > huidige_sl:
                update_trade_sl(trade_id, trail_sl)
                _log(f"Trade #{trade_id}: trailing SL naar {trail_sl:.4f}")
                return trail_sl
        else:
            trail_sl = round(close + 1.5 * atr_val, 6)
            if trail_sl < huidige_sl:
                update_trade_sl(trade_id, trail_sl)
                _log(f"Trade #{trade_id}: trailing SL naar {trail_sl:.4f}")
                return trail_sl
    except Exception:
        pass
    return huidige_sl


def _sluit_trade_markt(
    trade: dict,
    exit_price: float,
    reden: str,
    analyse: str,
    df_trade: pd.DataFrame | None = None,
) -> bool:
    asset = trade["asset"]
    entry = float(trade["entry_price"])
    richting = trade.get("direction", "long")
    review = f"{analyse} {generate_trade_review(trade, reden)}"
    if close_paper_trade(trade["id"], exit_price, reden, review):
        replay = _record_trade_replay(trade, df_trade, exit_price, reden)
        _update_leren(trade, reden)
        result_pct = (
            (exit_price - entry) / entry * 100
            if richting == "long"
            else (entry - exit_price) / entry * 100
        )
        _log(
            f"Trade #{trade['id']} {asset} {richting.upper()}: slim gesloten "
            f"({reden}) @ {exit_price:.4f} ({result_pct:+.2f}%, MFE {replay.get('max_favorable_r', 0):.2f}R)",
            "warning" if result_pct < 0 else "info",
        )
        if result_pct >= 0:
            melding_take_profit(trade["id"], asset, entry, exit_price, result_pct, 0, richting=richting)
        else:
            melding_stop_loss(trade["id"], asset, entry, exit_price, result_pct, 0, richting=richting)
        return True
    return False


def _slimme_exit_toegestaan(trade: dict, exit_reason: str, pnl_pct: float) -> tuple[bool, str, dict]:
    """
    Gebruik exit-learning voor niet-noodzakelijke exits.
    SL/TP blijven harde veiligheidsregels; deze check geldt voor time_stop/regime_flip.
    """
    profile = get_exit_strategy_profile(
        trade.get("asset", ""),
        trade.get("strategy") or "unknown",
        exit_reason,
    )
    if profile.get("action") == "avoid" and pnl_pct <= 0:
        return (
            False,
            f"Exit-learning vermijdt {exit_reason}: {profile['reason']}",
            profile,
        )
    return True, profile.get("reason", "Exitprofiel neutraal."), profile


def _controleer_slimme_exit(trade: dict, df_trade: pd.DataFrame, htf_trend: str) -> bool:
    """Sluit eerder als regime draait, momentum wegvalt of trade te lang niets doet."""
    try:
        df_full = add_all_indicators(df_trade)
        last = df_full.iloc[-1]
        live_price = float(last["close"])
        entry = float(trade["entry_price"])
        richting = trade.get("direction", "long")
        progress = _trade_progress_metrics(trade, df_trade)
        regime = classify_market_regime(
            df_trade,
            htf_trend=htf_trend,
            fear_greed=_fetch_fear_greed_safe()["value"],
        )
        momentum_bad = (
            richting == "long" and regime.get("direction") == "down"
        ) or (
            richting == "short" and regime.get("direction") == "up"
        )
        htf_bad = (
            richting == "long" and htf_trend == "downtrend"
        ) or (
            richting == "short" and htf_trend == "uptrend"
        )
        if momentum_bad or htf_bad:
            pnl_pct = (
                (live_price - entry) / entry * 100
                if richting == "long"
                else (entry - live_price) / entry * 100
            )
            exit_ok, exit_reden, exit_profile = _slimme_exit_toegestaan(trade, "regime_flip", pnl_pct)
            if not exit_ok:
                record_decision(
                    asset=trade.get("asset", ""),
                    strategy=trade.get("strategy"),
                    action="smart_exit_skipped",
                    reason=exit_reden,
                    context=_trade_context(trade),
                    trade_id=trade.get("id"),
                    details={"exit_profile": exit_profile, "pnl_pct": pnl_pct},
                )
                _log(f"Trade #{trade.get('id')}: regime_flip overgeslagen - {exit_reden}")
            else:
                return _sluit_trade_markt(
                    trade,
                    live_price,
                    "regime_flip",
                    f"Slimme exit: regime draaide tegen positie in ({regime.get('label')}, HTF {htf_trend}). {exit_reden}",
                    df_trade,
                )

        candles_open = _candles_open_sinds_entry(trade, df_trade)
        max_candles = int(ASSET_PARAMS.get(trade.get("asset"), {}).get("max_candles_open") or MAX_CANDLES_OPEN)
        risk_pct = abs(float(trade["stop_loss"]) - entry) / entry * 100
        pnl_pct = (
            (live_price - entry) / entry * 100
            if richting == "long"
            else (entry - live_price) / entry * 100
        )

        if progress.get("near_tp") and pnl_pct > 0:
            record_decision(
                asset=trade.get("asset", ""),
                strategy=trade.get("strategy"),
                action="smart_exit_skipped",
                reason="Time/exit overgeslagen: prijs ligt dicht bij TP.",
                context=_trade_context(trade),
                trade_id=trade.get("id"),
                details={"progress": progress, "pnl_pct": pnl_pct},
            )
            return False

        if candles_open >= 6 and pnl_pct < 0 and float(progress.get("max_favorable_r") or 0) < 0.30:
            exit_ok, exit_reden, exit_profile = _slimme_exit_toegestaan(trade, "no_progress", pnl_pct)
            if exit_ok:
                return _sluit_trade_markt(
                    trade,
                    live_price,
                    "no_progress",
                    (
                        "Slimme exit: na minimaal 30 minuten nooit 0.30R voordeel gezien "
                        f"en nu negatief ({pnl_pct:+.2f}%). {exit_reden}"
                    ),
                    df_trade,
                )
            record_decision(
                asset=trade.get("asset", ""),
                strategy=trade.get("strategy"),
                action="smart_exit_skipped",
                reason=exit_reden,
                context=_trade_context(trade),
                trade_id=trade.get("id"),
                details={"exit_profile": exit_profile, "progress": progress, "pnl_pct": pnl_pct},
            )

        if bool(trade.get("partial_tp_hit", 0)) and pnl_pct <= 0.05 and (momentum_bad or htf_bad):
            exit_ok, exit_reden, exit_profile = _slimme_exit_toegestaan(trade, "post_partial_momentum_loss", pnl_pct)
            if exit_ok:
                return _sluit_trade_markt(
                    trade,
                    live_price,
                    "post_partial_momentum_loss",
                    f"Slimme exit: na partial TP valt momentum weg ({regime.get('label')}, HTF {htf_trend}). {exit_reden}",
                    df_trade,
                )
            record_decision(
                asset=trade.get("asset", ""),
                strategy=trade.get("strategy"),
                action="smart_exit_skipped",
                reason=exit_reden,
                context=_trade_context(trade),
                trade_id=trade.get("id"),
                details={"exit_profile": exit_profile, "progress": progress, "pnl_pct": pnl_pct},
            )

        if candles_open >= max_candles:
            if pnl_pct < max(0.10, risk_pct * 0.25):
                exit_ok, exit_reden, exit_profile = _slimme_exit_toegestaan(trade, "time_stop", pnl_pct)
                if not exit_ok:
                    record_decision(
                        asset=trade.get("asset", ""),
                        strategy=trade.get("strategy"),
                        action="smart_exit_skipped",
                        reason=exit_reden,
                        context=_trade_context(trade),
                        trade_id=trade.get("id"),
                        details={"exit_profile": exit_profile, "pnl_pct": pnl_pct, "candles_open": candles_open},
                    )
                    _log(f"Trade #{trade.get('id')}: time_stop overgeslagen - {exit_reden}")
                    return False
                return _sluit_trade_markt(
                    trade,
                    live_price,
                    "time_stop",
                    f"Slimme exit: na {candles_open} candles onvoldoende progressie ({pnl_pct:+.2f}%). {exit_reden}",
                    df_trade,
                )
    except Exception as exc:
        _log(f"Trade #{trade.get('id')}: slimme exit-check mislukt - {exc}", "warning")
    return False


def _log_en_meld_sl(
    trade: dict,
    asset: str,
    entry: float,
    sl: float,
    richting: str,
    df_trade: pd.DataFrame | None = None,
):
    reflectie = f"{_genereer_sl_reflectie(trade)} {generate_trade_review(trade, 'sl')}"
    if close_paper_trade(trade["id"], sl, "sl", reflectie):
        replay = _record_trade_replay(trade, df_trade, sl, "sl")
        _update_leren(trade, "sl")
        result_pct = ((sl - entry) / entry * 100) if richting == "long" else ((entry - sl) / entry * 100)
        result_euro = result_pct / 100 * float(trade["capital"])
        partial_pnl = float(trade.get("partial_pnl_euro") or 0)
        totaal = result_euro + partial_pnl
        _log(f"Trade #{trade['id']} {asset} {richting.upper()}: STOP-LOSS @ {sl:.4f} "
             f"(netto: EUR{totaal:+.2f}, MFE {replay.get('max_favorable_r', 0):.2f}R)", "warning")
        melding_stop_loss(trade["id"], asset, entry, sl, result_pct, result_euro, richting=richting)
        return True
    return False


def _log_en_meld_tp(
    trade: dict,
    asset: str,
    entry: float,
    tp: float,
    richting: str,
    df_trade: pd.DataFrame | None = None,
):
    reflectie = f"Take-profit automatisch bereikt. {generate_trade_review(trade, 'tp')}"
    if close_paper_trade(trade["id"], tp, "tp", reflectie):
        replay = _record_trade_replay(trade, df_trade, tp, "tp")
        _update_leren(trade, "tp")
        result_pct = ((tp - entry) / entry * 100) if richting == "long" else ((entry - tp) / entry * 100)
        result_euro = result_pct / 100 * float(trade["capital"])
        partial_pnl = float(trade.get("partial_pnl_euro") or 0)
        totaal = result_euro + partial_pnl
        _log(f"Trade #{trade['id']} {asset} {richting.upper()}: TAKE-PROFIT @ {tp:.4f} "
             f"(netto: EUR{totaal:+.2f}, MFE {replay.get('max_favorable_r', 0):.2f}R)")
        melding_take_profit(trade["id"], asset, entry, tp, result_pct, result_euro, richting=richting)
        return True
    return False


def _controleer_open_trades():
    """Sluit open auto-trades automatisch als SL of TP geraakt is."""
    open_trades = _open_auto_trades()
    if not open_trades:
        return

    for trade in open_trades:
        asset = trade["asset"]
        try:
            df = fetch_live_crypto_candles(asset, 60, 20)
        except Exception as e:
            mark_api_health(asset, "down", str(e))
            record_decision(
                asset=asset,
                strategy=trade.get("strategy"),
                action="data_unavailable",
                reason=f"Prijs ophalen mislukt; open trade ongemoeid gelaten: {e}",
                context=_trade_context(trade),
                trade_id=trade.get("id"),
            )
            _log(f"Prijs ophalen mislukt voor {asset}: {e}", "warning")
            continue

        fresh, fresh_reason = latest_candle_is_fresh(df, max_age_seconds=1200)
        if not fresh:
            mark_api_health(asset, "stale", fresh_reason)
            record_decision(
                asset=asset,
                strategy=trade.get("strategy"),
                action="data_unavailable",
                reason=f"Prijsdata niet vers; open trade ongemoeid gelaten: {fresh_reason}",
                context=_trade_context(trade),
                trade_id=trade.get("id"),
            )
            _log(f"{asset}: prijsdata niet vers ({fresh_reason}); trade blijft open.", "warning")
            continue
        mark_api_health(asset, "ok", "SL/TP data ontvangen")

        df_trade = _candles_na_open(df, trade)
        if df_trade.empty:
            record_decision(
                asset=asset,
                strategy=trade.get("strategy"),
                action="skipped",
                reason="Nog geen nieuwe candle sinds het openen van de trade.",
                context=_trade_context(trade),
                trade_id=trade.get("id"),
            )
            continue

        sl       = float(trade["stop_loss"])
        tp       = float(trade["take_profit"])
        entry    = float(trade["entry_price"])
        richting = trade["direction"]
        partial  = bool(trade.get("partial_tp_hit", 0))
        gesloten = False

        for _, candle in df_trade.iterrows():
            laag = float(candle["low"])
            hoog = float(candle["high"])

            risico = abs(entry - sl)
            een_op_een = entry + risico if richting == "long" else entry - risico

            if richting == "long":
                if laag <= sl:
                    if _log_en_meld_sl(trade, asset, entry, sl, "long", df_trade):
                        gesloten = True
                        break
                    continue
                if hoog >= tp:
                    if _log_en_meld_tp(trade, asset, entry, tp, "long", df_trade):
                        gesloten = True
                        break
                    continue
                if not partial and hoog >= een_op_een:
                    gedeeltelijk_pnl = risico / entry * float(trade["capital"]) * 0.5
                    if update_trade_partial_tp(trade["id"], gedeeltelijk_pnl, entry):
                        sl = entry
                        partial = True
                        trade["capital"] = round(float(trade["capital"]) / 2, 2)
                        trade["partial_pnl_euro"] = gedeeltelijk_pnl
                        _log(
                            f"Trade #{trade['id']} {asset} LONG: GEDEELTELIJKE TP @ 1:1 "
                            f"(50% gesloten, SL naar break-even, +EUR{gedeeltelijk_pnl:.2f})"
                        )
                        if laag <= sl and _log_en_meld_sl(trade, asset, entry, sl, "long", df_trade):
                            gesloten = True
                            break

            elif richting == "short":
                if hoog >= sl:
                    if _log_en_meld_sl(trade, asset, entry, sl, "short", df_trade):
                        gesloten = True
                        break
                    continue
                if laag <= tp:
                    if _log_en_meld_tp(trade, asset, entry, tp, "short", df_trade):
                        gesloten = True
                        break
                    continue
                if not partial and laag <= een_op_een:
                    gedeeltelijk_pnl = risico / entry * float(trade["capital"]) * 0.5
                    if update_trade_partial_tp(trade["id"], gedeeltelijk_pnl, entry):
                        sl = entry
                        partial = True
                        trade["capital"] = round(float(trade["capital"]) / 2, 2)
                        trade["partial_pnl_euro"] = gedeeltelijk_pnl
                        _log(
                            f"Trade #{trade['id']} {asset} SHORT: GEDEELTELIJKE TP @ 1:1 "
                            f"(50% gesloten, SL naar break-even, +EUR{gedeeltelijk_pnl:.2f})"
                        )
                        if hoog >= sl and _log_en_meld_sl(trade, asset, entry, sl, "short", df_trade):
                            gesloten = True
                            break

        if gesloten:
            continue

        htf_trend = _get_htf_trend(asset)
        if _controleer_slimme_exit(trade, df_trade, htf_trend):
            continue

        if partial:
            _trail_stop(asset, trade["id"], richting, sl)


def _genereer_sl_reflectie(trade: dict) -> str:
    """Genereer automatisch een leermomentsamenvatting na een stop-loss."""
    entry = trade["entry_price"]
    sl    = trade["stop_loss"]
    verlies_pct = abs(sl - entry) / entry * 100
    reden = trade.get("reason", "")

    return (
        f"Stop-loss automatisch geraakt. Verlies: {verlies_pct:.2f}%. "
        f"Entry-reden was: {reden[:200] if reden else 'niet gespecificeerd'}. "
        f"Mogelijke oorzaken: te vroeg ingestapt, trend niet sterk genoeg, "
        f"of marktomstandigheden veranderden na entry."
    )


# ── Trade openen ──────────────────────────────────────────────────────────────

def _in_ny_open() -> bool:
    """Detecteer NY market open window (17:45–18:30 UTC) — extreme volatiliteit."""
    nu = datetime.now(timezone.utc)
    return (nu.hour == 17 and nu.minute >= 45) or (nu.hour == 18 and nu.minute < 30)


def _gecorreleerde_open(richting: str) -> int:
    """Tel hoeveel open trades dezelfde richting hebben in gecorreleerde assets."""
    return sum(
        1 for t in _open_auto_trades()
        if t.get("direction") == richting and t.get("asset") in GECORRELEERDE_ASSETS
    )


def _mag_handelen(asset: str) -> tuple[bool, str]:
    """Controleer alle discipline-regels voordat een trade wordt geopend."""
    # Cooling-periode
    if _in_cooling():
        leren = _laad_leren()
        return False, f"Bot in cooling-periode tot {leren.get('cooling_tot', '?')[:16]}"

    # NY market open — te volatiel voor nieuwe entries
    if _in_ny_open():
        return False, "NY market open (17:45-18:30 UTC) — te volatiel voor nieuwe entries."

    block_news, news_context = should_block_new_trades()
    if block_news:
        return False, f"Nieuws/event-filter actief: {news_context}"

    # Max dagverlies (dashboard kan deze limiet overschrijven)
    dagverlies = _dagverlies_vandaag()
    limiet = _effectief_dagverlies_limiet()
    if dagverlies < -limiet:
        return False, f"Dagverlies-limiet bereikt ({dagverlies:.2f}€ van max -{limiet:.0f}€)"

    # Max open trades
    open_trades = _open_auto_trades()
    if len(open_trades) >= MAX_OPEN_TRADES:
        return False, f"Maximaal {MAX_OPEN_TRADES} open trades bereikt"

    # Al een trade open in dit asset
    if any(t["asset"] == asset for t in open_trades):
        return False, f"Al een open trade in {asset}"

    return True, "ok"


def _confidence_size_multiplier(signal: dict) -> tuple[float, str]:
    """Bepaal beperkte inzet-schaal op basis van setup-confidence en remmen."""
    asset = signal.get("asset")
    confidence_score = int((signal.get("_confidence") or {}).get("score") or 0)
    multiplier = 0.50
    reden = "confidence onder minimum"

    for grens, factor, label in CONFIDENCE_SIZE_STEPS:
        if confidence_score >= grens:
            multiplier = factor
            reden = f"{label} ({confidence_score}/100)"
            break

    remmen: list[str] = []
    verliesreeks = _opeenvolgende_verliezen()
    if verliesreeks >= 2:
        multiplier = min(multiplier, 0.50)
        remmen.append(f"verliesreeks {verliesreeks} -> max 0.5x")
    elif verliesreeks == 1:
        multiplier = min(multiplier, 1.00)
        remmen.append("laatste trade verlies -> geen opschaling")

    spread_pct = float(((signal.get("_external_context") or {}).get("spread") or {}).get("spread_pct") or 0)
    if spread_pct > 0.04:
        multiplier = min(multiplier, 1.00)
        remmen.append(f"spread {spread_pct:.3f}% -> geen opschaling")

    learning_score = float((signal.get("_learning_score") or {}).get("score") or 0.5)
    if learning_score < 0.45:
        multiplier = min(multiplier, 0.50)
        remmen.append(f"lage leerscore {learning_score:.0%} -> max 0.5x")

    # Meta-label filter: alleen remmen als het model beschikbaar is én een lage winkans ziet.
    meta = signal.get("_meta_label") or {}
    if meta.get("available"):
        prob_win = float(meta.get("prob_win") or 0.5)
        if prob_win < 0.30:
            multiplier = min(multiplier, MAX_EXPLORATION_SIZE_MULTIPLIER)
            remmen.append(f"meta-label winkans {prob_win:.0%} -> max 0.25x")
        elif prob_win < 0.45:
            multiplier = min(multiplier, 0.50)
            remmen.append(f"meta-label winkans {prob_win:.0%} -> max 0.5x")

    proof = signal.get("_proof_details") or {}
    if proof.get("exploration"):
        multiplier = min(multiplier, MAX_EXPLORATION_SIZE_MULTIPLIER)
        remmen.append("exploratie-setup -> max 0.25x")
    if signal.get("_panic_exploration"):
        multiplier = min(multiplier, MAX_PANIC_EXPLORATION_SIZE_MULTIPLIER)
        remmen.append("extreme-fear BTC exploratie -> max 0.1x")

    profile = signal.get("_strategy_profile") or {}
    profile_cap = profile.get("size_cap")
    if profile_cap is not None:
        try:
            cap = float(profile_cap)
            if cap > 0:
                multiplier = min(multiplier, cap)
                remmen.append(f"strategieprofiel -> max {cap:g}x")
        except (TypeError, ValueError):
            pass

    time_profile = signal.get("_time_profile") or {}
    time_cap = time_profile.get("size_cap")
    if time_cap is not None:
        try:
            cap = float(time_cap)
            if cap > 0:
                multiplier = min(multiplier, cap)
                remmen.append(f"tijdprofiel -> max {cap:g}x")
        except (TypeError, ValueError):
            pass

    rules = signal.get("_asset_strategy_rules") or {}
    rule_cap = rules.get("size_cap")
    if rule_cap is not None:
        try:
            cap = float(rule_cap)
            if cap > 0:
                multiplier = min(multiplier, cap)
                remmen.append(f"asset/strategie-regel -> max {cap:g}x")
        except (TypeError, ValueError):
            pass

    if signal.get("strategie") in RANGE_STRATEGIES:
        multiplier = min(multiplier, MAX_EXPLORATION_SIZE_MULTIPLIER)
        remmen.append("range/squeeze-strategie -> max 0.25x")

    calibration = signal.get("_confidence_calibration") or {}
    calibration_cap = calibration.get("size_cap")
    if calibration_cap is not None:
        try:
            cap = float(calibration_cap)
            if cap > 0:
                multiplier = min(multiplier, cap)
                remmen.append(f"confidence-kalibratie -> max {cap:g}x")
        except (TypeError, ValueError):
            pass

    asset_mode = ALT_MARKT_MODUS.get(asset, "active")
    if asset_mode == "observe_or_explore":
        multiplier = min(multiplier, MAX_EXPLORATION_SIZE_MULTIPLIER)
        remmen.append("SOL alt-modus -> max 0.25x")
    elif asset_mode == "btc_confirmed" and BTC_MARKT.get("mode") == "cautious":
        multiplier = min(multiplier, 0.50)
        remmen.append("BTC cautious -> ETH max 0.5x")

    multiplier = min(multiplier, MAX_CONFIDENCE_SIZE_MULTIPLIER)
    if remmen:
        reden = f"{reden}; " + "; ".join(remmen)
    return multiplier, reden


def _confidence_bucket(score: float | int | None) -> str:
    try:
        value = int(score or 0)
    except (TypeError, ValueError):
        value = 0
    lower = max(0, min(90, (value // 10) * 10))
    upper = min(100, lower + 9)
    if lower == 90:
        upper = 100
    return f"{lower}-{upper}"


def _confidence_calibration_profile(asset: str, strategy: str, confidence_score: int) -> dict:
    """Vergelijk voorspelde confidence met echte uitkomst per bucket."""
    bucket = _confidence_bucket(confidence_score)
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT result_pct, market_context
        FROM paper_trades
        WHERE source='auto' AND status='closed' AND asset=? AND strategy=?
        ORDER BY id DESC LIMIT 200
        """,
        (asset, strategy),
    ).fetchall()
    conn.close()

    matches = []
    for row in rows:
        try:
            context = json.loads(row["market_context"] or "{}")
        except Exception:
            context = {}
        if context.get("confidence_bucket") == bucket:
            matches.append(float(row["result_pct"] or 0.0))

    if not matches:
        return {
            "bucket": bucket,
            "samples": 0,
            "winrate": 0.5,
            "avg_result_pct": 0.0,
            "confidence_adjustment": 0,
            "size_cap": None,
            "action": "neutral",
            "reason": f"Nog geen confidence-kalibratie voor bucket {bucket}.",
        }

    wins = sum(1 for value in matches if value > 0)
    winrate = wins / len(matches)
    avg_pct = sum(matches) / len(matches)
    adjustment = 0
    size_cap = None
    action = "neutral"
    if len(matches) >= 5 and winrate < 0.40 and avg_pct < 0:
        adjustment = -15
        size_cap = 0.50
        action = "caution"
    elif len(matches) >= 5 and winrate >= 0.60 and avg_pct > 0:
        adjustment = 5
        action = "trusted"

    return {
        "bucket": bucket,
        "samples": len(matches),
        "winrate": round(winrate, 4),
        "avg_result_pct": round(avg_pct, 4),
        "confidence_adjustment": adjustment,
        "size_cap": size_cap,
        "action": action,
        "reason": (
            f"Confidence bucket {bucket}: {len(matches)} trades, "
            f"winrate {winrate:.0%}, gem. {avg_pct:+.2f}%."
        ),
    }


def _kelly_fractie(trades: list[dict]) -> dict | None:
    """
    Bereken de (kwart-)Kelly risicofractie uit recente gesloten trades.

    Returns None als er te weinig data is. Anders een dict met o.a. de
    geadviseerde risicofractie (deel van het saldo dat je per trade riskeert).
    f* = p - (1-p)/b, met b = gem. winst% / gem. verlies%.
    """
    resultaten = [float(t.get("result_pct") or 0.0) for t in trades if t.get("result_pct") is not None]
    if len(resultaten) < MIN_KELLY_TRADES:
        return None

    wins = [r for r in resultaten if r > 0]
    losses = [-r for r in resultaten if r < 0]
    n = len(resultaten)
    p = len(wins) / n
    if not wins or not losses:
        # Geen verliezen (of geen winsten): geen betrouwbare odds → val terug op confidence.
        return None

    avg_win = sum(wins) / len(wins)
    avg_loss = sum(losses) / len(losses)
    if avg_loss <= 0:
        return None

    b = avg_win / avg_loss
    f_star = p - (1 - p) / b
    f_kwart = max(0.0, f_star) * KELLY_FRACTION
    return {
        "win_rate": round(p, 3),
        "avg_win_pct": round(avg_win, 3),
        "avg_loss_pct": round(avg_loss, 3),
        "payoff_b": round(b, 3),
        "kelly_full": round(f_star, 4),
        "kelly_fraction": round(f_kwart, 4),
        "sample": n,
    }


def _kelly_size_multiplier(signal: dict) -> tuple[float | None, str, dict]:
    """
    Vertaal de kwart-Kelly fractie naar een inzet-multiplier t.o.v. de 1%-basisrisico.

    Geeft (None, reden, {}) terug als er te weinig data is — dan blijft de bestaande
    confidence-multiplier leidend. De multiplier kan de inzet alleen binnen de
    bestaande envelope [0.25, 1.25] bijsturen; hij blaast nooit verder op.
    """
    kelly = _kelly_fractie(get_auto_trades(status="closed"))
    if kelly is None:
        return None, "te weinig trades voor Kelly", {}

    multiplier = kelly["kelly_fraction"] / RISICO_PCT if RISICO_PCT > 0 else 0.0
    multiplier = round(max(KELLY_MIN_MULTIPLIER, min(multiplier, MAX_CONFIDENCE_SIZE_MULTIPLIER)), 3)
    reden = (
        f"kwart-Kelly {kelly['kelly_fraction']*100:.2f}% risico "
        f"(WR {kelly['win_rate']:.0%}, payoff {kelly['payoff_b']:.2f}, n={kelly['sample']})"
    )
    return multiplier, reden, kelly


def _bereken_positiegrootte(signal: dict) -> tuple[float, dict]:
    """Bereken hoeveel kapitaal in deze trade wordt gezet (1% risico-regel).

    De positiegrootte = risico_euro / risk_per_unit, maar nooit meer dan
    het saldo gedeeld door het max aantal open trades, en nooit groter dan
    MAX_PORTFOLIO_FRACTION van het saldo.
    """
    saldo = _huidig_saldo()
    asset = signal.get("asset")
    risk_multiplier = ASSET_PARAMS.get(asset, {}).get("risk_multiplier", 1.0)
    confidence_multiplier, confidence_reason = _confidence_size_multiplier(signal)

    # Kwart-Kelly als veilige extra cap: neem de meest conservatieve van de twee.
    kelly_multiplier, kelly_reason, kelly_details = _kelly_size_multiplier(signal)
    effective_multiplier = confidence_multiplier
    size_reason = confidence_reason
    if kelly_multiplier is not None:
        effective_multiplier = min(confidence_multiplier, kelly_multiplier)
        size_reason = f"{confidence_reason}; {kelly_reason}; effectief min={effective_multiplier:.2f}x"

    # Zachte circuit breaker: halveer inzet zodra het dagverlies de zachte grens raakt
    # (de harde kill-switch bij MAX_DAGVERLIES blokkeert nieuwe trades volledig).
    soft_breaker = False
    if _dagverlies_vandaag() <= -(STARTKAPITAAL * SOFT_DAGVERLIES):
        effective_multiplier *= 0.5
        soft_breaker = True
        size_reason = f"{size_reason}; zachte drawdown-rem -50% (dagverlies ≥ {SOFT_DAGVERLIES:.0%})"

    risico_euro = saldo * RISICO_PCT * risk_multiplier * effective_multiplier
    entry = signal["entry"]
    sl    = signal["stop_loss"]
    risk_per_unit = abs(entry - sl) / entry if entry > 0 else 0.02
    if risk_per_unit <= 0:
        positie = risico_euro * 10
    else:
        positie = risico_euro / risk_per_unit

    # Nooit meer inzetten dan saldo / max trades (dashboard kan dit verlagen),
    # en nooit meer dan MAX_PORTFOLIO_FRACTION van het saldo in één positie.
    max_positie = min(_effectieve_max_positie(saldo), saldo * MAX_PORTFOLIO_FRACTION)
    positie = round(min(positie, max_positie), 2)
    details = {
        "base_risk_pct": RISICO_PCT,
        "asset_risk_multiplier": risk_multiplier,
        "confidence_multiplier": confidence_multiplier,
        "confidence_reason": confidence_reason,
        "kelly_multiplier": kelly_multiplier,
        "kelly": kelly_details,
        "effective_multiplier": round(effective_multiplier, 3),
        "soft_drawdown_breaker": soft_breaker,
        "size_reason": size_reason,
        "risk_euro": round(risico_euro, 2),
        "risk_per_unit": round(risk_per_unit, 6),
        "max_position": round(max_positie, 2),
    }
    return positie, details


def _verwachte_netto_reward_gate(signal: dict, kapitaal: float) -> tuple[bool, str, dict]:
    """Blokkeer setups waarvan de volledige TP na kosten te weinig netto edge heeft."""
    entry = float(signal.get("entry") or 0)
    tp = float(signal.get("take_profit") or 0)
    if entry <= 0 or tp <= 0 or kapitaal <= 0:
        return False, "Netto-edge: ongeldige entry/TP/positiegrootte.", {}

    reward_pct = abs(tp - entry) / entry * 100
    gross_reward_euro = reward_pct / 100 * float(kapitaal)
    estimated_costs_euro = round_trip_cost_euro(kapitaal)
    expected_net_reward_euro = gross_reward_euro - estimated_costs_euro
    reward_cost_ratio = (
        gross_reward_euro / estimated_costs_euro
        if estimated_costs_euro > 0 else float("inf")
    )
    min_net_reward = MIN_EXPECTED_NET_REWARD_EURO
    if signal.get("_panic_exploration"):
        min_net_reward = 0.50

    details = {
        "reward_pct": round(reward_pct, 4),
        "gross_reward_euro": round(gross_reward_euro, 2),
        "estimated_costs_euro": round(estimated_costs_euro, 2),
        "expected_net_reward_euro": round(expected_net_reward_euro, 2),
        "reward_cost_ratio": round(reward_cost_ratio, 2) if reward_cost_ratio != float("inf") else None,
        "min_expected_net_reward_euro": min_net_reward,
        "min_reward_cost_ratio": MIN_REWARD_COST_RATIO,
    }

    if expected_net_reward_euro < min_net_reward:
        return (
            False,
            (
                f"Netto-edge te dun: TP verwacht EUR{expected_net_reward_euro:.2f} "
                f"na EUR{estimated_costs_euro:.2f} kosten."
            ),
            details,
        )
    if reward_cost_ratio < MIN_REWARD_COST_RATIO:
        return (
            False,
            (
                f"Netto-edge te dun: bruto reward/kosten {reward_cost_ratio:.2f}x "
                f"(min {MIN_REWARD_COST_RATIO:.2f}x)."
            ),
            details,
        )
    return (
        True,
        (
            f"Netto-edge ok: TP verwacht EUR{expected_net_reward_euro:.2f} "
            f"na EUR{estimated_costs_euro:.2f} kosten."
        ),
        details,
    )


# Regimes waarin volatiliteit/onzekerheid hoog is: daar laten we de risico- en
# kosten-agents zwaarder meewegen, zodat de bot daar strenger wordt.
_HOOG_RISICO_REGIMES = frozenset({
    "panic", "euphoria", "breakout_up", "breakout_down", "reversal", "unknown",
})


def _committee_weights(regime_label: str) -> dict[str, float]:
    """
    Gewichten per agent voor de gewogen committee-score.

    In normale/rustige regimes wegen alle agents gelijk (1.0) — gedrag identiek aan
    een gewoon gemiddelde. In hoog-risico regimes tellen de Risk- en Cost/Edge-agents
    zwaarder, zodat een matige risico-inschatting de uitkomst sterker drukt.
    """
    weights = {
        "Setup Agent": 1.0,
        "Market Regime Agent": 1.0,
        "Risk Manager Agent": 1.0,
        "Cost/Edge Agent": 1.0,
        "Learning Reviewer Agent": 1.0,
        "News Watch Agent": 1.0,
    }
    if regime_label in _HOOG_RISICO_REGIMES:
        weights["Risk Manager Agent"] = 1.7
        weights["Cost/Edge Agent"] = 1.3
        weights["Market Regime Agent"] = 1.2
    return weights


def _agent_vote(agent: str, status: str, score: int, reason: str, details: dict | None = None) -> dict:
    return {
        "agent": agent,
        "status": status,
        "score": max(0, min(100, int(score))),
        "reason": reason,
        "details": details or {},
    }


def _agent_committee(
    signal: dict,
    size_details: dict,
    edge_details: dict,
    news_context: dict | None = None,
) -> dict:
    """Laat gespecialiseerde agents samen de laatste tradebeslissing verklaren."""
    votes: list[dict] = []
    strategy = signal.get("strategie", "?")
    richting = signal.get("richting", "long")
    entry = float(signal.get("entry") or 0)
    sl = float(signal.get("stop_loss") or 0)
    tp = float(signal.get("take_profit") or 0)
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    actual_rr = reward / risk if risk > 0 else 0.0

    if entry <= 0 or sl <= 0 or tp <= 0 or risk <= 0 or reward <= 0:
        votes.append(_agent_vote("Setup Agent", "block", 0, "Ongeldige setup-levels."))
    elif actual_rr < MIN_RR_START:
        votes.append(_agent_vote(
            "Setup Agent",
            "block",
            35,
            f"R/R {actual_rr:.2f} is te laag.",
            {"actual_rr": round(actual_rr, 3)},
        ))
    else:
        votes.append(_agent_vote(
            "Setup Agent",
            "pass",
            80 if actual_rr >= 2.4 else 70,
            f"Setup-levels kloppen; R/R {actual_rr:.2f}.",
            {"actual_rr": round(actual_rr, 3), "strategy": strategy, "direction": richting},
        ))

    regime = signal.get("_market_regime") or {}
    checklist = signal.get("_pre_trade_checklist") or {}
    htf_trend = (signal.get("_learning_context") or {}).get("htf_trend", "onduidelijk")
    if checklist.get("hard_blocks"):
        votes.append(_agent_vote(
            "Market Regime Agent",
            "block",
            int(checklist.get("score") or 0),
            f"Checklist blokkeert: {', '.join(checklist.get('hard_blocks', []))}.",
            {"regime": regime, "checklist": checklist},
        ))
    elif int((regime or {}).get("score") or 0) >= 60:
        votes.append(_agent_vote(
            "Market Regime Agent",
            "pass",
            int((regime or {}).get("score") or 70),
            f"Regime {regime.get('label', 'unknown')} past voldoende; HTF {htf_trend}.",
            {"regime": regime, "htf_trend": htf_trend},
        ))
    else:
        votes.append(_agent_vote(
            "Market Regime Agent",
            "warn",
            55,
            f"Regime {regime.get('label', 'unknown')} is matig; alleen voorzichtig.",
            {"regime": regime, "htf_trend": htf_trend},
        ))

    verliesreeks = _opeenvolgende_verliezen()
    fear_greed = float((signal.get("_learning_context") or {}).get("fear_greed_value") or BTC_MARKT.get("fear_greed") or 50)
    risk_blocks = []
    risk_warnings = []
    if verliesreeks >= 3:
        risk_blocks.append(f"verliesreeks {verliesreeks}")
    elif verliesreeks >= 2:
        risk_warnings.append(f"verliesreeks {verliesreeks}")
    if fear_greed <= 20 and richting == "short":
        risk_blocks.append("short in Extreme Fear")
    if fear_greed >= 80 and richting == "long":
        risk_blocks.append("long in Extreme Greed")
    if signal.get("asset") != "BTC-USD" and BTC_MARKT.get("mode") == "blocked":
        risk_blocks.append("alt terwijl BTC-first blocked is")

    if risk_blocks:
        votes.append(_agent_vote("Risk Manager Agent", "block", 25, "; ".join(risk_blocks)))
    elif risk_warnings or signal.get("_panic_exploration"):
        reason = "; ".join(risk_warnings) or "extreme-fear mini-exploratie"
        votes.append(_agent_vote(
            "Risk Manager Agent",
            "warn",
            65,
            reason,
            {"position_sizing": size_details, "btc_mode": BTC_MARKT},
        ))
    else:
        votes.append(_agent_vote(
            "Risk Manager Agent",
            "pass",
            78,
            "Risico binnen limieten.",
            {"position_sizing": size_details, "btc_mode": BTC_MARKT},
        ))

    expected_net = float(edge_details.get("expected_net_reward_euro") or 0.0)
    ratio = float(edge_details.get("reward_cost_ratio") or 0.0)
    if expected_net < float(edge_details.get("min_expected_net_reward_euro") or MIN_EXPECTED_NET_REWARD_EURO):
        votes.append(_agent_vote("Cost/Edge Agent", "block", 35, "Netto reward onder minimum.", edge_details))
    elif ratio < MIN_REWARD_COST_RATIO:
        votes.append(_agent_vote("Cost/Edge Agent", "block", 40, "Reward/kosten ratio te laag.", edge_details))
    else:
        votes.append(_agent_vote(
            "Cost/Edge Agent",
            "pass",
            85 if ratio >= 2.0 else 72,
            f"Netto edge positief: EUR{expected_net:.2f}, reward/kosten {ratio:.2f}x.",
            edge_details,
        ))

    score_info = signal.get("_learning_score") or {}
    profile = signal.get("_strategy_profile") or {}
    time_profile = signal.get("_time_profile") or {}
    learning_score = float(score_info.get("score") or 0.5)
    if score_info.get("should_block") or profile.get("action") == "block" or time_profile.get("action") == "block":
        votes.append(_agent_vote(
            "Learning Reviewer Agent",
            "block",
            25,
            "Learning/profiel blokkeert setup.",
            {"score_info": score_info, "strategy_profile": profile, "time_profile": time_profile},
        ))
    elif learning_score < 0.45 or profile.get("action") == "caution" or time_profile.get("action") == "caution":
        votes.append(_agent_vote(
            "Learning Reviewer Agent",
            "warn",
            62,
            f"Learning voorzichtig: score {learning_score:.0%}.",
            {"score_info": score_info, "strategy_profile": profile, "time_profile": time_profile},
        ))
    else:
        votes.append(_agent_vote(
            "Learning Reviewer Agent",
            "pass",
            78,
            f"Learning akkoord: score {learning_score:.0%}.",
            {"score_info": score_info, "strategy_profile": profile, "time_profile": time_profile},
        ))

    news_context = news_context or {}
    if news_context.get("block"):
        votes.append(_agent_vote("News Watch Agent", "block", 20, "Actieve nieuws/event blokkade.", news_context))
    else:
        risks = [
            (news_context.get("macro") or {}).get("risk"),
            (news_context.get("cryptopanic") or {}).get("risk"),
            (news_context.get("rss") or {}).get("risk"),
        ]
        if "caution" in risks:
            votes.append(_agent_vote("News Watch Agent", "warn", 65, "Nieuwsdruk vraagt voorzichtigheid.", news_context))
        else:
            votes.append(_agent_vote("News Watch Agent", "pass", 75, "Geen actieve nieuwsblokkade.", news_context))

    avg_score = round(sum(v["score"] for v in votes) / len(votes), 1) if votes else 0.0

    # Regime-gewogen score: in volatiele regimes drukken de risico-/kosten-agents zwaarder.
    regime_label = str((regime or {}).get("label") or "unknown")
    weights = _committee_weights(regime_label)
    gewogen_som = sum(v["score"] * weights.get(v["agent"], 1.0) for v in votes)
    gewicht_totaal = sum(weights.get(v["agent"], 1.0) for v in votes)
    weighted_score = round(gewogen_som / gewicht_totaal, 1) if gewicht_totaal else avg_score

    pass_count = sum(1 for v in votes if v["status"] == "pass")
    blocks = [v for v in votes if v["status"] == "block"]
    passed = (
        not blocks
        and weighted_score >= MIN_AGENT_COMMITTEE_SCORE
        and pass_count >= MIN_AGENT_PASSES
    )
    weeg_txt = "" if regime_label not in _HOOG_RISICO_REGIMES else f" (regime-gewogen, {regime_label})"
    reason = (
        f"Agent committee {weighted_score:.1f}/100{weeg_txt}, {pass_count}/{len(votes)} pass"
        + (f"; blocks: {', '.join(v['agent'] for v in blocks)}" if blocks else "")
    )
    return {
        "passed": passed,
        "score": weighted_score,
        "avg_score": avg_score,
        "weighted_score": weighted_score,
        "regime_weighted": regime_label in _HOOG_RISICO_REGIMES,
        "pass_count": pass_count,
        "min_score": MIN_AGENT_COMMITTEE_SCORE,
        "min_passes": MIN_AGENT_PASSES,
        "reason": reason,
        "votes": votes,
    }


def _pas_entry_aan_naar_live_prijs(signal: dict, live_prijs: float, max_slippage_r: float = 0.35) -> tuple[dict | None, str]:
    """
    Gebruik de actuele prijs als paper-entry, maar jaag geen setup achterna.
    Het signaal komt van de laatste gesloten candle; als de markt daarna te ver is
    doorgeschoten, is de echte R/R niet meer dezelfde setup.
    """
    adjusted = dict(signal)
    richting = adjusted.get("richting", "long")
    geplande_entry = float(adjusted["entry"])
    geplande_sl = float(adjusted["stop_loss"])
    rr = float(adjusted.get("risk_reward") or MIN_RR_START)
    risico = abs(geplande_entry - geplande_sl)

    if live_prijs <= 0 or risico <= 0:
        return None, "Ongeldige live prijs of risicoberekening."

    if richting == "long":
        adverse_move = live_prijs - geplande_entry
        if live_prijs <= geplande_sl:
            return None, "Live prijs ligt al op/onder de stop-loss."
        if adverse_move > max_slippage_r * risico:
            return None, f"Live prijs is te ver boven het signaal doorgeschoten ({adverse_move/risico:.1f}R)."
        adjusted["entry"] = round(live_prijs, 6)
        adjusted["stop_loss"] = round(live_prijs - risico, 6)
        adjusted["take_profit"] = round(live_prijs + rr * risico, 6)
    else:
        adverse_move = geplande_entry - live_prijs
        if live_prijs >= geplande_sl:
            return None, "Live prijs ligt al op/boven de stop-loss."
        if adverse_move > max_slippage_r * risico:
            return None, f"Live prijs is te ver onder het signaal doorgeschoten ({adverse_move/risico:.1f}R)."
        adjusted["entry"] = round(live_prijs, 6)
        adjusted["stop_loss"] = round(live_prijs + risico, 6)
        adjusted["take_profit"] = round(live_prijs - rr * risico, 6)

    adjusted["reden"] = list(adjusted.get("reden", [])) + [
        f"Paper-entry aangepast naar live prijs ({live_prijs:.4f})"
    ]
    return adjusted, "ok"


def _entry_execution_gate(signal: dict, live_prijs: float) -> tuple[bool, str, dict]:
    """Voorkom entries die de setup al hebben ingehaald of kapot gemaakt."""
    richting = signal.get("richting", "long")
    entry = float(signal.get("entry") or 0)
    sl = float(signal.get("stop_loss") or 0)
    tp = float(signal.get("take_profit") or 0)
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    details = {
        "planned_entry": entry,
        "live_price": live_prijs,
        "risk": risk,
        "reward": reward,
    }
    if entry <= 0 or sl <= 0 or tp <= 0 or risk <= 0 or reward <= 0:
        return False, "Entry-gate: ongeldige levels.", details

    if richting == "long":
        favorable_r = (live_prijs - entry) / risk
        adverse_r = (entry - live_prijs) / risk
        if live_prijs <= sl:
            return False, "Entry-gate: live prijs ligt al op/onder SL.", details
        if live_prijs >= tp:
            return False, "Entry-gate: live prijs heeft TP al bereikt.", details
    else:
        favorable_r = (entry - live_prijs) / risk
        adverse_r = (live_prijs - entry) / risk
        if live_prijs >= sl:
            return False, "Entry-gate: live prijs ligt al op/boven SL.", details
        if live_prijs <= tp:
            return False, "Entry-gate: live prijs heeft TP al bereikt.", details

    details.update({
        "favorable_r": round(favorable_r, 4),
        "adverse_r": round(adverse_r, 4),
        "reward_r": round(reward / risk, 4),
    })
    if favorable_r > 0.35:
        return False, f"Entry-gate: setup al {favorable_r:.2f}R doorgeschoten; wacht op betere entry.", details
    if adverse_r > 0.35:
        return False, f"Entry-gate: setup al {adverse_r:.2f}R tegen positie bewogen.", details
    return True, "Entry-gate ok.", details


def _pre_trade_checklist(
    *,
    asset: str,
    setup: dict,
    regime_ok: bool,
    htf_ok: bool,
    proof_details: dict,
    score_info: dict,
    quality: dict,
    confidence: dict,
    market_context: dict,
    strategy_profile: dict,
    time_profile: dict,
) -> dict:
    """Centrale checklist voor 'niet handelen tenzij genoeg klopt'."""
    score = 0
    items: list[str] = []
    hard_blocks: list[str] = []

    if regime_ok:
        score += 15
        items.append("regime ok")
    else:
        hard_blocks.append("regime past niet")

    if htf_ok:
        score += 15
        items.append("HTF aligned")
    elif setup.get("strategie") in RANGE_STRATEGIES:
        score += 5
        items.append("range-strategie mag HTF-neutraal zijn")
    else:
        items.append("HTF niet aligned")

    if float(score_info.get("score") or 0.5) >= 0.45:
        score += 12
        items.append("leerscore acceptabel")
    else:
        hard_blocks.append("leerscore te zwak")

    if int(quality.get("score") or 0) >= 65:
        score += 15
        items.append("setupkwaliteit acceptabel")
    else:
        hard_blocks.append("setupkwaliteit te laag")

    if int(confidence.get("score") or 0) >= 65:
        score += 15
        items.append("confidence acceptabel")
    else:
        items.append("confidence laag")

    if proof_details.get("exploration"):
        score += 5
        items.append("exploratie klein")
    elif float(proof_details.get("backtest_pnl") or 0) > 0:
        score += 10
        items.append("backtest positief")

    spread_pct = float((market_context.get("spread") or {}).get("spread_pct") or 0)
    if spread_pct <= 0.08:
        score += 8
        items.append("spread ok")
    else:
        hard_blocks.append(f"spread te hoog ({spread_pct:.3f}%)")

    if strategy_profile.get("action") == "block":
        hard_blocks.append("regime-geheugen blokkeert")
    elif strategy_profile.get("action") == "caution":
        score -= 8
        items.append("regime-geheugen voorzichtig")
    elif strategy_profile.get("action") == "boost":
        score += 5
        items.append("regime-geheugen positief")

    if time_profile.get("action") == "block":
        hard_blocks.append("tijdprofiel blokkeert")
    elif time_profile.get("action") == "caution":
        score -= 5
        items.append("tijdprofiel voorzichtig")

    btc_mode = BTC_MARKT.get("mode", "unknown")
    if asset == "BTC-USD" or btc_mode in {"active", "cautious"}:
        score += 10
        items.append("BTC-modus ok")
    else:
        hard_blocks.append(f"BTC-modus {btc_mode}")

    score = max(0, min(100, score))
    min_score = 58 if proof_details.get("exploration") or setup.get("strategie") in RANGE_STRATEGIES else 70
    passed = not hard_blocks and score >= min_score
    return {
        "score": score,
        "min_score": min_score,
        "passed": passed,
        "items": items,
        "hard_blocks": hard_blocks,
        "reason": (
            f"Checklist {score}/{min_score}: "
            + (", ".join(items[:6]) if items else "geen positieve checks")
            + (f"; blocks: {', '.join(hard_blocks)}" if hard_blocks else "")
        ),
    }


def _setup_heeft_bewijs(
    asset: str,
    df: pd.DataFrame,
    strategy: str,
    min_rr: float,
    score_info: dict,
) -> tuple[bool, str, dict]:
    """
    Laat alleen setups door die recent netto positief backtesten en waarvan de
    paper-score niet extreem negatief is zodra er genoeg live/bootstrap samples zijn.
    Matige brede leerdata wordt later meegenomen in confidence/positiegrootte, zodat
    de bot nog gecontroleerd kan exploreren met kleine inzet.
    """
    paper_samples = int(score_info.get("sample_count") or 0)
    paper_score = float(score_info.get("score") or 0.5)
    if paper_samples >= 10 and paper_score < HARD_BLOCK_PAPER_SCORE:
        return (
            False,
            f"Paper-score extreem laag: {paper_score:.0%} op {paper_samples} samples.",
            {"paper_score": paper_score, "paper_samples": paper_samples},
        )

    if strategy in RANGE_STRATEGIES and paper_score >= 0.35:
        return (
            True,
            "Exploratie-setup: range/squeeze mean-reversion, altijd kleine inzet.",
            {
                "backtest_trades": 0,
                "backtest_winrate": 0.0,
                "backtest_pnl": 0.0,
                "paper_score": paper_score,
                "paper_samples": paper_samples,
                "exploration": True,
                "range_strategy": True,
            },
        )

    try:
        recent_df = df.tail(260).reset_index(drop=True)
        last_ts = str(recent_df["timestamp"].iloc[-1]) if "timestamp" in recent_df.columns and len(recent_df) else len(recent_df)
        cache_key = (asset, len(recent_df), last_ts, round(float(min_rr), 2))
        cached = _proof_cache.get(cache_key)
        if cached is None:
            cached = {
                "walk": run_walk_forward_backtest(
                    recent_df,
                    min_rr=min_rr,
                    train_size=120,
                    test_size=50,
                    min_train_trades=1,
                    min_train_winrate=35.0,
                    min_train_pnl=-0.25,
                    initial_capital=STARTKAPITAAL,
                    risico_per_trade_pct=RISICO_PCT * 100,
                    max_candles_open=24,
                ),
                "res": run_backtest(
                    recent_df,
                    min_rr=min_rr,
                    initial_capital=STARTKAPITAAL,
                    risico_per_trade_pct=RISICO_PCT * 100,
                    max_candles_open=24,
                ),
            }
            _proof_cache[cache_key] = cached
        walk = cached["walk"]
        res = cached["res"]
    except Exception as exc:
        return False, f"Recente backtest kon niet draaien: {exc}", {}

    wf_stats = (walk.get("strategy_stats") or {}).get(strategy, {})
    wf_required = ASSET_PARAMS.get(asset, {}).get("walk_forward_min_trades", 1)
    wf_min_wr = ASSET_PARAMS.get(asset, {}).get("walk_forward_min_winrate", 40.0)
    wf_trades = int(wf_stats.get("trades") or 0)
    wf_winrate = float(wf_stats.get("winrate") or 0.0)
    wf_pnl = float(wf_stats.get("pnl") or 0.0)

    trades = [
        trade for trade in res.get("trades", [])
        if trade.get("strategie") == strategy
    ]
    required_trades = ASSET_PARAMS.get(asset, {}).get("min_backtest_trades", MIN_BACKTEST_TRADES)
    if len(trades) < required_trades:
        if len(trades) >= 1 and paper_score >= 0.35:
            trade = trades[0]
            details = {
                "backtest_trades": len(trades),
                "backtest_winrate": 100.0 if trade.get("result_pct", 0) > 0 else 0.0,
                "backtest_pnl": round(float(trade.get("result_pct") or 0), 3),
                "paper_score": paper_score,
                "paper_samples": paper_samples,
                "exploration": True,
            }
            return (
                True,
                f"Exploratie-setup: weinig backtest-data ({len(trades)}/{required_trades}), kleine inzet.",
                details,
            )
        if (
            asset == "BTC-USD"
            and strategy in PANIC_LONG_STRATEGIES
            and float(BTC_MARKT.get("fear_greed") or 50) <= 25
            and paper_score >= 0.35
        ):
            details = {
                "backtest_trades": len(trades),
                "backtest_winrate": 0.0,
                "backtest_pnl": 0.0,
                "paper_score": paper_score,
                "paper_samples": paper_samples,
                "exploration": True,
                "panic_candidate": True,
            }
            return (
                True,
                (
                    f"BTC extreme-fear exploratie: te weinig backtest-data "
                    f"({len(trades)}/{required_trades}), alleen door bij hoge confidence en mini-inzet."
                ),
                details,
            )
        return (
            False,
            f"Te weinig recente backtest-trades voor {strategy} ({len(trades)}/{required_trades}).",
            {"backtest_trades": len(trades)},
        )

    wins = sum(1 for trade in trades if trade.get("result_pct", 0) > 0)
    winrate = wins / len(trades) * 100
    pnl = sum(float(trade.get("result_pct") or 0) for trade in trades)
    costs = sum(float(trade.get("costs_euro") or 0) for trade in trades)
    details = {
        "backtest_trades": len(trades),
        "backtest_winrate": round(winrate, 1),
        "backtest_pnl": round(pnl, 3),
        "backtest_costs_euro": round(costs, 2),
        "walk_forward_trades": wf_trades,
        "walk_forward_winrate": round(wf_winrate, 1),
        "walk_forward_pnl": round(wf_pnl, 3),
        "paper_score": paper_score,
        "paper_samples": paper_samples,
    }

    if wf_trades >= wf_required:
        # Generalisatiecheck: out-of-sample (walk-forward) winrate mag niet ver onder
        # de in-sample winrate zakken — dat duidt op overfitting.
        overfit = winrate > 0 and wf_winrate < WF_GENERALIZATION_RATIO * winrate
        if overfit:
            details["generalization_ratio"] = round(wf_winrate / winrate, 2) if winrate else None
        if wf_winrate < wf_min_wr or wf_pnl <= -0.25 or overfit:
            if paper_score >= 0.45 and pnl > 0 and not overfit:
                details["exploration"] = True
                details["walk_forward_caution"] = True
                return (
                    True,
                    (
                        f"Walk-forward zwak ({wf_winrate:.0f}% winrate, {wf_pnl:+.2f}% P&L), "
                        "maar recente backtest/paper-score laten kleine exploratie toe."
                    ),
                    details,
                )
            if overfit:
                details["walk_forward_overfit"] = True
                return (
                    False,
                    (
                        f"Walk-forward overfitting voor {strategy}: out-of-sample "
                        f"{wf_winrate:.0f}% vs in-sample {winrate:.0f}% winrate."
                    ),
                    details,
                )
            return (
                False,
                f"Walk-forward out-of-sample zwak voor {strategy}: {wf_winrate:.0f}% winrate, {wf_pnl:+.2f}% P&L.",
                details,
            )
        details["walk_forward_ok"] = True
    elif len(trades) >= required_trades:
        details["walk_forward_insufficient"] = True

    if winrate < MIN_BACKTEST_WINRATE or pnl <= MIN_BACKTEST_PNL:
        if winrate >= 45.0 and pnl >= -0.75 and paper_score >= 0.35:
            details["exploration"] = True
            return (
                True,
                f"Exploratie-setup: backtest bijna neutraal ({winrate:.0f}% winrate, {pnl:+.2f}% P&L), kleine inzet.",
                details,
            )
        return (
            False,
            f"Recente netto backtest zwak voor {strategy}: {winrate:.0f}% winrate, {pnl:+.2f}% P&L.",
            details,
        )

    return (
        True,
        (
            f"Bewezen setup: backtest {winrate:.0f}% winrate, {pnl:+.2f}% netto P&L; "
            f"walk-forward {wf_winrate:.0f}%/{wf_pnl:+.2f}% ({wf_trades} trades)."
        ),
        details,
    )


def _panic_long_exploration_allowed(
    asset: str,
    strategy: str,
    richting: str,
    fear_greed: float,
    confidence: dict,
    quality: dict,
    score_info: dict,
) -> tuple[bool, str]:
    """Laat in extreme fear alleen zeer sterke BTC-long setups met mini-inzet toe."""
    if fear_greed > 25 or richting != "long":
        return True, "Geen extreme-fear longfilter actief."

    if asset != "BTC-USD":
        return False, f"Extreme Fear ({fear_greed}) - alt longs blijven geblokkeerd."
    if strategy not in PANIC_LONG_STRATEGIES:
        return False, f"Extreme Fear ({fear_greed}) - {strategy} is geen toegestane BTC long/reversal exploratie."

    confidence_score = int(confidence.get("score") or 0)
    if confidence_score < PANIC_LONG_CONFIDENCE_SCORE:
        return False, (
            f"Extreme Fear ({fear_greed}) - BTC long alleen vanaf confidence "
            f"{PANIC_LONG_CONFIDENCE_SCORE}, nu {confidence_score}."
        )

    quality_score = int(quality.get("score") or 0)
    if quality_score < 65:
        return False, f"Extreme Fear ({fear_greed}) - setupkwaliteit te laag voor panic-exploratie ({quality_score}/65)."

    paper_samples = int(score_info.get("sample_count") or 0)
    paper_score = float(score_info.get("score") or 0.5)
    if paper_samples >= 10 and paper_score < 0.35:
        return False, f"Extreme Fear ({fear_greed}) - paper-score te zwak voor panic-exploratie ({paper_score:.0%})."

    return True, (
        f"Extreme Fear ({fear_greed}) - BTC cautious exploration toegestaan: "
        f"confidence {confidence_score}/100, kwaliteit {quality_score}/100, mini-inzet."
    )


def _controleer_markt(asset: str):
    """Scan alle strategieen voor een markt en open de beste setup."""
    mag, reden = _mag_handelen(asset)
    if not mag:
        record_decision(asset, None, "skipped", reden)
        _log(f"{asset}: overgeslagen - {reden}")
        return

    mag_btc, btc_reden = _mag_asset_in_btc_first_modus(asset)
    if not mag_btc:
        record_decision(asset, None, "skipped", btc_reden)
        _log(f"{asset}: overgeslagen - {btc_reden}")
        return

    try:
        df = fetch_live_crypto_candles(asset, CANDLE_SECONDS, CANDLE_LIMIT)
    except Exception as e:
        mark_api_health(asset, "down", str(e))
        record_decision(
            asset=asset,
            strategy=None,
            action="data_unavailable",
            reason=f"Data ophalen mislukt; geen nieuwe trade geopend: {e}",
        )
        _log(f"{asset}: data ophalen mislukt - {e}", "warning")
        melding_verbindingsfout(asset, e)
        return

    if "_simulated" in df.columns:
        mark_api_health(asset, "down", "geen internetverbinding — gesimuleerde data")
        _log(f"{asset}: geen live data beschikbaar (DNS/netwerk) — cyclus overgeslagen", "warning")
        return

    fresh, fresh_reason = latest_candle_is_fresh(df, max_age_seconds=CANDLE_SECONDS * 3 + 120)
    if not fresh:
        # Eén herhalingspoging na 5 seconden wachten
        _log(f"{asset}: data niet vers ({fresh_reason}) — herhalingspoging over 5s...", "warning")
        time.sleep(5)
        try:
            df = fetch_live_crypto_candles(asset, CANDLE_SECONDS, CANDLE_LIMIT)
            if "_simulated" in df.columns:
                # Synthetische candles hebben verse timestamps en zouden anders door de
                # freshness-check glippen. Nooit handelen op gesimuleerde data.
                mark_api_health(asset, "down", "geen internetverbinding — gesimuleerde data")
                record_decision(
                    asset=asset,
                    strategy=None,
                    action="data_unavailable",
                    reason="Alleen gesimuleerde data beschikbaar na herhalingspoging; geen trade geopend.",
                )
                _log(f"{asset}: alleen gesimuleerde data na herhaling — cyclus overgeslagen", "warning")
                return
            fresh, fresh_reason = latest_candle_is_fresh(df, max_age_seconds=CANDLE_SECONDS * 3 + 120)
        except Exception as e:
            fresh, fresh_reason = False, str(e)

        if not fresh:
            mark_api_health(asset, "stale", fresh_reason)
            record_decision(
                asset=asset,
                strategy=None,
                action="data_unavailable",
                reason=f"Candledata niet vers na herhalingspoging; geen nieuwe trade geopend: {fresh_reason}",
            )
            _log(f"{asset}: data niet vers - {fresh_reason}", "warning")
            return
    mark_api_health(asset, "ok", "marktdata ontvangen")

    live_prijs = float(df.iloc[-1]["close"])

    # Verwijder de huidige nog-niet-gesloten candle — signalen alleen op bevestigde candles
    df = df.iloc[:-1].reset_index(drop=True)
    try:
        virtueel = evaluate_skipped_setups(asset, df)
        if virtueel:
            _log(f"{asset}: false-positive geheugen bijgewerkt ({virtueel} geweigerde setups geëvalueerd).")
    except Exception as e:
        _log(f"{asset}: false-positive evaluatie mislukt - {e}", "warning")

    leren      = _laad_leren()
    min_rr     = _actieve_min_rr(leren)
    gepauzeerd = leren.get("aanpassingen", {}).get("gepauzeerde_strategieen", [])

    try:
        boot = ensure_bootstrap(asset, df, timeframe=f"{CANDLE_SECONDS}s", min_rr=min_rr)
        if boot["ran"]:
            _log(f"{asset}: learning bootstrap bijgewerkt ({boot['samples']} samples, {boot['reason']}).")
    except Exception as e:
        record_decision(
            asset=asset,
            strategy=None,
            action="bootstrap_failed",
            reason=f"Bootstrap overgeslagen door fout: {e}",
        )
        _log(f"{asset}: learning bootstrap mislukt - {e}", "warning")

    # ── Hogere timeframe (1u) trendfilter ─────────────────────────────────────
    htf_trend = _get_htf_trend(asset)
    _log(f"{asset}: HTF (1u) trend = {htf_trend}")

    # ── Fear & Greed Index ─────────────────────────────────────────────────────
    fg = _fetch_fear_greed_safe()
    fg_waarde = fg["value"]
    fg_label  = fg["label"]
    _log(f"{asset}: Fear & Greed = {fg_waarde} ({fg_label})")

    # Liquidatiezones ophalen (alleen als Coinglass geconfigureerd is)
    liq_zones = []
    try:
        if heeft_coinglass_key():
            liq_zones = haal_liquidatie_zones(asset)
    except Exception:
        pass

    external_context = fetch_market_context(asset)
    news_context = news_watch_snapshot()

    try:
        analyse = analyze_market(df)
    except Exception:
        analyse = {}

    regime = classify_market_regime(df, htf_trend=htf_trend, fear_greed=fg_waarde)
    _log(
        f"{asset}: regime = {regime['label']} "
        f"(score {regime['score']}/100, {regime['reason']})"
    )
    if asset == "BTC-USD":
        global BTC_MARKT
        BTC_MARKT = _bepaal_btc_marktmodus(regime, htf_trend, fg_waarde, external_context)
        _log(f"BTC-first modus: {BTC_MARKT['mode']} - {BTC_MARKT['reason']}")

    setups = scan_alle_strategieen(df, min_rr=min_rr, liquidation_zones=liq_zones)
    # Strategie-whitelist (env TRADEAI_STRATEGIES, default = alle).
    setups = [s for s in setups if s.get("strategie") in ENABLED_STRATEGIES]
    if not setups:
        record_decision(asset, None, "no_setup", "Geen setup op alle strategieen.")
        _log(f"{asset}: geen setup op alle strategieen.")
        return

    kandidaten = []
    for setup in setups:
        strategy  = setup.get("strategie", "?")
        richting  = setup.get("richting", "long")
        setup["asset"] = asset
        setup["_market_regime"] = regime
        setup, asset_param_reden, asset_param_details = _pas_asset_strategy_params_toe(
            asset, setup, regime, htf_trend
        )
        if setup is None:
            record_decision(
                asset, strategy, "skipped", asset_param_reden,
                details=asset_param_details,
            )
            _log(f"{asset}: {strategy} overgeslagen - {asset_param_reden}")
            continue

        context = build_market_context(
            asset,
            setup,
            df,
            analyse=analyse,
            timeframe=f"{CANDLE_SECONDS}s",
            htf_trend=htf_trend,
            fear_greed=fg_waarde,
        )
        context["fear_greed_value"] = fg_waarde
        score_info = score_setup(context)
        setup["_learning_context"] = context
        setup["_learning_score"] = score_info
        # Meta-label filter (numpy-only, degradeert naar neutraal bij weinig data).
        try:
            setup["_meta_label"] = meta_label_score(context)
        except Exception:
            setup["_meta_label"] = {"available": False, "prob_win": 0.5}

        if strategy in gepauzeerd:
            reden = f"Strategie '{strategy}' is tijdelijk gepauzeerd door eerdere resultaten."
            record_decision(asset, strategy, "skipped", reden, context, score_info)
            _log(f"{asset}: overgeslagen - {reden}")
            continue

        if score_info["should_block"]:
            reden = f"Slechte leercontext geblokkeerd: {score_info['reason']}"
            record_decision(asset, strategy, "skipped", reden, context, score_info)
            _log(f"{asset}: {strategy} overgeslagen - {reden}", "warning")
            continue

        regime_ok, regime_reden = strategy_allowed_in_regime(strategy, regime)
        if not regime_ok:
            details = {"regime": regime}
            record_decision(
                asset, strategy, "skipped", regime_reden,
                context, score_info, details=details,
            )
            _log(f"{asset}: {strategy} overgeslagen - {regime_reden}")
            continue

        strategy_profile = get_strategy_regime_profile(
            asset, strategy, regime.get("label", "unknown")
        )
        setup["_strategy_profile"] = strategy_profile
        if strategy_profile.get("action") == "block":
            reden = f"Strategieprofiel blokkeert setup: {strategy_profile['reason']}"
            record_decision(
                asset, strategy, "skipped", reden,
                context, score_info,
                details={"regime": regime, "strategy_profile": strategy_profile},
            )
            _log(f"{asset}: {strategy} overgeslagen - {reden}", "warning")
            continue

        time_profile = get_time_strategy_profile(asset, strategy, context)
        setup["_time_profile"] = time_profile
        if time_profile.get("action") == "block":
            reden = f"Tijdprofiel blokkeert setup: {time_profile['reason']}"
            record_decision(
                asset, strategy, "skipped", reden,
                context, score_info,
                details={"regime": regime, "time_profile": time_profile},
            )
            _log(f"{asset}: {strategy} overgeslagen - {reden}", "warning")
            continue

        bewijs_ok, bewijs_reden, bewijs_details = _setup_heeft_bewijs(
            asset, df, strategy, min_rr, score_info
        )
        if not bewijs_ok:
            record_decision(
                asset, strategy, "skipped", bewijs_reden,
                context, score_info, details=bewijs_details,
            )
            _log(f"{asset}: {strategy} overgeslagen - {bewijs_reden}")
            continue
        setup["_proof_reason"] = bewijs_reden
        setup["_proof_details"] = bewijs_details

        kwaliteit = score_trade_quality(df, setup, liq_zones, external_context)
        setup["_quality"] = kwaliteit
        if kwaliteit.get("block"):
            reden = f"Setupkwaliteit te laag ({kwaliteit['score']}/100): {', '.join(kwaliteit['reasons'])}"
            record_decision(
                asset, strategy, "skipped", reden,
                context, score_info,
                details={"quality": kwaliteit, "regime": regime, "market_context": external_context},
            )
            _log(f"{asset}: {strategy} overgeslagen - {reden}")
            continue

        # ── HTF-trendfilter ────────────────────────────────────────────────────
        # Longs: alleen blokkeren bij expliciete downtrend
        # Shorts: vereisen een expliciete downtrend — "onduidelijk" is niet genoeg
        if htf_trend == "downtrend" and richting == "long":
            reden = f"HTF (1u) is '{htf_trend}' — geen long tegen duidelijke downtrend."
            record_decision(asset, strategy, "skipped", reden, context, score_info)
            _log(f"{asset}: {strategy} overgeslagen - {reden}")
            continue
        if htf_trend == "uptrend" and richting == "short":
            reden = f"HTF (1u) is '{htf_trend}' — geen short tegen duidelijke uptrend."
            record_decision(asset, strategy, "skipped", reden, context, score_info)
            _log(f"{asset}: {strategy} overgeslagen - {reden}")
            continue

        htf_ok = (
            (richting == "long" and htf_trend == "uptrend")
            or (richting == "short" and htf_trend == "downtrend")
        )
        confidence = calculate_confidence(
            regime_ok=regime_ok,
            htf_ok=htf_ok,
            proof_details=bewijs_details,
            score_info=score_info,
            quality=kwaliteit,
            market_context=external_context,
            strategy_profile=strategy_profile,
            time_profile=time_profile,
        )
        calibration = _confidence_calibration_profile(asset, strategy, int(confidence.get("score") or 0))
        if calibration.get("confidence_adjustment"):
            confidence = dict(confidence)
            confidence["score"] = max(0, min(100, int(confidence["score"]) + int(calibration["confidence_adjustment"])))
            confidence["reasons"] = list(confidence.get("reasons", [])) + [calibration["reason"]]
        setup["_confidence"] = confidence
        setup["_confidence_calibration"] = calibration
        setup["_external_context"] = external_context
        min_conf = _effectieve_min_confidence(asset)
        rules = setup.get("_asset_strategy_rules") or {}
        min_conf += int(rules.get("confidence_offset") or ASSET_PARAMS.get(asset, {}).get("confidence_offset") or 0)
        if (setup.get("_proof_details") or {}).get("exploration"):
            min_conf = min(min_conf, EXPLORATION_CONFIDENCE_SCORE)
        if strategy in RANGE_STRATEGIES:
            min_conf = min(min_conf, 50)
        if fg_waarde <= 25 and asset == "BTC-USD" and richting == "long":
            min_conf = max(min_conf, PANIC_LONG_CONFIDENCE_SCORE)
        if confidence["score"] < min_conf:
            reden = f"Confidence te laag ({confidence['score']}/{min_conf}): {', '.join(confidence['reasons'])}"
            record_decision(
                asset, strategy, "skipped", reden,
                context, score_info,
                details={
                    "confidence": confidence,
                    "quality": kwaliteit,
                    "regime": regime,
                    "proof": bewijs_details,
                    "strategy_profile": strategy_profile,
                    "time_profile": time_profile,
                    "market_context": external_context,
                    "setup": {k: v for k, v in setup.items() if not k.startswith("_")},
                },
            )
            _log(f"{asset}: {strategy} overgeslagen - {reden}")
            continue

        checklist = _pre_trade_checklist(
            asset=asset,
            setup=setup,
            regime_ok=regime_ok,
            htf_ok=htf_ok,
            proof_details=bewijs_details,
            score_info=score_info,
            quality=kwaliteit,
            confidence=confidence,
            market_context=external_context,
            strategy_profile=strategy_profile,
            time_profile=time_profile,
        )
        setup["_pre_trade_checklist"] = checklist
        if not checklist["passed"]:
            record_decision(
                asset, strategy, "skipped", checklist["reason"],
                context, score_info,
                details={
                    "checklist": checklist,
                    "confidence": confidence,
                    "calibration": calibration,
                    "quality": kwaliteit,
                    "regime": regime,
                    "proof": bewijs_details,
                    "strategy_profile": strategy_profile,
                    "time_profile": time_profile,
                    "market_context": external_context,
                    "setup": {k: v for k, v in setup.items() if not k.startswith("_")},
                },
            )
            _log(f"{asset}: {strategy} overgeslagen - {checklist['reason']}")
            continue

        # ── Fear & Greed filter ────────────────────────────────────────────────
        if fg_waarde >= 80 and richting == "long":
            reden = f"Extreme Greed ({fg_waarde}) — geen nieuwe longs."
            record_decision(asset, strategy, "skipped", reden, context, score_info)
            _log(f"{asset}: {strategy} overgeslagen - {reden}")
            continue
        if fg_waarde <= 25 and richting == "long":
            panic_ok, panic_reden = _panic_long_exploration_allowed(
                asset=asset,
                strategy=strategy,
                richting=richting,
                fear_greed=fg_waarde,
                confidence=confidence,
                quality=kwaliteit,
                score_info=score_info,
            )
            if not panic_ok:
                record_decision(asset, strategy, "skipped", panic_reden, context, score_info)
                _log(f"{asset}: {strategy} overgeslagen - {panic_reden}")
                continue
            proof_details = dict(setup.get("_proof_details") or {})
            proof_details["panic_exploration"] = True
            setup["_proof_details"] = proof_details
            setup["_panic_exploration"] = True
            setup["_proof_reason"] = f"{setup.get('_proof_reason', '')} {panic_reden}".strip()
            setup["reden"] = list(setup.get("reden", [])) + [panic_reden]
        if fg_waarde <= 20 and richting == "short":
            reden = f"Extreme Fear ({fg_waarde}) — geen nieuwe shorts."
            record_decision(asset, strategy, "skipped", reden, context, score_info)
            _log(f"{asset}: {strategy} overgeslagen - {reden}")
            continue

        if (analyse.get("volatiliteit") == "hoog"
                and leren.get("winrate_recent", 1.0) < 0.45):
            reden = "Hoge volatiliteit + lage recente winrate."
            record_decision(asset, strategy, "skipped", reden, context, score_info)
            _log(f"{asset}: overgeslagen - {reden}")
            continue

        kandidaten.append(setup)

    if not kandidaten:
        _log(f"{asset}: alle setups overgeslagen door leer- of disciplinefilters.")
        return

    signal = max(
        kandidaten,
        key=lambda s: (s.get("risk_reward") or 0) + (s["_learning_score"]["score"] - 0.5) * 2.0,
    )

    # ── Correlatie-filter ────────────────────────────────────────────────────────
    # BTC, ETH en SOL bewegen sterk samen. Max 1 gelijktijdige long en 1 short
    # in deze groep, anders stapel je hetzelfde risico drie keer op.
    if asset in GECORRELEERDE_ASSETS:
        richting_signal = signal.get("richting", "long")
        al_open = _gecorreleerde_open(richting_signal)
        if al_open >= 1:
            reden = (
                f"Correlatie-filter: al {al_open} open {richting_signal}-trade(s) "
                f"in gecorreleerde crypto-majors."
            )
            record_decision(asset, signal.get("strategie"), "skipped", reden,
                            signal["_learning_context"], signal["_learning_score"])
            _log(f"{asset}: overgeslagen - {reden}")
            return

    entry_ok, entry_reden, entry_details = _entry_execution_gate(signal, live_prijs)
    if not entry_ok:
        record_decision(asset, signal.get("strategie"), "skipped", entry_reden,
                        signal["_learning_context"], signal["_learning_score"],
                        details={"entry_gate": entry_details})
        _log(f"{asset}: {signal.get('strategie')} overgeslagen - {entry_reden}")
        return

    aangepast, reden = _pas_entry_aan_naar_live_prijs(signal, live_prijs)
    if aangepast is None:
        record_decision(asset, signal.get("strategie"), "skipped", reden,
                        signal["_learning_context"], signal["_learning_score"])
        _log(f"{asset}: {signal.get('strategie')} overgeslagen - {reden}")
        return
    signal = aangepast

    strat_nm = signal.get("strategie_naam", signal.get("strategie", "?"))
    kapitaal, size_details = _bereken_positiegrootte(signal)
    edge_ok, edge_reden, edge_details = _verwachte_netto_reward_gate(signal, kapitaal)
    signal["_net_edge"] = edge_details
    if not edge_ok:
        record_decision(
            asset,
            signal.get("strategie"),
            "skipped",
            edge_reden,
            signal["_learning_context"],
            signal["_learning_score"],
            details={
                "net_edge": edge_details,
                "position_sizing": size_details,
                "setup": {k: v for k, v in signal.items() if not k.startswith("_")},
            },
        )
        _log(f"{asset}: {signal.get('strategie')} overgeslagen - {edge_reden}")
        return
    committee = _agent_committee(signal, size_details, edge_details, news_context)
    signal["_agent_committee"] = committee
    if not committee["passed"]:
        record_decision(
            asset,
            signal.get("strategie"),
            "skipped",
            committee["reason"],
            signal["_learning_context"],
            signal["_learning_score"],
            details={
                "agent_committee": committee,
                "net_edge": edge_details,
                "position_sizing": size_details,
                "setup": {k: v for k, v in signal.items() if not k.startswith("_")},
            },
        )
        _log(f"{asset}: {signal.get('strategie')} overgeslagen - {committee['reason']}")
        return
    saldo    = _huidig_saldo()
    context = dict(signal["_learning_context"])
    context.update({
        "confidence_score": int((signal.get("_confidence") or {}).get("score") or 0),
        "confidence_bucket": _confidence_bucket((signal.get("_confidence") or {}).get("score")),
        "quality_score": int((signal.get("_quality") or {}).get("score") or 0),
        "checklist_score": int((signal.get("_pre_trade_checklist") or {}).get("score") or 0),
        "checklist_passed": bool((signal.get("_pre_trade_checklist") or {}).get("passed")),
        "calibration_action": (signal.get("_confidence_calibration") or {}).get("action", "neutral"),
        "agent_committee_score": committee["score"],
        "agent_committee_passed": committee["passed"],
    })
    signal["_learning_context"] = context
    score_info = signal["_learning_score"]

    richting_label = signal.get("richting", "long").upper()
    _log(
        f"{asset}: SETUP [{strat_nm}] {richting_label} @ {signal['entry']} | "
        f"SL {signal['stop_loss']} | TP {signal['take_profit']} | "
        f"R/R 1:{signal['risk_reward']} | Leerscore {score_info['score']:.0%} "
        f"({score_info['sample_count']} samples) | Confidence {signal.get('_confidence', {}).get('score', 0)}/100 | "
        f"Profiel: {(signal.get('_strategy_profile') or {}).get('action', 'neutral')} "
        f"({(signal.get('_strategy_profile') or {}).get('score', 0.5):.0%}, "
        f"{(signal.get('_strategy_profile') or {}).get('samples', 0)} samples) | "
        f"Tijd: {(signal.get('_time_profile') or {}).get('action', 'neutral')} "
        f"({(signal.get('_time_profile') or {}).get('score', 0.5):.0%}, "
        f"{(signal.get('_time_profile') or {}).get('samples', 0)} samples) | "
        f"{signal.get('_proof_reason', '')} | "
        f"{committee['reason']} | "
        f"{edge_reden} | "
        f"Inzet EUR{kapitaal:.2f} | Risico EUR{size_details['risk_euro']:.2f} "
        f"({size_details['risk_euro'] / saldo * 100:.2f}% saldo) "
        f"({size_details['confidence_multiplier']}x, "
        f"{size_details['confidence_reason']}; saldo: EUR{saldo:.2f})"
    )

    trade_id = add_paper_trade(
        asset       = asset,
        direction   = signal["richting"],
        entry_price = signal["entry"],
        stop_loss   = signal["stop_loss"],
        take_profit = signal["take_profit"],
        reason      = f"strategie:{signal.get('strategie','?')} | " + "; ".join(signal["reden"]),
        capital     = kapitaal,
        source      = "auto",
        strategy    = signal.get("strategie"),
        market_context   = context,
        learning_score   = score_info["score"],
        learning_samples = score_info["sample_count"],
        learning_reason  = score_info["reason"],
    )
    record_decision(
        asset=asset,
        strategy=signal.get("strategie"),
        action="opened",
        reason=score_info["reason"],
        context=context,
        score_info=score_info,
        trade_id=trade_id,
        details={
            "risk_reward": signal.get("risk_reward"),
            "strategy_name": strat_nm,
            "regime": signal.get("_market_regime", {}),
            "proof": signal.get("_proof_details", {}),
            "strategy_profile": signal.get("_strategy_profile", {}),
            "time_profile": signal.get("_time_profile", {}),
            "quality": signal.get("_quality", {}),
            "confidence": signal.get("_confidence", {}),
            "confidence_calibration": signal.get("_confidence_calibration", {}),
            "pre_trade_checklist": signal.get("_pre_trade_checklist", {}),
            "position_sizing": size_details,
            "net_edge": edge_details,
            "agent_committee": committee,
        },
    )
    _log(f"Trade #{trade_id} geopend: {asset} {richting_label} via [{strat_nm}] @ {signal['entry']}")
    melding_trade_geopend(
        trade_id=trade_id, asset=asset,
        entry=signal["entry"], sl=signal["stop_loss"], tp=signal["take_profit"],
        rr=signal.get("risk_reward", 0), kapitaal=kapitaal, strategie=strat_nm,
        richting=signal.get("richting", "long"),
        risico_euro=size_details["risk_euro"],
    )


# ── Dagelijkse samenvatting ───────────────────────────────────────────────────

def genereer_dagelijkse_samenvatting(datum: str = None) -> str:
    """Genereer een samenvatting van de afgelopen handelsdag."""
    gisteren = datum or (datetime.now(AMSTERDAM) - timedelta(days=1)).date().isoformat()
    alle = get_auto_trades()

    dag_trades = [
        t for t in alle
        if t.get("closed_at", "")[:10] == gisteren
        or t.get("created_at", "")[:10] == gisteren
    ]
    gesloten_dag = [t for t in dag_trades if t["status"] == "closed"]
    open_dag     = [t for t in dag_trades if t["status"] == "open"]

    wins   = [t for t in gesloten_dag if (t.get("result_pct") or 0) > 0]
    losses = [t for t in gesloten_dag if (t.get("result_pct") or 0) <= 0]
    totaal_pnl = sum(t.get("result_euro", 0) or 0 for t in gesloten_dag)
    winrate    = len(wins) / len(gesloten_dag) * 100 if gesloten_dag else 0
    saldo      = _huidig_saldo()
    leren      = _laad_leren()

    regels = [
        f"════════════════════════════════════════",
        f"  TRADEAI BOT — DAGRAPPORT {gisteren}",
        f"════════════════════════════════════════",
        f"",
        f"SALDO",
        f"  Gesimuleerd kapitaal:  €{saldo:>10,.2f}",
        f"  P&L vandaag:           €{totaal_pnl:>+10,.2f}",
        f"",
        f"TRADES",
        f"  Gesloten:  {len(gesloten_dag)}  (✓ {len(wins)} winst | ✗ {len(losses)} verlies)",
        f"  Open:      {len(open_dag)}",
        f"  Winrate:   {winrate:.0f}%",
        f"",
    ]

    if gesloten_dag:
        regels.append("DETAIL PER TRADE")
        for t in gesloten_dag:
            icon = "✓" if (t.get("result_pct") or 0) > 0 else "✗"
            regels.append(
                f"  {icon} #{t['id']} {t['asset']} {t.get('direction', 'long').upper()} "
                f"entry {t['entry_price']:.2f} → exit {t.get('exit_price', '?')} "
                f"| {t.get('result_pct', 0):+.2f}% | €{t.get('result_euro', 0):+.2f} "
                f"| reden: {t.get('exit_reason', '?')}"
            )
        regels.append("")

    # Wat ging goed
    regels.append("WAT GING GOED")
    if wins:
        for t in wins:
            regels.append(
                f"  ✓ Trade #{t['id']} {t['asset']}: +{t.get('result_pct', 0):.2f}% "
                f"— setup correct herkend, discipline aangehouden."
            )
    else:
        regels.append("  Geen winstgevende trades vandaag.")

    regels.append("")
    regels.append("LEERPUNTEN")
    if losses:
        for t in losses:
            reflectie = t.get("mistake_analysis") or "Automatisch stop-loss geraakt."
            regels.append(f"  ✗ Trade #{t['id']} {t['asset']}: {reflectie[:200]}")
    else:
        regels.append("  Geen verliezen vandaag — uitstekende discipline!")

    regels += [
        "",
        "BOTINSTELLINGEN (huidig)",
        f"  Min R/R:       1:{leren.get('min_rr', 2.0):.1f}",
        f"  Winrate recent: {leren.get('winrate_recent', 0.5)*100:.0f}%",
        f"  Cooling:       {'actief' if _in_cooling() else 'niet actief'}",
        "",
        "════════════════════════════════════════",
    ]

    inhoud = "\n".join(regels)
    save_daily_summary(gisteren, inhoud)
    return inhoud


# ── Hoofdcyclus ───────────────────────────────────────────────────────────────

_DRAWDOWN_ALERT_DAG = None  # datum waarop al een drawdown-alert is verstuurd (throttle)


def _bouw_llm_snapshot(asset: str, fg: dict) -> dict:
    """Bouw een compacte marktcontext voor het LLM-panel (faalt zacht)."""
    snap = {
        "fear_greed": fg.get("value"),
        "fear_greed_label": fg.get("label"),
        "btc_mode": BTC_MARKT.get("mode"),
    }
    try:
        df = fetch_live_crypto_candles(asset, CANDLE_SECONDS, CANDLE_LIMIT)
        if "_simulated" not in df.columns and len(df) > 60:
            df = add_all_indicators(df.iloc[:-1].reset_index(drop=True))
            regime = classify_market_regime(df, htf_trend="onduidelijk", fear_greed=fg.get("value", 50))
            last = df.iloc[-1]
            snap.update({
                "regime": regime.get("label"),
                "regime_score": regime.get("score"),
                "rsi": round(float(last.get("rsi14", 0) or 0), 1),
                "adx": round(float(last.get("adx14", 0) or 0), 1),
                "close": round(float(last["close"]), 4),
                "sma20": round(float(last.get("sma20", 0) or 0), 4),
                "sma50": round(float(last.get("sma50", 0) or 0), 4),
            })
    except Exception:
        pass
    return snap


def _draai_llm_panel_indien_nodig(max_leeftijd_seconden: int = 3600) -> None:
    """Draai het LLM-panel per asset, throttled (standaard ~1x per uur). Alleen als ingeschakeld."""
    if not llm_is_enabled():
        return
    fg = _fetch_fear_greed_safe()
    for asset in MARKETS:
        try:
            advies = get_latest_llm_advice(asset)
            if advies:
                leeftijd = (
                    datetime.now(timezone.utc)
                    - pd.to_datetime(advies["created_at"], utc=True).to_pydatetime()
                ).total_seconds()
                if leeftijd < max_leeftijd_seconden:
                    continue  # advies nog vers genoeg
            snapshot = _bouw_llm_snapshot(asset, fg)
            result = llm_run_panel(asset, snapshot)
            if result.get("available"):
                _log(
                    f"{asset}: LLM-advies bias={result['bias']:+.2f} "
                    f"conf={result['confidence']:.0%} Δmin_conf={result['min_confidence_delta']:+.1f}"
                )
        except Exception as e:
            _log(f"{asset}: LLM-panel mislukt - {e}", "warning")


def run_cycle():
    """Voer één volledige bot-cyclus uit."""
    global _DRAWDOWN_ALERT_DAG
    init_database()
    _htf_cache.clear()  # HTF-trend per cyclus opnieuw ophalen
    _proof_cache.clear()
    global BTC_MARKT
    BTC_MARKT = {"mode": "unknown", "reason": "Nog niet bepaald."}
    _log("=== Bot-cyclus gestart ===")

    # 1. Controleer SL/TP van open trades
    _controleer_open_trades()

    # 2. Controleer of dagverlies-limiet bereikt is
    dagverlies = _dagverlies_vandaag()
    limiet     = STARTKAPITAAL * MAX_DAGVERLIES
    if dagverlies < -limiet:
        _log(
            f"Dagverlies-limiet bereikt (€{dagverlies:.2f}). "
            f"Geen nieuwe trades voor de rest van de dag.", "warning"
        )
        vandaag = datetime.now(timezone.utc).date().isoformat()
        if _DRAWDOWN_ALERT_DAG != vandaag:
            melding_drawdown_limiet(dagverlies, limiet)
            _DRAWDOWN_ALERT_DAG = vandaag
        mark_heartbeat(f"gepauzeerd — dagverlies-limiet bereikt (€{dagverlies:.2f})")
        return

    # 2b. Optionele LLM-adviseslaag (standaard uit; throttled tot ~1x per uur per asset)
    _draai_llm_panel_indien_nodig()

    # 3. Controleer elk geconfigureerd market
    for asset in MARKETS:
        _controleer_markt(asset)

    saldo = _huidig_saldo()
    _log(f"Cyclus klaar. Huidig gesimuleerd saldo: €{saldo:,.2f}")
    mark_heartbeat(f"cyclus klaar — saldo €{saldo:,.2f}")
