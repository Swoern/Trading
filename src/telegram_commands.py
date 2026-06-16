"""
telegram_commands.py - Telegram command interface voor TradeAI.

Alle menu's, callback-acties en command-formatting staan hier zodat
notifier.py gericht kan blijven op uitgaande meldingen en polling.
"""
from __future__ import annotations

import html
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

from src.notifier import _config, melding_bot_herstart, melding_website_herstart

AMSTERDAM = ZoneInfo("Europe/Amsterdam")
PROJECT_DIR = Path(__file__).parent.parent
MARKETS = ("BTC-USD", "ETH-USD", "SOL-USD")
PRICE_CACHE_SECONDS = 45
_price_cache: dict[str, tuple[float, float, float]] = {}
ETA_CACHE_SECONDS = 45
_eta_cache: dict[str, tuple[float, float]] = {}


# -- Telegram API -------------------------------------------------------------

def _api_url(method: str) -> str | None:
    token = _config().get("token", "").strip()
    if not token:
        return None
    return f"https://api.telegram.org/bot{token}/{method}"


def _post(method: str, payload: dict) -> dict:
    url = _api_url(method)
    if not url:
        return {"ok": False, "description": "Telegram token ontbreekt."}
    try:
        response = requests.post(url, json=payload, timeout=10)
        try:
            return response.json()
        except Exception:
            return {"ok": response.ok, "description": response.text[:300]}
    except Exception as exc:
        return {"ok": False, "description": str(exc)}


def _send_message(chat_id: str, text: str, reply_markup: dict | None = None) -> bool:
    payload = {
        "chat_id": chat_id,
        "text": _limit_message(text),
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup
    return bool(_post("sendMessage", payload).get("ok"))


def _edit_message(chat_id: str, message_id: int, text: str, reply_markup: dict | None = None) -> bool:
    payload = {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": _limit_message(text),
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup
    result = _post("editMessageText", payload)
    if result.get("ok"):
        return True
    return _send_message(chat_id, text, reply_markup)


def _answer_callback(callback_query_id: str, text: str = "") -> bool:
    payload = {"callback_query_id": callback_query_id}
    if text:
        payload["text"] = text[:180]
    return bool(_post("answerCallbackQuery", payload).get("ok"))


# -- Formatting helpers -------------------------------------------------------

def _limit_message(text: str, limit: int = 4096) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _now() -> str:
    return datetime.now(AMSTERDAM).strftime("%H:%M")


def _escape(value) -> str:
    return html.escape(str(value), quote=False)


def _fmt_number(value: float, decimals: int = 2) -> str:
    return f"{value:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _fmt_eur(value: float) -> str:
    return f"EUR {_fmt_number(float(value), 2)}"


def _fmt_usd(value: float) -> str:
    return f"$ {_fmt_number(float(value), 2)}"


def _fmt_price(value: float) -> str:
    value = float(value)
    decimals = 4 if abs(value) < 100 else 2
    return _fmt_number(value, decimals)


def _fmt_pct(value: float, signed: bool = True) -> str:
    prefix = "+" if signed and value > 0 else ""
    return f"{prefix}{_fmt_number(float(value), 2)}%"


def _pnl_pct(direction: str, entry: float, live_price: float) -> float:
    if direction == "short":
        return (entry - live_price) / entry * 100
    return (live_price - entry) / entry * 100


def _trade_result_preview(trade: dict, exit_price: float) -> tuple[float, float]:
    entry = float(trade["entry_price"])
    capital = float(trade.get("capital") or 0)
    partial_pnl = float(trade.get("partial_pnl_euro") or 0)
    pct_on_remaining = _pnl_pct(trade.get("direction", "long"), entry, exit_price)
    euro = pct_on_remaining / 100 * capital + partial_pnl
    original = float(trade.get("original_capital") or 0)
    if original <= 0:
        original = capital * 2 if trade.get("partial_tp_hit") else capital
    pct = euro / original * 100 if original > 0 else pct_on_remaining
    return pct, euro


def _fetch_live_price(asset: str) -> float:
    from src.data_loader import fetch_live_crypto_candles

    df = fetch_live_crypto_candles(asset, 60, 2)
    return float(df.iloc[-1]["close"])


def _fetch_atr_per_minute(asset: str) -> float:
    from src.data_loader import fetch_live_crypto_candles
    from src.indicators import add_all_indicators

    cached = _eta_cache.get(asset)
    now = time.time()
    if cached and now - cached[0] < ETA_CACHE_SECONDS:
        return cached[1]

    df = fetch_live_crypto_candles(asset, 60, 80)
    df = add_all_indicators(df)
    atr = float(df.iloc[-1].get("atr14", 0) or 0)
    _eta_cache[asset] = (now, atr)
    return atr


def _format_eta_time(minutes: float) -> str:
    if minutes <= 1:
        return "<1 min"
    if minutes < 90:
        return f"~{int(round(minutes))} min"
    hours = minutes / 60
    if hours < 24:
        return f"~{hours:.1f} uur".replace(".", ",")
    return f"~{hours / 24:.1f} dagen".replace(".", ",")


def _estimate_trade_eta(trade: dict, live_price: float) -> str:
    """
    Ruwe looptijdschatting op basis van recente 1m ATR.
    Geen voorspelling: alleen afstand tot TP/SL gedeeld door gemiddeld candlebereik.
    """
    try:
        atr = _fetch_atr_per_minute(trade["asset"])
        if atr <= 0:
            return "TP/SL schatting: onbekend"

        tp = float(trade["take_profit"])
        sl = float(trade["stop_loss"])
        direction = trade.get("direction", "long")
        if direction == "short":
            tp_distance = max(live_price - tp, 0)
            sl_distance = max(sl - live_price, 0)
        else:
            tp_distance = max(tp - live_price, 0)
            sl_distance = max(live_price - sl, 0)

        tp_eta = _format_eta_time(tp_distance / atr)
        sl_eta = _format_eta_time(sl_distance / atr)
        return f"TP/SL schatting: TP {tp_eta} | SL {sl_eta}"
    except Exception:
        return "TP/SL schatting: onbekend"


def _fetch_daily_change_pct(asset: str) -> tuple[float, float]:
    from src.data_loader import fetch_live_crypto_candles

    cached = _price_cache.get(asset)
    now = time.time()
    if cached and now - cached[0] < PRICE_CACHE_SECONDS:
        return cached[1], cached[2]

    df = fetch_live_crypto_candles(asset, 86400, 2)
    current = float(df.iloc[-1]["close"])
    previous = float(df.iloc[-2]["close"]) if len(df) > 1 else current
    change = (current - previous) / previous * 100 if previous else 0.0
    _price_cache[asset] = (now, current, change)
    return current, change


# -- Keyboards ----------------------------------------------------------------

def _keyboard(rows: list[list[tuple[str, str]]]) -> dict:
    return {
        "inline_keyboard": [
            [{"text": text, "callback_data": data} for text, data in row]
            for row in rows
        ]
    }


def _keyboard_hoofd_menu() -> dict:
    return _keyboard([
        [("Live", "menu_info"), ("Trades", "menu_trades")],
        [("Beheer", "menu_beheer")],
    ])


def _keyboard_info_menu() -> dict:
    return _keyboard([
        [("Prijzen", "info_prijzen"), ("Posities", "info_posities")],
        [("Stats", "info_stats")],
        [("Terug", "menu_hoofd")],
    ])


def _keyboard_info_back() -> dict:
    return _keyboard([[("Terug", "menu_info")]])


def _keyboard_trades_menu(trades: list[dict]) -> dict:
    rows = []
    for trade in trades:
        text = (
            f"#{trade['id']} {trade['asset']} "
            f"{trade.get('direction', 'long').upper()} @ {_fmt_price(trade['entry_price'])}"
        )
        rows.append([(text, f"trade_select:{trade['id']}")])
    rows.append([("Terug", "menu_hoofd")])
    return _keyboard(rows)


def _keyboard_trade_detail(trade_id: int) -> dict:
    return _keyboard([
        [("Sluit trade", f"trade_close:{trade_id}")],
        [("Terug", "menu_trades")],
    ])


def _keyboard_trade_close_confirm(trade_id: int) -> dict:
    return _keyboard([
        [("Ja, sluiten", f"trade_close_confirm:{trade_id}")],
        [("Nee, annuleer", "menu_trades")],
    ])


def _keyboard_beheer_menu() -> dict:
    return _keyboard([
        [("Herstart bot", "restart_bot"), ("Dashboard", "restart_web")],
        [("Waarom geen trade?", "beheer_waarom"), ("Regime stats", "beheer_regime_stats")],
        [("Logs", "beheer_logs")],
        [("Terug", "menu_hoofd")],
    ])


def _keyboard_restart_confirm(target: str) -> dict:
    return _keyboard([
        [("Ja, herstarten", f"restart_confirm:{target}")],
        [("Annuleer", "menu_beheer")],
    ])


def _keyboard_beheer_back() -> dict:
    return _keyboard([[("Terug", "menu_beheer")]])


# -- Message formatters -------------------------------------------------------

def _format_hoofd_menu() -> str:
    return (
        f"<b>TradeAI</b> - {_now()}\n"
        f"Kies een onderdeel."
    )


def _format_info_menu() -> str:
    return (
        f"<b>Live info</b> - {_now()}\n"
        f"Kies wat je wilt zien."
    )


def _format_beheer_menu() -> str:
    return (
        f"<b>Beheer</b> - {_now()}\n"
        f"Herstarten vraagt altijd om bevestiging."
    )


def _format_live_prijzen() -> str:
    from src.data_loader import fetch_fear_greed

    lines = [f"<b>Prijzen</b> - {_now()}"]
    results: dict[str, tuple[float, float] | Exception] = {}
    with ThreadPoolExecutor(max_workers=len(MARKETS)) as executor:
        futures = {
            executor.submit(_fetch_daily_change_pct, asset): asset
            for asset in MARKETS
        }
        for future in as_completed(futures):
            asset = futures[future]
            try:
                results[asset] = future.result()
            except Exception as exc:
                results[asset] = exc

    for asset in MARKETS:
        result = results.get(asset)
        if isinstance(result, Exception):
            lines.append(f"{asset:<7} niet beschikbaar ({_escape(result)})")
        elif result:
            price, change = result
            direction = "omhoog" if change >= 0 else "omlaag"
            lines.append(f"{asset}: {_fmt_usd(price)} ({_fmt_pct(change)}, {direction})")
        else:
            lines.append(f"{asset:<7} niet beschikbaar")

    try:
        fg = fetch_fear_greed()
        lines.append(f"Fear & Greed: {fg.get('value', '?')} - {_escape(fg.get('label', 'onbekend'))}")
    except Exception as exc:
        lines.append(f"Fear & Greed: niet beschikbaar ({_escape(exc)})")

    return "\n".join(lines)


def _format_posities() -> str:
    from src.auto_trader import _open_auto_trades

    trades = _open_auto_trades()
    if not trades:
        return f"<b>Posities</b> - {_now()}\nGeen open posities."

    price_cache: dict[str, float] = {}
    lines = [f"<b>Posities ({len(trades)})</b> - {_now()}"]
    for trade in trades:
        asset = trade["asset"]
        direction = trade.get("direction", "long")
        entry = float(trade["entry_price"])
        capital = float(trade.get("capital") or 0)
        try:
            live = price_cache.setdefault(asset, _fetch_live_price(asset))
            pct = _pnl_pct(direction, entry, live)
            euro = pct / 100 * capital + float(trade.get("partial_pnl_euro") or 0)
            live_text = f"{_fmt_price(live)} ({_fmt_pct(pct)})"
            euro_text = _fmt_eur(euro)
        except Exception:
            live_text = "niet beschikbaar"
            euro_text = "onbekend"

        lines += [
            f"\n<b>#{trade['id']} {_escape(asset)} {direction.upper()}</b>",
            f"Entry {_fmt_price(entry)} | Nu {live_text}",
            f"P&L {euro_text}",
            f"SL {_fmt_price(trade['stop_loss'])} | TP {_fmt_price(trade['take_profit'])}",
            f"Inzet {_fmt_eur(capital)}",
        ]
    return "\n".join(lines).rstrip()


def _format_statistieken() -> str:
    from src.auto_trader import (
        _dagverlies_vandaag,
        _huidig_saldo,
        _in_cooling,
        _opeenvolgende_verliezen,
    )
    from src.database import get_auto_trades, get_bot_learning
    from src.bot_learning import get_health_overview

    saldo = _huidig_saldo()
    dagverlies = _dagverlies_vandaag()
    gesloten = get_auto_trades(status="closed")
    recent = gesloten[:10]
    wins_recent = sum(1 for trade in recent if (trade.get("result_pct") or 0) > 0)
    wins_total = sum(1 for trade in gesloten if (trade.get("result_pct") or 0) > 0)
    losses_total = len(gesloten) - wins_total
    leren = get_bot_learning()

    winrate_recent = f"{wins_recent}/{len(recent)}" if recent else "nog geen"
    cooling = "Ja" if _in_cooling() else "Nee"
    min_rr = float(leren.get("min_rr", 2.0))

    return (
        f"<b>Stats</b> - {_now()}\n"
        f"Saldo: {_fmt_eur(saldo)}\n"
        f"Vandaag: {_fmt_eur(dagverlies)}\n"
        f"Winrate 10: {winrate_recent}\n"
        f"Verliesreeks: {_opeenvolgende_verliezen()}\n"
        f"Cooling: {cooling}\n"
        f"Min R/R: 1:{_fmt_number(min_rr, 1)}\n"
        f"Gesloten: {len(gesloten)} | W {wins_total} | V {losses_total}\n"
        f"Status: {_format_health_kort(get_health_overview())}"
    )


def _format_health_kort(overview: dict) -> str:
    """Compacte gezondheidsregel: heartbeat-leeftijd + eventuele probleem-assets."""
    hb = overview.get("heartbeat")
    if not hb or hb.get("age_seconds") is None:
        return "⚪ geen heartbeat"
    age = int(hb["age_seconds"])
    age_txt = f"{age}s" if age < 120 else f"{age // 60}m"
    # Een cyclus duurt ~15 min; >35 min zonder heartbeat duidt op een probleem.
    bol = "🟢" if age < 35 * 60 else "🔴"
    problemen = [a["asset"] for a in overview.get("assets", []) if a.get("status") not in ("ok", "alive", None)]
    staart = f" | let op: {', '.join(problemen)}" if problemen else ""
    return f"{bol} heartbeat {age_txt} geleden{staart}"


def _format_trades_menu(trades: list[dict]) -> str:
    if not trades:
        return f"<b>Open trades</b> - {_now()}\nGeen open trades."
    return (
        f"<b>Open trades</b> - {_now()}\n"
        f"Kies een trade."
    )


def _format_trade_detail(trade: dict, live_price: float | None = None) -> str:
    direction = trade.get("direction", "long")
    entry = float(trade["entry_price"])
    if live_price is None:
        live_price = _fetch_live_price(trade["asset"])

    pct, euro = _trade_result_preview(trade, live_price)
    eta_text = _estimate_trade_eta(trade, live_price)
    return (
        f"<b>Trade #{trade['id']}</b>\n\n"
        f"Asset: {_escape(trade['asset'])} {direction.upper()}\n"
        f"Strategie: {_escape(trade.get('strategy') or 'onbekend')}\n"
        f"Entry: {_fmt_price(entry)}\n"
        f"Nu: {_fmt_price(live_price)}\n"
        f"Resultaat nu: {_fmt_pct(pct)} | {_fmt_eur(euro)}\n"
        f"Stop-loss: {_fmt_price(trade['stop_loss'])}\n"
        f"Take-profit: {_fmt_price(trade['take_profit'])}\n"
        f"{eta_text}\n"
        f"Inzet open: {_fmt_eur(trade.get('capital') or 0)}"
    )


def _format_close_confirm(trade: dict, live_price: float) -> str:
    pct, euro = _trade_result_preview(trade, live_price)
    return (
        f"<b>Trade sluiten?</b>\n\n"
        f"Weet je zeker dat je trade #{trade['id']} wilt sluiten op de huidige marktprijs?\n\n"
        f"{_escape(trade['asset'])} {trade.get('direction', 'long').upper()}\n"
        f"Entry: {_fmt_price(trade['entry_price'])}\n"
        f"Huidige prijs: {_fmt_price(live_price)}\n"
        f"Verwacht resultaat: {_fmt_pct(pct)} | {_fmt_eur(euro)}"
    )


def _format_logs() -> str:
    from src.database import get_bot_log

    rows = get_bot_log(10)
    if not rows:
        return f"<b>Bot logs</b> - {_now()}\n\nGeen logregels gevonden."
    log_lines = []
    for row in rows:
        ts = str(row.get("logged_at", ""))[11:19]
        msg = str(row.get("message", ""))[:320]
        log_lines.append(f"{ts} {msg}")
    return f"<b>Bot logs</b> - {_now()}\n\n<pre>{_escape(chr(10).join(log_lines))}</pre>"


def _format_waarom_geen_trade() -> str:
    from src.database import get_connection

    conn = get_connection()
    rows = conn.execute(
        """
        SELECT created_at, asset, strategy, action, score, sample_count, reason, details_json
        FROM bot_learning_decisions
        WHERE action IN ('skipped', 'no_setup', 'data_unavailable')
        ORDER BY id DESC LIMIT 8
        """
    ).fetchall()
    conn.close()
    if not rows:
        return f"<b>Waarom geen trade?</b> - {_now()}\n\nNog geen beslissingen gevonden."
    lines = [f"<b>Waarom geen trade?</b> - {_now()}"]
    for row in rows:
        ts = str(row["created_at"])[11:16]
        asset = row["asset"] or "-"
        strategy = row["strategy"] or "-"
        score = "" if row["score"] is None else f" | score {float(row['score']):.0%}"
        reason = str(row["reason"] or "")[:180]
        lines.append(f"\n{ts} {_escape(asset)} {_escape(strategy)}{score}\n{_escape(reason)}")
    return "\n".join(lines)


def _format_regime_stats() -> str:
    from src.bot_learning import get_regime_strategy_stats

    rows = get_regime_strategy_stats(10)
    if not rows:
        return f"<b>Regime stats</b> - {_now()}\n\nNog geen gesloten trades met regime-data."
    lines = [f"<b>Regime stats</b> - {_now()}"]
    for row in rows:
        lines.append(
            f"\n{_escape(row['strategy'])} / {_escape(row['regime'])}\n"
            f"{row['trades']} trades | W {row['wins']} | WR {row['winrate']}% | "
            f"P&L {_fmt_eur(row['pnl_euro'])}"
        )
    return "\n".join(lines)


def _format_restart_confirm(target: str) -> str:
    label = "bot" if target == "bot" else "dashboard"
    return (
        f"<b>{label.capitalize()} herstarten?</b>\n\n"
        f"Weet je zeker dat je de {label} wilt herstarten?\n"
        f"Dit onderdeel kan kort offline zijn."
    )


# -- Actions ------------------------------------------------------------------

def _load_open_trade(trade_id: int) -> dict | None:
    from src.database import get_trade_by_id

    trade = get_trade_by_id(trade_id)
    if not trade or trade.get("status") != "open":
        return None
    return trade


def _close_trade(trade_id: int) -> str:
    from src.auto_trader import _update_leren
    from src.database import close_paper_trade, get_trade_by_id

    trade = _load_open_trade(trade_id)
    if not trade:
        return f"<b>Trade #{trade_id}</b>\n\nDeze trade is niet meer open."

    live_price = _fetch_live_price(trade["asset"])
    pct, euro = _trade_result_preview(trade, live_price)
    ok = close_paper_trade(
        trade_id,
        live_price,
        "manual_telegram",
        "Handmatig gesloten via Telegram.",
    )
    if not ok:
        return f"<b>Trade #{trade_id}</b>\n\nTrade kon niet worden gesloten. Mogelijk is hij al gesloten."

    closed = get_trade_by_id(trade_id) or trade
    try:
        _update_leren(closed, "manual")
    except Exception:
        pass

    return (
        f"<b>Trade gesloten</b>\n\n"
        f"#{trade_id} {_escape(trade['asset'])} {trade.get('direction', 'long').upper()}\n"
        f"Entry: {_fmt_price(trade['entry_price'])}\n"
        f"Exit: {_fmt_price(live_price)}\n"
        f"Resultaat: {_fmt_pct(pct)} | {_fmt_eur(euro)}"
    )


def _restart_dashboard() -> str:
    try:
        subprocess.run(
            ["sudo", "/bin/systemctl", "restart", "tradeai-web"],
            cwd=str(PROJECT_DIR),
            timeout=30,
            check=False,
        )
        melding_website_herstart()
        return "<b>Dashboard herstart</b>\n\nHet dashboard wordt opnieuw gestart."
    except Exception as exc:
        return f"<b>Dashboard herstart mislukt</b>\n\n{_escape(exc)}"


def _restart_bot_async() -> str:
    command = "sleep 2; sudo /bin/systemctl restart tradeai"
    try:
        subprocess.Popen(
            ["sh", "-c", command],
            cwd=str(PROJECT_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        melding_bot_herstart()
        return "<b>Bot herstart</b>\n\nDe bot wordt herstart en is over ongeveer 30 seconden terug online."
    except Exception as exc:
        return f"<b>Bot herstart mislukt</b>\n\n{_escape(exc)}"


# -- Dispatch -----------------------------------------------------------------

def _chat_id_from_message(message: dict) -> str:
    return str(message.get("chat", {}).get("id", ""))


def _message_id_from_callback(callback_query: dict) -> int | None:
    return callback_query.get("message", {}).get("message_id")


def verwerk_bericht(message: dict, expected_chat_id: str | None = None) -> bool:
    chat_id = _chat_id_from_message(message)
    if expected_chat_id and chat_id != str(expected_chat_id):
        return False

    text = str(message.get("text", "")).strip().lower()
    if text not in ("/start", "/menu", "/hoofdmenu"):
        return False

    return _send_message(chat_id, _format_hoofd_menu(), _keyboard_hoofd_menu())


def verwerk_callback(callback_query: dict, expected_chat_id: str | None = None) -> bool:
    callback_id = callback_query.get("id", "")
    _answer_callback(callback_id)

    message = callback_query.get("message", {})
    chat_id = str(message.get("chat", {}).get("id", ""))
    message_id = _message_id_from_callback(callback_query)
    data = str(callback_query.get("data", ""))

    if expected_chat_id and chat_id != str(expected_chat_id):
        return False
    if not chat_id or message_id is None:
        return False

    try:
        if data == "menu_hoofd":
            return _edit_message(chat_id, message_id, _format_hoofd_menu(), _keyboard_hoofd_menu())
        if data == "menu_info":
            return _edit_message(chat_id, message_id, _format_info_menu(), _keyboard_info_menu())
        if data == "menu_trades":
            from src.auto_trader import _open_auto_trades

            trades = _open_auto_trades()
            return _edit_message(chat_id, message_id, _format_trades_menu(trades), _keyboard_trades_menu(trades))
        if data == "menu_beheer":
            return _edit_message(chat_id, message_id, _format_beheer_menu(), _keyboard_beheer_menu())

        if data == "info_prijzen":
            return _edit_message(chat_id, message_id, _format_live_prijzen(), _keyboard_info_back())
        if data == "info_posities":
            return _edit_message(chat_id, message_id, _format_posities(), _keyboard_info_back())
        if data == "info_stats":
            return _edit_message(chat_id, message_id, _format_statistieken(), _keyboard_info_back())

        if data.startswith("trade_select:"):
            trade_id = int(data.split(":", 1)[1])
            trade = _load_open_trade(trade_id)
            if not trade:
                text = f"<b>Trade #{trade_id}</b>\n\nDeze trade is niet meer open."
                return _edit_message(chat_id, message_id, text, _keyboard_trades_menu([]))
            live_price = _fetch_live_price(trade["asset"])
            return _edit_message(chat_id, message_id, _format_trade_detail(trade, live_price), _keyboard_trade_detail(trade_id))

        if data.startswith("trade_close:"):
            trade_id = int(data.split(":", 1)[1])
            trade = _load_open_trade(trade_id)
            if not trade:
                return _edit_message(chat_id, message_id, f"<b>Trade #{trade_id}</b>\n\nDeze trade is niet meer open.", _keyboard_trades_menu([]))
            live_price = _fetch_live_price(trade["asset"])
            return _edit_message(chat_id, message_id, _format_close_confirm(trade, live_price), _keyboard_trade_close_confirm(trade_id))

        if data.startswith("trade_close_confirm:"):
            trade_id = int(data.split(":", 1)[1])
            text = _close_trade(trade_id)
            from src.auto_trader import _open_auto_trades

            return _edit_message(chat_id, message_id, text, _keyboard_trades_menu(_open_auto_trades()))

        if data == "beheer_logs":
            return _edit_message(chat_id, message_id, _format_logs(), _keyboard_beheer_back())
        if data == "beheer_waarom":
            return _edit_message(chat_id, message_id, _format_waarom_geen_trade(), _keyboard_beheer_back())
        if data == "beheer_regime_stats":
            return _edit_message(chat_id, message_id, _format_regime_stats(), _keyboard_beheer_back())
        if data == "restart_bot":
            return _edit_message(chat_id, message_id, _format_restart_confirm("bot"), _keyboard_restart_confirm("bot"))
        if data == "restart_web":
            return _edit_message(chat_id, message_id, _format_restart_confirm("dashboard"), _keyboard_restart_confirm("web"))
        if data == "restart_confirm:bot":
            text = _restart_bot_async()
            return _edit_message(chat_id, message_id, text, _keyboard_beheer_menu())
        if data == "restart_confirm:web":
            text = _restart_dashboard()
            return _edit_message(chat_id, message_id, text, _keyboard_beheer_menu())

        return _edit_message(chat_id, message_id, _format_hoofd_menu(), _keyboard_hoofd_menu())
    except Exception as exc:
        text = f"<b>Telegram actie mislukt</b>\n\n{_escape(exc)}"
        return _edit_message(chat_id, message_id, text, _keyboard_hoofd_menu())
