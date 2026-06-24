"""
notifier.py — Telegram-meldingen voor de TradeAI bot.

Configuratie staat in telegram_config.json (naast dit project):
    {
        "token": "...",
        "chat_id": "..."
    }

Als het bestand ontbreekt of leeg is, worden meldingen stil overgeslagen.

Ondersteunde commando's via Telegram:
    "hoe gaat het"  →  mini-rapport met saldo, open trades en winrate
"""
import json
import threading
import time
import requests
from datetime import datetime
from pathlib import Path
from src.secure_config import load_tradeai_config

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

_CONFIG_PAD = Path(__file__).parent.parent / "telegram_config.json"
_START_MELDING_PAD = Path(__file__).parent.parent / ".telegram_last_start"
_AMSTERDAM  = ZoneInfo("Europe/Amsterdam")


def _config() -> dict:
    return load_tradeai_config(_CONFIG_PAD)


def stuur_melding(tekst: str) -> bool:
    """Stuur een Telegram-bericht. Geeft True terug bij succes, anders False."""
    cfg = _config()
    token   = cfg.get("token", "").strip()
    chat_id = cfg.get("chat_id", "").strip()
    if not token or not chat_id:
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": tekst, "parse_mode": "HTML"},
            timeout=10,
        )
        return r.ok
    except Exception:
        return False


# ── Kant-en-klare meldingen ───────────────────────────────────────────────────

def melding_bot_gestart():
    nu = datetime.now(_AMSTERDAM).strftime("%Y-%m-%d %H:%M")
    stuur_melding(
        f"🟢 <b>TradeAI bot gestart</b>\n"
        f"Tijd: {nu}\n"
        f"Markten: BTC-USD · ETH-USD · SOL-USD\n"
        f"Check-interval: elke 5 minuten"
    )


def melding_bot_gestopt():
    nu = datetime.now(_AMSTERDAM).strftime("%Y-%m-%d %H:%M")
    stuur_melding(
        f"🔴 <b>TradeAI bot gestopt</b>\n"
        f"Tijd: {nu}\n"
        f"Open trades blijven geregistreerd in de database."
    )


def melding_trade_geopend(trade_id: int, asset: str, entry: float,
                           sl: float, tp: float, rr: float,
                           kapitaal: float, strategie: str,
                           richting: str = "long",
                           risico_euro: float | None = None):
    icoon = "📈" if richting == "long" else "📉"
    risico_regel = ""
    if risico_euro is not None:
        risico_regel = f"\nRisico:      EUR{risico_euro:,.2f}"
    stuur_melding(
        f"{icoon} <b>Trade geopend #{trade_id}</b>\n"
        f"Asset:       {asset} {richting.upper()}\n"
        f"Strategie:   {strategie}\n"
        f"Entry:       {entry:,.4f}\n"
        f"Stop-loss:   {sl:,.4f}\n"
        f"Take-profit: {tp:,.4f}\n"
        f"R/R:         1:{rr:.1f}\n"
        f"Inzet:       €{kapitaal:,.2f}"
        f"{risico_regel}"
    )


def melding_take_profit(trade_id: int, asset: str, entry: float,
                         exit_price: float, result_pct: float, result_euro: float,
                         richting: str = "long"):
    stuur_melding(
        f"✅ <b>Take-profit bereikt #{trade_id}</b>\n"
        f"Asset:  {asset} {richting.upper()}\n"
        f"Entry:  {entry:,.4f}\n"
        f"Exit:   {exit_price:,.4f}\n"
        f"Winst:  +{result_pct:.2f}%  |  +€{result_euro:,.2f}"
    )


def melding_stop_loss(trade_id: int, asset: str, entry: float,
                       exit_price: float, result_pct: float, result_euro: float,
                       richting: str = "long"):
    stuur_melding(
        f"❌ <b>Stop-loss geraakt #{trade_id}</b>\n"
        f"Asset:   {asset} {richting.upper()}\n"
        f"Entry:   {entry:,.4f}\n"
        f"Exit:    {exit_price:,.4f}\n"
        f"Verlies: {result_pct:.2f}%  |  €{result_euro:,.2f}"
    )


def melding_verbindingsfout(asset: str, fout: str):
    stuur_melding(
        f"⚠️ <b>Verbindingsfout</b>\n"
        f"Asset: {asset}\n"
        f"Fout:  {str(fout)[:200]}"
    )


def melding_drawdown_limiet(dagverlies: float, limiet: float):
    nu = datetime.now(_AMSTERDAM).strftime("%Y-%m-%d %H:%M")
    stuur_melding(
        f"🛑 <b>Dagverlies-limiet bereikt</b>\n"
        f"Tijd: {nu}\n"
        f"Verlies vandaag: €{dagverlies:,.2f} (limiet €{limiet:,.0f})\n"
        f"Geen nieuwe trades meer vandaag — open trades blijven gemonitord."
    )


def melding_bot_herstart():
    nu = datetime.now(_AMSTERDAM).strftime("%Y-%m-%d %H:%M")
    stuur_melding(
        f"🔄 <b>TradeAI bot hergestart</b>\n"
        f"Tijd: {nu}\n"
        f"Herstart geïnitieerd via het dashboard."
    )


def melding_website_herstart():
    nu = datetime.now(_AMSTERDAM).strftime("%Y-%m-%d %H:%M")
    stuur_melding(
        f"🌐 <b>TradeAI website hergestart</b>\n"
        f"Tijd: {nu}\n"
        f"Herstart geïnitieerd via het dashboard."
    )


def melding_dagrapport(inhoud: str):
    tekst = f"📊 <b>Dagrapport TradeAI</b>\n\n<pre>{inhoud}</pre>"
    if len(tekst) > 4096:
        tekst = tekst[:4090] + "…"
    stuur_melding(tekst)


# ── Mini-rapport op verzoek ───────────────────────────────────────────────────

def _genereer_mini_rapport() -> str:
    """Haal live data op uit de database en maak een kort statusrapport."""
    try:
        from src.database import get_auto_trades, get_bot_learning
        from src.auto_trader import _huidig_saldo, _in_cooling

        saldo   = _huidig_saldo()
        leren   = get_bot_learning()
        open_t  = get_auto_trades(status="open")
        gesloten = get_auto_trades(status="closed")

        # Winrate van de laatste 10 trades
        recente  = gesloten[:10]
        wins     = sum(1 for t in recente if (t.get("result_pct") or 0) > 0)
        winrate  = f"{wins}/{len(recente)}" if recente else "nog geen"

        # P&L vandaag
        vandaag  = datetime.now(_AMSTERDAM).date().isoformat()
        pnl_dag  = sum(
            t.get("result_euro", 0) or 0
            for t in gesloten
            if t.get("closed_at", "")[:10] == vandaag
        )

        # Open trades samenvatting
        if open_t:
            open_regels = "\n".join(
                f"  #{t['id']} {t['asset']} @ {t['entry_price']:,.4f}"
                for t in open_t
            )
        else:
            open_regels = "  Geen open trades"

        cooling = "🟡 Ja" if _in_cooling() else "🟢 Nee"

        nu = datetime.now(_AMSTERDAM).strftime("%H:%M")
        return (
            f"🤖 <b>TradeAI status — {nu}</b>\n\n"
            f"💰 Saldo:       €{saldo:,.2f}\n"
            f"📅 P&L vandaag: €{pnl_dag:+,.2f}\n"
            f"🎯 Winrate:     {winrate} (laatste {len(recente)})\n"
            f"⏸ Cooling:      {cooling}\n"
            f"📊 Min R/R:     1:{leren.get('min_rr', 1.5):.1f}\n\n"
            f"<b>Open trades ({len(open_t)}):</b>\n{open_regels}"
        )
    except Exception as e:
        return f"⚠️ Rapport ophalen mislukt: {e}"


# ── Telegram polling (luistert naar inkomende berichten) ──────────────────────

def _format_eur(value: float) -> str:
    return f"EUR {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _format_price(value: float) -> str:
    return f"{value:,.4f}".replace(",", "X").replace(".", ",").replace("X", ".")


def melding_bot_gestart():
    """Stuur een compacte startmelding, maximaal eens per 15 minuten."""
    try:
        nu_ts = time.time()
        vorige = float(_START_MELDING_PAD.read_text().strip())
        if nu_ts - vorige < 900:
            return
    except Exception:
        nu_ts = time.time()
    try:
        _START_MELDING_PAD.write_text(str(nu_ts))
    except Exception:
        pass

    nu = datetime.now(_AMSTERDAM).strftime("%Y-%m-%d %H:%M")
    stuur_melding(
        f"<b>TradeAI bot gestart</b>\n"
        f"Tijd: {nu}\n"
        f"Markten: BTC-USD · ETH-USD · SOL-USD\n"
        f"Check-interval: elke 15 minuten\n"
        f"Menu: /menu"
    )


def _genereer_mini_rapport() -> str:
    """Compact statusrapport voor het oude 'hoe gaat het' commando."""
    try:
        from src.database import get_auto_trades, get_bot_learning
        from src.auto_trader import _huidig_saldo, _in_cooling

        saldo = _huidig_saldo()
        leren = get_bot_learning()
        open_t = get_auto_trades(status="open")
        gesloten = get_auto_trades(status="closed")

        recente = gesloten[:10]
        wins = sum(1 for trade in recente if (trade.get("result_pct") or 0) > 0)
        winrate = f"{wins}/{len(recente)}" if recente else "nog geen"

        vandaag = datetime.now(_AMSTERDAM).date().isoformat()
        pnl_dag = sum(
            trade.get("result_euro", 0) or 0
            for trade in gesloten
            if trade.get("closed_at", "")[:10] == vandaag
        )

        if open_t:
            open_regels = "\n".join(
                f"#{trade['id']} {trade['asset']} {trade.get('direction', 'long').upper()} @ {_format_price(trade['entry_price'])}"
                for trade in open_t
            )
        else:
            open_regels = "Geen open trades"

        cooling = "Ja" if _in_cooling() else "Nee"
        nu = datetime.now(_AMSTERDAM).strftime("%H:%M")
        min_rr = f"{leren.get('min_rr', 2.0):.1f}".replace(".", ",")
        return (
            f"<b>TradeAI status</b> - {nu}\n"
            f"Saldo: {_format_eur(saldo)}\n"
            f"Vandaag: {_format_eur(pnl_dag)}\n"
            f"Winrate 10: {winrate}\n"
            f"Cooling: {cooling}\n"
            f"Min R/R: 1:{min_rr}\n"
            f"Open trades ({len(open_t)}):\n{open_regels}\n\n"
            f"Menu: /menu"
        )
    except Exception as e:
        return f"Rapport ophalen mislukt: {e}"


_OFFSET_FILE = Path(__file__).parent.parent / "telegram_offset.txt"
_polling_started = False


def _load_offset() -> int | None:
    try:
        return int(_OFFSET_FILE.read_text().strip())
    except Exception:
        return None


def _save_offset(offset: int) -> None:
    try:
        _OFFSET_FILE.write_text(str(offset))
    except Exception:
        pass


def _polling_loop():
    """
    Draait in een aparte thread. Pollt Telegram elke 5 seconden op
    inkomende berichten en inline keyboard callbacks.
    """
    cfg     = _config()
    token   = cfg.get("token", "").strip()
    chat_id = str(cfg.get("chat_id", "")).strip()
    if not token or not chat_id:
        return

    offset = _load_offset()
    while True:
        try:
            params = {"timeout": 10, "allowed_updates": ["message", "callback_query"]}
            if offset:
                params["offset"] = offset
            r = requests.get(
                f"https://api.telegram.org/bot{token}/getUpdates",
                params=params,
                timeout=15,
            )
            if not r.ok:
                time.sleep(1)
                continue

            for update in r.json().get("result", []):
                offset = update["update_id"] + 1
                _save_offset(offset)

                callback = update.get("callback_query")
                if callback:
                    van_cq = str(callback.get("message", {}).get("chat", {}).get("id", ""))
                    if van_cq == chat_id:
                        try:
                            from src.telegram_commands import verwerk_callback
                            verwerk_callback(callback, expected_chat_id=chat_id)
                        except Exception:
                            pass
                    continue

                msg   = update.get("message", {})
                tekst = msg.get("text", "").strip().lower()
                van   = str(msg.get("chat", {}).get("id", ""))

                if van != chat_id:
                    continue

                if tekst in ("/start", "/menu", "/hoofdmenu"):
                    try:
                        from src.telegram_commands import verwerk_bericht
                        verwerk_bericht(msg, expected_chat_id=chat_id)
                    except Exception:
                        pass
                    continue

                if "hoe gaat het" in tekst:
                    rapport = _genereer_mini_rapport()
                    stuur_melding(rapport)

        except Exception:
            pass

        time.sleep(0.2)


def start_polling():
    """Start de Telegram-listener als achtergrond-thread. Roept zichzelf nooit twee keer aan."""
    global _polling_started
    if _polling_started:
        return
    _polling_started = True
    t = threading.Thread(target=_polling_loop, daemon=True, name="telegram-polling")
    t.start()
