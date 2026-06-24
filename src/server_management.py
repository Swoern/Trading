"""
src/server_management.py - Server Management dashboard met live systeemmetrics.
"""
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import streamlit as st

from src.database import get_connection
from src.notifier import melding_bot_herstart, melding_website_herstart

LOCK_FILE = Path(__file__).parent.parent / "bot.lock"


@st.cache_data(ttl=15, show_spinner=False)
def _bot_status() -> dict:
    """Controleer of de bot-process echt draait via bot.lock en recente logs."""
    draait = False
    pid    = None

    if LOCK_FILE.exists():
        try:
            pid = int(LOCK_FILE.read_text().strip())
            if psutil.pid_exists(pid):
                proc = psutil.Process(pid)
                draait = "python" in proc.name().lower()
        except Exception:
            pass

    # Fallback: kijk of de laatste log-regel recent is (< 20 min)
    laatste_log_sec = None
    try:
        conn = get_connection()
        row  = conn.execute(
            "SELECT logged_at FROM bot_logs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        if row:
            ts = datetime.fromisoformat(row["logged_at"])
            # Logs zijn opgeslagen als naive local time — vergelijk met naive now()
            if ts.tzinfo is not None:
                ts = ts.replace(tzinfo=None)
            laatste_log_sec = (datetime.now() - ts).total_seconds()
            if not draait and laatste_log_sec < 1200:  # 20 min
                draait = True
    except Exception:
        pass

    # Open trades tellen
    open_trades = 0
    try:
        conn = get_connection()
        open_trades = conn.execute(
            "SELECT count(*) FROM paper_trades WHERE status='open' AND source='auto'"
        ).fetchone()[0]
        conn.close()
    except Exception:
        pass

    return {
        "draait":          draait,
        "pid":             pid,
        "laatste_log_sec": laatste_log_sec,
        "open_trades":     open_trades,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_uptime(seconds: float) -> str:
    dagen  = int(seconds // 86400)
    uren   = int((seconds % 86400) // 3600)
    if dagen > 0:
        return f"{dagen} d {uren} u"
    minuten = int((seconds % 3600) // 60)
    if uren > 0:
        return f"{uren} u {minuten} min"
    return f"{int(minuten)} min"


def _pi_temperatuur() -> float | None:
    """Lees CPU-temperatuur op Raspberry Pi (of Linux)."""
    thermal = Path("/sys/class/thermal/thermal_zone0/temp")
    if thermal.exists():
        try:
            return int(thermal.read_text().strip()) / 1000.0
        except Exception:
            pass
    # psutil sensors (Linux, sommige Windows met WMI)
    try:
        temps = psutil.sensors_temperatures()
        for entries in temps.values():
            if entries:
                return entries[0].current
    except Exception:
        pass
    return None


@st.cache_data(ttl=30, show_spinner=False)
def _systeemmetrics() -> dict:
    """Haal live systeemdata op. Wordt elke 30 seconden ververst."""
    cpu      = psutil.cpu_percent(interval=0.5)
    ram      = psutil.virtual_memory()
    boot_sec = time.time() - psutil.boot_time()

    # Schijf: / op Linux/Pi, C:/ op Windows
    schijf_pad = "C:/" if platform.system() == "Windows" else "/"
    try:
        disk = psutil.disk_usage(schijf_pad)
        disk_gebruikt = round(disk.percent, 1)
        disk_vrij     = round(100 - disk.percent, 1)
        disk_vrij_gb  = round(disk.free / 1024**3, 1)
    except Exception:
        disk_gebruikt = disk_vrij = disk_vrij_gb = None

    temp = _pi_temperatuur()

    return {
        "cpu":           round(cpu, 1),
        "ram":           round(ram.percent, 1),
        "ram_gebruikt_gb": round(ram.used / 1024**3, 1),
        "ram_totaal_gb": round(ram.total / 1024**3, 1),
        "uptime_sec":    boot_sec,
        "disk_gebruikt": disk_gebruikt,
        "disk_vrij":     disk_vrij,
        "disk_vrij_gb":  disk_vrij_gb,
        "temp":          temp,
    }


@st.cache_data(ttl=15, show_spinner=False)
def _recente_logs(n: int = 20) -> list[str]:
    """Lees de laatste N regels uit de echte bot-log in de database."""
    try:
        conn = get_connection()
        rows = conn.execute(
            "SELECT logged_at, message FROM bot_logs ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        conn.close()
        return [f"[{r['logged_at'][:19]}] {r['message']}" for r in rows]
    except Exception:
        return ["(Geen logs beschikbaar)"]


def _voer_uit(commando: str) -> tuple[bool, str]:
    """Voer een shell-commando uit en geef (succes, uitvoer) terug."""
    try:
        result = subprocess.run(
            commando,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        uitvoer = (result.stdout + result.stderr).strip()
        return result.returncode == 0, uitvoer or "OK"
    except subprocess.TimeoutExpired:
        return False, "Commando duurde te lang (timeout 30s)."
    except Exception as e:
        return False, str(e)


def _card(label: str, value: str, sub: str = "", tone: str = "neutral") -> str:
    tone_class = {
        "good":    "value-positive",
        "bad":     "value-negative",
        "neutral": "value-neutral",
        "warning": "value-warning",
    }.get(tone, "value-neutral")
    return (
        '<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value {tone_class}">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        '</div>'
    )


def _cpu_tone(pct: float) -> str:
    if pct < 50: return "good"
    if pct < 80: return "warning"
    return "bad"


def _ram_tone(pct: float) -> str:
    if pct < 70: return "good"
    if pct < 88: return "warning"
    return "bad"


def _disk_tone(pct: float) -> str:
    if pct < 70: return "good"
    if pct < 88: return "warning"
    return "bad"


def _temp_tone(temp: float) -> str:
    if temp < 60: return "good"
    if temp < 75: return "warning"
    return "bad"


# ── Render ────────────────────────────────────────────────────────────────────

def render_server_management() -> None:
    st.markdown(
        """
        <div class="page-header">
            <h1>&#x1F5A5;&#xFE0F; Server Management</h1>
            <p>Live systeemmetrics en beheeracties voor de TradeAI omgeving.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m = _systeemmetrics()

    # ── Status ────────────────────────────────────────────────────────────────
    bot = _bot_status()
    c1, c2, c3 = st.columns(3)

    # Toon "Herstarten..." gedurende 60 seconden na een herstart-verzoek
    bot_restarting = st.session_state.get("bot_restart_ts", 0)
    website_restarting = st.session_state.get("website_restart_ts", 0)
    nu_ts = time.time()

    if nu_ts - bot_restarting < 60:
        seconden_geleden = int(nu_ts - bot_restarting)
        c2.markdown(
            _card("BOT SERVICE", "Herstarten...",
                  f"Herstart geïnitieerd {seconden_geleden}s geleden", "warning"),
            unsafe_allow_html=True,
        )
    elif bot["draait"]:
        pid_sub = f"PID {bot['pid']}" if bot["pid"] else "Actief (via log)"
        if bot["laatste_log_sec"] is not None:
            minuten = int(bot["laatste_log_sec"] // 60)
            pid_sub += f" · laatste log {minuten} min geleden"
        c2.markdown(_card("BOT SERVICE", "Online", pid_sub, "good"),
                    unsafe_allow_html=True)
    else:
        sub = "bot.lock niet gevonden of process gestopt"
        if bot["laatste_log_sec"] is not None:
            minuten = int(bot["laatste_log_sec"] // 60)
            sub = f"Laatste activiteit {minuten} min geleden"
        c2.markdown(_card("BOT SERVICE", "Gestopt", sub, "bad"),
                    unsafe_allow_html=True)

    if nu_ts - website_restarting < 60:
        seconden_geleden = int(nu_ts - website_restarting)
        c1.markdown(
            _card("Website", "Herstarten...",
                  f"Herstart geïnitieerd {seconden_geleden}s geleden", "warning"),
            unsafe_allow_html=True,
        )
    else:
        c1.markdown(_card("Website", "Online", "Streamlit dashboard bereikbaar", "good"),
                    unsafe_allow_html=True)

    open_txt = f"{bot['open_trades']} open trade{'s' if bot['open_trades'] != 1 else ''}"
    c3.markdown(_card("Omgeving", "Paper mode", open_txt, "neutral"),
                unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-title">Systeemmetrics</div>', unsafe_allow_html=True)

    # ── Rij 1: uptime / CPU / RAM ─────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)

    uptime_str = _format_uptime(m["uptime_sec"])
    m1.markdown(
        _card("Server uptime", uptime_str, f"Systeem: {platform.system()}"),
        unsafe_allow_html=True,
    )

    m2.markdown(
        _card("CPU-gebruik", f"{m['cpu']}%", "Live meting (0.5s gem.)",
              _cpu_tone(m["cpu"])),
        unsafe_allow_html=True,
    )

    m3.markdown(
        _card("RAM-gebruik", f"{m['ram']}%",
              f"{m['ram_gebruikt_gb']} GB / {m['ram_totaal_gb']} GB",
              _ram_tone(m["ram"])),
        unsafe_allow_html=True,
    )

    # ── Rij 2: schijf / vrij / temperatuur ───────────────────────────────────
    m4, m5, m6 = st.columns(3)

    if m["disk_gebruikt"] is not None:
        m4.markdown(
            _card("Opslag gebruikt", f"{m['disk_gebruikt']}%", "Van totale schijfruimte",
                  _disk_tone(m["disk_gebruikt"])),
            unsafe_allow_html=True,
        )
        m5.markdown(
            _card("Opslag vrij", f"{m['disk_vrij']}%", f"{m['disk_vrij_gb']} GB beschikbaar",
                  "good" if m["disk_vrij"] > 20 else "warning"),
            unsafe_allow_html=True,
        )
    else:
        m4.markdown(_card("Opslag gebruikt", "N/B", "Schijf niet uitleesbaar"), unsafe_allow_html=True)
        m5.markdown(_card("Opslag vrij",     "N/B", "Schijf niet uitleesbaar"), unsafe_allow_html=True)

    if m["temp"] is not None:
        temp_str = f"{m['temp']:.1f}°C"
        m6.markdown(
            _card("Temperatuur", temp_str,
                  "Normaal bereik" if m["temp"] < 60 else "Let op: warm!",
                  _temp_tone(m["temp"])),
            unsafe_allow_html=True,
        )
    else:
        os_naam = platform.system()
        m6.markdown(
            _card("Temperatuur", "N/B", f"Niet uitleesbaar op {os_naam}"),
            unsafe_allow_html=True,
        )

    st.caption("Metrics worden elke 30 seconden automatisch ververst.")

    # ── Beheer ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-title">Beheer</div>', unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3)

    with b1:
        if st.button("🔁 Restart bot", type="secondary", use_container_width=True):
            ok, uitvoer = _voer_uit("sudo systemctl restart tradeai")
            if ok:
                st.session_state["bot_restart_ts"] = time.time()
                st.cache_data.clear()
                st.toast("Bot wordt hergestart...", icon="🔁")
                st.success("Bot-service hergestart.")
                melding_bot_herstart()
                st.rerun()
            else:
                st.error(f"Mislukt: {uitvoer}")

    with b2:
        if st.button("🌐 Restart website", type="secondary", use_container_width=True):
            ok, uitvoer = _voer_uit("sudo systemctl restart tradeai-web")
            if ok:
                st.session_state["website_restart_ts"] = time.time()
                st.toast("Website wordt hergestart...", icon="🌐")
                st.success("Website-service hergestart.")
                melding_website_herstart()
                st.rerun()
            else:
                st.error(f"Mislukt: {uitvoer}")

    with b3:
        if st.button("⬆️ Update project", type="primary", use_container_width=True):
            with st.spinner("Git pull uitvoeren..."):
                ok, uitvoer = _voer_uit(
                    "git -C /home/admin/TRADEAI pull --ff-only"
                )
            if ok:
                st.success(f"Update geslaagd:\n```\n{uitvoer}\n```")
                st.info("Herstart de bot en website om de update toe te passen.")
            else:
                st.error(f"Update mislukt:\n```\n{uitvoer}\n```")

    # ── Logs ──────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-title">Recente logs</div>', unsafe_allow_html=True)

    logs = _recente_logs(25)
    st.markdown(
        '<div class="log-block">' + "<br>".join(logs) + "</div>",
        unsafe_allow_html=True,
    )
