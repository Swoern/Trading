"""
app.py — TradeAI Coach Dashboard (Streamlit)

Start met:  streamlit run app.py

⚠️  Dit systeem geeft GEEN financieel advies.
    Geen echte orders, geen echt geld. Alleen leren via paper trading.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
from html import escape

# Zorg dat de src/ map gevonden wordt
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

from src.data_loader import (
    load_csv,
    generate_sample_data,
    get_available_csv_files,
    fetch_live_crypto_candles,
)
from src.indicators import add_all_indicators
from src.strategy import analyze_market, generate_signal
from src.backtest import run_backtest
from src.paper_trading import (
    sluit_paper_trade,
    get_open_trades,
    get_closed_trades,
)
from src.evaluator import evalueer_trades, weekoverzicht
from src.database import init_database, get_bot_log, get_auto_trades
from src.server_management import render_server_management

try:
    from zoneinfo import ZoneInfo as _ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo as _ZoneInfo
_AMSTERDAM = _ZoneInfo("Europe/Amsterdam")

# ── Pagina-instelling ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TradeAI Coach",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_database()

# ── Session state: navigatie ──────────────────────────────────────────────────
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "bot_dashboard"

START_BALANCE = 10_000.0

DEFAULT_SETTINGS = {
    "data_mode": "Live crypto",
    "selected_file": None,
    "product_id": "BTC-USD",
    "timeframe_label": "5 minuten",
    "refresh_sec": 60,
    "auto_refresh": True,
}

for key, value in DEFAULT_SETTINGS.items():
    st.session_state.setdefault(key, value)

VALID_TABS = {
    "bot_dashboard",
    "server_mgmt",
    "analyse",
    "backtesting",
    "leergeheugen",
    "settings",
}

try:
    requested_tab = st.query_params.get("tab")
    if isinstance(requested_tab, list):
        requested_tab = requested_tab[0] if requested_tab else None
    if requested_tab in VALID_TABS:
        st.session_state.active_tab = requested_tab
except Exception:
    pass


def format_eur(value: float) -> str:
    """Formatteer eurobedragen compact en Nederlands leesbaar."""
    formatted = f"€{value:,.2f}"
    return formatted.replace(",", "X").replace(".", ",").replace("X", ".")


def format_duration(delta: timedelta | None) -> str:
    if not delta:
        return "-"
    total_seconds = max(0, int(delta.total_seconds()))
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days} d")
    if hours or days:
        parts.append(f"{hours} u")
    parts.append(f"{minutes} min")
    return " ".join(parts)


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def load_file_bot_logs(limit: int | None = 30) -> list[dict]:
    log_path = Path("bot.log")
    if not log_path.exists():
        return []
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []

    entries = []
    selected_lines = lines if limit is None else lines[-limit:]
    for line in selected_lines:
        logged_at = parse_dt(line[:19]) if len(line) >= 19 else None
        message = line[20:].strip() if logged_at else line.strip()
        entries.append({
            "logged_at": logged_at.isoformat() if logged_at else "",
            "message": message,
            "_dt": logged_at,
        })
    return entries


def load_bot_activity(limit: int = 10) -> list[dict]:
    entries = []
    for row in get_bot_log(limit):
        dt = parse_dt(row.get("logged_at"))
        entries.append({
            "logged_at": row.get("logged_at", ""),
            "message": row.get("message", ""),
            "_dt": dt,
        })
    entries.extend(load_file_bot_logs(limit * 3))
    entries = [e for e in entries if e.get("message")]
    entries.sort(key=lambda e: e.get("_dt") or datetime.min, reverse=True)
    return entries[:limit]


def get_bot_status(activity: list[dict]) -> dict:
    now = datetime.now()
    latest = next((item for item in activity if item.get("_dt")), None)
    latest_dt = latest.get("_dt") if latest else None
    latest_msg = (latest.get("message", "") if latest else "").lower()
    is_recent = latest_dt is not None and (now - latest_dt) <= timedelta(minutes=20)
    is_stopped = "gestopt" in latest_msg
    is_active = bool(is_recent and not is_stopped)

    started_at = None
    file_history = load_file_bot_logs(limit=None)
    status_history = reversed(file_history) if file_history else activity
    for item in status_history:
        msg = item.get("message", "").lower()
        if item.get("_dt") and ("bot gestart" in msg or "tradeai bot gestart" in msg):
            started_at = item["_dt"]
            break

    uptime = now - started_at if is_active and started_at else None
    return {
        "active": is_active,
        "uptime": format_duration(uptime),
        "last_check": latest_dt.strftime("%H:%M:%S") if latest_dt else "-",
    }


def _render_health_panel() -> None:
    """Compacte gezondheidsstatus: heartbeat-leeftijd + per-asset databron-status."""
    try:
        from src.bot_learning import get_health_overview
        overview = get_health_overview()
    except Exception:
        return

    hb = overview.get("heartbeat")
    assets = overview.get("assets", [])
    if not hb and not assets:
        return

    if hb and hb.get("age_seconds") is not None:
        age = int(hb["age_seconds"])
        age_txt = f"{age}s geleden" if age < 120 else f"{age // 60}m geleden"
        gezond = age < 35 * 60
        kop = f"{'🟢' if gezond else '🔴'} Bot-gezondheid — laatste cyclus {age_txt}"
    else:
        kop = "⚪ Bot-gezondheid — nog geen heartbeat"

    with st.expander(kop, expanded=False):
        if not assets:
            st.caption("Nog geen databron-status geregistreerd.")
            return
        _emoji = {"ok": "🟢", "alive": "🟢", "stale": "🟡", "down": "🔴"}
        for a in assets:
            age = a.get("age_seconds")
            age_txt = "-" if age is None else (f"{int(age)}s" if age < 120 else f"{int(age) // 60}m")
            st.markdown(
                f"{_emoji.get(a.get('status'), '⚪')} **{a['asset']}** — "
                f"{a.get('status', '?')} ({age_txt} geleden) · {a.get('message', '')}"
            )


def _render_llm_advice_panel() -> None:
    """Toon de laatste adviezen van de optionele LLM-adviseslaag (Fase 3)."""
    try:
        from src.database import get_llm_advice_log
        log = get_llm_advice_log(limit=6)
    except Exception:
        return
    if not log:
        return  # laag staat uit of heeft nog geen advies gegeven

    with st.expander("🧠 AI-adviespanel (laatste adviezen)", expanded=False):
        for a in log:
            bias = float(a.get("bias") or 0)
            richting = "📈 bullish" if bias > 0.15 else ("📉 bearish" if bias < -0.15 else "➖ neutraal")
            st.markdown(
                f"**{a['asset']}** · {richting} (bias {bias:+.2f}, "
                f"vertrouwen {float(a.get('confidence') or 0):.0%}, "
                f"Δdrempel {float(a.get('min_conf_delta') or 0):+.1f}) "
                f"· _{(a.get('rationale') or '').strip()[:160]}_"
            )


@st.cache_data(ttl=30, show_spinner=False)
def get_latest_price(asset: str) -> float | None:
    try:
        candles = fetch_live_crypto_candles(asset, 60, limit=3)
        return float(candles["close"].iloc[-1])
    except Exception:
        return None


def metric_card(
    label: str,
    value: str,
    sub: str = "",
    value_class: str = "value-neutral",
    icon: str = "",
    visual: str = "",
) -> str:
    icon_html = f'<div class="kpi-icon">{escape(icon)}</div>' if icon else ""
    visual_html = f'<div class="kpi-visual">{visual}</div>' if visual else ""
    return (
        '<div class="kpi-card">'
        '<div class="kpi-main">'
        '<div class="kpi-heading">'
        f'{icon_html}<div class="kpi-label">{escape(label)}</div>'
        '</div>'
        f'<div class="kpi-value {value_class}">{escape(value)}</div>'
        f'<div class="kpi-sub">{escape(sub)}</div>'
        '</div>'
        f'{visual_html}'
        '</div>'
    )


def status_card(label: str, value: str, sub: str = "", value_class: str = "value-neutral") -> str:
    live_class = " status-live" if "Actief" in value or "Online" in value else ""
    return (
        f'<div class="status-card{live_class}">'
        f'<div class="status-label">{escape(label)}</div>'
        f'<div class="status-value {value_class}">{escape(value)}</div>'
        f'<div class="status-sub">{escape(sub)}</div>'
        '</div>'
    )


def style_cells(df: pd.DataFrame, fn, subset: list[str]):
    styler = df.style
    if hasattr(styler, "map"):
        return styler.map(fn, subset=subset)
    return styler.applymap(fn, subset=subset)


def build_strategy_risk_dashboard(trades: list[dict]) -> pd.DataFrame:
    rows = []
    for strategy, group in pd.DataFrame(trades).groupby("strategy", dropna=False):
        strategy_name = strategy or "onbekend"
        result_euro = pd.to_numeric(group.get("result_euro"), errors="coerce").fillna(0)
        result_pct = pd.to_numeric(group.get("result_pct"), errors="coerce").fillna(0)
        wins = result_euro[result_euro > 0]
        losses = result_euro[result_euro <= 0]
        total = len(group)
        winrate = len(wins) / total if total else 0
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = abs(float(losses.mean())) if len(losses) else 0.0
        expectancy = winrate * avg_win - (1 - winrate) * avg_loss
        cumulative = result_euro.cumsum()
        peak = cumulative.cummax()
        drawdown = cumulative - peak
        max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
        status = "Goed" if total >= 3 and expectancy > 0 else ("Oppassen" if total >= 3 else "Te weinig data")

        best_regime = "-"
        worst_regime = "-"
        if "market_context" in group.columns:
            regime_rows = []
            for _, trade in group.iterrows():
                try:
                    import json
                    context = json.loads(trade.get("market_context") or "{}")
                except Exception:
                    context = {}
                regime_rows.append({
                    "regime": context.get("regime_label") or "unknown",
                    "result_euro": float(trade.get("result_euro") or 0),
                })
            regime_df = pd.DataFrame(regime_rows)
            if not regime_df.empty:
                regime_sum = regime_df.groupby("regime")["result_euro"].sum().sort_values()
                worst_regime = str(regime_sum.index[0])
                best_regime = str(regime_sum.index[-1])

        rows.append({
            "Strategie": strategy_name,
            "Trades": total,
            "Winrate": winrate * 100,
            "Totaal €": float(result_euro.sum()),
            "Gem. winst": avg_win,
            "Gem. verlies": -avg_loss,
            "Verwachting/trade": expectancy,
            "Max drawdown": max_drawdown,
            "Beste regime": best_regime,
            "Slechtste regime": worst_regime,
            "Status": status,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["Verwachting/trade", "Totaal €"], ascending=False)


def render_recent_closed_cards(trades: list[dict], limit: int = 10) -> None:
    def exit_label(trade: dict, result_euro: float) -> str:
        reason = str(trade.get("exit_reason") or "-")
        if reason == "sl" and trade.get("partial_tp_hit"):
            return "partial TP + trailing SL" if result_euro > 0 else "partial TP + SL"
        if reason == "sl" and result_euro > 0:
            return "trailing SL"
        if reason == "sl":
            return "stop-loss"
        if reason == "tp":
            return "take-profit"
        if reason == "time_stop":
            return "time-stop"
        if reason == "regime_flip":
            return "regime flip"
        if reason == "no_progress":
            return "geen progressie"
        if reason == "post_partial_momentum_loss":
            return "momentum weg na partial"
        return reason

    cards = []
    for trade in trades[:limit]:
        result_euro = float(trade.get("result_euro") or 0)
        result_pct = float(trade.get("result_pct") or 0)
        tone = "win" if result_euro > 0 else "loss"
        direction = str(trade.get("direction") or "").upper()
        closed = str(trade.get("closed_at") or "")[:16].replace("T", " ")
        strategy = trade.get("strategy") or "-"
        reason = exit_label(trade, result_euro)
        entry = float(trade.get("entry_price") or 0)
        exit_price = float(trade.get("exit_price") or 0)
        cards.append(
            '<div class="closed-trade-card">'
            '<div class="closed-main">'
            f'<span class="closed-asset">{escape(str(trade.get("asset") or "-"))}</span>'
            f'<span class="closed-chip">{escape(direction)}</span>'
            f'<span class="closed-chip">{escape(str(strategy))}</span>'
            '</div>'
            f'<div class="closed-result {tone}">{result_euro:+.2f} EUR <span>{result_pct:+.2f}%</span></div>'
            '<div class="closed-meta">'
            f'<span>Entry <b>{entry:,.4f}</b></span>'
            f'<span>Exit <b>{exit_price:,.4f}</b></span>'
            f'<span>Reden <b>{escape(str(reason))}</b></span>'
            f'<span>{escape(closed)}</span>'
            '</div>'
            '</div>'
        )
    st.markdown('<div class="closed-trade-list">' + "".join(cards) + '</div>', unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        '<div class="page-header">'
        f'<h1>{escape(title)}</h1>'
        f'<p>{escape(subtitle)}</p>'
        '</div>',
        unsafe_allow_html=True,
    )


def section_title(title: str) -> None:
    st.markdown(f'<div class="section-title">{escape(title)}</div>', unsafe_allow_html=True)


def compact_price(value: float | None) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):,.2f}"
    except Exception:
        return "-"


def last_valid(series) -> float | None:
    try:
        cleaned = series.dropna()
        if cleaned.empty:
            return None
        return float(cleaned.iloc[-1])
    except Exception:
        return None


def timeframe_short(label: str) -> str:
    return {
        "1 minuut": "1m",
        "5 minuten": "5m",
        "15 minuten": "15m",
        "1 uur": "1h",
        "6 uur": "6h",
        "1 dag": "1D",
    }.get(label, label)


def chart_link(range_value: str | None = None, indicators: bool | None = None) -> str:
    range_value = range_value or chart_range
    indicators_value = show_chart_indicators if indicators is None else indicators
    return f"/?tab=bot_dashboard&chart_range={escape(range_value)}&chart_indicators={'1' if indicators_value else '0'}"


def dashboard_topbar(status: dict, asset: str) -> None:
    live_text = "Bot online" if status["active"] else "Bot gestopt"
    live_class = "topbar-dot live" if status["active"] else "topbar-dot"
    st.markdown(
        '<div class="dashboard-topbar">'
        '<div class="topbar-left">'
        f'<span class="{live_class}"></span>'
        f'<strong>{escape(live_text)}</strong>'
        f'<span class="topbar-chip">{escape(asset)}</span>'
        '</div>'
        '<div class="topbar-center">'
        f'<span>Laatste check {escape(status["last_check"])}</span>'
        '</div>'
        '<div class="topbar-right">'
        '<span>Paper mode</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def signal_panel(signal: dict, analyse: dict | None = None) -> str:
    is_setup = signal.get("type") == "setup"
    badge = (signal.get("richting") or "WACHT").upper() if is_setup else "WACHT"
    badge_class = "signal-badge long" if is_setup else "signal-badge wait"
    strength = 72 if is_setup else 38
    trend = (analyse or {}).get("trend", "-")
    volume = (analyse or {}).get("volume_status", "-")

    def level(label: str, value, tone: str = "") -> str:
        shown = "-" if value is None else f"{value:,.4f}" if isinstance(value, (int, float)) else str(value)
        return (
            '<div class="signal-row">'
            f'<span>{escape(label)}</span>'
            f'<strong class="{tone}">{escape(shown)}</strong>'
            '</div>'
        )

    reason = signal.get("reden") or signal.get("blokkades") or [signal.get("uitleg", "Geen details beschikbaar.")]
    reason_text = str(reason[0]) if reason else "Geen details beschikbaar."

    return (
        '<div class="signal-panel">'
        '<div class="signal-head">'
        '<h3>Huidig signaal</h3>'
        f'<span class="{badge_class}">{escape(badge)}</span>'
        '</div>'
        '<div class="signal-strength">'
        '<div><span>Sterkte</span><strong>' + str(strength) + '%</strong></div>'
        f'<div class="signal-meter"><span style="width:{strength}%"></span></div>'
        '</div>'
        + level("Entry prijs", signal.get("entry"))
        + level("Stop-Loss", signal.get("stop_loss"), "value-negative")
        + level("Take-Profit", signal.get("take_profit"), "value-positive")
        + level("Risk/Reward", f"1 : {signal.get('risk_reward')}" if signal.get("risk_reward") else "-")
        + level("Trend", trend.capitalize(), "value-positive" if trend == "uptrend" else "")
        + level("Volume", volume.capitalize())
        + '<div class="signal-note">'
        '<strong>Waarom dit signaal?</strong>'
        f'<p>{escape(reason_text)}</p>'
        '</div>'
        '</div>'
    )


SPARK_UP = (
    '<svg class="sparkline" viewBox="0 0 92 40" aria-hidden="true">'
    '<path d="M4 31 L18 25 L31 29 L44 18 L57 22 L70 9 L88 14" />'
    '</svg>'
)
SPARK_DOWN = (
    '<svg class="sparkline down" viewBox="0 0 92 40" aria-hidden="true">'
    '<path d="M4 10 L18 14 L31 12 L44 19 L57 17 L70 25 L88 31" />'
    '</svg>'
)
DONUT_VISUAL = (
    '<div class="donut-visual" aria-hidden="true">'
    '<span></span>'
    '</div>'
)
BARS_VISUAL = (
    '<div class="bars-visual" aria-hidden="true">'
    '<span></span><span></span><span></span>'
    '</div>'
)


def polish_chart(fig: go.Figure, height: int, top: int = 28, showlegend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", color="#0F172A", size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            font=dict(size=11, color="#475569"),
        ),
        margin=dict(l=8, r=8, t=top, b=8),
        showlegend=showlegend,
        hoverlabel=dict(
            bgcolor="#0F172A",
            bordercolor="#0F172A",
            font=dict(color="#FFFFFF", family="Inter, Segoe UI, Arial, sans-serif"),
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#EEF2F7",
        zeroline=False,
        linecolor="#E2E8F0",
        tickfont=dict(color="#64748B", size=11),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#EEF2F7",
        zeroline=False,
        linecolor="#E2E8F0",
        tickfont=dict(color="#64748B", size=11),
    )
    return fig

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --bg: #F8FAFC;
        --card: #FFFFFF;
        --text: #0F172A;
        --muted: #64748B;
        --accent: #10B981;
        --accent-hover: #059669;
        --border: #E2E8F0;
        --error: #EF4444;
        --warning: #F59E0B;
        --blue: #2563EB;
    }
    .stApp {
        background: var(--bg);
        color: var(--text);
    }
    [data-testid="stAppViewContainer"] > .main {
        background: var(--bg);
    }
    [data-testid="stMainBlockContainer"] {
        padding-top: 1.35rem;
        padding-bottom: 2rem;
        max-width: 1420px;
    }
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    #MainMenu,
    footer {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }
    [data-testid="stSidebar"] {
        background: #F1F5F9;
        border-right: 1px solid var(--border);
        width: 232px !important;
        min-width: 232px !important;
        max-width: 232px !important;
        height: 100vh !important;
        overflow: hidden !important;
        display: block !important;
        visibility: visible !important;
        transform: translateX(0) !important;
        left: 0 !important;
        transition: none !important;
    }
    [data-testid="stSidebar"][aria-expanded="false"],
    [data-testid="stSidebar"]:not([aria-expanded="true"]) {
        width: 232px !important;
        min-width: 232px !important;
        max-width: 232px !important;
        transform: translateX(0) !important;
        display: block !important;
        visibility: visible !important;
    }
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stBaseButton-headerNoPadding"],
    [data-testid="stExpandSidebarButton"],
    [data-testid="stMainMenuButton"],
    [data-testid="stBaseButton-header"],
    button[kind="header"],
    button[kind="headerNoPadding"] {
        display: none !important;
        visibility: hidden !important;
    }
    [data-testid="stSidebar"]::-webkit-scrollbar,
    [data-testid="stSidebar"] *::-webkit-scrollbar {
        width: 0 !important;
        height: 0 !important;
        display: none !important;
    }
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] * {
        scrollbar-width: none;
        -ms-overflow-style: none;
    }
    [data-testid="stSidebar"] > div:first-child {
        height: 100vh !important;
        overflow: hidden !important;
    }
    [data-testid="stSidebar"] * {
        color: var(--text);
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.18rem;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding: 0.35rem 0.85rem 0.75rem 0.85rem;
    }
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 0 0 8px 0;
        border-bottom: 1px solid #CBD5E1;
        margin-bottom: 6px;
    }
    .sidebar-logo {
        width: 34px;
        height: 34px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        background: #E0F2FE;
        border: 1px solid #BAE6FD;
        color: #0369A1;
        font-size: 17px;
        flex: 0 0 auto;
    }
    .sidebar-brand-text {
        min-width: 0;
    }
    .sidebar-brand h2 {
        margin: 0;
        font-size: 16px;
        line-height: 1.15;
        letter-spacing: 0;
        color: var(--text);
    }
    .sidebar-brand p {
        margin: 4px 0 0 0;
        font-size: 11px;
        color: var(--muted);
    }
    .sidebar-section-label {
        margin: 11px 0 4px 4px;
        color: #64748B;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .sidebar-footer {
        margin-top: 12px;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        background: #FFFFFF;
        padding: 10px 11px;
    }
    .sidebar-footer strong {
        display: block;
        color: var(--text);
        font-size: 12px;
        line-height: 1.2;
        margin-bottom: 3px;
    }
    .sidebar-footer span {
        color: var(--muted);
        font-size: 11px;
        line-height: 1.25;
    }
    [data-testid="stSidebar"] button,
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
        width: 100% !important;
        min-height: 39px !important;
        justify-content: flex-start !important;
        border-radius: 8px !important;
        border: 1px solid transparent !important;
        background: transparent !important;
        color: var(--text) !important;
        font-weight: 600 !important;
        padding: 10px 12px !important;
        text-align: left !important;
        display: flex !important;
        align-items: center !important;
        box-shadow: none !important;
        transition: background 140ms ease, border-color 140ms ease, color 140ms ease, transform 140ms ease !important;
    }
    [data-testid="stSidebar"] button p,
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] p,
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] p {
        width: 100%;
        margin: 0 !important;
        font-size: 14px !important;
        line-height: 1.1 !important;
        text-align: left !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    .page-header {
        margin: 0 0 18px 0;
    }
    .page-header h1 {
        margin: 0;
        color: var(--text);
        font-size: 30px;
        line-height: 1.2;
        letter-spacing: 0;
    }
    .page-header p {
        margin: 6px 0 0 0;
        color: var(--muted);
        font-size: 14px;
    }
    div[data-testid="stAlert"] {
        border-radius: 8px;
        border: 1px solid var(--border);
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
    }
    [data-testid="stMetric"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 14px 16px;
    }
    [data-testid="stMetricLabel"] p {
        color: var(--muted) !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--text) !important;
    }
    [data-testid="stBaseButton-secondary"] {
        background: #FFFFFF !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        box-shadow: none !important;
    }
    [data-testid="stBaseButton-secondary"]:hover {
        background: #F8FAFC !important;
        border-color: #BFDBFE !important;
        color: #1D4ED8 !important;
    }
    [data-testid="stBaseButton-primary"] {
        background: var(--accent) !important;
        color: #FFFFFF !important;
        border: 1px solid var(--accent) !important;
        border-radius: 8px !important;
        box-shadow: none !important;
    }
    [data-testid="stBaseButton-primary"]:hover {
        background: var(--accent-hover) !important;
        border-color: var(--accent-hover) !important;
    }
    div[data-testid="stSlider"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stRadio"] label {
        color: var(--text) !important;
    }
    hr {
        border-color: var(--border) !important;
    }
    /* ── Bestaande stijlen ─────────────────────────────────────── */
    .metric-box {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 12px 16px;
        margin: 4px 0;
    }
    .warn-box {
        background: #FFFBEB;
        border-left: 4px solid var(--warning);
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 0.85em;
    }

    .dashboard-title {
        border: none;
        border-radius: 0;
        background: transparent;
        padding: 0;
        margin: 0 0 18px 0;
    }
    .dashboard-title h2 {
        margin: 0;
        color: var(--text);
        font-size: 24px;
        line-height: 1.2;
        letter-spacing: 0;
    }
    .dashboard-title p {
        margin: 6px 0 0 0;
        color: var(--muted);
        font-size: 14px;
    }
    .status-card,
    .kpi-card {
        min-height: 96px;
        border: 1px solid var(--border);
        border-radius: 8px;
        background: var(--card);
        padding: 14px 16px;
    }
    .kpi-card {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 14px;
    }
    .kpi-main {
        min-width: 0;
    }
    .kpi-heading {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 10px;
    }
    .kpi-icon {
        width: 36px;
        height: 36px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        background: #ECFDF5;
        color: #047857;
        font-size: 17px;
        flex: 0 0 auto;
    }
    .status-label,
    .kpi-label {
        color: var(--muted);
        font-size: 12px;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.04em;
        margin-bottom: 10px;
    }
    .status-value,
    .kpi-value {
        color: var(--text);
        font-size: 24px;
        font-weight: 750;
        line-height: 1.1;
        letter-spacing: 0;
    }
    .status-sub,
    .kpi-sub {
        color: var(--muted);
        font-size: 12px;
        margin-top: 8px;
    }
    .kpi-visual {
        flex: 0 0 auto;
        min-width: 72px;
        display: flex;
        justify-content: flex-end;
    }
    .sparkline {
        width: 88px;
        height: 42px;
        overflow: visible;
    }
    .sparkline path {
        fill: none;
        stroke: #10B981;
        stroke-width: 2.4;
        stroke-linecap: round;
        stroke-linejoin: round;
    }
    .sparkline.down path {
        stroke: #EF4444;
    }
    .donut-visual {
        width: 54px;
        height: 54px;
        border-radius: 50%;
        background: conic-gradient(#10B981 0 108deg, #E2E8F0 108deg 360deg);
        display: grid;
        place-items: center;
    }
    .donut-visual span {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        background: #FFFFFF;
        display: block;
    }
    .bars-visual {
        height: 48px;
        display: flex;
        align-items: flex-end;
        gap: 9px;
        padding-right: 4px;
    }
    .bars-visual span {
        width: 7px;
        border-radius: 999px 999px 2px 2px;
        background: #7AA7FF;
    }
    .bars-visual span:nth-child(1) { height: 23px; }
    .bars-visual span:nth-child(2) { height: 35px; }
    .bars-visual span:nth-child(3) { height: 28px; }
    .value-positive { color: var(--accent); }
    .value-negative { color: var(--error); }
    .value-warning { color: var(--warning); }
    .value-neutral { color: var(--text); }
    .section-title {
        margin: 24px 0 10px 0;
        color: var(--text);
        font-size: 18px;
        font-weight: 700;
    }
    .closed-trade-list {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(310px, 1fr));
        gap: 10px;
        margin-bottom: 12px;
    }
    .closed-trade-card {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 12px 14px;
    }
    .closed-main,
    .closed-meta {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }
    .closed-asset {
        font-weight: 800;
        color: var(--text);
    }
    .closed-chip {
        border: 1px solid #CBD5E1;
        border-radius: 999px;
        padding: 2px 7px;
        font-size: 11px;
        color: #475569;
        background: #F8FAFC;
    }
    .closed-result {
        margin: 9px 0 8px 0;
        font-size: 20px;
        line-height: 1.15;
        font-weight: 800;
    }
    .closed-result span {
        font-size: 13px;
        font-weight: 700;
        margin-left: 6px;
    }
    .closed-result.win { color: #059669; }
    .closed-result.loss { color: #DC2626; }
    .closed-meta {
        color: #64748B;
        font-size: 12px;
    }
    .closed-meta b {
        color: #0F172A;
        font-weight: 700;
    }
    .market-card {
        background: #FFFFFF;
        border: 1px solid #E5EAF2;
        border-radius: 10px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        overflow: hidden;
        margin-top: 18px;
    }
    .market-card-head {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 14px 16px 4px 16px;
    }
    .market-card-head h3 {
        margin: 0;
        font-size: 15px;
        line-height: 1.2;
        color: #111827;
        font-weight: 800;
    }
    .market-info-dot {
        width: 15px;
        height: 15px;
        border: 1px solid #CBD5E1;
        border-radius: 50%;
        color: #64748B;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 10px;
        font-weight: 800;
    }
    .chart-toolbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        padding: 8px 16px 12px 16px;
        border-bottom: 1px solid #EEF2F7;
        flex-wrap: wrap;
    }
    .chart-toolbar-left,
    .chart-toolbar-right {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }
    .chart-asset-label,
    .chart-icon-btn,
    .chart-indicator-btn,
    .chart-range-group {
        min-height: 34px;
        border: 1px solid #E5EAF2;
        background: #FFFFFF;
        color: #334155;
        border-radius: 6px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 1px 1px rgba(15, 23, 42, 0.02);
    }
    .chart-asset-label {
        padding: 0 10px;
        gap: 8px;
        font-size: 12px;
        font-weight: 700;
        color: #0F172A;
        background: #F8FAFC;
    }
    .chart-icon-btn {
        width: 34px;
        color: #475569;
    }
    .chart-indicator-btn {
        padding: 0 12px;
        gap: 7px;
        font-size: 12px;
        font-weight: 700;
        text-decoration: none !important;
    }
    .chart-indicator-btn.active {
        background: #F8FAFC;
        color: #0F172A;
    }
    .chart-range-group {
        overflow: hidden;
        gap: 0;
    }
    .chart-range-group a {
        padding: 8px 10px;
        font-size: 12px;
        color: #64748B;
        border-right: 1px solid #EEF2F7;
        text-decoration: none !important;
    }
    .chart-range-group a:last-child {
        border-right: 0;
    }
    .chart-range-group .active {
        background: #F8FAFC;
        color: #0F172A;
        font-weight: 800;
    }
    .market-chart-body {
        padding: 0 8px 0 8px;
    }
    .market-card-foot {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        border-top: 1px solid #EEF2F7;
        padding: 11px 16px 13px 16px;
        color: #64748B;
        font-size: 12px;
        flex-wrap: wrap;
    }
    .market-live-status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }
    .market-live-dot {
        width: 8px;
        height: 8px;
        border-radius: 999px;
        background: #10B981;
        box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.12);
    }

    /* ── Custom nav-balk ───────────────────────────────────────── */

    [data-testid="stMarkdown"]:has(#nav-bar-marker)
    + [data-testid="stHorizontalBlock"] {
        gap: 12px !important;
        border-bottom: 1px solid #2d3150;
        padding-bottom: 14px;
        margin-bottom: 18px;
    }

    /* Alle knoppen in de nav-rij: tab-uiterlijk */
    [data-testid="stMarkdown"]:has(#nav-bar-marker)
    + [data-testid="stHorizontalBlock"] button {
        width: 100% !important;
        min-height: 50px !important;
        background: #111827 !important;
        border: 1px solid #2d3150 !important;
        border-radius: 7px !important;
        color: #e2e8f0 !important;
        padding: 9px 12px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        white-space: nowrap !important;
        box-shadow: none !important;
        transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease !important;
    }

    /* Hover */
    [data-testid="stMarkdown"]:has(#nav-bar-marker)
    + [data-testid="stHorizontalBlock"] button:hover {
        color: #ffffff !important;
        background: #182235 !important;
        border-color: rgba(66,165,245,0.55) !important;
    }

    /* Actieve tab: blauwe underline */
    [data-testid="stMarkdown"]:has(.nav-active)
    + [data-testid="stButton"] button {
        color: #ffffff !important;
        background: rgba(66,165,245,0.16) !important;
        border-color: #42a5f5 !important;
    }

    /* ── Server Management logs ────────────────────────────────── */
    .log-block {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 14px 16px;
        font-family: "Courier New", monospace;
        font-size: 12px;
        color: #334155;
        line-height: 1.8;
        max-height: 260px;
        overflow-y: auto;
    }
    [data-testid="stSidebar"] button:hover,
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover,
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover {
        background: #F1F5F9 !important;
        border-color: #E2E8F0 !important;
        color: #0F172A !important;
        transform: translateX(1px);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdown"]:has(.nav-active)
    + [data-testid="stButton"] button {
        background: #E0F2FE !important;
        border-color: #BAE6FD !important;
        color: #0369A1 !important;
        font-weight: 750 !important;
    }

    /* Premium control-room polish */
    :root {
        --bg: #F6F8FB;
        --sidebar: #111827;
        --sidebar-soft: #172033;
        --card: #FFFFFF;
        --text: #0F172A;
        --muted: #64748B;
        --accent: #10B981;
        --accent-hover: #059669;
        --border: #E5EAF1;
        --soft-border: rgba(15, 23, 42, 0.08);
        --error: #EF4444;
        --warning: #F59E0B;
        --blue: #2563EB;
        --shadow: 0 14px 32px rgba(15, 23, 42, 0.07);
    }
    .stApp {
        background:
            linear-gradient(180deg, #F8FAFC 0%, #F6F8FB 44%, #F3F7FB 100%);
    }
    [data-testid="stMainBlockContainer"] {
        padding: 1.05rem 2rem 2rem 2rem;
        max-width: none;
    }
    [data-testid="stSidebar"] {
        background: var(--sidebar) !important;
        border-right: 1px solid rgba(255,255,255,0.08) !important;
    }
    [data-testid="stSidebar"] * {
        color: #E5E7EB !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding: 0.8rem 0.8rem 1rem 0.8rem;
    }
    .sidebar-shell {
        min-height: calc(100vh - 1.8rem);
        display: flex;
        flex-direction: column;
        padding: 6px 0 8px 0;
    }
    .sidebar-brand {
        border-bottom: 1px solid rgba(255,255,255,0.11);
        padding: 10px 4px 18px 4px;
        margin-bottom: 14px;
        display: grid;
        grid-template-columns: 32px 1fr 22px;
        align-items: center;
        gap: 12px;
    }
    .sidebar-logo {
        width: 32px;
        height: 32px;
        background: transparent;
        border: 0;
        color: #34D399 !important;
        box-shadow: none;
        display: inline-flex;
        align-items: center;
        justify-content: center;
    }
    .sidebar-logo svg,
    .sidebar-collapse svg,
    .sidebar-link svg,
    .paper-more svg {
        width: 18px;
        height: 18px;
        stroke: currentColor;
        fill: none;
        stroke-width: 1.8;
        stroke-linecap: round;
        stroke-linejoin: round;
    }
    .sidebar-logo svg {
        width: 24px;
        height: 24px;
        fill: #34D399;
        stroke: #34D399;
    }
    .sidebar-collapse {
        color: #CBD5E1 !important;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        opacity: 0.9;
    }
    .sidebar-brand h2 {
        color: #F8FAFC !important;
        font-weight: 760;
        font-size: 16px;
        margin: 0;
        line-height: 1.1;
    }
    .sidebar-brand p,
    .sidebar-section-label,
    .sidebar-footer span {
        color: #94A3B8 !important;
    }
    .sidebar-brand p {
        display: none;
    }
    .sidebar-nav {
        display: flex;
        flex-direction: column;
        gap: 6px;
    }
    .sidebar-section-label {
        margin: 22px 10px 8px 10px;
        color: #7B879A !important;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .sidebar-link {
        position: relative;
        min-height: 44px;
        border: 1px solid transparent;
        border-radius: 8px;
        color: #CBD5E1 !important;
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 0 12px;
        font-size: 14px;
        font-weight: 650;
        text-decoration: none !important;
        transition: background 180ms ease, border-color 180ms ease, color 180ms ease, transform 180ms ease;
    }
    .sidebar-link:hover {
        background: rgba(255,255,255,0.06);
        border-color: rgba(148,163,184,0.16);
        color: #FFFFFF !important;
        transform: translateX(2px);
    }
    .sidebar-link.active {
        background: linear-gradient(90deg, rgba(255,255,255,0.10), rgba(255,255,255,0.055));
        border-color: rgba(52,211,153,0.18);
        color: #6EE7B7 !important;
        box-shadow: inset -3px 0 0 #34D399;
    }
    .sidebar-link.active svg {
        color: #34D399;
    }
    .sidebar-spacer {
        flex: 1;
        min-height: 120px;
    }
    .sidebar-footer {
        margin: 18px 6px 0 6px;
        background: linear-gradient(145deg, rgba(255,255,255,0.085), rgba(255,255,255,0.035));
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 18px 18px;
        box-shadow: 0 18px 36px rgba(2, 6, 23, 0.24);
    }
    .paper-status {
        display: flex;
        align-items: center;
        gap: 10px;
        color: #6EE7B7 !important;
        font-weight: 800;
        font-size: 14px;
        margin-bottom: 16px;
    }
    .paper-dot {
        width: 12px;
        height: 12px;
        border-radius: 999px;
        background: #34D399;
        box-shadow: 0 0 0 6px rgba(52,211,153,0.13);
        flex: 0 0 auto;
    }
    .sidebar-footer p {
        margin: 0;
        color: #CBD5E1 !important;
        font-size: 13px;
        line-height: 1.55;
    }
    .paper-more {
        margin-top: 16px;
        color: #93C5FD !important;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 13px;
        font-weight: 700;
        text-decoration: none !important;
    }
    [data-testid="stSidebar"] button,
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
        color: #CBD5E1 !important;
        border-radius: 8px !important;
        transition: background 180ms ease, border-color 180ms ease, color 180ms ease, transform 180ms ease !important;
    }
    [data-testid="stSidebar"] button:hover,
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover,
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover {
        background: var(--sidebar-soft) !important;
        border-color: rgba(148,163,184,0.18) !important;
        color: #FFFFFF !important;
        transform: translateX(2px);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdown"]:has(.nav-active)
    + [data-testid="stButton"] button {
        background: rgba(16,185,129,0.13) !important;
        border-color: rgba(16,185,129,0.42) !important;
        color: #A7F3D0 !important;
        box-shadow: inset 3px 0 0 #10B981 !important;
    }
    @keyframes soft-enter {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes live-pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(16,185,129,0.22), var(--shadow); }
        50% { box-shadow: 0 0 0 7px rgba(16,185,129,0), var(--shadow); }
    }
    .dashboard-topbar {
        min-height: 58px;
        margin: 0 0 24px 0;
        padding: 0 20px;
        border: 1px solid var(--soft-border);
        border-radius: 8px;
        background: rgba(255,255,255,0.96);
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.055);
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        align-items: center;
        gap: 16px;
        animation: soft-enter 360ms cubic-bezier(.2,.8,.2,1) both;
    }
    .topbar-left,
    .topbar-center,
    .topbar-right {
        display: flex;
        align-items: center;
        gap: 10px;
        color: #334155;
        font-size: 14px;
    }
    .topbar-center {
        justify-content: center;
        color: #475569;
    }
    .topbar-right {
        justify-content: flex-end;
        color: #0F766E;
        font-weight: 700;
    }
    .topbar-dot {
        width: 11px;
        height: 11px;
        border-radius: 999px;
        background: #94A3B8;
        box-shadow: 0 0 0 5px rgba(148,163,184,0.12);
    }
    .topbar-dot.live {
        background: #10B981;
        box-shadow: 0 0 0 5px rgba(16,185,129,0.12);
    }
    .topbar-chip,
    .signal-badge {
        border-radius: 8px;
        padding: 6px 10px;
        font-size: 12px;
        font-weight: 800;
    }
    .topbar-chip {
        background: #ECFDF5;
        color: #047857;
    }
    .page-header,
    .dashboard-title,
    .dashboard-topbar,
    .status-card,
    .kpi-card,
    [data-testid="stMetric"],
    [data-testid="stDataFrame"],
    [data-testid="stPlotlyChart"],
    div[data-testid="stAlert"],
    .log-block {
        animation: soft-enter 420ms cubic-bezier(.2,.8,.2,1) both;
    }
    .status-card,
    .kpi-card,
    [data-testid="stMetric"],
    [data-testid="stPlotlyChart"],
    [data-testid="stDataFrame"],
    .log-block {
        border: 1px solid var(--soft-border);
        border-radius: 8px;
        background: rgba(255,255,255,0.94);
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.055);
        transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
    }
    .status-card:hover,
    .kpi-card:hover,
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(37, 99, 235, 0.18);
        box-shadow: var(--shadow);
    }
    .status-card.status-live {
        border-color: rgba(16,185,129,0.26);
        animation: soft-enter 420ms cubic-bezier(.2,.8,.2,1) both, live-pulse 2.8s ease-in-out infinite;
    }
    .page-header,
    .dashboard-title {
        padding: 0 0 2px 0;
    }
    .page-header h1,
    .dashboard-title h2 {
        font-size: clamp(26px, 2vw, 34px);
        font-weight: 780;
        letter-spacing: 0;
    }
    .page-header p,
    .dashboard-title p {
        max-width: 760px;
        line-height: 1.55;
    }
    .section-title {
        margin-top: 28px;
        font-size: 16px;
        letter-spacing: 0;
    }
    .status-label,
    .kpi-label {
        color: #64748B;
        letter-spacing: 0.035em;
    }
    .status-value,
    .kpi-value {
        font-size: clamp(22px, 2vw, 28px);
        font-weight: 780;
    }
    [data-testid="stPlotlyChart"] {
        padding: 10px 10px 4px 10px;
        overflow: hidden;
    }
    .modebar-container,
    .modebar {
        display: none !important;
        visibility: hidden !important;
    }
    .market-panel-title {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 26px 0 8px 0;
    }
    .market-panel-title h3 {
        margin: 0;
        font-size: 18px;
        color: #0F172A;
    }
    .market-panel-title span {
        color: #64748B;
        font-size: 12px;
        font-weight: 700;
    }
    .signal-panel {
        min-height: 440px;
        border: 1px solid var(--soft-border);
        border-radius: 8px;
        background: rgba(255,255,255,0.96);
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.055);
        padding: 18px 18px 16px 18px;
        animation: soft-enter 460ms cubic-bezier(.2,.8,.2,1) both;
    }
    .signal-head,
    .signal-strength > div,
    .signal-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
    }
    .signal-head h3 {
        margin: 0;
        color: #0F172A;
        font-size: 18px;
    }
    .signal-badge.long {
        background: #DCFCE7;
        color: #047857;
    }
    .signal-badge.wait {
        background: #FEF3C7;
        color: #B45309;
    }
    .signal-strength {
        margin: 26px 0 22px 0;
    }
    .signal-strength span,
    .signal-row span {
        color: #64748B;
        font-size: 13px;
    }
    .signal-strength strong {
        color: #059669;
    }
    .signal-meter {
        margin-top: 9px;
        height: 8px;
        border-radius: 999px;
        background: #E2E8F0;
        overflow: hidden;
    }
    .signal-meter span {
        display: block;
        height: 100%;
        border-radius: inherit;
        background: #10B981;
    }
    .signal-row {
        min-height: 36px;
        border-top: 1px solid #EEF2F7;
    }
    .signal-row strong {
        color: #0F172A;
        font-size: 13px;
        text-align: right;
    }
    .signal-note {
        margin-top: 18px;
        border: 1px solid rgba(16,185,129,0.18);
        border-radius: 8px;
        background: #ECFDF5;
        padding: 14px;
    }
    .signal-note strong {
        display: block;
        color: #0F172A;
        margin-bottom: 6px;
    }
    .signal-note p {
        margin: 0;
        color: #475569;
        font-size: 13px;
        line-height: 1.45;
    }
    .log-block {
        max-height: 300px;
        background: #0F172A;
        border-color: rgba(148,163,184,0.18);
        color: #CBD5E1;
    }
    div[data-testid="stAlert"] {
        box-shadow: none;
    }
    @media (max-width: 900px) {
        [data-testid="stSidebar"],
        [data-testid="stSidebar"][aria-expanded="false"],
        [data-testid="stSidebar"]:not([aria-expanded="true"]) {
            width: 190px !important;
            min-width: 190px !important;
            max-width: 190px !important;
        }
        [data-testid="stMainBlockContainer"] {
            padding-left: 0.85rem;
            padding-right: 0.85rem;
        }
        .kpi-card {
            align-items: flex-start;
        }
        .kpi-visual {
            display: none;
        }
        .status-card,
        .kpi-card {
            min-height: 88px;
        }
        [data-testid="stSidebar"] button,
        [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
        [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
            padding: 9px 10px !important;
        }
        [data-testid="stSidebar"] button p {
            font-size: 13px !important;
        }
    }
    @media (max-width: 620px) {
        [data-testid="stSidebar"],
        [data-testid="stSidebar"][aria-expanded="false"],
        [data-testid="stSidebar"]:not([aria-expanded="true"]) {
            width: 168px !important;
            min-width: 168px !important;
            max-width: 168px !important;
        }
        .sidebar-brand {
            gap: 7px;
        }
        .sidebar-logo {
            width: 30px;
            height: 30px;
            font-size: 14px;
        }
        .sidebar-brand h2 {
            font-size: 13px;
        }
        .sidebar-brand p,
        .sidebar-footer span {
            font-size: 10px;
        }
        .dashboard-topbar {
            grid-template-columns: 1fr;
            align-items: flex-start;
            padding: 14px;
        }
        .topbar-center,
        .topbar-right {
            justify-content: flex-start;
        }
    }
    @media (prefers-reduced-motion: reduce) {
        *,
        *::before,
        *::after {
            animation-duration: 0.001ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.001ms !important;
            scroll-behavior: auto !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ── Navigatie en instellingen ────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def laad_data(pad: str) -> pd.DataFrame:
    return load_csv(pad)


@st.cache_data(ttl=15, show_spinner=False)
def laad_live_data(product: str, candle_seconds: int) -> pd.DataFrame:
    return fetch_live_crypto_candles(product, candle_seconds, limit=300)


csv_files = get_available_csv_files("data")
if not csv_files:
    with st.spinner("Sample dataset wordt aangemaakt..."):
        generate_sample_data("data/sample_btc.csv")
    csv_files = get_available_csv_files("data")

if st.session_state.selected_file not in csv_files:
    st.session_state.selected_file = csv_files[0] if csv_files else None

SIDEBAR_ICONS = {
    "bot_dashboard": '<svg viewBox="0 0 24 24"><path d="M7 8.5h10a3 3 0 0 1 3 3v4a3 3 0 0 1-3 3H7a3 3 0 0 1-3-3v-4a3 3 0 0 1 3-3Z"/><path d="M9 12.5h.01M15 12.5h.01M9 15.5c1.9 1 4.1 1 6 0"/><path d="M12 8.5V5.5"/><path d="M9.8 5.5h4.4"/></svg>',
    "server_mgmt": '<svg viewBox="0 0 24 24"><rect x="5" y="4" width="14" height="6" rx="1.5"/><rect x="3" y="14" width="18" height="6" rx="1.5"/><path d="M7 7h.01M7 17h.01M10 17h7M12 10v4"/></svg>',
    "analyse": '<svg viewBox="0 0 24 24"><path d="M4 19V5"/><path d="M4 19h16"/><path d="M8 16V11"/><path d="M12 16V7"/><path d="M16 16v-4"/></svg>',
    "backtesting": '<svg viewBox="0 0 24 24"><path d="M4 17l5-5 4 4 7-8"/><path d="M4 20h16"/><path d="M4 4v16"/></svg>',
    "leergeheugen": '<svg viewBox="0 0 24 24"><rect x="5" y="5" width="14" height="14" rx="2"/><path d="M9 9h6M9 13h6M9 17h3"/><path d="M8 3v4M16 3v4"/></svg>',
    "settings": '<svg viewBox="0 0 24 24"><path d="M12 15.2a3.2 3.2 0 1 0 0-6.4 3.2 3.2 0 0 0 0 6.4Z"/><path d="M19.4 15a1.8 1.8 0 0 0 .36 1.98l.04.04a2.1 2.1 0 0 1-2.97 2.97l-.04-.04a1.8 1.8 0 0 0-1.98-.36 1.8 1.8 0 0 0-1.1 1.66V21.3a2.1 2.1 0 0 1-4.2 0v-.06a1.8 1.8 0 0 0-1.1-1.66 1.8 1.8 0 0 0-1.98.36l-.04.04a2.1 2.1 0 0 1-2.97-2.97l.04-.04A1.8 1.8 0 0 0 4.6 15a1.8 1.8 0 0 0-1.66-1.1H2.9a2.1 2.1 0 0 1 0-4.2h.06A1.8 1.8 0 0 0 4.6 8a1.8 1.8 0 0 0-.36-1.98l-.04-.04a2.1 2.1 0 0 1 2.97-2.97l.04.04A1.8 1.8 0 0 0 9.2 3.4a1.8 1.8 0 0 0 1.1-1.66V1.7a2.1 2.1 0 0 1 4.2 0v.06a1.8 1.8 0 0 0 1.1 1.66 1.8 1.8 0 0 0 1.98-.36l.04-.04a2.1 2.1 0 0 1 2.97 2.97l-.04.04A1.8 1.8 0 0 0 19.4 8a1.8 1.8 0 0 0 1.66 1.1h.06a2.1 2.1 0 0 1 0 4.2h-.06A1.8 1.8 0 0 0 19.4 15Z"/></svg>',
}

nav_sections = [
    ("MAIN", [
        ("bot_dashboard", "Bot Dashboard"),
        ("server_mgmt", "Server Management"),
    ]),
    ("ANALYSE", [
        ("analyse", "Analyse"),
        ("backtesting", "Backtesting"),
        ("leergeheugen", "Leergeheugen"),
    ]),
    ("SYSTEM", [
        ("settings", "Instellingen"),
    ]),
]

with st.sidebar:
    nav_html = [
        '<div class="sidebar-shell">',
        '<div class="sidebar-brand">',
        '<span class="sidebar-logo"><svg viewBox="0 0 24 24"><path d="M7 12.7 13.5 3h4.1L11 12.1h5.1L8.9 21l1.7-6.8H6.4z"/></svg></span>',
        '<span class="sidebar-brand-text"><h2>TradeAI Coach</h2><p>Paper trading dashboard</p></span>',
        '<span class="sidebar-collapse"><svg viewBox="0 0 24 24"><path d="m15 6-6 6 6 6"/><path d="m20 6-6 6 6 6"/></svg></span>',
        '</div>',
        '<div class="sidebar-nav">',
    ]
    for section, items in nav_sections:
        nav_html.append(f'<div class="sidebar-section-label">{escape(section)}</div>')
        for tab_key, text in items:
            active_cls = " active" if st.session_state.active_tab == tab_key else ""
            nav_html.append(
                f'<a class="sidebar-link{active_cls}" href="/?tab={escape(tab_key)}" target="_self">'
                f'{SIDEBAR_ICONS[tab_key]}<span>{escape(text)}</span>'
                '</a>'
            )
    nav_html.extend([
        '</div>',
        '<div class="sidebar-spacer"></div>',
        '<div class="sidebar-footer">',
        '<div class="paper-status"><span class="paper-dot"></span><span>Paper mode</span></div>',
        '<p>Dit is paper trading.<br>Geen echt geld, geen echte orders.</p>',
        '<a class="paper-more" href="/?tab=settings" target="_self">Meer info <svg viewBox="0 0 24 24"><path d="m9 18 6-6-6-6"/></svg></a>',
        '</div>',
        '</div>',
    ])
    st.markdown("".join(nav_html), unsafe_allow_html=True)

active = st.session_state.active_tab
chart_range = "1D"
show_chart_indicators = True
try:
    requested_range = st.query_params.get("chart_range", "1D")
    if isinstance(requested_range, list):
        requested_range = requested_range[0] if requested_range else "1D"
    if requested_range in {"1D", "7D", "1M", "3M", "1Y"}:
        chart_range = requested_range
    requested_indicators = st.query_params.get("chart_indicators", "1")
    if isinstance(requested_indicators, list):
        requested_indicators = requested_indicators[0] if requested_indicators else "1"
    show_chart_indicators = requested_indicators != "0"
except Exception:
    pass

timeframe_map = {
    "1 minuut": 60,
    "5 minuten": 300,
    "15 minuten": 900,
    "1 uur": 3600,
    "6 uur": 21600,
    "1 dag": 86400,
}
try:
    from src.markets import get_markets
    # De bot-markten staan voorop; extra opties voor handmatige analyse erachter.
    _bot_markets = get_markets()
    product_options = _bot_markets + [
        p for p in ["XRP-USD", "DOGE-USD", "ADA-USD"] if p not in _bot_markets
    ]
except Exception:
    product_options = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD"]

if active == "settings":
    page_header("⚙️ Instellingen", "Databron, live market en refresh-gedrag voor alle pagina's.")
    st.warning(
        "⚠️ Dit systeem geeft geen financieel advies. Paper trading is bedoeld om te leren; "
        "er worden geen echte orders geplaatst."
    )

    col_data, col_live = st.columns(2)
    with col_data:
        st.subheader("Databron")
        st.radio("Databron", ["Live crypto", "CSV-bestand"], key="data_mode")
        st.selectbox(
            "CSV dataset",
            csv_files,
            key="selected_file",
            format_func=lambda x: Path(x).stem,
            disabled=st.session_state.data_mode != "CSV-bestand",
        )
        if st.button("🔄 Nieuwe sample data aanmaken", disabled=st.session_state.data_mode != "CSV-bestand"):
            with st.spinner("Genereren..."):
                generate_sample_data("data/sample_btc.csv")
            st.cache_data.clear()
            st.rerun()
        st.caption("Eigen CSV: zet je bestand in `data/` met `timestamp, open, high, low, close, volume`.")

    with col_live:
        st.subheader("Live crypto")
        st.selectbox("Live market", product_options, key="product_id")
        st.selectbox("Timeframe", list(timeframe_map), key="timeframe_label")
        st.slider("Ververs elke", 15, 300, key="refresh_sec", step=15, format="%d sec")
        st.toggle("Auto-refresh", key="auto_refresh")
        if st.button("Nu verversen"):
            laad_live_data.clear()
            st.rerun()

data_source = "live" if st.session_state.data_mode == "Live crypto" else "csv"
selected_file = st.session_state.selected_file
asset_name = Path(selected_file).stem.upper() if selected_file else "DATA"

try:
    df = laad_data(selected_file)
except Exception as e:
    st.error(f"Fout bij laden van CSV-data: {e}")
    st.stop()

if data_source == "live":
    granularity = timeframe_map[st.session_state.timeframe_label]
    if st.session_state.auto_refresh and st_autorefresh is not None:
        st_autorefresh(interval=st.session_state.refresh_sec * 1000, key="live_data_refresh")
    try:
        df = laad_live_data(st.session_state.product_id, granularity)
        asset_name = st.session_state.product_id
        if "_simulated" in df.columns:
            api_errors = df["_errors"].iloc[0] if "_errors" in df.columns else "onbekend"
            st.warning(
                "Live data is niet beschikbaar — de grafiek toont **gesimuleerde data**. "
                f"API-fout: `{api_errors}`",
                icon="⚠️",
            )
            df = df.drop(columns=[c for c in ["_simulated", "_errors"] if c in df.columns])
    except Exception as e:
        st.error(f"Live data laden mislukt: {e}")
        st.info("Gebruik tijdelijk CSV-data via Instellingen.")
        st.stop()


# ════════════════════════════════════════════════════════════════════════════
# SERVER MANAGEMENT
# ════════════════════════════════════════════════════════════════════════════
if active == "settings":
    latest_ts = df["timestamp"].max()
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("Actieve bron", st.session_state.data_mode, "Gebruikt door analyse/backtest"), unsafe_allow_html=True)
    c2.markdown(metric_card("Asset", asset_name, "Huidige dataset of live market"), unsafe_allow_html=True)
    c3.markdown(metric_card("Candles", f"{len(df):,}", "Beschikbaar voor analyses"), unsafe_allow_html=True)
    c4.markdown(metric_card("Laatste candle", latest_ts.strftime("%d-%m %H:%M"), "Timestamp uit de data"), unsafe_allow_html=True)

    if data_source == "live" and st.session_state.auto_refresh and st_autorefresh is None:
        st.info("Auto-refresh wordt actief na installatie van `streamlit-autorefresh`.")

elif active == "server_mgmt":
    render_server_management()

# ════════════════════════════════════════════════════════════════════════════
# MARKTANALYSE
# ════════════════════════════════════════════════════════════════════════════
elif active == "analyse":
    page_header(f"📊 Marktanalyse — {asset_name}", "Prijsactie, trend, volume en huidig signaal.")
    if data_source == "live":
        st.caption("Live candles worden automatisch vernieuwd zolang auto-refresh aan staat.")

    col_grafiek, col_info = st.columns([2, 1])

    with col_grafiek:
        df_ind  = add_all_indicators(df)
        recent  = df_ind.tail(80)

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=recent["timestamp"],
            open=recent["open"], high=recent["high"],
            low=recent["low"],  close=recent["close"],
            name="Prijs",
            increasing_line_color="#10B981",
            decreasing_line_color="#EF4444",
        ))
        fig.add_trace(go.Scatter(
            x=recent["timestamp"], y=recent["sma20"],
            line=dict(color="#F59E0B", width=1.8),
            name="SMA 20",
        ))
        fig.add_trace(go.Scatter(
            x=recent["timestamp"], y=recent["sma50"],
            line=dict(color="#2563EB", width=1.8),
            name="SMA 50",
        ))
        # Support & resistance als achtergrondband
        fig.add_trace(go.Scatter(
            x=recent["timestamp"], y=recent["resistance"],
            line=dict(color="rgba(239,68,68,0.55)", width=1, dash="dot"),
            name="Resistance",
        ))
        fig.add_trace(go.Scatter(
            x=recent["timestamp"], y=recent["support"],
            line=dict(color="rgba(16,185,129,0.55)", width=1, dash="dot"),
            name="Support",
            fill="tonexty",
            fillcolor="rgba(37,99,235,0.05)",
        ))
        polish_chart(fig, height=460, top=34, showlegend=True)
        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Volume barchart
        vol_colors = [
            "#10B981" if c >= o else "#EF4444"
            for o, c in zip(recent["open"], recent["close"])
        ]
        fig_vol = go.Figure(go.Bar(
            x=recent["timestamp"], y=recent["volume"],
            marker_color=vol_colors, name="Volume",
        ))
        fig_vol.add_trace(go.Scatter(
            x=recent["timestamp"], y=recent["vol_ma20"],
            line=dict(color="#64748B", width=1),
            name="Gem. volume",
        ))
        polish_chart(fig_vol, height=150, top=4, showlegend=False)
        st.plotly_chart(fig_vol, use_container_width=True, config={"displayModeBar": False})

    with col_info:
        st.subheader("Analyse samenvatting")

        try:
            analyse = analyze_market(df)

            # Trend
            trend_emoji = {"uptrend": "🟢", "downtrend": "🔴", "onduidelijk": "🟡"}
            t = analyse["trend"]
            st.markdown(f"**Trend:** {trend_emoji.get(t, '⚪')} `{t.upper()}`")
            st.caption(analyse["trend_tekst"])
            st.divider()

            # Volatiliteit
            st.markdown(f"**Volatiliteit:** `{analyse['volatiliteit'].upper()}`")
            st.caption(analyse["volatiliteit_tekst"])
            st.divider()

            # Volume
            st.markdown(f"**Volume:** `{analyse['volume_status'].upper()}`")
            st.caption(analyse["volume_tekst"])
            st.divider()

            # RSI
            rsi_val = analyse["rsi"]
            if not pd.isna(rsi_val):
                rsi_kleur = "red" if rsi_val > 70 else ("green" if rsi_val < 30 else "white")
                st.markdown(f"**RSI:** `{rsi_val:.0f}`")
                st.progress(min(int(rsi_val), 100))
            st.caption(analyse["rsi_tekst"])
            st.divider()

            # Support / Resistance
            st.caption(analyse["positie_tekst"])
            st.divider()

            # Conclusie
            if analyse["is_duidelijk"]:
                st.success(f"✅ {analyse['conclusie']}")
            else:
                st.warning(f"⏳ {analyse['conclusie']}")

        except Exception as e:
            st.error(f"Analysefout: {e}")

    # Signaal
    section_title("Huidig signaal")

    try:
        signal = generate_signal(df)

        if signal["type"] == "setup":
            st.success(f"✅ SETUP GEVONDEN — {signal['richting'].upper()}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Entry", f"{signal['entry']:,.4f}")
            c2.metric(
                "Stop-Loss", f"{signal['stop_loss']:,.4f}",
                delta=f"{(signal['stop_loss'] - signal['entry']) / signal['entry'] * 100:.1f}%",
                delta_color="inverse",
            )
            c3.metric(
                "Take-Profit", f"{signal['take_profit']:,.4f}",
                delta=f"+{(signal['take_profit'] - signal['entry']) / signal['entry'] * 100:.1f}%",
            )
            c4.metric("Risk/Reward", f"1:{signal['risk_reward']}")

            with st.expander("Redenen voor dit signaal"):
                for r in signal["reden"]:
                    st.markdown(f"• {r}")

        elif signal["type"] == "wacht":
            st.warning("⏳ GEEN SETUP — Beter wachten")
            for b in signal.get("blokkades", []):
                st.caption(f"• {b}")
        else:
            st.info(signal.get("uitleg", "Geen informatie beschikbaar."))

        with st.expander("Volledige uitleg signaal"):
            st.code(signal["uitleg"], language=None)

    except Exception as e:
        st.error(f"Signaalkout: {e}")


# ════════════════════════════════════════════════════════════════════════════
# BACKTESTING
# ════════════════════════════════════════════════════════════════════════════
elif active == "backtesting":
    page_header("🔄 Backtesting", "Test de strategie op historische data zonder echt geld.")
    st.info(
        "Test de strategie op historische data. "
        "Dit simuleert trades ZONDER echt geld. "
        "Goede backtestresultaten zijn **geen** garantie voor de toekomst."
    )

    col_rr, col_cap, col_risk = st.columns(3)
    with col_rr:
        min_rr       = st.slider("Minimum Risk/Reward", 1.5, 4.0, 2.0, 0.5,
                                  help="Hoe hoger, hoe minder maar kwalitatievere trades.")
    with col_cap:
        start_kap    = st.number_input("Gesimuleerd startkapitaal (€)",
                                        1_000, 100_000, 10_000, 1_000)
    with col_risk:
        risico_pct   = st.slider("Risico per trade (%)", 0.5, 3.0, 1.0, 0.5,
                                  help="Hoeveel % van het kapitaal per trade riskeer je?")

    if st.button("▶️ Start backtest", type="primary"):
        with st.spinner("Backtest wordt uitgevoerd — even geduld..."):
            try:
                res = run_backtest(
                    df,
                    min_rr=min_rr,
                    initial_capital=start_kap,
                    risico_per_trade_pct=risico_pct,
                )

                if res["total_trades"] == 0:
                    st.warning(res["boodschap"])
                else:
                    st.success(res["boodschap"])

                    # KPI's
                    c1, c2, c3, c4, c5, c6 = st.columns(6)
                    c1.metric("Trades",       res["total_trades"])
                    c2.metric("Winrate",      f"{res['winrate']}%")
                    c3.metric("Totaal P&L",   f"{res['total_pnl']:+.1f}%")
                    c4.metric("Max Drawdown", f"{res['max_drawdown']:.1f}%",
                               delta_color="inverse")
                    c5.metric("Gem. winst",   f"+{res['avg_win']:.2f}%")
                    c6.metric("Gem. verlies", f"{res['avg_loss']:.2f}%",
                               delta_color="inverse")

                    st.caption(
                        f"Eindkapitaal (simulatie): €{res['eind_kapitaal']:,.2f}  "
                        f"(start: €{start_kap:,.2f})"
                    )

                    # P&L grafiek
                    df_t = res["trades_df"]
                    fig_pnl = go.Figure()
                    fig_pnl.add_trace(go.Scatter(
                        y=df_t["cumulatief_pnl"],
                        mode="lines+markers",
                        name="Cumulatief P&L %",
                        line=dict(
                            color="#10B981" if res["total_pnl"] >= 0 else "#EF4444",
                            width=2,
                        ),
                    ))
                    fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray")
                    polish_chart(fig_pnl, height=310, top=34, showlegend=False)
                    fig_pnl.update_layout(title="Cumulatief resultaat (%)")
                    st.plotly_chart(fig_pnl, use_container_width=True, config={"displayModeBar": False})

                    # Trades tabel
                    st.subheader("Alle backtest-trades")
                    show = df_t[[
                        "entry_datum", "exit_datum",
                        "entry_prijs", "exit_prijs",
                        "result_pct", "exit_reden",
                    ]].copy()
                    show.columns = [
                        "Entry", "Exit",
                        "Entry prijs", "Exit prijs",
                        "Resultaat %", "Reden",
                    ]
                    show["Resultaat %"] = show["Resultaat %"].round(2)
                    st.dataframe(
                        style_cells(
                            show,
                            lambda v: "color: #10B981" if isinstance(v, float) and v > 0
                            else ("color: #EF4444" if isinstance(v, float) and v < 0 else ""),
                            ["Resultaat %"],
                        ),
                        use_container_width=True,
                        height=300,
                    )

            except Exception as e:
                st.error(f"Backtest mislukt: {e}")
                import traceback
                with st.expander("Technische foutmelding"):
                    st.code(traceback.format_exc())


# ════════════════════════════════════════════════════════════════════════════
# BOT DASHBOARD (voorheen Paper Trading)
# ════════════════════════════════════════════════════════════════════════════
elif active == "bot_dashboard":
    bot_activity = load_bot_activity(limit=12)
    bot_status = get_bot_status(bot_activity)
    open_trades = get_open_trades()
    closed_trades = get_closed_trades()

    # KPI's alleen op basis van bot-trades (source='auto')
    auto_closed = get_auto_trades(status="closed")
    auto_open   = get_auto_trades(status="open")
    today = datetime.now(_AMSTERDAM).date().isoformat()

    closed_pnl = sum(t.get("result_euro", 0) or 0 for t in auto_closed)
    pnl_today = sum(
        t.get("result_euro", 0) or 0
        for t in auto_closed
        if (t.get("closed_at") or "")[:10] == today
    )
    wins = sum(1 for t in auto_closed if (t.get("result_pct") or 0) > 0)
    winrate = (wins / len(auto_closed) * 100) if auto_closed else 0
    balance = START_BALANCE + closed_pnl

    status_label = "🟢 Actief" if bot_status["active"] else "⚫ Gestopt"
    status_class = "value-positive" if bot_status["active"] else "value-neutral"
    pnl_class = "value-positive" if pnl_today > 0 else ("value-negative" if pnl_today < 0 else "value-neutral")
    balance_class = "value-positive" if balance >= START_BALANCE else "value-negative"

    dashboard_topbar(bot_status, asset_name)

    st.markdown(
        """
        <div class="dashboard-title">
            <h2>🤖 Bot Dashboard</h2>
            <p>Overzicht van bot prestaties, markt en signalen.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
    kpi_1.markdown(
        metric_card(
            "Gesimuleerd saldo",
            format_eur(balance),
            "Startsaldo: EUR 10.000",
            balance_class,
            icon="💳",
            visual=SPARK_UP if balance >= START_BALANCE else SPARK_DOWN,
        ),
        unsafe_allow_html=True,
    )
    kpi_2.markdown(
        metric_card(
            "P&L vandaag",
            format_eur(pnl_today),
            "Gesloten trades vandaag",
            pnl_class,
            icon="↗",
            visual=SPARK_UP if pnl_today >= 0 else SPARK_DOWN,
        ),
        unsafe_allow_html=True,
    )
    kpi_3.markdown(
        metric_card(
            "Winrate",
            f"{winrate:.1f}%",
            f"{wins} van {len(auto_closed)} bot-trades",
            icon="◎",
            visual=DONUT_VISUAL,
        ),
        unsafe_allow_html=True,
    )
    kpi_4.markdown(
        metric_card(
            "Open trades",
            str(len(auto_open)),
            "Actieve bot-posities",
            icon="▣",
            visual=BARS_VISUAL,
        ),
        unsafe_allow_html=True,
    )

    _render_health_panel()
    _render_llm_advice_panel()

    try:
        df_ind = add_all_indicators(df)
        range_limits = {"1D": 288, "7D": 2016, "1M": 8640, "3M": 25920, "1Y": 105120}
        recent = df_ind.tail(min(len(df_ind), range_limits.get(chart_range, 288)))
        last_close = float(recent["close"].iloc[-1])
        last_support = last_valid(recent["support"])
        last_resistance = last_valid(recent["resistance"])
        last_update = pd.to_datetime(recent["timestamp"].iloc[-1]).strftime("%H:%M:%S")
        indicator_href = chart_link(indicators=not show_chart_indicators)

        st.markdown(
            '<div class="market-card">'
            '<div class="market-card-head">'
            '<h3>Market Intelligence</h3><span class="market-info-dot">i</span>'
            '</div>'
            '<div class="chart-toolbar">'
            '<div class="chart-toolbar-left">'
            f'<span class="chart-asset-label">{escape(asset_name)} · {escape(timeframe_short(st.session_state.timeframe_label))} candles</span>'
            f'<a class="chart-indicator-btn {"active" if show_chart_indicators else ""}" href="{indicator_href}" target="_self"><b>ƒx</b> Indicatoren</a>'
            '</div>'
            '<div class="chart-toolbar-right"></div>'
            '</div>'
            '<div class="market-chart-body">',
            unsafe_allow_html=True,
        )

        fig_market = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.035,
            row_heights=[0.76, 0.24],
        )
        fig_market.add_trace(go.Candlestick(
            x=recent["timestamp"],
            open=recent["open"],
            high=recent["high"],
            low=recent["low"],
            close=recent["close"],
            name="Prijs",
            increasing_line_color="#10B981",
            increasing_fillcolor="rgba(16,185,129,0.78)",
            decreasing_line_color="#EF4444",
            decreasing_fillcolor="rgba(239,68,68,0.72)",
        ), row=1, col=1)
        if show_chart_indicators:
            fig_market.add_trace(go.Scatter(
                x=recent["timestamp"], y=recent["sma20"],
                line=dict(color="#2DBE7E", width=1.45),
                name="SMA 20",
            ), row=1, col=1)
            fig_market.add_trace(go.Scatter(
                x=recent["timestamp"], y=recent["sma50"],
                line=dict(color="#6EA1FF", width=1.45),
                name="SMA 50",
            ), row=1, col=1)
            fig_market.add_trace(go.Scatter(
                x=recent["timestamp"], y=recent["resistance"],
                line=dict(color="rgba(239,68,68,0.75)", width=1.2, dash="dash"),
                name="Support / Resistance",
                legendgroup="sr",
            ), row=1, col=1)
            fig_market.add_trace(go.Scatter(
                x=recent["timestamp"], y=recent["support"],
                line=dict(color="rgba(16,185,129,0.75)", width=1.2, dash="dash"),
                name="Support / Resistance",
                legendgroup="sr",
                showlegend=False,
            ), row=1, col=1)
        vol_colors = ["#6ED7B7" if c >= o else "#F99A9D" for o, c in zip(recent["open"], recent["close"])]
        fig_market.add_trace(go.Bar(
            x=recent["timestamp"],
            y=recent["volume"],
            marker_color=vol_colors,
            opacity=0.82,
            name="Volume",
        ), row=2, col=1)

        if show_chart_indicators and last_resistance is not None:
            fig_market.add_hline(
                y=last_resistance,
                line_dash="dash",
                line_color="rgba(239,68,68,0.72)",
                line_width=1,
                row=1,
                col=1,
            )
            fig_market.add_annotation(
                xref="paper", yref="y",
                x=1.005, y=last_resistance,
                text=f"Resistance {compact_price(last_resistance)}",
                showarrow=False,
                xanchor="left",
                bgcolor="#FFF1F2",
                bordercolor="#FDA4AF",
                borderwidth=1,
                borderpad=5,
                font=dict(size=10, color="#DC2626"),
            )
        if show_chart_indicators and last_support is not None:
            fig_market.add_hline(
                y=last_support,
                line_dash="dash",
                line_color="rgba(16,185,129,0.72)",
                line_width=1,
                row=1,
                col=1,
            )
            fig_market.add_annotation(
                xref="paper", yref="y",
                x=1.005, y=last_support,
                text=f"Support {compact_price(last_support)}",
                showarrow=False,
                xanchor="left",
                bgcolor="#ECFDF5",
                bordercolor="#86EFAC",
                borderwidth=1,
                borderpad=5,
                font=dict(size=10, color="#047857"),
            )
        fig_market.add_annotation(
            xref="paper", yref="y",
            x=1.005, y=last_close,
            text=compact_price(last_close),
            showarrow=False,
            xanchor="left",
            bgcolor="#10B981",
            bordercolor="#10B981",
            borderwidth=1,
            borderpad=5,
            font=dict(size=10, color="#FFFFFF"),
        )

        polish_chart(fig_market, height=470, top=8, showlegend=True)
        fig_market.update_layout(
            xaxis_rangeslider_visible=False,
            margin=dict(l=6, r=104, t=8, b=58),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.16,
                xanchor="left",
                x=0,
                font=dict(size=11, color="#475569"),
                traceorder="normal",
            ),
        )
        fig_market.update_xaxes(
            showgrid=True,
            gridcolor="#EDF2F7",
            zeroline=False,
            rangeslider_visible=False,
        )
        fig_market.update_yaxes(
            title_text="",
            side="right",
            showgrid=True,
            gridcolor="#EDF2F7",
            zeroline=False,
            tickfont=dict(size=11, color="#64748B"),
            row=1,
            col=1,
        )
        fig_market.update_yaxes(
            title_text="",
            side="right",
            showgrid=True,
            gridcolor="#EDF2F7",
            zeroline=False,
            tickfont=dict(size=11, color="#64748B"),
            row=2,
            col=1,
        )
        st.plotly_chart(
            fig_market,
            use_container_width=True,
            config={
                "displaylogo": False,
                "responsive": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            },
        )

        st.markdown(
            '</div>'
            '<div class="market-card-foot">'
            f'<span class="market-live-status"><span class="market-live-dot"></span>Laatste update: {escape(last_update)}</span>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.warning(f"Market Intelligence kon niet worden geladen: {e}")

    if auto_closed:
        section_title("Risico-dashboard per strategie")
        risk_df = build_strategy_risk_dashboard(auto_closed)
        if risk_df.empty:
            st.info("Nog niet genoeg gesloten auto-trades voor strategie-risico.")
        else:
            st.dataframe(
                style_cells(
                    risk_df,
                    lambda v: "color: #10B981; font-weight: 700" if isinstance(v, (int, float)) and v > 0
                    else ("color: #EF4444; font-weight: 700" if isinstance(v, (int, float)) and v < 0 else ""),
                    ["Totaal €", "Gem. winst", "Gem. verlies", "Verwachting/trade", "Max drawdown"],
                ).format({
                    "Winrate": "{:.0f}%",
                    "Totaal €": "€{:+.2f}",
                    "Gem. winst": "€{:.2f}",
                    "Gem. verlies": "€{:.2f}",
                    "Verwachting/trade": "€{:+.2f}",
                    "Max drawdown": "€{:.2f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

    st.markdown('<div class="section-title">Open trades</div>', unsafe_allow_html=True)
    if not open_trades:
        st.info("Geen open trades op dit moment.")
    else:
        rows = []
        for trade in open_trades:
            current_price = get_latest_price(trade["asset"]) if "-" in trade["asset"] else None
            if current_price is None and trade["asset"] == asset_name:
                current_price = float(df["close"].iloc[-1])
            if current_price is None:
                current_price = float(trade["entry_price"])

            if trade["direction"] == "long":
                unrealized_pct = (current_price - trade["entry_price"]) / trade["entry_price"] * 100
            else:
                unrealized_pct = (trade["entry_price"] - current_price) / trade["entry_price"] * 100
            unrealized_eur = unrealized_pct / 100 * (trade.get("capital") or 0)

            rows.append({
                "Asset": trade["asset"],
                "Richting": trade["direction"].upper(),
                "Entry": round(trade["entry_price"], 4),
                "Huidige prijs": round(current_price, 4),
                "Ongerealiseerd %": round(unrealized_pct, 2),
                "Ongerealiseerd €": round(unrealized_eur, 2),
            })

        open_df = pd.DataFrame(rows)
        st.dataframe(
            style_cells(
                open_df,
                lambda v: "color: #10B981" if isinstance(v, (int, float)) and v > 0
                else ("color: #EF4444" if isinstance(v, (int, float)) and v < 0 else ""),
                ["Ongerealiseerd %", "Ongerealiseerd €"],
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown('<div class="section-title">Recente gesloten trades</div>', unsafe_allow_html=True)
    if not closed_trades:
        st.info("Nog geen gesloten trades.")
        st.stop()
    else:
        render_recent_closed_cards(closed_trades[:10])
        st.stop()
        recent_closed = pd.DataFrame(closed_trades[:10])
        columns = [
            "asset", "direction", "entry_price", "exit_price",
            "result_pct", "result_euro", "exit_reason", "closed_at",
        ]
        recent_closed = recent_closed[[c for c in columns if c in recent_closed.columns]].rename(columns={
            "asset": "Asset",
            "direction": "Richting",
            "entry_price": "Entry",
            "exit_price": "Exit",
            "result_pct": "Resultaat %",
            "result_euro": "Resultaat €",
            "exit_reason": "Reden",
            "closed_at": "Gesloten",
        })
        if "Gesloten" in recent_closed.columns:
            recent_closed["Gesloten"] = recent_closed["Gesloten"].astype(str).str[:16].str.replace("T", " ")
        if "Richting" in recent_closed.columns:
            recent_closed["Richting"] = recent_closed["Richting"].astype(str).str.upper()

        st.dataframe(
            style_cells(
                recent_closed,
                lambda v: "color: #10B981" if isinstance(v, (int, float)) and v > 0
                else ("color: #EF4444" if isinstance(v, (int, float)) and v < 0 else ""),
                [c for c in ["Resultaat %", "Resultaat €"] if c in recent_closed.columns],
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown('<div class="section-title">Recente bot-activiteit</div>', unsafe_allow_html=True)
    if bot_activity:
        log_lines = []
        for item in bot_activity[:8]:
            stamp = item["_dt"].strftime("%H:%M:%S") if item.get("_dt") else "--:--:--"
            log_lines.append(escape(f"{stamp}  {item['message']}"))
        st.markdown(
            '<div class="log-block">' + "<br>".join(log_lines) + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Nog geen bot-activiteit gevonden in database of bot.log.")

    st.stop()

    st.info(
        "Oefen met nep-trades. Je riskeert **geen echt geld**. "
        "Schrijf altijd op waarom je een trade opent — dat is de kern van leren."
    )

    subtab_nieuw, subtab_open, subtab_history = st.tabs([
        "➕ Nieuwe trade",
        "📋 Open trades",
        "📜 Geschiedenis",
    ])

    # ── Nieuwe handmatige trade ──────────────────────────────────────────────
    with subtab_nieuw:
        st.subheader("Handmatige paper trade openen")

        huidig_signaal = None
        try:
            huidig_signaal = generate_signal(df)
        except Exception:
            pass

        if huidig_signaal and huidig_signaal["type"] == "setup":
            st.success(
                f"✅ Er is momenteel een setup: "
                f"{huidig_signaal['richting'].upper()} @ {huidig_signaal['entry']}"
            )
            st.caption("Gebruik de levels hieronder als startpunt.")
            default_entry = huidig_signaal["entry"]
            default_sl    = huidig_signaal["stop_loss"]
            default_tp    = huidig_signaal["take_profit"]
        else:
            default_entry = float(df["close"].iloc[-1])
            default_sl    = round(default_entry * 0.97, 4)
            default_tp    = round(default_entry * 1.06, 4)

        with st.form("nieuwe_paper_trade", clear_on_submit=True):
            col_a, col_b = st.columns(2)

            with col_a:
                pt_asset    = st.text_input("Asset", value=asset_name)
                pt_direction = st.selectbox("Richting", ["long", "short"])
                pt_entry    = st.number_input(
                    "Entry prijs", value=float(default_entry), format="%.4f", step=0.01
                )
                pt_kapitaal = st.number_input(
                    "Gesimuleerd bedrag (€)", min_value=10.0, value=500.0, step=50.0
                )

            with col_b:
                pt_sl = st.number_input(
                    "Stop-Loss prijs", value=float(default_sl), format="%.4f", step=0.01
                )
                pt_tp = st.number_input(
                    "Take-Profit prijs", value=float(default_tp), format="%.4f", step=0.01
                )

            pt_reden = st.text_area(
                "Waarom open je deze trade?",
                placeholder=(
                    "Beschrijf wat je ziet. Bijvoorbeeld:\n"
                    "- Uptrend op SMA20 en SMA50\n"
                    "- Prijs pullback naar support\n"
                    "- Volume boven gemiddelde"
                ),
                height=120,
            )

            # Live R/R preview
            if pt_direction == "long":
                risk   = pt_entry - pt_sl
                reward = pt_tp - pt_entry
            else:
                risk   = pt_sl - pt_entry
                reward = pt_entry - pt_tp

            rr_preview = reward / risk if risk > 0 else 0
            kleur_rr   = "🟢" if rr_preview >= 2.0 else ("🟡" if rr_preview >= 1.5 else "🔴")
            st.info(
                f"{kleur_rr} Risk/Reward: 1:{rr_preview:.2f}  |  "
                f"Risico: {risk:.4f}  |  Reward: {reward:.4f}"
            )

            verzend = st.form_submit_button("📝 Trade openen", type="primary")

            if verzend:
                res = open_manual_trade(
                    asset       = pt_asset,
                    direction   = pt_direction,
                    entry_price = pt_entry,
                    stop_loss   = pt_sl,
                    take_profit = pt_tp,
                    reden       = pt_reden,
                    kapitaal    = pt_kapitaal,
                )
                if res["success"]:
                    st.success(res["boodschap"])
                else:
                    st.error(res["boodschap"])

    # ── Open trades ──────────────────────────────────────────────────────────
    with subtab_open:
        st.subheader("Open paper trades")

        if st.button("🔄 Vernieuwen", key="refresh_open"):
            st.rerun()

        open_trades = get_open_trades()

        if not open_trades:
            st.info("Geen open trades. Open er een via '➕ Nieuwe trade'.")
        else:
            for trade in open_trades:
                huidige_prijs = float(df["close"].iloc[-1])
                if trade["direction"] == "long":
                    ongerealiseerd = (huidige_prijs - trade["entry_price"]) / trade["entry_price"] * 100
                else:
                    ongerealiseerd = (trade["entry_price"] - huidige_prijs) / trade["entry_price"] * 100
                kleur_ong = "🟢" if ongerealiseerd >= 0 else "🔴"

                with st.expander(
                    f"Trade #{trade['id']} | {trade['asset']} "
                    f"{trade['direction'].upper()} @ {trade['entry_price']:.4f} "
                    f"| {kleur_ong} {ongerealiseerd:+.2f}% (ongerealiseerd)"
                ):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Entry",       f"{trade['entry_price']:.4f}")
                    c2.metric("Stop-Loss",   f"{trade['stop_loss']:.4f}")
                    c3.metric("Take-Profit", f"{trade['take_profit']:.4f}")

                    risk_euro = trade["capital"] * abs(
                        trade["entry_price"] - trade["stop_loss"]
                    ) / trade["entry_price"]
                    st.caption(
                        f"Gesimuleerd bedrag: €{trade['capital']:.0f}  |  "
                        f"Max. risico: ~€{risk_euro:.2f}  |  "
                        f"Geopend: {trade['created_at'][:10]}"
                    )
                    if trade.get("reason"):
                        st.caption(f"📝 Reden: {trade['reason']}")

                    st.divider()

                    with st.form(f"sluit_form_{trade['id']}"):
                        c_ep, c_er = st.columns(2)
                        exit_price  = c_ep.number_input(
                            "Exit prijs", value=huidige_prijs,
                            format="%.4f", key=f"ep_{trade['id']}"
                        )
                        exit_reason = c_er.selectbox(
                            "Sluit-reden",
                            ["manual", "sl", "tp"],
                            format_func=lambda x: {
                                "manual": "Handmatig gesloten",
                                "sl": "Stop-loss geraakt",
                                "tp": "Take-profit geraakt",
                            }[x],
                            key=f"er_{trade['id']}"
                        )
                        fout = st.text_area(
                            "Wat ging er fout of wat leerde je?",
                            placeholder="Schrijf hier je reflectie na het sluiten van de trade.",
                            key=f"fout_{trade['id']}",
                        )

                        if st.form_submit_button("❎ Trade sluiten"):
                            r = sluit_paper_trade(
                                trade["id"], exit_price, exit_reason, fout
                            )
                            if r["success"]:
                                st.success(r["boodschap"])
                                st.rerun()
                            else:
                                st.error(r["boodschap"])

    # ── Handelsgeschiedenis ───────────────────────────────────────────────────
    with subtab_history:
        st.subheader("Gesloten trades")

        gesloten = get_closed_trades()

        if not gesloten:
            st.info("Nog geen gesloten trades.")
        else:
            df_g = pd.DataFrame(gesloten)

            kolommen = [
                "id", "asset", "direction",
                "entry_price", "exit_price",
                "result_pct", "result_euro",
                "exit_reason", "created_at",
            ]
            beschikbaar = [c for c in kolommen if c in df_g.columns]
            df_show = df_g[beschikbaar].rename(columns={
                "id":          "Trade#",
                "asset":       "Asset",
                "direction":   "Richting",
                "entry_price": "Entry",
                "exit_price":  "Exit",
                "result_pct":  "Resultaat %",
                "result_euro": "Resultaat €",
                "exit_reason": "Reden",
                "created_at":  "Datum",
            })
            if "Datum" in df_show.columns:
                df_show["Datum"] = df_show["Datum"].str[:10]

            st.dataframe(
                style_cells(
                    df_show,
                    lambda v: "color: #10B981" if isinstance(v, (int, float)) and v > 0
                    else ("color: #EF4444" if isinstance(v, (int, float)) and v < 0 else ""),
                    [c for c in ["Resultaat %", "Resultaat €"] if c in df_show.columns],
                ),
                use_container_width=True,
            )


# ════════════════════════════════════════════════════════════════════════════
# LEERGEHEUGEN (voorheen Evaluatie)
# ════════════════════════════════════════════════════════════════════════════
elif active == "leergeheugen":
    page_header("🧠 Leergeheugen & Evaluatie", "Gesloten trades, patronen en verbeterpunten.")
    st.info(
        "Het systeem analyseert je trades en stelt verbeteringen **voor**. "
        "**Jij** beslist wat je aanpast. Regels worden nooit automatisch gewijzigd."
    )

    eval_data = evalueer_trades()

    if eval_data["totaal"] == 0:
        st.warning(
            "Nog geen gesloten trades om te analyseren. "
            "Sluit eerst een paar paper trades via het Bot Dashboard."
        )
    else:
        # KPI's
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Totaal trades",  eval_data["totaal"])
        c2.metric("Winrate",        f"{eval_data['winrate']}%")
        c3.metric("Gem. winst",     f"+{eval_data['gem_winst']:.2f}%")
        c4.metric("Gem. verlies",   f"{eval_data['gem_verlies']:.2f}%",
                   delta_color="inverse")

        st.divider()

        # Suggesties
        st.subheader("💡 Suggesties & Verbeterpunten")

        icon_map = {
            "waarschuwing": ("⚠️", "warning"),
            "tip":          ("💡", "info"),
            "goed":         ("✅", "success"),
            "info":         ("ℹ️", "info"),
        }

        for s in eval_data["suggesties"]:
            icon, fn_name = icon_map.get(s["type"], ("•", "info"))
            getattr(st, fn_name)(f"{icon} {s['tekst']}")

        st.divider()

        # Trade-voor-trade
        st.subheader("📋 Trade-voor-trade analyse")

        for t in eval_data["trades"]:
            winstgevend = t["resultaat"] and t["resultaat"] > 0
            kleur    = "🟢" if winstgevend else "🔴"
            res_pct  = f"{t['resultaat']:+.2f}%" if t["resultaat"] is not None else "?"
            res_eur  = t.get("result_euro") or 0
            res_eur_str = f"€{res_eur:+,.2f}"
            strategie = t.get("strategy") or "-"

            with st.expander(
                f"{kleur} Trade #{t['id']} | {t['asset']} "
                f"{t['richting'].upper()} | {res_pct} | {res_eur_str}"
            ):
                # Rij 1: kernresultaten
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Resultaat %", res_pct)
                c2.metric("Resultaat €", res_eur_str)
                c3.metric("Inzet", f"€{t.get('capital', 0):,.0f}")
                c4.metric("Duur", t.get("duur", "-"))

                # Rij 2: prijsdetails
                c5, c6, c7, c8 = st.columns(4)
                entry = t.get("entry_price")
                exit_ = t.get("exit_price")
                sl    = t.get("stop_loss")
                tp    = t.get("take_profit")
                c5.markdown(f"**Entry:** `{entry:,.4f}`"    if entry else "**Entry:** -")
                c6.markdown(f"**Exit:** `{exit_:,.4f}`"     if exit_ else "**Exit:** -")
                c7.markdown(f"**Stop-loss:** `{sl:,.4f}`"   if sl    else "**Stop-loss:** -")
                c8.markdown(f"**Take-profit:** `{tp:,.4f}`" if tp    else "**Take-profit:** -")

                # Rij 3: meta
                c9, c10, c11 = st.columns(3)
                c9.markdown(f"**Strategie:** `{strategie}`")
                c10.markdown(f"**Exit reden:** `{t['exit_reden']}`")
                geopend_str = (t.get("created_at") or "")[:16].replace("T", " ")
                c11.markdown(f"**Geopend:** `{geopend_str}`")

                st.divider()
                if t.get("fout_analyse"):
                    st.markdown("**Reflectie:**")
                    st.info(t["fout_analyse"])
                else:
                    st.caption(
                        "Je hebt nog geen reflectie ingevuld voor deze trade. "
                        "Doe dat via 'Open trades' → trade sluiten."
                    )

        st.divider()

        # Weekoverzicht
        st.subheader("📆 Weekoverzicht")
        overzicht = weekoverzicht()
        st.code(overzicht, language=None)

        if st.download_button(
            "⬇️ Download weekoverzicht (txt)",
            data=overzicht,
            file_name="weekoverzicht.txt",
            mime="text/plain",
        ):
            pass


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "📊 TradeAI Coach v1.0  |  Geen financieel advies  |  "
    "Paper trading only  |  Geen echte orders  |  "
    "Gebouwd met Python & Streamlit"
)
