"""
app.py — TradeAI Coach Dashboard (Streamlit)

Start met:  streamlit run app.py

⚠️  Dit systeem geeft GEEN financieel advies.
    Geen echte orders, geen echt geld. Alleen leren via paper trading.
"""
import sys
from pathlib import Path

# Zorg dat de src/ map gevonden wordt
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_csv, generate_sample_data, get_available_csv_files
from src.indicators import add_all_indicators
from src.strategy import analyze_market, generate_signal
from src.backtest import run_backtest
from src.paper_trading import (
    open_manual_trade,
    sluit_paper_trade,
    get_open_trades,
    get_closed_trades,
)
from src.evaluator import evalueer_trades, weekoverzicht
from src.database import init_database

# ── Pagina-instelling ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TradeAI Coach",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_database()

# ── CSS: kleine stijlaanpassing ───────────────────────────────────────────────
st.markdown("""
<style>
    .metric-box {
        background: #1e2130;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 4px 0;
    }
    .warn-box {
        background: #2d1f00;
        border-left: 4px solid #ffa500;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)

# ── Veiligheidswaarschuwing (altijd zichtbaar) ────────────────────────────────
st.warning(
    "⚠️  **WAARSCHUWING** — Dit systeem geeft **GEEN financieel advies**. "
    "Resultaten uit het verleden geven geen garantie voor de toekomst. "
    "Paper trading is uitsluitend bedoeld om te **leren en te oefenen**. "
    "Automatisch handelen met echt geld is **niet mogelijk** in deze versie."
)

st.title("📊 TradeAI Coach")
st.caption("Paper trading & backtesting — Versie 1.0 | Geen echte orders mogelijk")

# ── Sidebar: Data laden ───────────────────────────────────────────────────────
with st.sidebar:
    st.title("📁 Dataset")

    csv_files = get_available_csv_files("data")

    if not csv_files:
        st.info("Geen CSV gevonden. Sample data wordt aangemaakt...")
        with st.spinner("Genereren..."):
            generate_sample_data("data/sample_btc.csv")
        csv_files = get_available_csv_files("data")

    selected_file = st.selectbox("Kies een dataset:", csv_files,
                                  format_func=lambda x: Path(x).stem)
    asset_name = Path(selected_file).stem.upper()

    @st.cache_data(show_spinner=False)
    def laad_data(pad: str) -> pd.DataFrame:
        return load_csv(pad)

    try:
        df = laad_data(selected_file)
        st.success(f"✅ {len(df):,} candles geladen")
        st.caption(
            f"Van: {df['timestamp'].min().date()}  \n"
            f"Tot: {df['timestamp'].max().date()}"
        )
    except Exception as e:
        st.error(f"Fout bij laden: {e}")
        st.stop()

    st.divider()

    if st.button("🔄 Nieuwe sample data aanmaken"):
        with st.spinner("Genereren..."):
            generate_sample_data("data/sample_btc.csv")
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption(
        "**Eigen CSV toevoegen:**  \n"
        "Zet je bestand in de `data/` map.  \n"
        "Kolommen: `timestamp, open, high, low, close, volume`"
    )

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_analyse, tab_backtest, tab_paper, tab_eval = st.tabs([
    "📊 Analyse",
    "🔄 Backtesting",
    "📝 Paper Trading",
    "📚 Evaluatie",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — MARKTANALYSE
# ════════════════════════════════════════════════════════════════════════════
with tab_analyse:
    st.header(f"Marktanalyse — {asset_name}")

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
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ))
        fig.add_trace(go.Scatter(
            x=recent["timestamp"], y=recent["sma20"],
            line=dict(color="#ff9800", width=1.8),
            name="SMA 20",
        ))
        fig.add_trace(go.Scatter(
            x=recent["timestamp"], y=recent["sma50"],
            line=dict(color="#42a5f5", width=1.8),
            name="SMA 50",
        ))
        # Support & resistance als achtergrondband
        fig.add_trace(go.Scatter(
            x=recent["timestamp"], y=recent["resistance"],
            line=dict(color="rgba(239,83,80,0.3)", width=1, dash="dot"),
            name="Resistance",
        ))
        fig.add_trace(go.Scatter(
            x=recent["timestamp"], y=recent["support"],
            line=dict(color="rgba(38,166,154,0.3)", width=1, dash="dot"),
            name="Support",
            fill="tonexty",
            fillcolor="rgba(100,100,100,0.05)",
        ))
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=460,
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Volume barchart
        vol_colors = [
            "#26a69a" if c >= o else "#ef5350"
            for o, c in zip(recent["open"], recent["close"])
        ]
        fig_vol = go.Figure(go.Bar(
            x=recent["timestamp"], y=recent["volume"],
            marker_color=vol_colors, name="Volume",
        ))
        fig_vol.add_trace(go.Scatter(
            x=recent["timestamp"], y=recent["vol_ma20"],
            line=dict(color="white", width=1),
            name="Gem. volume",
        ))
        fig_vol.update_layout(
            height=150, template="plotly_dark",
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig_vol, use_container_width=True)

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
    st.subheader("Huidig signaal")

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
# TAB 2 — BACKTESTING
# ════════════════════════════════════════════════════════════════════════════
with tab_backtest:
    st.header("Backtesting")
    st.info(
        "Test de strategie op historische data. "
        "Dit simuleert trades ZONDER echt geld. "
        "Goede backtestresultaten zijn **geen** garantie voor de toekomst."
    )

    col_params, col_leeg = st.columns([1, 2])
    with col_params:
        min_rr       = st.slider("Minimum Risk/Reward", 1.5, 4.0, 2.0, 0.5,
                                  help="Hoe hoger, hoe minder maar kwalitatievere trades.")
        start_kap    = st.number_input("Gesimuleerd startkapitaal (€)",
                                        1_000, 100_000, 10_000, 1_000)
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
                            color="#26a69a" if res["total_pnl"] >= 0 else "#ef5350",
                            width=2,
                        ),
                    ))
                    fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig_pnl.update_layout(
                        title="Cumulatief resultaat (%)",
                        height=300,
                        template="plotly_dark",
                        margin=dict(l=0, r=0, t=30, b=0),
                    )
                    st.plotly_chart(fig_pnl, use_container_width=True)

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
                        show.style.applymap(
                            lambda v: "color: #26a69a" if isinstance(v, float) and v > 0
                            else ("color: #ef5350" if isinstance(v, float) and v < 0 else ""),
                            subset=["Resultaat %"],
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
# TAB 3 — PAPER TRADING
# ════════════════════════════════════════════════════════════════════════════
with tab_paper:
    st.header("Paper Trading")
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
                df_show.style.applymap(
                    lambda v: "color: #26a69a" if isinstance(v, (int, float)) and v > 0
                    else ("color: #ef5350" if isinstance(v, (int, float)) and v < 0 else ""),
                    subset=[c for c in ["Resultaat %", "Resultaat €"] if c in df_show.columns],
                ),
                use_container_width=True,
            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — EVALUATIE
# ════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.header("Evaluatie & Leren")
    st.info(
        "Het systeem analyseert je trades en stelt verbeteringen **voor**. "
        "**Jij** beslist wat je aanpast. Regels worden nooit automatisch gewijzigd."
    )

    eval_data = evalueer_trades()

    if eval_data["totaal"] == 0:
        st.warning(
            "Nog geen gesloten trades om te analyseren. "
            "Sluit eerst een paar paper trades via het Paper Trading tabblad."
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
            kleur = "🟢" if t["resultaat"] and t["resultaat"] > 0 else "🔴"
            res_str = f"{t['resultaat']:+.2f}%" if t["resultaat"] is not None else "?"
            with st.expander(
                f"{kleur} Trade #{t['id']} | {t['asset']} "
                f"{t['richting'].upper()} | {res_str}"
            ):
                c_r, c_e = st.columns(2)
                c_r.markdown(f"**Resultaat:** `{res_str}`")
                c_e.markdown(f"**Exit reden:** `{t['exit_reden']}`")

                if t.get("fout_analyse"):
                    st.markdown("**Jouw reflectie:**")
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
