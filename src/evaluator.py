"""
evaluator.py — Analyseer gesloten paper trades en identificeer patronen.

Het systeem stelt verbeteringen VOOR maar past regels NOOIT automatisch aan.
Jij bent altijd degene die beslist. Het systeem leert je te denken als een trader.
"""
import pandas as pd
from src.database import get_paper_trades


def evalueer_trades() -> dict:
    """
    Analyseer alle gesloten paper trades.

    Geeft terug:
      - Statistieken (winrate, gemiddelde winst/verlies, totaal P&L)
      - Suggesties voor verbetering (op basis van patronen)
      - Trade-voor-trade uitleg
    """
    trades = get_paper_trades(status="closed")

    if not trades:
        return {
            "trades":      [],
            "totaal":      0,
            "winrate":     0.0,
            "gem_winst":   0.0,
            "gem_verlies": 0.0,
            "totaal_pnl":  0.0,
            "suggesties":  [],
            "samenvatting": "Nog geen gesloten trades om te evalueren.",
        }

    df = pd.DataFrame(trades)

    totaal        = len(df)
    winst_trades  = df[df["result_pct"] > 0]
    verlies_trades = df[df["result_pct"] < 0]

    winrate    = len(winst_trades) / totaal * 100 if totaal > 0 else 0.0
    gem_winst  = float(winst_trades["result_pct"].mean())  if len(winst_trades)   > 0 else 0.0
    gem_verlies = float(verlies_trades["result_pct"].mean()) if len(verlies_trades) > 0 else 0.0
    totaal_pnl = float(df["result_pct"].sum())

    # ── Suggesties op basis van patronen ────────────────────────────
    suggesties = []

    sl_trades = df[df["exit_reason"] == "sl"]
    tp_trades = df[df["exit_reason"] == "tp"]

    if totaal >= 5 and len(sl_trades) > len(tp_trades) * 1.5:
        suggesties.append({
            "type":  "waarschuwing",
            "tekst": (
                f"Je stop-loss wordt vaker geraakt ({len(sl_trades)}×) dan je take-profit "
                f"({len(tp_trades)}×). Overweeg: betere entry-timing afwachten, "
                f"of een iets ruimere stop-loss plaatsen op basis van ATR."
            ),
        })

    if totaal >= 5 and winrate < 35:
        suggesties.append({
            "type":  "waarschuwing",
            "tekst": (
                f"Winrate is laag: {winrate:.0f}%. "
                f"Kijk terug naar je trades: waren de setups echt duidelijk? "
                f"Wacht vaker op hogere kwaliteit setups."
            ),
        })

    if totaal >= 3 and abs(gem_verlies) > abs(gem_winst) * 1.2:
        suggesties.append({
            "type":  "tip",
            "tekst": (
                f"Je gemiddelde verlies ({gem_verlies:.1f}%) is groter dan je gemiddelde "
                f"winst ({gem_winst:.1f}%). "
                f"Zorg dat je risk/reward altijd minimaal 1:2 is."
            ),
        })

    if totaal >= 5 and winrate > 60 and gem_winst < abs(gem_verlies):
        suggesties.append({
            "type":  "tip",
            "tekst": (
                f"Hoge winrate ({winrate:.0f}%), maar je winsten zijn kleiner dan je verliezen. "
                f"Overweeg je take-profit verder weg te plaatsen."
            ),
        })

    if totaal >= 5 and winrate > 55 and totaal_pnl > 0:
        suggesties.append({
            "type":  "goed",
            "tekst": (
                f"Goede prestaties! Winrate {winrate:.0f}% en positief P&L ({totaal_pnl:.1f}%). "
                f"Blijf je strategie consistent toepassen."
            ),
        })

    # Fout-patroon: veel trades met lege fout-analyse
    geen_analyse = df[df["mistake_analysis"].isna() | (df["mistake_analysis"] == "")]
    if len(geen_analyse) > totaal * 0.5 and totaal >= 3:
        suggesties.append({
            "type":  "tip",
            "tekst": (
                "Je vult de fout-analyse weinig in. "
                "Schrijf na elke trade op wat je leerde — dit is de snelste manier om beter te worden."
            ),
        })

    if not suggesties:
        suggesties.append({
            "type":  "info",
            "tekst": (
                f"Je hebt {totaal} trades gedaan. "
                f"Doe er ten minste 10 om specifiekere feedback te krijgen."
            ),
        })

    # ── Per-trade overzicht ──────────────────────────────────────────
    trade_lijst = []
    for _, t in df.iterrows():
        trade_lijst.append({
            "id":           t["id"],
            "asset":        t["asset"],
            "richting":     t["direction"],
            "resultaat":    t["result_pct"],
            "result_euro":  t.get("result_euro", 0),
            "exit_reden":   t.get("exit_reason", "onbekend"),
            "fout_analyse": t.get("mistake_analysis", ""),
        })

    samenvatting = (
        f"Totaal trades:  {totaal}\n"
        f"Winrate:        {winrate:.1f}%\n"
        f"Totaal P&L:     {totaal_pnl:+.2f}%\n"
        f"Gem. winst:    +{gem_winst:.2f}%\n"
        f"Gem. verlies:   {gem_verlies:.2f}%"
    )

    return {
        "trades":      trade_lijst,
        "totaal":      totaal,
        "winrate":     round(winrate, 1),
        "gem_winst":   round(gem_winst, 2),
        "gem_verlies": round(gem_verlies, 2),
        "totaal_pnl":  round(totaal_pnl, 2),
        "suggesties":  suggesties,
        "samenvatting": samenvatting,
    }


def weekoverzicht() -> str:
    """
    Geef een kort tekstoverzicht van je trades en lessen.
    Gebruik dit elke week om te reflecteren.
    """
    data = evalueer_trades()

    if data["totaal"] == 0:
        return (
            "📊 WEEKOVERZICHT\n"
            "━" * 32 + "\n"
            "Nog geen trades om te evalueren.\n"
            "Open je eerste paper trade via het dashboard!"
        )

    icon_map = {"waarschuwing": "⚠️", "tip": "💡", "goed": "✅", "info": "ℹ️"}

    regels = [
        "📊 WEEKOVERZICHT",
        "━" * 32,
        data["samenvatting"],
        "",
        "📝 VERBETERPUNTEN:",
    ]

    for s in data["suggesties"]:
        icon = icon_map.get(s["type"], "•")
        regels.append(f"{icon} {s['tekst']}")

    regels.extend([
        "",
        "━" * 32,
        "⚠️  Vergeet niet:",
        "  • Dit is paper trading — oefen, analyseer, leer.",
        "  • Pas strategie-regels NOOIT automatisch aan.",
        "  • Denk eerst na, beslis dan zelf.",
    ])

    return "\n".join(regels)
