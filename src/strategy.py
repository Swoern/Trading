"""
strategy.py — Strategie-regels en marktanalyse.

De strategie is bewust eenvoudig gehouden voor beginners:
  - Alleen LONG kijken bij een bevestigde uptrend
  - Geen trade midden in een range
  - Minimale risk/reward van 1:2
  - Stop-loss en take-profit altijd bepaald via ATR
  - Geen trade bij onduidelijke markt of laag volume

⚠️  DIT SYSTEEM GEEFT GEEN FINANCIEEL ADVIES.
    Resultaten uit het verleden geven geen garantie voor de toekomst.
    Paper trading is bedoeld om te oefenen, niet om echt geld mee te verdienen.
"""
import pandas as pd
import numpy as np
from src.indicators import add_all_indicators

WAARSCHUWING = (
    "⚠️  WAARSCHUWING: Dit systeem geeft GEEN financieel advies. "
    "Resultaten uit het verleden geven geen garantie voor de toekomst. "
    "Paper trading is uitsluitend bedoeld voor leren en oefenen."
)


def analyze_market(df: pd.DataFrame) -> dict:
    """
    Analyseer de laatste candle en geef een begrijpelijke uitleg
    over trend, volatiliteit, volume, support/resistance en RSI.

    Geeft een dictionary terug met tekst-uitleg per onderdeel.
    """
    df = add_all_indicators(df)

    # Gebruik de laatste gesloten candle (niet de huidige open candle)
    last = df.iloc[-1]

    result = {}

    # ── Trend ──────────────────────────────────────────────────────
    if pd.isna(last["sma20"]) or pd.isna(last["sma50"]):
        trend = "onduidelijk"
        trend_tekst = "Niet genoeg data om de trend te bepalen (min. 50 candles nodig)."
    elif last["close"] > last["sma20"] and last["sma20"] > last["sma50"]:
        trend = "uptrend"
        trend_tekst = (
            f"De markt is in een UPTREND. "
            f"Prijs ({last['close']:.2f}) staat boven SMA20 ({last['sma20']:.2f}) "
            f"en SMA20 staat boven SMA50 ({last['sma50']:.2f}). "
            f"Dit is het basisvereiste voor een long-setup."
        )
    elif last["close"] < last["sma20"] and last["sma20"] < last["sma50"]:
        trend = "downtrend"
        trend_tekst = (
            f"De markt is in een DOWNTREND. "
            f"Prijs ({last['close']:.2f}) staat onder SMA20 ({last['sma20']:.2f}) "
            f"en SMA20 staat onder SMA50 ({last['sma50']:.2f}). "
            f"In deze versie worden short-setups nog niet ondersteund."
        )
    else:
        trend = "onduidelijk"
        trend_tekst = (
            "De trend is ONDUIDELIJK. De moving averages kruisen elkaar of "
            "bewegen zijwaarts. Beter wachten tot er een duidelijke richting is."
        )

    result["trend"]      = trend
    result["trend_tekst"] = trend_tekst

    # ── Volatiliteit ───────────────────────────────────────────────
    atr_pct = last.get("atr_pct", 0) if not pd.isna(last.get("atr_pct", np.nan)) else 0

    if atr_pct < 1.5:
        volatiliteit = "laag"
        vol_tekst = f"Volatiliteit is LAAG ({atr_pct:.1f}%). De markt beweegt rustig."
    elif atr_pct < 4.0:
        volatiliteit = "normaal"
        vol_tekst = f"Volatiliteit is NORMAAL ({atr_pct:.1f}%). Gewone marktbeweging."
    else:
        volatiliteit = "hoog"
        vol_tekst = (
            f"Volatiliteit is HOOG ({atr_pct:.1f}%). De markt beweegt sterk. "
            f"Wees voorzichtiger met positiegrootte."
        )

    result["volatiliteit"]      = volatiliteit
    result["volatiliteit_tekst"] = vol_tekst

    # ── Volume ─────────────────────────────────────────────────────
    vol_ma = last.get("vol_ma20", 0)
    huidige_vol = last["volume"]

    if pd.isna(vol_ma) or vol_ma == 0:
        volume_status = "onbekend"
        vol_volume = "Volume-gemiddelde kon niet worden berekend."
    elif huidige_vol > vol_ma * 1.5:
        volume_status = "hoog"
        vol_volume = "Volume is HOOG. Er is veel activiteit — dit bevestigt de beweging."
    elif huidige_vol >= vol_ma * 0.8:
        volume_status = "normaal"
        vol_volume = "Volume is NORMAAL."
    else:
        volume_status = "laag"
        vol_volume = (
            "Volume is LAAG. Er is weinig activiteit. "
            "Signalen zijn minder betrouwbaar bij laag volume."
        )

    result["volume_status"] = volume_status
    result["volume_tekst"]  = vol_volume

    # ── Support & Resistance ───────────────────────────────────────
    support    = last.get("support", np.nan)
    resistance = last.get("resistance", np.nan)
    prijs      = last["close"]

    if pd.isna(support) or pd.isna(resistance) or resistance == support:
        positie        = 0.5
        positie_tekst  = "Support/resistance kon niet worden berekend."
    else:
        range_size = resistance - support
        positie    = (prijs - support) / range_size

        if positie < 0.25:
            positie_tekst = (
                f"Prijs ({prijs:.2f}) zit dicht bij SUPPORT ({support:.2f}). "
                f"Dit is een mogelijke bodem — interessant voor een long-setup."
            )
        elif positie > 0.75:
            positie_tekst = (
                f"Prijs ({prijs:.2f}) zit dicht bij RESISTANCE ({resistance:.2f}). "
                f"Mogelijk plafond — wees voorzichtig met longs hier."
            )
        else:
            positie_tekst = (
                f"Prijs ({prijs:.2f}) zit IN HET MIDDEN van de range "
                f"({support:.2f} – {resistance:.2f}). "
                f"Geen goede plek voor een entry — wacht op de rand."
            )

    result["support"]           = support
    result["resistance"]        = resistance
    result["positie_in_range"]  = positie
    result["positie_tekst"]     = positie_tekst

    # ── RSI ────────────────────────────────────────────────────────
    rsi_val = last.get("rsi14", np.nan)

    if pd.isna(rsi_val):
        rsi_tekst = "RSI kon niet worden berekend."
    elif rsi_val > 70:
        rsi_tekst = f"RSI = {rsi_val:.0f} — OVERBOUGHT. Markt is mogelijk te duur op korte termijn."
    elif rsi_val < 30:
        rsi_tekst = f"RSI = {rsi_val:.0f} — OVERSOLD. Markt is mogelijk te goedkoop op korte termijn."
    else:
        rsi_tekst = f"RSI = {rsi_val:.0f} — Neutraal gebied. Geen extreme omstandigheden."

    result["rsi"]      = rsi_val
    result["rsi_tekst"] = rsi_tekst

    # ── Conclusie ─────────────────────────────────────────────────
    is_duidelijk = (
        trend == "uptrend"
        and volume_status not in ("laag", "onbekend")
        and (positie < 0.30 or positie > 0.70)
        and not pd.isna(rsi_val)
        and rsi_val < 75
    )

    if is_duidelijk:
        conclusie = (
            "De markt geeft een redelijk duidelijk beeld. "
            "Er is mogelijk een setup te vinden. Bekijk het signaal hieronder."
        )
    else:
        conclusie = (
            "De markt is momenteel ONDUIDELIJK of de omstandigheden zijn niet ideaal. "
            "Beter even wachten op een betere kans."
        )

    result["is_duidelijk"] = is_duidelijk
    result["conclusie"]    = conclusie
    result["last_close"]   = prijs
    result["last_atr"]     = last.get("atr14", np.nan)
    result["sma20"]        = last.get("sma20", np.nan)
    result["sma50"]        = last.get("sma50", np.nan)

    return result


def generate_signal(df: pd.DataFrame, min_rr: float = 2.0) -> dict:
    """
    Genereer een handelssignaal op basis van de strategie-regels.

    Regels (in volgorde):
      1. Alleen long bij bevestigde uptrend (close > SMA20 > SMA50)
      2. Geen trade midden in een range (positie 30–70%)
      3. Volume moet aanwezig zijn
      4. RSI niet boven de 75 (overbought)
      5. Risk/reward minimaal 1:min_rr
      6. Stop-loss = 1.5 × ATR onder entry
      7. Take-profit = min_rr × risk boven entry

    Geeft altijd een dict terug, nooit een exception.
    """
    signal = {
        "type":        "geen",
        "richting":    None,
        "entry":       None,
        "stop_loss":   None,
        "take_profit": None,
        "risk_reward": None,
        "reden":       [],
        "blokkades":   [],
        "uitleg":      "",
    }

    if len(df) < 55:
        signal["type"]   = "wacht"
        signal["uitleg"] = "Niet genoeg data (min. 55 candles nodig)."
        return signal

    try:
        analyse = analyze_market(df)
    except Exception as e:
        signal["type"]   = "fout"
        signal["uitleg"] = f"Analysefout: {e}"
        return signal

    df_ind = add_all_indicators(df)
    last   = df_ind.iloc[-1]

    blokkades = []

    # Regel 1: Trend
    if analyse["trend"] != "uptrend":
        blokkades.append(f"Trend is '{analyse['trend']}' — alleen long bij uptrend.")

    # Regel 2: Niet midden in range
    pos = analyse["positie_in_range"]
    if 0.30 <= pos <= 0.70:
        blokkades.append(
            f"Prijs zit midden in de range ({pos*100:.0f}%) — wacht op de rand."
        )

    # Regel 3: Volume
    if analyse["volume_status"] in ("laag", "onbekend"):
        blokkades.append("Volume is laag — signaal minder betrouwbaar.")

    # Regel 4: RSI niet overbought
    rsi_val = analyse["rsi"]
    if not pd.isna(rsi_val) and rsi_val > 75:
        blokkades.append(f"RSI ({rsi_val:.0f}) is boven 75 — markt is overbought.")

    # Regel 5: ATR beschikbaar?
    atr_val = analyse["last_atr"]
    if pd.isna(atr_val) or atr_val == 0:
        blokkades.append("ATR niet beschikbaar — stop-loss kan niet worden berekend.")

    if blokkades:
        signal["type"]      = "wacht"
        signal["blokkades"] = blokkades
        signal["uitleg"]    = (
            "Geen setup gevonden. Redenen:\n"
            + "\n".join(f"  • {b}" for b in blokkades)
        )
        return signal

    # Bereken entry, stop-loss en take-profit
    entry     = last["close"]
    stop_loss = entry - 1.5 * atr_val
    risk      = entry - stop_loss
    take_profit = entry + min_rr * risk
    rr        = (take_profit - entry) / risk if risk > 0 else 0

    if rr < min_rr:
        signal["type"]      = "wacht"
        signal["blokkades"] = [
            f"Risk/Reward ({rr:.1f}) is lager dan het minimum van 1:{min_rr}."
        ]
        signal["uitleg"] = f"Setup gevonden maar R/R is te laag ({rr:.1f}). Wacht op betere kans."
        return signal

    signal["type"]        = "setup"
    signal["richting"]    = "long"
    signal["entry"]       = round(entry, 4)
    signal["stop_loss"]   = round(stop_loss, 4)
    signal["take_profit"] = round(take_profit, 4)
    signal["risk_reward"] = round(rr, 2)
    signal["reden"] = [
        f"Uptrend bevestigd (close > SMA20 > SMA50)",
        f"Prijs nabij support of boven midrange ({pos*100:.0f}%)",
        f"Volume: {analyse['volume_status']}",
        f"RSI: {rsi_val:.0f} (niet overbought)",
        f"Risk/Reward: 1:{rr:.1f}",
    ]
    signal["uitleg"] = (
        f"Mogelijke LONG setup gevonden.\n"
        f"  Entry:       {entry:.4f}\n"
        f"  Stop-Loss:   {stop_loss:.4f}  "
        f"({(entry - stop_loss) / entry * 100:.1f}% risico)\n"
        f"  Take-Profit: {take_profit:.4f}  "
        f"(+{(take_profit - entry) / entry * 100:.1f}% winst)\n"
        f"  Risk/Reward: 1:{rr:.1f}\n\n"
        f"Dit is een VOORSTEL, geen advies. "
        f"Analyseer altijd zelf voordat je een paper trade opent."
    )

    return signal
