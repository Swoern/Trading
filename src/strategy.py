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

STRATEGIE_NAMEN: dict[str, str] = {
    "trend_long":     "Trend Long",
    "pullback":       "Pullback Long",
    "breakout":       "Breakout",
    "trend_short":    "Trend Short",
    "pullback_short": "Pullback Short",
    "breakdown":      "Breakdown",
    "macd_bull_div":  "MACD Bull Divergentie",
    "macd_bear_div":  "MACD Bear Divergentie",
    "range_long":     "Range Long",
    "range_short":    "Range Short",
}

# Alleen trend-strategieën worden geblokkeerd bij ADX < 18 (choppige markt).
# Divergentie-strategieën mogen ook in ranging markten actief zijn.
_TREND_STRATEGIEEN = frozenset({
    "trend_long", "pullback", "breakout",
    "trend_short", "pullback_short", "breakdown",
})

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

    # ── MACD ───────────────────────────────────────────────────────
    macd_line_val   = last.get("macd_line", np.nan)
    macd_signal_val = last.get("macd_signal", np.nan)
    macd_hist_val   = last.get("macd_hist", np.nan)

    if pd.isna(macd_line_val) or pd.isna(macd_signal_val):
        macd_tekst = "MACD kon niet worden berekend (te weinig data)."
    elif macd_line_val > macd_signal_val and macd_hist_val > 0:
        macd_tekst = (
            f"MACD is BULLISH. Lijn ({macd_line_val:.4f}) staat boven signaal "
            f"({macd_signal_val:.4f}). Momentum is opwaarts."
        )
    elif macd_line_val < macd_signal_val and macd_hist_val < 0:
        macd_tekst = (
            f"MACD is BEARISH. Lijn ({macd_line_val:.4f}) staat onder signaal "
            f"({macd_signal_val:.4f}). Momentum is neerwaarts."
        )
    else:
        macd_tekst = (
            f"MACD is NEUTRAAL of in een kruispunt "
            f"(lijn: {macd_line_val:.4f}, signaal: {macd_signal_val:.4f})."
        )

    bull_div = bool(last.get("macd_bull_div", False))
    bear_div = bool(last.get("macd_bear_div", False))
    if bull_div:
        macd_tekst += " ⚡ BULLISH DIVERGENTIE gedetecteerd — trend verzwakt neerwaarts."
    if bear_div:
        macd_tekst += " ⚡ BEARISH DIVERGENTIE gedetecteerd — trend verzwakt opwaarts."

    result["macd_line"]   = macd_line_val
    result["macd_signal"] = macd_signal_val
    result["macd_hist"]   = macd_hist_val
    result["macd_bull_div"] = bull_div
    result["macd_bear_div"] = bear_div
    result["macd_tekst"]  = macd_tekst

    # ── Bollinger Bands ────────────────────────────────────────────
    bb_pct_val   = last.get("bb_pct", np.nan)
    bb_width_val = last.get("bb_width", np.nan)
    bb_upper_val = last.get("bb_upper", np.nan)
    bb_lower_val = last.get("bb_lower", np.nan)

    if pd.isna(bb_pct_val):
        bb_tekst = "Bollinger Bands konden niet worden berekend."
    elif bb_pct_val > 1.0:
        bb_tekst = (
            f"Prijs ({prijs:.2f}) staat BOVEN de bovenste Bollinger Band ({bb_upper_val:.2f}). "
            f"Sterke uitbraak of overbought — wees voorzichtig."
        )
    elif bb_pct_val < 0.0:
        bb_tekst = (
            f"Prijs ({prijs:.2f}) staat ONDER de onderste Bollinger Band ({bb_lower_val:.2f}). "
            f"Sterke daling of oversold — mogelijke bodem."
        )
    elif bb_pct_val > 0.8:
        bb_tekst = f"Prijs ({prijs:.2f}) zit dicht bij de BOVENSTE band ({bb_upper_val:.2f}) — wees alert op weerstand."
    elif bb_pct_val < 0.2:
        bb_tekst = f"Prijs ({prijs:.2f}) zit dicht bij de ONDERSTE band ({bb_lower_val:.2f}) — mogelijke steun."
    else:
        bb_tekst = f"Prijs ({prijs:.2f}) zit in het MIDDEN van de Bollinger Bands. Geen extreme positie."

    if not pd.isna(bb_width_val) and bb_width_val < 0.03:
        bb_tekst += " 📉 SQUEEZE gedetecteerd (banden nauw) — grote beweging op komst."

    result["bb_pct"]   = bb_pct_val
    result["bb_width"] = bb_width_val
    result["bb_upper"] = bb_upper_val
    result["bb_lower"] = bb_lower_val
    result["bb_tekst"] = bb_tekst

    # ── Candlestick Patronen ───────────────────────────────────────
    candle_hammer     = bool(last.get("candle_hammer", False))
    candle_doji       = bool(last.get("candle_doji", False))
    candle_bull_engulf = bool(last.get("candle_bull_engulf", False))
    candle_bear_engulf = bool(last.get("candle_bear_engulf", False))

    patronen = []
    if candle_hammer:      patronen.append("🔨 Hammer (bullish reversal)")
    if candle_bull_engulf: patronen.append("📈 Bullish Engulfing")
    if candle_bear_engulf: patronen.append("📉 Bearish Engulfing")
    if candle_doji:        patronen.append("➕ Doji (onzekerheid)")

    result["candle_patronen"] = patronen
    result["candle_tekst"]    = ", ".join(patronen) if patronen else "Geen bijzonder patroon."

    # ── Fibonacci ──────────────────────────────────────────────────
    from src.indicators import fibonacci_levels as _fib_fn
    fib = _fib_fn(df, lookback=50)
    fib_tekst = "Fibonacci kon niet worden berekend (te weinig data)."
    if fib:
        nabij = []
        for lvl in ("38.2", "50.0", "61.8"):
            fib_prijs = fib.get(lvl, float("nan"))
            if not pd.isna(fib_prijs):
                afstand = abs(prijs - fib_prijs) / prijs * 100
                if afstand < 0.5:
                    nabij.append(f"Fib {lvl}% ({fib_prijs:.2f})")
        fib_tekst = (
            f"Fib-niveaus (laatste 50 bars): "
            f"38.2%={fib.get('38.2', 0):.2f}  "
            f"50%={fib.get('50.0', 0):.2f}  "
            f"61.8%={fib.get('61.8', 0):.2f}"
        )
        if nabij:
            fib_tekst += f" ⚡ Prijs nabij: {', '.join(nabij)}"
    result["fib"]      = fib
    result["fib_tekst"] = fib_tekst

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


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-STRATEGIE SCANNER
# ══════════════════════════════════════════════════════════════════════════════

def _setup(
    strategie: str,
    richting: str,
    entry: float,
    stop_loss: float,
    take_profit: float,
    reden: list[str],
    min_rr: float,
) -> dict | None:
    """Bouw een setup-dict en valideer de R/R. Geeft None terug als R/R te laag."""
    risk = abs(entry - stop_loss)
    if risk == 0:
        return None
    rr = abs(take_profit - entry) / risk
    if rr < min_rr:
        return None
    return {
        "strategie":   strategie,
        "type":        "setup",
        "richting":    richting,
        "entry":       round(entry, 6),
        "stop_loss":   round(stop_loss, 6),
        "take_profit": round(take_profit, 6),
        "risk_reward": round(rr, 2),
        "reden":       reden,
        "blokkades":   [],
        "uitleg":      (
            f"{STRATEGIE_NAMEN.get(strategie, strategie)} — {richting.upper()}\n"
            f"  Entry: {entry:.4f}  SL: {stop_loss:.4f}  TP: {take_profit:.4f}  R/R: 1:{rr:.1f}"
        ),
    }


def _signal_trend_long(last: pd.Series, prev: pd.Series, min_rr: float) -> dict | None:
    """Trend-long: uptrend bevestigd, prijs boven SMA20 > SMA50, ADX sterk genoeg."""
    if pd.isna(last["sma20"]) or pd.isna(last["sma50"]) or pd.isna(last["atr14"]):
        return None
    if not (last["close"] > last["sma20"] > last["sma50"]):
        return None
    adx_val = last.get("adx14", float("nan"))
    if not pd.isna(adx_val) and adx_val < 18:
        return None
    rsi_val = last.get("rsi14", 50)
    if not pd.isna(rsi_val) and not (40 <= rsi_val <= 70):
        return None
    # Bollinger: niet boven bovenste band (overbought op BB)
    bb_pct = last.get("bb_pct", float("nan"))
    if not pd.isna(bb_pct) and bb_pct > 1.0:
        return None
    entry   = last["close"]
    sl      = entry - 2.0 * last["atr14"]
    tp      = entry + min_rr * (entry - sl)
    macd_line   = last.get("macd_line", float("nan"))
    macd_signal = last.get("macd_signal", float("nan"))
    macd_info = (
        f"MACD bullish ({macd_line:.4f} > {macd_signal:.4f})"
        if not pd.isna(macd_line) and macd_line > macd_signal
        else "MACD neutraal/bearish"
    )
    return _setup("trend_long", "long", entry, sl, tp, [
        f"Uptrend: close > SMA20 ({last['sma20']:.2f}) > SMA50 ({last['sma50']:.2f})",
        f"RSI: {rsi_val:.0f}" if not pd.isna(rsi_val) else "RSI: n/b",
        f"ADX: {adx_val:.0f}" if not pd.isna(adx_val) else "ADX: n/b",
        macd_info,
    ], min_rr)


def _signal_pullback(last: pd.Series, prev: pd.Series, min_rr: float) -> dict | None:
    """Pullback long: uptrend, prijs teruggevallen naar EMA20, herstelcandle of hammer."""
    if pd.isna(last["ema20"]) or pd.isna(last["sma50"]) or pd.isna(last["atr14"]):
        return None
    if not (last["sma20"] > last["sma50"]):
        return None
    dist = abs(last["close"] - last["ema20"])
    if dist > last["atr14"]:
        return None
    rsi_val = last.get("rsi14", 50)
    if not pd.isna(rsi_val) and not (30 <= rsi_val <= 58):
        return None
    # Candlestick bevestiging: groen-na-rood, bull engulfing, of hammer
    groen_na_rood = prev["close"] < prev["open"] and last["close"] > last["open"]
    bull_engulf   = bool(last.get("candle_bull_engulf", False))
    hammer        = bool(last.get("candle_hammer", False))
    if not (groen_na_rood or bull_engulf or hammer):
        return None
    entry = last["close"]
    sl    = entry - 2.0 * last["atr14"]
    tp    = entry + min_rr * (entry - sl)
    candle_info = (
        "Bullish Engulfing" if bull_engulf
        else "Hammer" if hammer
        else "Herstelcandle (groen na rood)"
    )
    # Fibonacci: check of entry nabij een sleutelzone ligt
    fib_info = ""
    for lvl, col in (("38.2", "fib_38_2"), ("50.0", "fib_50_0"), ("61.8", "fib_61_8")):
        fib_prijs = last.get(col, float("nan"))
        if not pd.isna(fib_prijs) and abs(entry - fib_prijs) / entry < 0.005:
            fib_info = f"Nabij Fibonacci {lvl}% ({fib_prijs:.2f})"
            break
    return _setup("pullback", "long", entry, sl, tp, [
        f"Pullback naar EMA20 ({last['ema20']:.2f}) in uptrend",
        candle_info,
        f"RSI: {rsi_val:.0f}" if not pd.isna(rsi_val) else "RSI: n/b",
        *([fib_info] if fib_info else []),
    ], min_rr)


def _signal_breakout(last: pd.Series, prev: pd.Series, min_rr: float) -> dict | None:
    """Breakout: sluit boven resistance met verhoogd volume en MACD bevestiging."""
    if pd.isna(last["resistance"]) or pd.isna(last["atr14"]):
        return None
    res = last["resistance"]
    if not (prev["close"] <= res < last["close"]):
        return None
    vol_ma = last.get("vol_ma20", 0)
    if not pd.isna(vol_ma) and vol_ma > 0 and last["volume"] < 1.3 * vol_ma:
        return None
    rsi_val = last.get("rsi14", 50)
    if not pd.isna(rsi_val) and rsi_val < 45:
        return None
    # MACD moet positief zijn (bevestigt momentum bij uitbraak)
    macd_hist = last.get("macd_hist", float("nan"))
    if not pd.isna(macd_hist) and macd_hist < 0:
        return None
    entry = last["close"]
    sl    = res - 0.5 * last["atr14"]
    tp    = entry + min_rr * (entry - sl)
    bb_info = ""
    bb_pct = last.get("bb_pct", float("nan"))
    if not pd.isna(bb_pct) and bb_pct > 1.0:
        bb_info = "Prijs boven BB bovenband (sterke uitbraak)"
    return _setup("breakout", "long", entry, sl, tp, [
        f"Breakout boven resistance ({res:.2f})",
        f"Volume: {last['volume']:.0f} (gem. {vol_ma:.0f})" if not pd.isna(vol_ma) else "Volume: n/b",
        f"RSI: {rsi_val:.0f}" if not pd.isna(rsi_val) else "RSI: n/b",
        f"MACD histogram: {macd_hist:.4f} (positief)" if not pd.isna(macd_hist) else "MACD: n/b",
        *([bb_info] if bb_info else []),
    ], min_rr)


def _signal_trend_short(last: pd.Series, prev: pd.Series, min_rr: float) -> dict | None:
    """Trend-short: downtrend bevestigd, close < SMA20 < SMA50, ADX sterk.

    Strenge eisen om shortten in een bull-markt te voorkomen:
      - ADX >= 30 (was 25) — trend moet écht sterk zijn
      - RSI 25–48 (was 30–60) — momentum al duidelijk neerwaarts
      - Volume boven gemiddelde — bevestigt de druk
      - MACD moet bearish zijn (verplicht, was optioneel)
      - Vorige candle ook rood (candle-bevestiging)
    """
    if pd.isna(last["sma20"]) or pd.isna(last["sma50"]) or pd.isna(last["atr14"]):
        return None
    if not (last["close"] < last["sma20"] < last["sma50"]):
        return None
    # Strengere ADX-eis: echte trend, geen korte correctie
    adx_val = last.get("adx14", float("nan"))
    if not pd.isna(adx_val) and adx_val < 30:
        return None
    # RSI moet al duidelijk neerwaarts zijn — niet boven 48
    rsi_val = last.get("rsi14", 50)
    if not pd.isna(rsi_val) and not (25 <= rsi_val <= 48):
        return None
    # Volume moet boven gemiddelde zijn
    vol_ma = last.get("vol_ma20", float("nan"))
    if not pd.isna(vol_ma) and vol_ma > 0 and last["volume"] < vol_ma:
        return None
    # MACD verplicht bearish
    macd_line   = last.get("macd_line", float("nan"))
    macd_signal_v = last.get("macd_signal", float("nan"))
    if not pd.isna(macd_line) and not pd.isna(macd_signal_v) and macd_line >= macd_signal_v:
        return None
    # Candle-bevestiging: vorige candle ook rood (momentum bevestigd)
    if prev["close"] >= prev["open"]:
        return None
    # Bollinger: niet onder onderste band (oversold op BB)
    bb_pct = last.get("bb_pct", float("nan"))
    if not pd.isna(bb_pct) and bb_pct < 0.0:
        return None
    entry = last["close"]
    sl    = entry + 2.0 * last["atr14"]
    tp    = entry - min_rr * (sl - entry)
    macd_line   = last.get("macd_line", float("nan"))
    macd_signal = last.get("macd_signal", float("nan"))
    macd_info = (
        f"MACD bearish ({macd_line:.4f} < {macd_signal:.4f})"
        if not pd.isna(macd_line) and macd_line < macd_signal
        else "MACD neutraal/bullish"
    )
    return _setup("trend_short", "short", entry, sl, tp, [
        f"Downtrend: close < SMA20 ({last['sma20']:.2f}) < SMA50 ({last['sma50']:.2f})",
        f"RSI: {rsi_val:.0f}" if not pd.isna(rsi_val) else "RSI: n/b",
        f"ADX: {adx_val:.0f}" if not pd.isna(adx_val) else "ADX: n/b",
        macd_info,
    ], min_rr)


def _signal_pullback_short(last: pd.Series, prev: pd.Series, min_rr: float) -> dict | None:
    """Pullback short: downtrend, prijs bounced naar EMA20 en faalt — bearish candle bevestigt."""
    if pd.isna(last["ema20"]) or pd.isna(last["sma50"]) or pd.isna(last["atr14"]):
        return None
    if not (last["sma20"] < last["sma50"]):
        return None
    dist = abs(last["close"] - last["ema20"])
    if dist > last["atr14"]:
        return None
    # RSI mag niet te hoog zijn — anders is de bounce nog steeds sterk
    rsi_val = last.get("rsi14", 50)
    if not pd.isna(rsi_val) and not (38 <= rsi_val <= 55):
        return None
    # MACD moet bearish zijn voor extra bevestiging
    macd_line  = last.get("macd_line", float("nan"))
    macd_sig   = last.get("macd_signal", float("nan"))
    if not pd.isna(macd_line) and not pd.isna(macd_sig) and macd_line >= macd_sig:
        return None
    # Candlestick bevestiging: rood-na-groen of bear engulfing
    rood_na_groen = prev["close"] > prev["open"] and last["close"] < last["open"]
    bear_engulf   = bool(last.get("candle_bear_engulf", False))
    if not (rood_na_groen or bear_engulf):
        return None
    entry = last["close"]
    sl    = entry + 2.0 * last["atr14"]
    tp    = entry - min_rr * (sl - entry)
    candle_info = "Bearish Engulfing" if bear_engulf else "Rode candle na groene bounce"
    fib_info = ""
    for lvl, col in (("61.8", "fib_61_8"), ("50.0", "fib_50_0"), ("38.2", "fib_38_2")):
        fib_prijs = last.get(col, float("nan"))
        if not pd.isna(fib_prijs) and abs(entry - fib_prijs) / entry < 0.005:
            fib_info = f"Nabij Fibonacci {lvl}% ({fib_prijs:.2f}) — bounce-zone"
            break
    return _setup("pullback_short", "short", entry, sl, tp, [
        f"Pullback naar EMA20 ({last['ema20']:.2f}) in downtrend — bounce mislukt",
        candle_info,
        f"RSI: {rsi_val:.0f}" if not pd.isna(rsi_val) else "RSI: n/b",
        *([fib_info] if fib_info else []),
    ], min_rr)


def _signal_breakdown(last: pd.Series, prev: pd.Series, min_rr: float) -> dict | None:
    """Breakdown: sluit onder support met verhoogd volume en MACD bevestiging."""
    if pd.isna(last["support"]) or pd.isna(last["atr14"]):
        return None
    sup = last["support"]
    if not (prev["close"] >= sup > last["close"]):
        return None
    # Geen breakdown traden in een duidelijke uptrend
    sma20 = last.get("sma20", float("nan"))
    sma50 = last.get("sma50", float("nan"))
    if not pd.isna(sma20) and not pd.isna(sma50) and last["close"] > sma20 and sma20 > sma50:
        return None
    vol_ma = last.get("vol_ma20", 0)
    if not pd.isna(vol_ma) and vol_ma > 0 and last["volume"] < 1.3 * vol_ma:
        return None
    rsi_val = last.get("rsi14", 50)
    if not pd.isna(rsi_val) and rsi_val > 55:
        return None
    # MACD moet negatief zijn (bevestigt neerwaarts momentum bij breakdown)
    macd_hist = last.get("macd_hist", float("nan"))
    if not pd.isna(macd_hist) and macd_hist > 0:
        return None
    entry = last["close"]
    sl    = sup + 0.5 * last["atr14"]
    tp    = entry - min_rr * (sl - entry)
    bb_info = ""
    bb_pct = last.get("bb_pct", float("nan"))
    if not pd.isna(bb_pct) and bb_pct < 0.0:
        bb_info = "Prijs onder BB onderband (sterke breakdown)"
    return _setup("breakdown", "short", entry, sl, tp, [
        f"Breakdown onder support ({sup:.2f})",
        f"Volume: {last['volume']:.0f} (gem. {vol_ma:.0f})" if not pd.isna(vol_ma) else "Volume: n/b",
        f"RSI: {rsi_val:.0f}" if not pd.isna(rsi_val) else "RSI: n/b",
        f"MACD histogram: {macd_hist:.4f} (negatief)" if not pd.isna(macd_hist) else "MACD: n/b",
        *([bb_info] if bb_info else []),
    ], min_rr)


def _signal_macd_bull_div(last: pd.Series, prev: pd.Series, min_rr: float) -> dict | None:
    """MACD bullish divergentie: prijs maakt lagere low, histogram stijgt — trend keert mogelijk om."""
    if not bool(last.get("macd_bull_div", False)):
        return None
    if pd.isna(last.get("atr14", float("nan"))):
        return None
    rsi_val = last.get("rsi14", float("nan"))
    # Alleen geldig als RSI niet overbought (dan is er ruimte voor herstel)
    if not pd.isna(rsi_val) and rsi_val > 65:
        return None
    # Niet in een duidelijke zware downtrend (SMA's mogen niet ver onder prijs liggen)
    sma20 = last.get("sma20", float("nan"))
    sma50 = last.get("sma50", float("nan"))
    if not pd.isna(sma20) and not pd.isna(sma50) and last["close"] < sma50 * 0.97:
        return None
    entry = last["close"]
    sl    = entry - 2.0 * last["atr14"]
    tp    = entry + min_rr * (entry - sl)
    macd_hist = last.get("macd_hist", float("nan"))
    return _setup("macd_bull_div", "long", entry, sl, tp, [
        "MACD bullish divergentie: prijs lager, histogram hoger",
        f"RSI: {rsi_val:.0f}" if not pd.isna(rsi_val) else "RSI: n/b",
        f"MACD histogram: {macd_hist:.4f}" if not pd.isna(macd_hist) else "MACD: n/b",
    ], min_rr)


def _signal_macd_bear_div(last: pd.Series, prev: pd.Series, min_rr: float) -> dict | None:
    """MACD bearish divergentie: prijs maakt hogere high, histogram daalt — trend verzwakt."""
    if not bool(last.get("macd_bear_div", False)):
        return None
    if pd.isna(last.get("atr14", float("nan"))):
        return None
    rsi_val = last.get("rsi14", float("nan"))
    # Alleen geldig als RSI niet oversold
    if not pd.isna(rsi_val) and rsi_val < 35:
        return None
    # Blokkeer in duidelijke uptrend (close > SMA20 > SMA50)
    sma20 = last.get("sma20", float("nan"))
    sma50 = last.get("sma50", float("nan"))
    if not pd.isna(sma20) and not pd.isna(sma50) and last["close"] > sma20 and sma20 > sma50:
        return None
    entry = last["close"]
    sl    = entry + 2.0 * last["atr14"]
    tp    = entry - min_rr * (sl - entry)
    macd_hist = last.get("macd_hist", float("nan"))
    return _setup("macd_bear_div", "short", entry, sl, tp, [
        "MACD bearish divergentie: prijs hoger, histogram lager",
        f"RSI: {rsi_val:.0f}" if not pd.isna(rsi_val) else "RSI: n/b",
        f"MACD histogram: {macd_hist:.4f}" if not pd.isna(macd_hist) else "MACD: n/b",
    ], min_rr)


def _signal_range_long(last: pd.Series, prev: pd.Series, min_rr: float) -> dict | None:
    """Range mean-reversion long: kopen bij steun/onderste band in rustige markt."""
    required = ("support", "resistance", "atr14", "bb_pct", "bb_width")
    if any(pd.isna(last.get(col, float("nan"))) for col in required):
        return None
    support = float(last["support"])
    resistance = float(last["resistance"])
    entry = float(last["close"])
    atr = float(last["atr14"])
    if resistance <= support or atr <= 0:
        return None

    range_size = resistance - support
    positie = (entry - support) / range_size
    bb_pct = float(last.get("bb_pct", 0.5))
    bb_width = float(last.get("bb_width", 1.0))
    adx_val = last.get("adx14", float("nan"))
    rsi_val = last.get("rsi14", 50)

    if positie > 0.28 and bb_pct > 0.25:
        return None
    if bb_width > 0.025:
        return None
    if not pd.isna(adx_val) and adx_val > 35:
        return None
    if not pd.isna(rsi_val) and not (28 <= rsi_val <= 48):
        return None

    groen_na_rood = prev["close"] < prev["open"] and last["close"] >= last["open"]
    hammer = bool(last.get("candle_hammer", False))
    if not (groen_na_rood or hammer or bb_pct <= 0.15):
        return None

    sl = min(support - 0.35 * atr, entry - 0.8 * atr)
    tp = min(resistance, entry + min_rr * (entry - sl))
    return _setup("range_long", "long", entry, sl, tp, [
        f"Range long bij support ({support:.2f})",
        f"Positie in range: {positie*100:.0f}%",
        f"Bollinger positie: {bb_pct:.2f}",
        f"RSI: {rsi_val:.0f}" if not pd.isna(rsi_val) else "RSI: n/b",
        "Kleine mean-reversion exploratie",
    ], min_rr)


def _signal_range_short(last: pd.Series, prev: pd.Series, min_rr: float) -> dict | None:
    """Range mean-reversion short: verkopen bij weerstand/bovenste band in rustige markt."""
    required = ("support", "resistance", "atr14", "bb_pct", "bb_width")
    if any(pd.isna(last.get(col, float("nan"))) for col in required):
        return None
    support = float(last["support"])
    resistance = float(last["resistance"])
    entry = float(last["close"])
    atr = float(last["atr14"])
    if resistance <= support or atr <= 0:
        return None

    range_size = resistance - support
    positie = (entry - support) / range_size
    bb_pct = float(last.get("bb_pct", 0.5))
    bb_width = float(last.get("bb_width", 1.0))
    adx_val = last.get("adx14", float("nan"))
    rsi_val = last.get("rsi14", 50)

    if positie < 0.72 and bb_pct < 0.75:
        return None
    if bb_width > 0.025:
        return None
    if not pd.isna(adx_val) and adx_val > 35:
        return None
    if not pd.isna(rsi_val) and not (52 <= rsi_val <= 72):
        return None

    rood_na_groen = prev["close"] > prev["open"] and last["close"] <= last["open"]
    bear_engulf = bool(last.get("candle_bear_engulf", False))
    if not (rood_na_groen or bear_engulf or bb_pct >= 0.85):
        return None

    sl = max(resistance + 0.35 * atr, entry + 0.8 * atr)
    tp = max(support, entry - min_rr * (sl - entry))
    return _setup("range_short", "short", entry, sl, tp, [
        f"Range short bij resistance ({resistance:.2f})",
        f"Positie in range: {positie*100:.0f}%",
        f"Bollinger positie: {bb_pct:.2f}",
        f"RSI: {rsi_val:.0f}" if not pd.isna(rsi_val) else "RSI: n/b",
        "Kleine mean-reversion exploratie",
    ], min_rr)


_STRATEGIE_FNS = [
    _signal_trend_long,
    _signal_pullback,
    _signal_breakout,
    _signal_trend_short,
    _signal_pullback_short,
    _signal_breakdown,
    _signal_macd_bull_div,
    _signal_macd_bear_div,
    _signal_range_long,
    _signal_range_short,
]


# Strategieën waarvoor trendcontinuïteit vereist is (3 opeenvolgende candles)
_TREND_CONTINUITEIT_CHECK = frozenset({
    "trend_long", "trend_short", "pullback", "pullback_short",
})


def _trend_bevestigd(df_ind: pd.DataFrame, richting: str, lookback: int = 3) -> bool:
    """
    Controleer of de SMA-trend de laatste N candles consistent is.
    Voorkomt entries direct na een trendwisseling (whipsaw-filter).
    """
    if len(df_ind) < lookback:
        return False
    for i in range(1, lookback + 1):
        c = df_ind.iloc[-i]
        sma20 = c.get("sma20", float("nan"))
        sma50 = c.get("sma50", float("nan"))
        close = c.get("close", float("nan"))
        if pd.isna(sma20) or pd.isna(sma50):
            return False
        if richting == "long"  and not (close > sma20 > sma50):
            return False
        if richting == "short" and not (close < sma20 < sma50):
            return False
    return True


def scan_alle_strategieen(
    df: pd.DataFrame,
    min_rr: float = 2.0,
    liquidation_zones: list | None = None,
) -> list[dict]:
    """
    Scan alle strategieën op het gegeven DataFrame.
    Geeft een lijst van setup-dicts terug (alleen echte setups, geen 'wacht').

    Globale filters:
      - ADX < 18 → trend-strategieën overgeslagen (choppy markt)
      - Trend-continuïteit: voor trend/pullback-strategieën moet de SMA-trend
        de laatste 3 candles consistent zijn (whipsaw-filter)
    """
    if len(df) < 60:
        return []

    try:
        df_ind = add_all_indicators(df)
    except Exception:
        return []

    last = df_ind.iloc[-1]
    prev = df_ind.iloc[-2]

    # Globale choppy-markt filter — verhoogd van 18 naar 22
    adx_val  = last.get("adx14", float("nan"))
    is_choppy = not pd.isna(adx_val) and adx_val < 18

    setups: list[dict] = []
    for fn in _STRATEGIE_FNS:
        try:
            naam = fn.__name__.replace("_signal_", "")
            if is_choppy and naam in _TREND_STRATEGIEEN:
                continue
            result = fn(last, prev, min_rr)
            if result is None:
                continue
            # Trend-continuïteit check: eis 3 opeenvolgende candles in dezelfde trendrichting
            if naam in _TREND_CONTINUITEIT_CHECK:
                if not _trend_bevestigd(df_ind, result["richting"], lookback=3):
                    continue
            setups.append(result)
        except Exception:
            continue

    return setups
