"""
indicators.py — Technische indicatoren berekend zonder externe bibliotheken.

Elke functie heeft een uitleg zodat je begrijpt wat hij doet.
"""
import pandas as pd
import numpy as np


# ── Trendlijnen ──────────────────────────────────────────────────────────────

def sma(series: pd.Series, period: int) -> pd.Series:
    """
    Simple Moving Average (SMA) — gewoon gemiddelde over N candles.
    Gebruik: SMA20 en SMA50 om de trendrichting te bepalen.
    Als close > SMA20 > SMA50 → uptrend.
    """
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """
    Exponential Moving Average (EMA) — recentere candles wegen zwaarder.
    Reageert sneller op prijsveranderingen dan SMA.
    """
    return series.ewm(span=period, adjust=False).mean()


# ── Volatiliteit ─────────────────────────────────────────────────────────────

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range (ATR) — maat voor hoe veel de markt beweegt.
    Grotere ATR = wildere markt. Gebruik voor stop-loss bepaling.
    Vuistregel: stop-loss = 1.5 × ATR onder entry.
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    return tr.rolling(window=period, min_periods=period).mean()


# ── Momentum ─────────────────────────────────────────────────────────────────

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI) — geeft een waarde van 0 tot 100.
    > 70 : mogelijk overbought (te duur, kans op daling)
    < 30 : mogelijk oversold  (te goedkoop, kans op stijging)
    50   : neutraal
    """
    delta = series.diff()
    winst = delta.clip(lower=0).rolling(window=period, min_periods=period).mean()
    verlies = (-delta.clip(upper=0)).rolling(window=period, min_periods=period).mean()
    rs = winst / verlies.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def vwap(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Rolling Volume Weighted Average Price over `period` candles.
    Gemiddelde prijs gewogen met volume — waar het 'echte' handelsgewicht ligt.
    Causaal (trailing window).
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3
    pv = (typical * df["volume"]).rolling(window=period, min_periods=period).sum()
    vol = df["volume"].rolling(window=period, min_periods=period).sum()
    return pv / vol.replace(0, np.nan)


def stoch_rsi(series: pd.Series, period: int = 14, smooth: int = 3) -> pd.Series:
    """
    Stochastic RSI — waar staat de RSI binnen zijn eigen recente bereik (0-1)?
    Gevoeliger dan kale RSI voor over-/onderkochte extremen. Causaal.
    """
    r = rsi(series, period)
    laag = r.rolling(window=period, min_periods=period).min()
    hoog = r.rolling(window=period, min_periods=period).max()
    stoch = (r - laag) / (hoog - laag).replace(0, np.nan)
    return stoch.rolling(window=smooth, min_periods=1).mean()


# ── Support & Resistance ──────────────────────────────────────────────────────

def support_resistance(df: pd.DataFrame, lookback: int = 20):
    """
    Simpele support en resistance op basis van rolling min/max.
    Support    = laagste punt van de laatste N candles (vloer, kopers stappen in)
    Resistance = hoogste punt van de laatste N candles (plafond, verkopers stappen in)
    """
    support    = df["low"].rolling(window=lookback, min_periods=lookback).min()
    resistance = df["high"].rolling(window=lookback, min_periods=lookback).max()
    return support, resistance


# ── Volume ────────────────────────────────────────────────────────────────────

def volume_ma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Gemiddeld handelsvolume over N perioden. Hogere volumepieken = sterkere beweging."""
    return df["volume"].rolling(window=period, min_periods=period).mean()


# ── Trendsterkte ─────────────────────────────────────────────────────────────

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average Directional Index (ADX) — meet de STERKTE van een trend (niet richting).
    < 20 : Zijwaartse/choppige markt — geen duidelijke trend
    20-40: Matige trend aanwezig
    > 40 : Sterke trend
    Gebruik: sla trades over als ADX < 18 (markt gaat nergens heen).
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    up_move   = high - high.shift(1)
    down_move = low.shift(1) - low

    dm_plus  = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    dm_minus = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr_s    = tr.ewm(span=period, adjust=False).mean()
    di_plus  = 100 * dm_plus.ewm(span=period, adjust=False).mean()  / atr_s
    di_minus = 100 * dm_minus.ewm(span=period, adjust=False).mean() / atr_s

    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    return dx.ewm(span=period, adjust=False).mean()


# ── Fibonacci Retracement ────────────────────────────────────────────────────

def fibonacci_levels(df: pd.DataFrame, lookback: int = 50) -> dict:
    """
    Bereken Fibonacci retracement-niveaus op basis van de swing high en low
    van de laatste 'lookback' candles.

    Niveaus: 23.6%, 38.2%, 50%, 61.8%, 78.6%
    Hoe te gebruiken: in een uptrend zijn 38.2%, 50% en 61.8% pullback-zones
    waar kopers historisch instappen. In een downtrend zijn dit bounce-zones
    voor shorts.
    """
    if len(df) < lookback:
        return {}
    window = df.tail(lookback)
    hoog   = float(window["high"].max())
    laag   = float(window["low"].min())
    rng    = hoog - laag
    if rng == 0:
        return {}
    return {
        "hoog":  round(hoog, 6),
        "laag":  round(laag, 6),
        "23.6":  round(hoog - 0.236 * rng, 6),
        "38.2":  round(hoog - 0.382 * rng, 6),
        "50.0":  round(hoog - 0.500 * rng, 6),
        "61.8":  round(hoog - 0.618 * rng, 6),
        "78.6":  round(hoog - 0.786 * rng, 6),
    }


# ── MACD ─────────────────────────────────────────────────────────────────────

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal_period: int = 9):
    """
    MACD — Moving Average Convergence Divergence.
    macd_line   = EMA(fast) - EMA(slow)
    signal_line = EMA(macd_line, signal_period)
    histogram   = macd_line - signal_line
    Bullish cross: histogram wordt positief (macd kruist signal omhoog).
    Bearish cross: histogram wordt negatief (macd kruist signal omlaag).
    """
    ema_fast    = series.ewm(span=fast, adjust=False).mean()
    ema_slow    = series.ewm(span=slow, adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


# ── Bollinger Bands ───────────────────────────────────────────────────────────

def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    """
    Bollinger Bands — dynamische bandbreedte rond een SMA.
    upper/lower = SMA ± 2×std
    bb_pct  : 0 = onderste band, 1 = bovenste band (prijs positie binnen de bands)
    bb_width: (upper - lower) / middle — klein = squeeze (lage volatiliteit, uitbraak verwacht)
    """
    middle   = series.rolling(window=period, min_periods=period).mean()
    std      = series.rolling(window=period, min_periods=period).std()
    upper    = middle + std_dev * std
    lower    = middle - std_dev * std
    bb_pct   = (series - lower) / (upper - lower).replace(0, np.nan)
    bb_width = (upper - lower) / middle.replace(0, np.nan)
    return upper, middle, lower, bb_pct, bb_width


# ── Pivot Support & Resistance ────────────────────────────────────────────────

def pivot_support_resistance(df: pd.DataFrame, lookback: int = 5):
    """
    Pivot-gebaseerde support en resistance via swing hoog/laag detectie.
    Een swing low is de laagste bar in een symmetrisch venster van 2×lookback+1 candles.
    Resultaten worden met 'lookback' bars vertraagd — geen toekomstdata gebruikt.
    Veel nauwkeuriger dan een simpele rolling min/max.
    """
    highs = df["high"]
    lows  = df["low"]
    w = 2 * lookback + 1

    roll_min   = lows.rolling(window=w, center=True, min_periods=w).min()
    roll_max   = highs.rolling(window=w, center=True, min_periods=w).max()
    swing_low  = lows.where(lows == roll_min)
    swing_high = highs.where(highs == roll_max)

    # Vertraging zodat de pivot pas 'zichtbaar' is nadat hij bevestigd is
    support    = swing_low.shift(lookback).ffill()
    resistance = swing_high.shift(lookback).ffill()
    return support, resistance


# ── Candlestick Patronen ──────────────────────────────────────────────────────

def candlestick_patterns(df: pd.DataFrame):
    """
    Detecteer basale candlestick patronen op basis van OHLC-data.
    hammer         : lange onderste wick, kleine body bovenin — bullish reversal signaal
    doji           : body < 10% van de range — onzekerheid, mogelijke ommekeer
    bull_engulfing : groene candle omhult vorige rode candle body — bullish
    bear_engulfing : rode candle omhult vorige groene candle body — bearish
    """
    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]

    body       = (c - o).abs()
    range_     = (h - l).replace(0, np.nan)
    lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l
    upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)

    hammer = (
        (body / range_ <= 0.35) &
        (lower_wick >= 2.0 * body) &
        (upper_wick <= body) &
        (c >= o)
    )
    doji = body / range_ <= 0.10
    bull_engulf = (
        (o.shift(1) > c.shift(1)) &
        (c > o) &
        (o <= c.shift(1)) &
        (c >= o.shift(1))
    )
    bear_engulf = (
        (c.shift(1) > o.shift(1)) &
        (c < o) &
        (o >= c.shift(1)) &
        (c <= o.shift(1))
    )
    return (
        hammer.fillna(False),
        doji.fillna(False),
        bull_engulf.fillna(False),
        bear_engulf.fillna(False),
    )


# ── MACD Divergentie ──────────────────────────────────────────────────────────

def macd_divergence(df: pd.DataFrame, lookback: int = 10):
    """
    MACD Divergentie — signaleert verzwakkende trend (mogelijke ommekeer).
    Bullish: prijs maakt lagere low, MACD histogram maakt hogere low (kracht neemt toe).
    Bearish: prijs maakt hogere high, MACD histogram maakt lagere high (kracht neemt af).
    Vereist dat 'macd_hist' al aanwezig is in het DataFrame.
    """
    if "macd_hist" not in df.columns:
        false = pd.Series(False, index=df.index)
        return false, false

    close = df["close"]
    hist  = df["macd_hist"]

    bull_div = (
        (close < close.shift(lookback)) &
        (hist  > hist.shift(lookback)) &
        (hist  < 0)
    )
    bear_div = (
        (close > close.shift(lookback)) &
        (hist  < hist.shift(lookback)) &
        (hist  > 0)
    )
    return bull_div.fillna(False), bear_div.fillna(False)


# ── Alles in één keer toevoegen ───────────────────────────────────────────────

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Voeg alle indicatoren toe aan het DataFrame en geef het terug.
    Originele data wordt NIET aangepast (copy).
    """
    df = df.copy()

    df["sma20"]    = sma(df["close"], 20)
    df["sma50"]    = sma(df["close"], 50)
    df["ema20"]    = ema(df["close"], 20)
    df["ema50"]    = ema(df["close"], 50)
    df["ema200"]   = ema(df["close"], 200)
    df["atr14"]    = atr(df, 14)
    df["rsi14"]    = rsi(df["close"], 14)
    df["vol_ma20"] = volume_ma(df, 20)
    df["adx14"]    = adx(df, 14)
    df["atr_pct"]  = df["atr14"] / df["close"] * 100
    df["vwap20"]   = vwap(df, 20)
    df["stoch_rsi"] = stoch_rsi(df["close"], 14)

    # Pivot-gebaseerde support/resistance (vervangt simpele rolling min/max)
    df["support"], df["resistance"] = pivot_support_resistance(df, lookback=5)

    # MACD (eerst berekenen, divergentie heeft macd_hist nodig)
    df["macd_line"], df["macd_signal"], df["macd_hist"] = macd(df["close"])

    # Bollinger Bands
    (df["bb_upper"], df["bb_mid"], df["bb_lower"],
     df["bb_pct"], df["bb_width"]) = bollinger_bands(df["close"])

    # Candlestick patronen
    (df["candle_hammer"], df["candle_doji"],
     df["candle_bull_engulf"], df["candle_bear_engulf"]) = candlestick_patterns(df)

    # MACD divergentie (vereist macd_hist)
    df["macd_bull_div"], df["macd_bear_div"] = macd_divergence(df)

    # Fibonacci retracement-niveaus (als scalar kolommen op elke rij)
    fib = fibonacci_levels(df, lookback=50)
    for level in ("38.2", "50.0", "61.8"):
        col = f"fib_{level.replace('.', '_')}"
        df[col] = fib.get(level, float("nan"))

    return df
