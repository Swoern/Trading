"""Tests voor src/indicators.py — beschermt de kern-wiskunde tegen regressies."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import indicators as ind  # noqa: E402


def _ohlcv(rows=150, seed=7):
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2026-01-01", tz="UTC")
    price = 100.0
    data = []
    for i in range(rows):
        price *= 1 + rng.normal(0.0005, 0.01)
        o = price * (1 + rng.normal(0, 0.002))
        c = price
        h = max(o, c) * (1 + abs(rng.normal(0, 0.003)))
        low = min(o, c) * (1 - abs(rng.normal(0, 0.003)))
        data.append({"timestamp": base + pd.Timedelta(minutes=5 * i),
                     "open": o, "high": h, "low": low, "close": c,
                     "volume": rng.uniform(100, 1000)})
    return pd.DataFrame(data)


class BasicIndicatorTests(unittest.TestCase):
    def test_sma_known_values(self):
        s = pd.Series([1, 2, 3, 4, 5], dtype=float)
        out = ind.sma(s, 3)
        self.assertTrue(np.isnan(out.iloc[1]))
        self.assertAlmostEqual(out.iloc[2], 2.0)
        self.assertAlmostEqual(out.iloc[4], 4.0)

    def test_rsi_bounds_and_extremes(self):
        rng = np.random.default_rng(1)
        # Realistische trends mét pullbacks (zoals echte data) — geen deel-door-nul.
        up = pd.Series(np.cumsum(rng.normal(0.8, 0.5, 90)) + 100)
        down = pd.Series(np.cumsum(rng.normal(-0.8, 0.5, 90)) + 100)
        r_up = ind.rsi(up, 14).iloc[-1]
        r_down = ind.rsi(down, 14).iloc[-1]
        self.assertGreater(r_up, 60)        # sterke uptrend -> RSI hoog
        self.assertLess(r_down, 40)         # sterke downtrend -> RSI laag
        self.assertGreater(r_up, r_down)
        r_all = ind.rsi(up, 14).dropna()
        self.assertTrue(((r_all >= 0) & (r_all <= 100)).all())

    def test_atr_positive(self):
        df = _ohlcv()
        a = ind.atr(df, 14).dropna()
        self.assertTrue((a > 0).all())

    def test_macd_histogram_consistency(self):
        s = _ohlcv()["close"]
        line, signal, hist = ind.macd(s)
        # histogram == line - signal
        self.assertTrue(np.allclose((line - signal).dropna(), hist.dropna()))

    def test_bollinger_ordering(self):
        s = _ohlcv()["close"]
        upper, mid, lower, pct, width = ind.bollinger_bands(s)
        valid = upper.dropna().index
        self.assertTrue((upper.loc[valid] >= mid.loc[valid]).all())
        self.assertTrue((mid.loc[valid] >= lower.loc[valid]).all())

    def test_adx_range(self):
        df = _ohlcv()
        a = ind.adx(df, 14).dropna()
        self.assertTrue(((a >= 0) & (a <= 100)).all())


class CausalityTests(unittest.TestCase):
    """Bewijst dat indicatoren geen toekomstdata gebruiken — fundament voor de
    snelle backtester (precompute één keer == per-candle herberekenen)."""

    def test_support_resistance_is_causal(self):
        full = _ohlcv(160)
        full_ind = ind.add_all_indicators(full)
        for i in (80, 110, 140, 155):
            prefix = ind.add_all_indicators(full.iloc[:i + 1])
            for col in ("support", "resistance"):
                a = prefix[col].iloc[-1]
                b = full_ind[col].iloc[i]
                if pd.isna(a) and pd.isna(b):
                    continue
                self.assertAlmostEqual(a, b, places=6,
                                       msg=f"{col} op rij {i} verschilt -> lookahead!")

    def test_all_indicators_columns_present(self):
        df = ind.add_all_indicators(_ohlcv())
        for col in ("sma20", "sma50", "ema20", "atr14", "rsi14", "adx14",
                    "macd_line", "macd_hist", "bb_upper", "bb_lower",
                    "support", "resistance"):
            self.assertIn(col, df.columns)
        # Na warmup zijn de kern-indicatoren gevuld op de laatste rij.
        last = df.iloc[-1]
        for col in ("sma20", "rsi14", "atr14", "macd_line"):
            self.assertFalse(pd.isna(last[col]), f"{col} is NaN op laatste rij")


if __name__ == "__main__":
    unittest.main()
