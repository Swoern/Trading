"""Tests voor src/strategy.py — setup-correctheid + precomputed-vlag."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import strategy  # noqa: E402
from src.indicators import add_all_indicators  # noqa: E402


def _trending_ohlcv(rows=200, drift=0.004, seed=3):
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2026-01-01", tz="UTC")
    price = 100.0
    data = []
    for i in range(rows):
        price *= 1 + rng.normal(drift, 0.008)
        o = price * (1 + rng.normal(0, 0.002))
        c = price
        h = max(o, c) * (1 + abs(rng.normal(0, 0.003)))
        low = min(o, c) * (1 - abs(rng.normal(0, 0.003)))
        data.append({"timestamp": base + pd.Timedelta(minutes=5 * i),
                     "open": o, "high": h, "low": low, "close": c,
                     "volume": rng.uniform(100, 1000)})
    return pd.DataFrame(data)


class ScanTests(unittest.TestCase):
    def test_returns_list_and_too_short_is_empty(self):
        self.assertEqual(strategy.scan_alle_strategieen(_trending_ohlcv(30)), [])
        out = strategy.scan_alle_strategieen(_trending_ohlcv(200))
        self.assertIsInstance(out, list)

    def test_setups_are_valid(self):
        for seed in range(8):  # meerdere markten/condities
            for s in strategy.scan_alle_strategieen(_trending_ohlcv(200, seed=seed)):
                for k in ("strategie", "richting", "entry", "stop_loss", "take_profit"):
                    self.assertIn(k, s)
                entry, sl, tp = s["entry"], s["stop_loss"], s["take_profit"]
                rr = s.get("risk_reward", 0)
                self.assertGreaterEqual(round(rr, 3), 2.0 - 1e-6)  # min_rr default 2.0
                if s["richting"] == "long":
                    self.assertLess(sl, entry)
                    self.assertGreater(tp, entry)
                else:
                    self.assertGreater(sl, entry)
                    self.assertLess(tp, entry)

    def test_precomputed_matches_recompute(self):
        """De precomputed-vlag (waar de snelle backtester op leunt) moet identieke
        setups geven als opnieuw berekenen."""
        for seed in range(6):
            df = _trending_ohlcv(200, seed=seed)
            a = strategy.scan_alle_strategieen(df, min_rr=2.0)
            b = strategy.scan_alle_strategieen(add_all_indicators(df), min_rr=2.0, precomputed=True)
            self.assertEqual(
                [(x["strategie"], round(x["entry"], 6), x["richting"]) for x in a],
                [(x["strategie"], round(x["entry"], 6), x["richting"]) for x in b],
                msg=f"precomputed != recompute (seed {seed})",
            )

    def test_min_rr_respected(self):
        for s in strategy.scan_alle_strategieen(_trending_ohlcv(200), min_rr=3.0):
            self.assertGreaterEqual(round(s.get("risk_reward", 0), 3), 3.0 - 1e-6)


if __name__ == "__main__":
    unittest.main()
