"""Tests voor src/markets.py — configureerbare marktlijst."""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import markets  # noqa: E402


class MarketsTests(unittest.TestCase):
    def setUp(self):
        self._orig = os.environ.get("TRADEAI_MARKETS")

    def tearDown(self):
        if self._orig is None:
            os.environ.pop("TRADEAI_MARKETS", None)
        else:
            os.environ["TRADEAI_MARKETS"] = self._orig

    def test_default_includes_core_and_majors(self):
        os.environ.pop("TRADEAI_MARKETS", None)
        m = markets.get_markets()
        for core in ("BTC-USD", "ETH-USD", "SOL-USD"):
            self.assertIn(core, m)
        self.assertGreaterEqual(len(m), 4)
        self.assertEqual(m[0], markets.ANCHOR_ASSET)

    def test_env_override(self):
        os.environ["TRADEAI_MARKETS"] = "eth-usd, sol-usd"
        m = markets.get_markets()
        # Anker wordt altijd toegevoegd, ook al staat het niet in de env-lijst.
        self.assertIn("BTC-USD", m)
        self.assertIn("ETH-USD", m)
        self.assertIn("SOL-USD", m)

    def test_env_dedupes_and_uppercases(self):
        os.environ["TRADEAI_MARKETS"] = "btc-usd,btc-usd,xrp-usd"
        m = markets.get_markets()
        self.assertEqual(m.count("BTC-USD"), 1)
        self.assertIn("XRP-USD", m)

    def test_satellite_detection(self):
        self.assertFalse(markets.is_satellite("BTC-USD"))
        self.assertFalse(markets.is_satellite("SOL-USD"))
        self.assertTrue(markets.is_satellite("XRP-USD"))
        self.assertTrue(markets.is_satellite("DOGE-USD"))

    def test_empty_env_falls_back_to_anchor(self):
        os.environ["TRADEAI_MARKETS"] = "   "
        m = markets.get_markets()
        self.assertIn(markets.ANCHOR_ASSET, m)


if __name__ == "__main__":
    unittest.main()
