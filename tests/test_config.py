"""Tests voor src/config.py — env-overschrijfbare knoppen + strategie-whitelist."""
import importlib
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self._env = {k: os.environ.get(k) for k in
                     ("TRADEAI_MIN_RR", "TRADEAI_STRATEGIES", "TRADEAI_RISICO_PCT")}

    def tearDown(self):
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        import src.config as config
        importlib.reload(config)

    def _reload(self):
        import src.config as config
        return importlib.reload(config)

    def test_defaults_unchanged(self):
        for k in ("TRADEAI_MIN_RR", "TRADEAI_STRATEGIES", "TRADEAI_RISICO_PCT"):
            os.environ.pop(k, None)
        c = self._reload()
        self.assertEqual(c.MIN_RR_START, 2.0)
        self.assertEqual(c.RISICO_PCT, 0.01)
        self.assertEqual(c.ENABLED_STRATEGIES, frozenset(c.ALL_STRATEGIES))

    def test_env_overrides(self):
        os.environ["TRADEAI_MIN_RR"] = "1.5"
        os.environ["TRADEAI_RISICO_PCT"] = "0.02"
        c = self._reload()
        self.assertEqual(c.MIN_RR_START, 1.5)
        self.assertEqual(c.RISICO_PCT, 0.02)

    def test_strategy_whitelist(self):
        os.environ["TRADEAI_STRATEGIES"] = "trend_long, trend_short , macd_bear_div"
        c = self._reload()
        self.assertEqual(c.ENABLED_STRATEGIES, frozenset({"trend_long", "trend_short", "macd_bear_div"}))

    def test_invalid_whitelist_falls_back_to_all(self):
        os.environ["TRADEAI_STRATEGIES"] = "bestaat_niet,ook_niet"
        c = self._reload()
        self.assertEqual(c.ENABLED_STRATEGIES, frozenset(c.ALL_STRATEGIES))


if __name__ == "__main__":
    unittest.main()
