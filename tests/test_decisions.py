"""Tests voor Fase 2 — slimmere beslissingen: Kelly-sizing, committee-weging, breaker."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.auto_trader as auto_trader  # noqa: E402


def _trades(result_pcts):
    return [{"result_pct": r} for r in result_pcts]


class KellyFractionTests(unittest.TestCase):
    def test_none_with_too_few_trades(self):
        self.assertIsNone(auto_trader._kelly_fractie(_trades([1.0] * 5)))

    def test_quarter_kelly_known_values(self):
        # 10 winsten +2%, 10 verliezen -1% → p=0.5, b=2, f*=0.25, kwart=0.0625
        trades = _trades([2.0] * 10 + [-1.0] * 10)
        kelly = auto_trader._kelly_fractie(trades)
        self.assertIsNotNone(kelly)
        self.assertAlmostEqual(kelly["win_rate"], 0.5, places=3)
        self.assertAlmostEqual(kelly["payoff_b"], 2.0, places=3)
        self.assertAlmostEqual(kelly["kelly_full"], 0.25, places=4)
        self.assertAlmostEqual(kelly["kelly_fraction"], 0.0625, places=4)

    def test_none_without_losses(self):
        self.assertIsNone(auto_trader._kelly_fractie(_trades([1.0] * 20)))


class KellyMultiplierTests(unittest.TestCase):
    def setUp(self):
        self._orig = auto_trader.get_auto_trades

    def tearDown(self):
        auto_trader.get_auto_trades = self._orig

    def test_strong_edge_caps_at_max(self):
        auto_trader.get_auto_trades = lambda status=None: _trades([2.0] * 12 + [-1.0] * 8)
        mult, reason, details = auto_trader._kelly_size_multiplier({})
        self.assertEqual(mult, auto_trader.MAX_CONFIDENCE_SIZE_MULTIPLIER)

    def test_weak_edge_floors(self):
        # p=0.3, b=1 → f* negatief → vloer
        auto_trader.get_auto_trades = lambda status=None: _trades([1.0] * 6 + [-1.0] * 14)
        mult, reason, details = auto_trader._kelly_size_multiplier({})
        self.assertEqual(mult, auto_trader.KELLY_MIN_MULTIPLIER)

    def test_none_with_too_few(self):
        auto_trader.get_auto_trades = lambda status=None: _trades([1.0, -1.0])
        mult, reason, details = auto_trader._kelly_size_multiplier({})
        self.assertIsNone(mult)


class CommitteeWeightTests(unittest.TestCase):
    def test_calm_regime_equal_weights(self):
        w = auto_trader._committee_weights("range")
        self.assertTrue(all(v == 1.0 for v in w.values()))

    def test_panic_regime_boosts_risk_and_cost(self):
        w = auto_trader._committee_weights("panic")
        self.assertGreater(w["Risk Manager Agent"], 1.0)
        self.assertGreater(w["Cost/Edge Agent"], 1.0)
        self.assertEqual(w["Setup Agent"], 1.0)


class SoftDrawdownBreakerTests(unittest.TestCase):
    def setUp(self):
        self._saldo = auto_trader._huidig_saldo
        self._losses = auto_trader._opeenvolgende_verliezen
        self._dag = auto_trader._dagverlies_vandaag
        self._trades_fn = auto_trader.get_auto_trades
        self._maxpos = auto_trader._effectieve_max_positie
        auto_trader._huidig_saldo = lambda: 10_000.0
        auto_trader._opeenvolgende_verliezen = lambda: 0
        auto_trader.get_auto_trades = lambda status=None: []  # geen Kelly-data
        auto_trader._effectieve_max_positie = lambda saldo: saldo  # geen DB-afhankelijke cap

    def tearDown(self):
        auto_trader._huidig_saldo = self._saldo
        auto_trader._opeenvolgende_verliezen = self._losses
        auto_trader._dagverlies_vandaag = self._dag
        auto_trader.get_auto_trades = self._trades_fn
        auto_trader._effectieve_max_positie = self._maxpos

    def _signal(self):
        return {
            "asset": "BTC-USD",
            "entry": 100.0,
            "stop_loss": 95.0,
            "_confidence": {"score": 94},
            "_external_context": {"spread": {"spread_pct": 0.01}},
        }

    def test_no_breaker_below_threshold(self):
        auto_trader._dagverlies_vandaag = lambda: -500.0  # 5% < 15%
        kapitaal, details = auto_trader._bereken_positiegrootte(self._signal())
        self.assertFalse(details["soft_drawdown_breaker"])
        self.assertEqual(kapitaal, 2000.0)

    def test_breaker_halves_position(self):
        auto_trader._dagverlies_vandaag = lambda: -1600.0  # 16% ≥ 15%
        kapitaal, details = auto_trader._bereken_positiegrootte(self._signal())
        self.assertTrue(details["soft_drawdown_breaker"])
        # 10000 * 1% * 0.8 (BTC) * 1.25 * 0.5 = 50 risico → 50 / 0.05 = 1000
        self.assertEqual(kapitaal, 1000.0)


if __name__ == "__main__":
    unittest.main()
