"""Tests voor src/monte_carlo.py — Monte Carlo robuustheidsanalyse."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.monte_carlo import monte_carlo  # noqa: E402


class MonteCarloTests(unittest.TestCase):
    def test_too_few_data(self):
        self.assertTrue(monte_carlo([10.0]).get("te_weinig_data"))

    def test_reproducible_with_seed(self):
        pnls = [50, -30, 40, -20, 60, -50, 30, -10, 45, -25]
        a = monte_carlo(pnls, seed=1)
        b = monte_carlo(pnls, seed=1)
        self.assertEqual(a["return_median_pct"], b["return_median_pct"])
        self.assertEqual(a["prob_profit"], b["prob_profit"])

    def test_winning_set_high_prob_profit(self):
        # Duidelijk winstgevende set -> hoge kans op winst.
        pnls = [50] * 8 + [-10] * 2
        mc = monte_carlo(pnls, n_iter=500)
        self.assertGreater(mc["prob_profit"], 0.9)
        self.assertGreater(mc["return_median_pct"], 0)

    def test_losing_set_low_prob_profit(self):
        pnls = [-50] * 8 + [10] * 2
        mc = monte_carlo(pnls, n_iter=500)
        self.assertLess(mc["prob_profit"], 0.1)
        self.assertLess(mc["return_median_pct"], 0)

    def test_ci_ordering(self):
        pnls = [30, -20, 25, -15, 40, -30, 20, -10]
        mc = monte_carlo(pnls)
        # 99%-band breder dan 95%-band
        self.assertLessEqual(mc["return_ci99"][0], mc["return_ci95"][0])
        self.assertGreaterEqual(mc["return_ci99"][1], mc["return_ci95"][1])


if __name__ == "__main__":
    unittest.main()
