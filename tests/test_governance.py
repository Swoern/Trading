"""Tests voor src/governance.py — leaderboard + lifecycle + decay."""
import sys
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import governance as g  # noqa: E402


class DecayTests(unittest.TestCase):
    def test_decay_tiers(self):
        self.assertEqual(g.confidence_decay(10), 1.0)
        self.assertEqual(g.confidence_decay(45), 0.98)
        self.assertEqual(g.confidence_decay(75), 0.95)
        self.assertEqual(g.confidence_decay(120), 0.90)
        self.assertEqual(g.confidence_decay(None), 1.0)


class StatusTests(unittest.TestCase):
    def test_candidate_too_few(self):
        self.assertEqual(g.assign_status(3, 0.5, 1.0), "CANDIDATE")

    def test_retired_significant_loss(self):
        self.assertEqual(g.assign_status(50, -0.3, -3.0), "RETIRED")

    def test_active_significant_win(self):
        self.assertEqual(g.assign_status(40, 0.4, 2.5), "ACTIVE")

    def test_deprecated_proven_negative(self):
        self.assertEqual(g.assign_status(35, -0.1, -1.0), "DEPRECATED")

    def test_watchlist_positive_not_significant(self):
        self.assertEqual(g.assign_status(35, 0.2, 1.0), "WATCHLIST")

    def test_challenger_provisional(self):
        self.assertEqual(g.assign_status(15, 0.2, 0.5), "CHALLENGER")


class LeaderboardTests(unittest.TestCase):
    def test_ranking_and_fields(self):
        now = datetime(2026, 6, 17, tzinfo=timezone.utc)
        recent = (now - timedelta(days=2)).isoformat()
        data = {
            "winner": [{"result_pct": 2.0, "closed_at": recent} for _ in range(40)]
                      + [{"result_pct": -1.0, "closed_at": recent} for _ in range(10)],
            "loser": [{"result_pct": -1.0, "closed_at": recent} for _ in range(40)]
                     + [{"result_pct": 0.5, "closed_at": recent} for _ in range(10)],
        }
        board = g.strategy_leaderboard(data, now=now)
        self.assertEqual(board[0]["strategy"], "winner")  # hoogste score bovenaan
        self.assertGreater(board[0]["governance_score"], board[1]["governance_score"])
        self.assertIn(board[0]["status"], ("ACTIVE", "WATCHLIST"))
        self.assertEqual(board[1]["status"], "RETIRED")
        # Contributie telt op tot ~100%
        self.assertAlmostEqual(sum(r["contribution_pct"] for r in board), 100.0, delta=0.5)

    def test_old_trades_get_decay(self):
        now = datetime(2026, 6, 17, tzinfo=timezone.utc)
        old = (now - timedelta(days=100)).isoformat()
        data = {"oud": [{"result_pct": 1.0, "closed_at": old} for _ in range(40)]}
        board = g.strategy_leaderboard(data, now=now)
        self.assertEqual(board[0]["confidence_decay"], 0.90)


if __name__ == "__main__":
    unittest.main()
