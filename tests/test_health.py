"""Tests voor de heartbeat-/health-overview-laag (Fase 1 betrouwbaarheid)."""
import os
import tempfile
import unittest
from pathlib import Path


class HealthOverviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "tradeai-test.db"
        os.environ["TRADEAI_DB_PATH"] = str(self.db_path)

        import src.database as database
        database.DB_PATH = self.db_path
        database.init_database()

        import src.bot_learning as bot_learning
        self.learning = bot_learning

    def tearDown(self):
        self.tmp.cleanup()

    def test_heartbeat_appears_in_overview(self):
        self.learning.mark_heartbeat("cyclus klaar")
        overview = self.learning.get_health_overview()
        self.assertIsNotNone(overview["heartbeat"])
        self.assertEqual(overview["heartbeat"]["status"], "alive")
        self.assertIsNotNone(overview["heartbeat"]["age_seconds"])
        self.assertLess(overview["heartbeat"]["age_seconds"], 60)

    def test_assets_separated_from_heartbeat(self):
        self.learning.mark_api_health("BTC-USD", "ok", "marktdata ontvangen")
        self.learning.mark_api_health("ETH-USD", "stale", "data oud")
        self.learning.mark_heartbeat()

        overview = self.learning.get_health_overview()
        assets = {a["asset"]: a for a in overview["assets"]}
        self.assertIn("BTC-USD", assets)
        self.assertIn("ETH-USD", assets)
        self.assertEqual(assets["ETH-USD"]["status"], "stale")
        # De heartbeat-rij hoort NIET tussen de assets te staan.
        self.assertNotIn(self.learning.HEARTBEAT_KEY, assets)

    def test_overview_empty_without_data(self):
        overview = self.learning.get_health_overview()
        self.assertIsNone(overview["heartbeat"])
        self.assertEqual(overview["assets"], [])


if __name__ == "__main__":
    unittest.main()
