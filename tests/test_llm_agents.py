"""Tests voor src/llm_agents.py — LLM-adviseslaag, met nep-client (geen echte API)."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import llm_agents  # noqa: E402


class _FakePart:
    def __init__(self, text):
        self.text = text


class _FakeMessages:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0

    def create(self, model=None, max_tokens=None, messages=None):
        self.calls += 1
        reply = self._replies.pop(0) if self._replies else ""

        class _Resp:
            content = [_FakePart(reply)]
        return _Resp()


class _FakeClient:
    def __init__(self, replies):
        self.messages = _FakeMessages(replies)


class EnabledTests(unittest.TestCase):
    def setUp(self):
        self._flag = os.environ.get("TRADEAI_LLM_ENABLED")
        self._key = os.environ.get("TRADEAI_ANTHROPIC_KEY")

    def tearDown(self):
        for name, val in (("TRADEAI_LLM_ENABLED", self._flag), ("TRADEAI_ANTHROPIC_KEY", self._key)):
            if val is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = val

    def test_disabled_by_default(self):
        os.environ.pop("TRADEAI_LLM_ENABLED", None)
        self.assertFalse(llm_agents.is_enabled())

    def test_enabled_needs_flag_and_key(self):
        os.environ["TRADEAI_LLM_ENABLED"] = "1"
        os.environ.pop("TRADEAI_ANTHROPIC_KEY", None)
        os.environ.pop("ANTHROPIC_API_KEY", None)
        self.assertFalse(llm_agents.is_enabled())  # flag aan maar geen key
        os.environ["TRADEAI_ANTHROPIC_KEY"] = "test-key"
        self.assertTrue(llm_agents.is_enabled())


class ParseAdviceTests(unittest.TestCase):
    def test_parses_and_clamps(self):
        text = ('Hier is mijn oordeel. {"bias": 2.5, "confidence": 1.8, '
                '"min_confidence_delta": 50, "rationale": "te heet"}')
        advice = llm_agents.parse_advice(text)
        self.assertEqual(advice["bias"], 1.0)          # geclamped naar max 1
        self.assertEqual(advice["confidence"], 1.0)    # geclamped naar max 1
        self.assertEqual(advice["min_confidence_delta"], llm_agents.MIN_CONF_DELTA_CAP)
        self.assertEqual(advice["rationale"], "te heet")

    def test_neutral_on_garbage(self):
        advice = llm_agents.parse_advice("geen json hier")
        self.assertEqual(advice["bias"], 0.0)
        self.assertEqual(advice["min_confidence_delta"], 0.0)

    def test_floor_clamp(self):
        advice = llm_agents.parse_advice('{"min_confidence_delta": -99}')
        self.assertEqual(advice["min_confidence_delta"], llm_agents.MIN_CONF_DELTA_FLOOR)


class BoundedDeltaTests(unittest.TestCase):
    def test_none_advice_zero(self):
        self.assertEqual(llm_agents.bounded_min_confidence_delta(None), 0.0)

    def test_reads_db_column_name(self):
        # DB-rij gebruikt 'min_conf_delta'
        self.assertEqual(llm_agents.bounded_min_confidence_delta({"min_conf_delta": 5.0}), 5.0)

    def test_clamps_out_of_range(self):
        self.assertEqual(
            llm_agents.bounded_min_confidence_delta({"min_confidence_delta": 999}),
            llm_agents.MIN_CONF_DELTA_CAP,
        )


class ConflictDetectionTests(unittest.TestCase):
    def test_parse_verdict(self):
        v = llm_agents.parse_analyst_verdict("Bla bla. VERDICT: bias=0.6, vertrouwen=0.8")
        self.assertIsNotNone(v)
        self.assertAlmostEqual(v["bias"], 0.6)
        self.assertAlmostEqual(v["confidence"], 0.8)

    def test_parse_verdict_none_when_missing(self):
        self.assertIsNone(llm_agents.parse_analyst_verdict("geen verdict hier"))

    def test_direction_split_detected(self):
        views = {
            "technical": "VERDICT: bias=0.7, vertrouwen=0.8",
            "sentiment": "VERDICT: bias=-0.6, vertrouwen=0.7",
            "risk": "VERDICT: bias=0.1, vertrouwen=0.5",
        }
        conflict = llm_agents.detect_panel_conflict(views, {"bias": 0.2})
        self.assertIn("analyst_direction_split", conflict)

    def test_coordinator_contradiction_detected(self):
        views = {
            "technical": "VERDICT: bias=0.6, vertrouwen=0.8",
            "sentiment": "VERDICT: bias=0.5, vertrouwen=0.7",
        }
        # Analisten duidelijk bullish, coördinator bearish → conflict.
        conflict = llm_agents.detect_panel_conflict(views, {"bias": -0.5})
        self.assertIn("coordinator_contradicts_analysts", conflict)

    def test_no_conflict_on_agreement(self):
        views = {
            "technical": "VERDICT: bias=0.5, vertrouwen=0.7",
            "sentiment": "VERDICT: bias=0.4, vertrouwen=0.6",
        }
        self.assertEqual(llm_agents.detect_panel_conflict(views, {"bias": 0.45}), [])

    def test_neutralize_only_makes_stricter(self):
        advice = {"bias": 0.8, "confidence": 0.9, "min_confidence_delta": -2.0, "rationale": "x"}
        veilig = llm_agents._neutraliseer_bij_conflict(advice, ["analyst_direction_split"])
        self.assertEqual(veilig["bias"], 0.0)
        self.assertLessEqual(veilig["confidence"], 0.30)
        self.assertGreaterEqual(veilig["min_confidence_delta"], 0.0)  # nooit losser
        self.assertIn("panel-conflict", veilig["rationale"])


class RunPanelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "tradeai-test.db"
        os.environ["TRADEAI_DB_PATH"] = str(self.db_path)
        import src.database as database
        database.DB_PATH = self.db_path
        database.init_database()
        self.database = database

    def tearDown(self):
        self.tmp.cleanup()

    def test_run_panel_with_fake_client_persists(self):
        client = _FakeClient([
            "Technisch: bullish. VERDICT: bias=0.5, vertrouwen=0.7",
            "Sentiment: neutraal. VERDICT: bias=0.1, vertrouwen=0.5",
            "Risico: matig. VERDICT: bias=-0.2, vertrouwen=0.6",
            '{"bias": 0.3, "confidence": 0.6, "min_confidence_delta": 3, "rationale": "voorzichtig long"}',
        ])
        result = llm_agents.run_panel("BTC-USD", {"fear_greed": 55, "regime": "trend_up"}, client=client)
        self.assertTrue(result["available"])
        self.assertEqual(result["bias"], 0.3)
        self.assertEqual(result["min_confidence_delta"], 3.0)
        self.assertEqual(client.messages.calls, 4)  # 3 analisten + coördinator

        # Advies is opgeslagen en weer op te halen.
        stored = self.database.get_latest_llm_advice("BTC-USD")
        self.assertIsNotNone(stored)
        self.assertEqual(stored["min_conf_delta"], 3.0)

    def test_run_panel_neutralizes_conflicting_panel(self):
        # Analisten oneens (sterk bull vs sterk bear) + coördinator gokt toch bullish.
        client = _FakeClient([
            "VERDICT: bias=0.8, vertrouwen=0.9",
            "VERDICT: bias=-0.7, vertrouwen=0.8",
            "VERDICT: bias=0.0, vertrouwen=0.5",
            '{"bias": 0.7, "confidence": 0.8, "min_confidence_delta": -2, "rationale": "long"}',
        ])
        result = llm_agents.run_panel("BTC-USD", {"fear_greed": 50}, client=client)
        self.assertTrue(result["available"])
        self.assertIn("analyst_direction_split", result["conflict"])
        self.assertEqual(result["bias"], 0.0)                 # geneutraliseerd
        self.assertGreaterEqual(result["min_confidence_delta"], 0.0)  # nooit losser

    def test_run_panel_disabled_returns_neutral(self):
        os.environ.pop("TRADEAI_LLM_ENABLED", None)
        result = llm_agents.run_panel("BTC-USD", {"fear_greed": 50})
        self.assertFalse(result["available"])
        self.assertEqual(result["min_confidence_delta"], 0.0)

    def test_run_panel_survives_client_error(self):
        class _BoomClient:
            class messages:
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError("API down")
        result = llm_agents.run_panel("BTC-USD", {"fear_greed": 50}, client=_BoomClient())
        self.assertFalse(result["available"])
        self.assertEqual(result["min_confidence_delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
