"""Tests voor src/meta_label.py — numpy-only meta-labeling filter."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import meta_label  # noqa: E402


def _make_samples(n_per_class=60):
    """Twee duidelijk scheidbare contexten: 'goed' wint vaak, 'slecht' verliest vaak."""
    samples = []
    for i in range(n_per_class):
        # 'goede' context: wint 85% van de tijd
        samples.append({
            "context": {"asset": "BTC-USD", "strategy": "trend_long",
                        "regime": "trend_up", "rsi_zone": "rsi_neutral"},
            "outcome": 1 if i % 100 < 85 else 0,
            "weight": 1.0,
        })
        # 'slechte' context: wint 15% van de tijd
        samples.append({
            "context": {"asset": "BTC-USD", "strategy": "range_short",
                        "regime": "trend_up", "rsi_zone": "rsi_overbought"},
            "outcome": 1 if i % 100 < 15 else 0,
            "weight": 1.0,
        })
    return samples


class FeatureExtractionTests(unittest.TestCase):
    def test_excludes_freeform_keys(self):
        feats = meta_label.extract_features({
            "asset": "BTC-USD", "context_key": "abc|123", "timeframe": "300s",
            "regime": "trend_up",
        })
        self.assertIn("asset=BTC-USD", feats)
        self.assertIn("regime=trend_up", feats)
        self.assertNotIn("context_key=abc|123", feats)
        self.assertFalse(any(f.startswith("timeframe=") for f in feats))

    def test_bool_becomes_int_feature(self):
        feats = meta_label.extract_features({"is_weekend": True})
        self.assertIn("is_weekend=1", feats)


class TrainingTests(unittest.TestCase):
    def test_none_with_too_few_samples(self):
        self.assertIsNone(meta_label.train_model([{"context": {"a": "b"}, "outcome": 1}]))

    def test_learns_separable_pattern(self):
        model = meta_label.train_model(_make_samples(), min_samples=40)
        self.assertIsNotNone(model)

        goed = model.predict_proba({"asset": "BTC-USD", "strategy": "trend_long",
                                    "regime": "trend_up", "rsi_zone": "rsi_neutral"})
        slecht = model.predict_proba({"asset": "BTC-USD", "strategy": "range_short",
                                      "regime": "trend_up", "rsi_zone": "rsi_overbought"})
        self.assertGreater(goed, 0.6)
        self.assertLess(slecht, 0.4)
        self.assertGreater(goed, slecht)

    def test_serialization_roundtrip(self):
        model = meta_label.train_model(_make_samples(), min_samples=40)
        restored = meta_label.MetaLabelModel.from_dict(model.to_dict())
        ctx = {"asset": "BTC-USD", "strategy": "trend_long", "regime": "trend_up"}
        self.assertAlmostEqual(model.predict_proba(ctx), restored.predict_proba(ctx), places=6)


class ScoreApiTests(unittest.TestCase):
    def tearDown(self):
        meta_label.reset_cache()

    def test_score_degrades_without_data(self):
        meta_label.reset_cache()
        # Geen DB-samples beschikbaar in deze omgeving → neutraal.
        orig = meta_label._load_samples_from_db
        meta_label._load_samples_from_db = lambda: []
        try:
            result = meta_label.meta_label_score({"asset": "BTC-USD"})
        finally:
            meta_label._load_samples_from_db = orig
        self.assertFalse(result["available"])
        self.assertEqual(result["prob_win"], 0.5)

    def test_score_available_with_data(self):
        meta_label.reset_cache()
        orig = meta_label._load_samples_from_db
        meta_label._load_samples_from_db = lambda: _make_samples()
        try:
            result = meta_label.meta_label_score(
                {"asset": "BTC-USD", "strategy": "trend_long", "regime": "trend_up",
                 "rsi_zone": "rsi_neutral"},
                min_samples=40,
            )
        finally:
            meta_label._load_samples_from_db = orig
        self.assertTrue(result["available"])
        self.assertGreater(result["prob_win"], 0.5)


if __name__ == "__main__":
    unittest.main()
