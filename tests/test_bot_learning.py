import os
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd


def _sample_ohlcv(rows=90):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    data = []
    price = 100.0
    for i in range(rows):
        price *= 1 + (0.002 if i % 7 else -0.001)
        open_ = price * (1 - 0.001)
        close = price
        high = max(open_, close) * 1.003
        low = min(open_, close) * 0.997
        data.append({
            "timestamp": base + timedelta(minutes=15 * i),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1000 + i * 10,
        })
    return pd.DataFrame(data)


class FeatureEngineeringTests(unittest.TestCase):
    def test_extra_features_are_added(self):
        from src.features import add_all_features

        df = add_all_features(_sample_ohlcv())
        expected = {
            "ma10", "ma30", "ema10", "ema30",
            "yesterday_close", "yesterday_close_logr", "yesterday_volume_logr",
            "return_1", "log_return_1", "gap_pct",
            "momentum_3", "momentum_10", "momentum_20",
            "volatility_10", "volatility_20", "volatility_30",
            "zscore_20", "obv", "obv_slope", "skew_20",
            "day_of_week", "day_of_month", "month_number", "hour_utc",
            "is_weekend", "session_bucket", "fear_greed_change",
        }
        self.assertTrue(expected.issubset(set(df.columns)))
        self.assertFalse(pd.isna(df["momentum_10"].iloc[-1]))
        self.assertFalse(pd.isna(df["zscore_20"].iloc[-1]))

    def test_market_regime_classifies_trend_and_filters_strategy(self):
        from src.market_regime import classify_market_regime, strategy_allowed_in_regime

        df = _sample_ohlcv(rows=120)
        regime = classify_market_regime(df, htf_trend="uptrend", fear_greed=50)

        self.assertIn(regime["label"], {
            "trend_up", "breakout_up", "range", "squeeze", "reversal", "euphoria"
        })
        ok, _ = strategy_allowed_in_regime("trend_long", {"label": "trend_up"})
        self.assertTrue(ok)
        ok, _ = strategy_allowed_in_regime("trend_long", {"label": "trend_down"})
        self.assertFalse(ok)


class BotLearningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "tradeai-test.db"
        os.environ["TRADEAI_DB_PATH"] = str(self.db_path)

        import src.database as database
        database.DB_PATH = self.db_path
        database.init_database()

        import src.bot_learning as bot_learning

        self.database = database
        self.learning = bot_learning

    def tearDown(self):
        self.tmp.cleanup()

    def _context(self, asset="BTC-USD", strategy="trend_long"):
        context = {
            "asset": asset,
            "strategy": strategy,
            "direction": "long",
            "timeframe": "300s",
            "trend": "uptrend",
            "volatility": "normaal",
            "volume": "normaal",
            "rsi_zone": "rsi_neutral",
            "adx_zone": "adx_trend",
            "range_zone": "near_support",
            "rr_zone": "rr_2_25",
            "hour_block": "utc_12_18",
        }
        context["context_key"] = self.learning.make_context_key(context)
        return context

    def test_migrations_are_idempotent(self):
        self.database.init_database()
        self.database.init_database()

        conn = self.database.get_connection()
        columns = {row[1] for row in conn.execute("PRAGMA table_info(paper_trades)").fetchall()}
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        conn.close()

        self.assertIn("market_context", columns)
        self.assertIn("learning_score", columns)
        self.assertIn("gross_result_euro", columns)
        self.assertIn("costs_euro", columns)
        self.assertIn("bot_learning_samples", tables)
        self.assertIn("bot_learning_decisions", tables)
        self.assertIn("bot_api_health", tables)

    def test_score_is_neutral_with_too_little_data(self):
        score = self.learning.score_setup(self._context())

        self.assertEqual(score["score"], self.learning.NEUTRAL_SCORE)
        self.assertEqual(score["sample_count"], 0)
        self.assertFalse(score["should_block"])

    def test_market_context_includes_feature_buckets(self):
        import src.bot_learning as bot_learning

        signal = {
            "strategie": "trend_long",
            "richting": "long",
            "risk_reward": 2.0,
        }
        context = bot_learning.build_market_context(
            "BTC-USD",
            signal,
            _sample_ohlcv(),
            analyse={"trend": "uptrend", "volatiliteit": "normaal", "volume_status": "normaal"},
            timeframe="900s",
            htf_trend="uptrend",
            fear_greed=50,
        )

        for key in (
            "momentum_zone",
            "feature_volatility_zone",
            "zscore_zone",
            "obv_zone",
            "gap_zone",
            "skew_zone",
            "session_bucket",
            "weekend_zone",
            "regime_label",
            "regime_direction",
        ):
            self.assertIn(key, context)
            self.assertTrue(context[key])

    def test_bad_context_blocks_only_after_enough_samples(self):
        context = self._context()
        for _ in range(self.learning.MIN_BLOCK_SAMPLES - 1):
            self.learning.record_learning_sample(
                context, source="live", result_pct=-1.0, refresh=False
            )
        self.learning.refresh_scorecards()
        early_score = self.learning.score_setup(context)
        self.assertFalse(early_score["should_block"])

        self.learning.record_learning_sample(
            context, source="live", result_pct=-1.0, refresh=False
        )
        self.learning.refresh_scorecards()
        score = self.learning.score_setup(context)

        self.assertLess(score["score"], self.learning.BLOCK_SCORE_THRESHOLD)
        self.assertTrue(score["should_block"])

    def test_reset_learning_does_not_delete_trades(self):
        context = self._context()
        trade_id = self.database.add_paper_trade(
            asset="BTC-USD",
            direction="long",
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            source="auto",
            strategy="trend_long",
            market_context=context,
        )
        self.learning.record_learning_sample(
            context, source="live", result_pct=-1.0, trade_id=trade_id
        )

        deleted = self.learning.reset_learning(asset="BTC-USD", reason="unit test")
        score = self.learning.score_setup(context)

        self.assertEqual(deleted, 1)
        self.assertIsNotNone(self.database.get_trade_by_id(trade_id))
        self.assertEqual(score["sample_count"], 0)

    def test_api_failure_leaves_open_trade_untouched(self):
        import src.auto_trader as auto_trader

        context = self._context()
        trade_id = self.database.add_paper_trade(
            asset="BTC-USD",
            direction="long",
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            source="auto",
            strategy="trend_long",
            market_context=context,
        )

        original_fetch = auto_trader.fetch_live_crypto_candles

        def failing_fetch(*args, **kwargs):
            raise RuntimeError("Coinbase offline")

        auto_trader.fetch_live_crypto_candles = failing_fetch
        try:
            auto_trader._controleer_open_trades()
        finally:
            auto_trader.fetch_live_crypto_candles = original_fetch

        trade = self.database.get_trade_by_id(trade_id)
        conn = self.database.get_connection()
        decision = conn.execute(
            "SELECT * FROM bot_learning_decisions ORDER BY id DESC LIMIT 1"
        ).fetchone()
        health = conn.execute(
            "SELECT * FROM bot_api_health WHERE asset = 'BTC-USD'"
        ).fetchone()
        conn.close()

        self.assertEqual(trade["status"], "open")
        self.assertEqual(decision["action"], "data_unavailable")
        self.assertEqual(health["status"], "down")

    def test_open_trade_ignores_candles_before_entry_time(self):
        import pandas as pd
        import src.auto_trader as auto_trader

        context = self._context()
        trade_id = self.database.add_paper_trade(
            asset="BTC-USD",
            direction="long",
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            source="auto",
            strategy="trend_long",
            market_context=context,
        )
        trade = self.database.get_trade_by_id(trade_id)
        opened_at = datetime.fromisoformat(trade["created_at"]).astimezone(timezone.utc)

        df = pd.DataFrame([
            {
                "timestamp": opened_at - timedelta(minutes=1),
                "open": 100.0,
                "high": 101.0,
                "low": 90.0,
                "close": 100.0,
                "volume": 1000.0,
            },
            {
                "timestamp": opened_at + timedelta(seconds=1),
                "open": 100.0,
                "high": 104.0,
                "low": 99.0,
                "close": 101.0,
                "volume": 1000.0,
            },
        ])

        original_fetch = auto_trader.fetch_live_crypto_candles
        auto_trader.fetch_live_crypto_candles = lambda *args, **kwargs: df
        try:
            auto_trader._controleer_open_trades()
        finally:
            auto_trader.fetch_live_crypto_candles = original_fetch

        trade = self.database.get_trade_by_id(trade_id)
        self.assertEqual(trade["status"], "open")

    def test_time_stop_counts_strategy_candles_not_one_minute_checks(self):
        import pandas as pd
        import src.auto_trader as auto_trader

        context = self._context()
        trade_id = self.database.add_paper_trade(
            asset="BTC-USD",
            direction="long",
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            source="auto",
            strategy="trend_long",
            market_context=context,
        )
        opened_at = datetime.now(timezone.utc) - timedelta(minutes=30)
        conn = self.database.get_connection()
        conn.execute(
            "UPDATE paper_trades SET created_at = ? WHERE id = ?",
            (opened_at.isoformat(), trade_id),
        )
        conn.commit()
        conn.close()

        df = pd.DataFrame([
            {
                "timestamp": opened_at + timedelta(minutes=i),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1000.0,
            }
            for i in range(31)
        ])

        trade = self.database.get_trade_by_id(trade_id)
        self.assertEqual(auto_trader._candles_open_sinds_entry(trade, df), 6)

    def test_active_min_rr_keeps_learned_higher_threshold(self):
        import src.auto_trader as auto_trader

        self.assertEqual(auto_trader._actieve_min_rr({"min_rr": 2.4}), 2.4)
        self.assertEqual(auto_trader._actieve_min_rr({"min_rr": 1.6}), 2.0)

    def test_trade_replay_measures_progress_in_r(self):
        import src.auto_trader as auto_trader

        opened_at = datetime.now(timezone.utc) - timedelta(minutes=10)
        trade = {
            "created_at": opened_at.isoformat(),
            "entry_price": 100.0,
            "stop_loss": 95.0,
            "take_profit": 110.0,
            "direction": "long",
        }
        df = pd.DataFrame([
            {
                "timestamp": opened_at + timedelta(minutes=i),
                "open": 100.0,
                "high": 105.5 if i == 3 else 101.0,
                "low": 97.5 if i == 1 else 99.0,
                "close": 104.0,
                "volume": 1000.0,
            }
            for i in range(6)
        ])

        replay = auto_trader._trade_progress_metrics(trade, df)

        self.assertGreaterEqual(replay["max_favorable_r"], 1.0)
        self.assertGreaterEqual(replay["max_adverse_r"], 0.5)
        self.assertEqual(replay["candles_to_1r"], 3)

    def test_anomaly_monitor_flags_time_stop_worse_than_short_sl(self):
        import src.auto_trader as auto_trader

        context = self._context(asset="ETH-USD", strategy="macd_bear_div")
        trade_id = self.database.add_paper_trade(
            asset="ETH-USD",
            direction="short",
            entry_price=100.0,
            stop_loss=105.0,
            take_profit=90.0,
            source="auto",
            strategy="macd_bear_div",
            market_context=context,
        )
        trade = self.database.get_trade_by_id(trade_id)

        anomalies = auto_trader._check_trade_anomalies(
            trade,
            exit_price=106.0,
            exit_reason="time_stop",
            replay={"candles_open": 3},
        )

        self.assertIn("time_stop_worse_than_short_sl", anomalies)
        conn = self.database.get_connection()
        decision = conn.execute(
            "SELECT action FROM bot_learning_decisions ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        self.assertEqual(decision["action"], "trade_anomaly")

    def test_entry_execution_gate_blocks_chasing(self):
        import src.auto_trader as auto_trader

        signal = {
            "richting": "long",
            "entry": 100.0,
            "stop_loss": 95.0,
            "take_profit": 110.0,
        }

        ok, reason, details = auto_trader._entry_execution_gate(signal, 102.0)

        self.assertFalse(ok)
        self.assertIn("doorgeschoten", reason)
        self.assertGreater(details["favorable_r"], 0.35)

    def test_pre_trade_checklist_blocks_weak_learning(self):
        import src.auto_trader as auto_trader

        original_btc = dict(auto_trader.BTC_MARKT)
        auto_trader.BTC_MARKT.update({"mode": "active"})
        try:
            result = auto_trader._pre_trade_checklist(
                asset="BTC-USD",
                setup={"strategie": "trend_long"},
                regime_ok=True,
                htf_ok=True,
                proof_details={"backtest_pnl": 1.0},
                score_info={"score": 0.2},
                quality={"score": 80},
                confidence={"score": 85},
                market_context={"spread": {"spread_pct": 0.01}},
                strategy_profile={"action": "neutral"},
                time_profile={"action": "neutral"},
            )
        finally:
            auto_trader.BTC_MARKT.clear()
            auto_trader.BTC_MARKT.update(original_btc)

        self.assertFalse(result["passed"])
        self.assertIn("leerscore te zwak", result["hard_blocks"])

    def test_confidence_calibration_caps_overconfident_bucket(self):
        import src.auto_trader as auto_trader

        context = self._context(asset="BTC-USD", strategy="trend_long")
        context["confidence_bucket"] = "80-89"
        for _ in range(5):
            trade_id = self.database.add_paper_trade(
                asset="BTC-USD",
                direction="long",
                entry_price=100.0,
                stop_loss=95.0,
                take_profit=110.0,
                source="auto",
                strategy="trend_long",
                market_context=context,
            )
            self.database.close_paper_trade(trade_id, 95.0, "sl")

        profile = auto_trader._confidence_calibration_profile("BTC-USD", "trend_long", 85)

        self.assertEqual(profile["action"], "caution")
        self.assertEqual(profile["confidence_adjustment"], -15)
        self.assertEqual(profile["size_cap"], 0.5)

    def test_stop_loss_takes_precedence_over_smart_exit(self):
        import pandas as pd
        import src.auto_trader as auto_trader

        context = self._context(asset="ETH-USD", strategy="macd_bear_div")
        trade_id = self.database.add_paper_trade(
            asset="ETH-USD",
            direction="short",
            entry_price=100.0,
            stop_loss=105.0,
            take_profit=90.0,
            capital=1000.0,
            source="auto",
            strategy="macd_bear_div",
            market_context=context,
        )
        opened_at = datetime.now(timezone.utc) - timedelta(minutes=30)
        conn = self.database.get_connection()
        conn.execute(
            "UPDATE paper_trades SET created_at = ? WHERE id = ?",
            (opened_at.isoformat(), trade_id),
        )
        conn.commit()
        conn.close()

        df = pd.DataFrame([
            {
                "timestamp": opened_at + timedelta(minutes=i),
                "open": 100.0,
                "high": 106.0 if i == 1 else 101.0,
                "low": 99.0,
                "close": 106.0 if i == 19 else 100.0,
                "volume": 1000.0,
            }
            for i in range(20)
        ])
        df.loc[df.index[-1], "timestamp"] = datetime.now(timezone.utc)

        original_fetch = auto_trader.fetch_live_crypto_candles
        original_update = auto_trader._update_leren
        original_smart_exit = auto_trader._controleer_slimme_exit
        original_notify_sl = auto_trader.melding_stop_loss

        def smart_exit_should_not_run(*args, **kwargs):
            raise AssertionError("smart exit ran before hard stop-loss")

        auto_trader.fetch_live_crypto_candles = lambda *args, **kwargs: df
        auto_trader._update_leren = lambda *args, **kwargs: None
        auto_trader._controleer_slimme_exit = smart_exit_should_not_run
        auto_trader.melding_stop_loss = lambda *args, **kwargs: None
        try:
            auto_trader._controleer_open_trades()
        finally:
            auto_trader.fetch_live_crypto_candles = original_fetch
            auto_trader._update_leren = original_update
            auto_trader._controleer_slimme_exit = original_smart_exit
            auto_trader.melding_stop_loss = original_notify_sl

        trade = self.database.get_trade_by_id(trade_id)
        self.assertEqual(trade["status"], "closed")
        self.assertEqual(trade["exit_reason"], "sl")
        self.assertEqual(trade["exit_price"], 105.0)

    def test_partial_take_profit_counts_in_closed_result(self):
        context = self._context()
        trade_id = self.database.add_paper_trade(
            asset="BTC-USD",
            direction="long",
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            capital=1000.0,
            source="auto",
            strategy="trend_long",
            market_context=context,
        )

        self.assertTrue(self.database.update_trade_partial_tp(trade_id, 25.0, 100.0))
        self.assertTrue(self.database.close_paper_trade(trade_id, 100.0, "sl"))

        trade = self.database.get_trade_by_id(trade_id)
        self.assertEqual(trade["gross_result_euro"], 25.0)
        self.assertGreater(trade["costs_euro"], 0)
        self.assertLess(trade["result_euro"], 25.0)
        self.assertGreater(trade["result_pct"], 0)

    def test_backtest_records_costs(self):
        from src.backtest import run_backtest

        df = _sample_ohlcv(rows=180)
        res = run_backtest(df, fee_rate=0.001, spread_rate=0.0005, slippage_rate=0.0005)

        self.assertIn("kosten_euro", res)
        self.assertIn("fee_rate", res)
        if res["trades"]:
            self.assertIn("costs_euro", res["trades"][0])

    def test_walk_forward_backtest_shape(self):
        from src.backtest import run_walk_forward_backtest

        res = run_walk_forward_backtest(_sample_ohlcv(rows=320), train_size=120, test_size=60)

        self.assertIn("windows", res)
        self.assertIn("total_trades", res)
        self.assertIn("winrate", res)

    def test_confidence_and_quality_shape(self):
        from src.setup_confidence import calculate_confidence
        from src.trade_quality import score_trade_quality

        signal = {
            "asset": "BTC-USD",
            "strategie": "trend_long",
            "richting": "long",
            "entry": 110.0,
            "stop_loss": 106.0,
            "take_profit": 118.0,
        }
        quality = score_trade_quality(_sample_ohlcv(rows=120), signal)
        confidence = calculate_confidence(
            regime_ok=True,
            htf_ok=True,
            proof_details={"backtest_pnl": 1.0, "backtest_winrate": 50},
            score_info={"score": 0.6, "sample_count": 5},
            quality=quality,
            market_context={"spread": {"spread_pct": 0.01}},
        )

        self.assertIn("score", quality)
        self.assertIn("score", confidence)
        self.assertGreaterEqual(confidence["score"], 0)

    def test_confidence_scales_position_size_with_safety_caps(self):
        import src.auto_trader as auto_trader

        original_saldo = auto_trader._huidig_saldo
        original_losses = auto_trader._opeenvolgende_verliezen
        auto_trader._huidig_saldo = lambda: 10_000.0
        auto_trader._opeenvolgende_verliezen = lambda: 0
        try:
            signal = {
                "asset": "BTC-USD",
                "entry": 100.0,
                "stop_loss": 95.0,
                "_confidence": {"score": 94},
                "_external_context": {"spread": {"spread_pct": 0.01}},
            }
            kapitaal, details = auto_trader._bereken_positiegrootte(signal)
            self.assertEqual(details["confidence_multiplier"], 1.25)
            self.assertEqual(kapitaal, 2000.0)

            signal["_confidence"] = {"score": 80}
            kapitaal, details = auto_trader._bereken_positiegrootte(signal)
            self.assertEqual(details["confidence_multiplier"], 0.5)
            self.assertEqual(kapitaal, 800.0)
        finally:
            auto_trader._huidig_saldo = original_saldo
            auto_trader._opeenvolgende_verliezen = original_losses

    def test_net_reward_gate_blocks_cost_thin_setup(self):
        import src.auto_trader as auto_trader

        signal = {
            "asset": "BTC-USD",
            "entry": 100.0,
            "stop_loss": 99.0,
            "take_profit": 100.2,
        }

        ok, reason, details = auto_trader._verwachte_netto_reward_gate(signal, 1000.0)

        self.assertFalse(ok)
        self.assertIn("Netto-edge te dun", reason)
        self.assertLess(details["expected_net_reward_euro"], 0)

    def test_net_reward_gate_allows_cost_covered_setup(self):
        import src.auto_trader as auto_trader

        signal = {
            "asset": "BTC-USD",
            "entry": 100.0,
            "stop_loss": 99.0,
            "take_profit": 101.0,
        }

        ok, reason, details = auto_trader._verwachte_netto_reward_gate(signal, 1000.0)

        self.assertTrue(ok)
        self.assertIn("Netto-edge ok", reason)
        self.assertGreater(details["expected_net_reward_euro"], 0)

    def test_agent_committee_passes_strong_candidate(self):
        import src.auto_trader as auto_trader

        original_btc = dict(auto_trader.BTC_MARKT)
        auto_trader.BTC_MARKT.update({"mode": "active", "fear_greed": 50})
        try:
            signal = {
                "asset": "BTC-USD",
                "strategie": "trend_long",
                "richting": "long",
                "entry": 100.0,
                "stop_loss": 95.0,
                "take_profit": 112.0,
                "_market_regime": {"label": "trend_up", "score": 85},
                "_pre_trade_checklist": {"score": 90, "passed": True, "hard_blocks": []},
                "_learning_context": {"htf_trend": "uptrend", "fear_greed_value": 50},
                "_learning_score": {"score": 0.65, "sample_count": 12, "should_block": False},
                "_strategy_profile": {"action": "neutral"},
                "_time_profile": {"action": "neutral"},
            }
            committee = auto_trader._agent_committee(
                signal,
                {"confidence_multiplier": 1.0},
                {
                    "expected_net_reward_euro": 10.0,
                    "min_expected_net_reward_euro": 1.5,
                    "reward_cost_ratio": 2.5,
                },
                {"block": False, "macro": {"risk": "normal"}, "cryptopanic": {"risk": "unknown"}, "rss": {"risk": "unknown"}},
            )
        finally:
            auto_trader.BTC_MARKT.clear()
            auto_trader.BTC_MARKT.update(original_btc)

        self.assertTrue(committee["passed"])
        self.assertGreaterEqual(committee["pass_count"], auto_trader.MIN_AGENT_PASSES)
        self.assertIn("Setup Agent", {vote["agent"] for vote in committee["votes"]})

    def test_news_watch_rss_detects_negative_headlines(self):
        import src.news_filter as news_filter

        class FakeResponse:
            content = b"""
            <rss><channel>
              <item><title>Major crypto exchange hack hits market</title></item>
              <item><title>SEC lawsuit expands against token issuer</title></item>
              <item><title>Network outage triggers liquidation fears</title></item>
            </channel></rss>
            """

            def raise_for_status(self):
                return None

        original_get = news_filter.requests.get
        original_urls = os.environ.get("TRADEAI_NEWS_RSS_URLS")
        original_token = os.environ.get("TRADEAI_CRYPTOPANIC_TOKEN")
        news_filter._NEWS_CACHE["snapshot"] = None
        news_filter._NEWS_CACHE["expires_at"] = 0.0
        os.environ["TRADEAI_NEWS_RSS_URLS"] = "https://example.test/rss"
        os.environ.pop("TRADEAI_CRYPTOPANIC_TOKEN", None)
        news_filter.requests.get = lambda *args, **kwargs: FakeResponse()
        try:
            snapshot = news_filter.news_watch_snapshot(force_refresh=True)
        finally:
            news_filter.requests.get = original_get
            if original_urls is None:
                os.environ.pop("TRADEAI_NEWS_RSS_URLS", None)
            else:
                os.environ["TRADEAI_NEWS_RSS_URLS"] = original_urls
            if original_token is None:
                os.environ.pop("TRADEAI_CRYPTOPANIC_TOKEN", None)
            else:
                os.environ["TRADEAI_CRYPTOPANIC_TOKEN"] = original_token
            news_filter._NEWS_CACHE["snapshot"] = None
            news_filter._NEWS_CACHE["expires_at"] = 0.0

        self.assertTrue(snapshot["block"])
        self.assertEqual(snapshot["rss"]["risk"], "high")

    def test_false_positive_memory_records_virtual_sample(self):
        import json
        import src.bot_learning as bot_learning

        context = self._context()
        setup = {
            "richting": "long",
            "entry": 100.0,
            "stop_loss": 95.0,
            "take_profit": 110.0,
        }
        created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        conn = self.database.get_connection()
        conn.execute(
            """
            INSERT INTO bot_learning_decisions
                (created_at, asset, strategy, action, context_key, context_json, reason, details_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at.isoformat(),
                "BTC-USD",
                "trend_long",
                "skipped",
                context["context_key"],
                json.dumps(context),
                "unit test skip",
                json.dumps({"setup": setup}),
            ),
        )
        conn.commit()
        conn.close()

        df = pd.DataFrame([
            {"timestamp": created_at + timedelta(minutes=5), "open": 100, "high": 104, "low": 99, "close": 101, "volume": 1},
            {"timestamp": created_at + timedelta(minutes=10), "open": 101, "high": 111, "low": 100, "close": 110, "volume": 1},
            {"timestamp": created_at + timedelta(minutes=15), "open": 110, "high": 112, "low": 109, "close": 111, "volume": 1},
        ])

        updated = bot_learning.evaluate_skipped_setups("BTC-USD", df)
        conn = self.database.get_connection()
        sample = conn.execute(
            "SELECT * FROM bot_learning_samples WHERE source='skipped_virtual'"
        ).fetchone()
        conn.close()

        self.assertEqual(updated, 1)
        self.assertIsNotNone(sample)
        self.assertGreater(sample["result_pct"], 0)

    def test_legacy_learning_schema_is_supported(self):
        conn = self.database.get_connection()
        conn.execute("DROP TABLE bot_learning")
        conn.execute(
            """
            CREATE TABLE bot_learning (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bijgewerkt TEXT NOT NULL,
                min_rr REAL DEFAULT 2.0,
                cooling_tot TEXT,
                winrate_recent REAL DEFAULT 0.5,
                aanpassingen TEXT DEFAULT '{}'
            )
            """
        )
        conn.commit()
        conn.close()

        self.database.save_bot_learning({
            "min_rr": 2.4,
            "cooling_tot": None,
            "winrate_recent": 0.3,
            "aanpassingen": {"test": True},
        })
        data = self.database.get_bot_learning()

        self.assertEqual(data["min_rr"], 2.4)
        self.assertEqual(data["winrate_recent"], 0.3)
        self.assertTrue(data["aanpassingen"]["test"])

    def test_bootstrap_thresholds(self):
        should_run, _ = self.learning.should_bootstrap("BTC-USD", "300s")
        self.assertTrue(should_run)

        context = self._context()
        for _ in range(self.learning.BOOTSTRAP_MIN_SAMPLES_PER_ASSET):
            self.learning.record_learning_sample(
                context, source="bootstrap", result_pct=1.0, refresh=False
            )
        self.learning.refresh_scorecards()

        conn = self.database.get_connection()
        conn.execute(
            """
            INSERT INTO bot_bootstrap_runs
                (run_at, asset, timeframe, samples_created, live_closed_count, reason)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (datetime.now(timezone.utc).isoformat(), "BTC-USD", "300s", 100, 0, "unit test"),
        )
        conn.commit()
        conn.close()

        should_run, _ = self.learning.should_bootstrap("BTC-USD", "300s")
        self.assertFalse(should_run)

        conn = self.database.get_connection()
        conn.execute(
            "UPDATE bot_bootstrap_runs SET run_at = ?",
            ((datetime.now(timezone.utc) - timedelta(days=31)).isoformat(),),
        )
        conn.commit()
        conn.close()

        should_run, reason = self.learning.should_bootstrap("BTC-USD", "300s")
        self.assertTrue(should_run)
        self.assertIn("ouder", reason)


if __name__ == "__main__":
    unittest.main()
