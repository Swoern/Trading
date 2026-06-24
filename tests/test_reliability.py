"""Tests voor src/reliability.py — retry/backoff-gedrag zonder echte netwerkcalls."""
import sys
import unittest
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reliability import http_get_json, with_retry, RETRYABLE_STATUS  # noqa: E402


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {"ok": True}

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.exceptions.HTTPError(f"HTTP {self.status_code}")
            err.response = self
            raise err

    def json(self):
        return self._json


class _FakeSession:
    """Levert vooraf bepaalde responses/excepties per opeenvolgende get()-call."""

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class HttpGetJsonTests(unittest.TestCase):
    def setUp(self):
        self.slept = []

    def _sleep(self, secs):
        self.slept.append(secs)

    def test_succeeds_first_try(self):
        session = _FakeSession([_FakeResponse(200, {"value": 42})])
        result = http_get_json("http://x", session=session, sleep=self._sleep)
        self.assertEqual(result, {"value": 42})
        self.assertEqual(session.calls, 1)
        self.assertEqual(self.slept, [])

    def test_retries_on_retryable_status_then_succeeds(self):
        session = _FakeSession([
            _FakeResponse(503),
            _FakeResponse(200, {"ok": 1}),
        ])
        result = http_get_json("http://x", session=session, sleep=self._sleep,
                               retries=2, base_delay=1.0)
        self.assertEqual(result, {"ok": 1})
        self.assertEqual(session.calls, 2)
        # Eén backoff-slaap van base_delay * 2**0 = 1.0
        self.assertEqual(self.slept, [1.0])

    def test_retries_on_connection_error(self):
        session = _FakeSession([
            requests.exceptions.ConnectionError("boom"),
            _FakeResponse(200, {"ok": True}),
        ])
        result = http_get_json("http://x", session=session, sleep=self._sleep)
        self.assertEqual(result, {"ok": True})
        self.assertEqual(session.calls, 2)

    def test_does_not_retry_on_404(self):
        session = _FakeSession([
            _FakeResponse(404),
            _FakeResponse(200, {"unreached": True}),
        ])
        with self.assertRaises(requests.exceptions.HTTPError):
            http_get_json("http://x", session=session, sleep=self._sleep)
        self.assertEqual(session.calls, 1)          # geen retry
        self.assertEqual(self.slept, [])

    def test_exhausts_retries_and_raises(self):
        session = _FakeSession([_FakeResponse(500) for _ in range(5)])
        with self.assertRaises(Exception):
            http_get_json("http://x", session=session, sleep=self._sleep,
                          retries=2, base_delay=0.5)
        self.assertEqual(session.calls, 3)          # 1 + 2 retries
        # Exponential backoff: 0.5, 1.0
        self.assertEqual(self.slept, [0.5, 1.0])

    def test_429_is_retryable(self):
        self.assertIn(429, RETRYABLE_STATUS)


class WithRetryTests(unittest.TestCase):
    def test_decorator_retries_then_succeeds(self):
        slept = []
        calls = {"n": 0}

        @with_retry(retries=3, base_delay=0.1, sleep=slept.append)
        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ValueError("nog niet")
            return "klaar"

        self.assertEqual(flaky(), "klaar")
        self.assertEqual(calls["n"], 3)
        self.assertEqual(slept, [0.1, 0.2])

    def test_decorator_raises_after_exhaustion(self):
        @with_retry(retries=1, base_delay=0.0, sleep=lambda _: None)
        def always_fails():
            raise RuntimeError("kapot")

        with self.assertRaises(RuntimeError):
            always_fails()


if __name__ == "__main__":
    unittest.main()
