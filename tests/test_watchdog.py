"""Tests voor de watchdog beslis-logica (zelfherstel 24/7)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from watchdog import should_restart, MAX_HEARTBEAT_AGE  # noqa: E402


class WatchdogTests(unittest.TestCase):
    def test_restart_when_service_inactive(self):
        restart, reden = should_restart(age_seconds=10, service_active=False)
        self.assertTrue(restart)
        self.assertIn("niet actief", reden)

    def test_restart_when_heartbeat_stale(self):
        restart, reden = should_restart(age_seconds=MAX_HEARTBEAT_AGE + 60, service_active=True)
        self.assertTrue(restart)
        self.assertIn("hangt", reden)

    def test_ok_when_fresh(self):
        restart, reden = should_restart(age_seconds=120, service_active=True)
        self.assertFalse(restart)
        self.assertIn("gezond", reden)

    def test_no_restart_when_no_heartbeat_but_active(self):
        # Verse start zonder heartbeat: niet herstarten (vermijd restart-loop).
        restart, reden = should_restart(age_seconds=None, service_active=True)
        self.assertFalse(restart)

    def test_boundary_just_under_max(self):
        restart, _ = should_restart(age_seconds=MAX_HEARTBEAT_AGE - 1, service_active=True)
        self.assertFalse(restart)


if __name__ == "__main__":
    unittest.main()
