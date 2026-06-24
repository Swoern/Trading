"""
watchdog.py — zelfherstel voor 24/7-draaien.

systemd's Restart=on-failure vangt een GECRASHTE bot, maar niet een bot die
HANGT (proces leeft, maar schrijft geen heartbeat meer). Deze watchdog checkt de
heartbeat-leeftijd in de DB en herstart de bot als die te oud is of de service
uit staat. Bedoeld om elke ~10 min via cron te draaien op de Pi.

Cron (admin-crontab, geen sudo nodig voor de cron zelf):
    */10 * * * * /usr/bin/python3 /home/admin/TRADEAI/tools/watchdog.py >> /home/admin/TRADEAI/watchdog.log 2>&1

De herstart gebruikt `sudo systemctl restart tradeai` (NOPASSWD reeds toegestaan).
"""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Heartbeat ouder dan dit = bot hangt (cyclus = 15 min, dus 2+ gemist).
MAX_HEARTBEAT_AGE = 35 * 60
SERVICE = "tradeai"
DB_PATH = os.environ.get("TRADEAI_DB_PATH", str(Path(__file__).resolve().parents[1] / "tradeai.db"))


def heartbeat_age_seconds(db_path: str = DB_PATH) -> float | None:
    """Leeftijd (s) van de laatste heartbeat, of None als onbekend."""
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        row = conn.execute(
            "SELECT checked_at FROM bot_api_health WHERE asset='_heartbeat'"
        ).fetchone()
        conn.close()
    except Exception:
        return None
    if not row or not row[0]:
        return None
    try:
        text = str(row[0]).replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds()
    except Exception:
        return None


def should_restart(age_seconds: float | None, service_active: bool,
                   max_age: int = MAX_HEARTBEAT_AGE) -> tuple[bool, str]:
    """Pure beslissing — testbaar zonder systeem."""
    if not service_active:
        return True, "service niet actief"
    if age_seconds is None:
        # Service draait maar (nog) geen heartbeat: niet herstarten (vermijd loop bij verse start).
        return False, "geen heartbeat-info; service actief — afwachten"
    if age_seconds > max_age:
        return True, f"heartbeat {int(age_seconds)}s oud (> {max_age}s) — bot hangt"
    return False, f"gezond (heartbeat {int(age_seconds)}s geleden)"


def _service_active(service: str = SERVICE) -> bool:
    try:
        out = subprocess.run(["systemctl", "is-active", service],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() == "active"
    except Exception:
        return False


def _restart(service: str = SERVICE) -> bool:
    try:
        subprocess.run(["sudo", "-n", "/bin/systemctl", "restart", service],
                       capture_output=True, text=True, timeout=30, check=True)
        return True
    except Exception:
        return False


def main() -> int:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    age = heartbeat_age_seconds()
    active = _service_active()
    restart, reden = should_restart(age, active)
    if restart:
        ok = _restart()
        print(f"[{now}] WATCHDOG herstart {SERVICE}: {reden} — {'gelukt' if ok else 'MISLUKT'}")
        return 0 if ok else 1
    print(f"[{now}] OK: {reden}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
