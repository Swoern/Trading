"""
secure_config.py - Secrets laden zonder ze hard aan code te koppelen.

Om compatibel te blijven leest de bot nog steeds telegram_config.json, maar
environment variables krijgen voorrang. Op Linux waarschuwen we als het bestand
breder leesbaar is dan nodig.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path


PROJECT_DIR = Path(__file__).parent.parent


def load_config_file(path: Path | None = None) -> dict:
    cfg_path = path or PROJECT_DIR / "telegram_config.json"
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def check_secret_file_permissions(path: Path | None = None) -> str | None:
    cfg_path = path or PROJECT_DIR / "telegram_config.json"
    if os.name == "nt" or not cfg_path.exists():
        return None
    try:
        mode = stat.S_IMODE(cfg_path.stat().st_mode)
    except Exception:
        return None
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        return f"{cfg_path} is te breed leesbaar; gebruik chmod 600."
    return None


def load_tradeai_config(path: Path | None = None) -> dict:
    cfg = load_config_file(path)
    env_map = {
        "token": "TRADEAI_TELEGRAM_TOKEN",
        "chat_id": "TRADEAI_TELEGRAM_CHAT_ID",
        "coinglass_key": "TRADEAI_COINGLASS_KEY",
    }
    for key, env_name in env_map.items():
        value = os.environ.get(env_name, "").strip()
        if value:
            cfg[key] = value
    warning = check_secret_file_permissions(path)
    if warning:
        cfg["_security_warning"] = warning
    return cfg

