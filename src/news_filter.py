"""
news_filter.py - Simpele nieuws/event-rem.

Zonder expliciete event-config blokkeert deze module geen trades op vaste tijden.
Vaste CPI/FOMC-uren bestaan wel vaak, maar niet elke dag; een dagelijks hard block
maakt de bot onnodig voorzichtig. Met TRADEAI_MACRO_EVENT_WINDOWS kunnen echte
eventvensters worden opgegeven, bijvoorbeeld:
2026-06-10T12:15:00+00:00/2026-06-10T12:45:00+00:00,2026-06-17T17:45:00+00:00/2026-06-17T18:15:00+00:00
"""
from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import requests

_NEWS_CACHE: dict = {"expires_at": 0.0, "snapshot": None}

NEGATIVE_NEWS_KEYWORDS = (
    "hack",
    "exploit",
    "breach",
    "lawsuit",
    "sec",
    "ban",
    "halt",
    "outage",
    "insolvent",
    "insolvency",
    "bankrupt",
    "liquidation",
    "liquidations",
    "probe",
    "investigation",
    "charged",
    "sanction",
    "delist",
)


def _cache_ttl_seconds() -> int:
    try:
        return max(30, int(os.environ.get("TRADEAI_NEWS_CACHE_SECONDS", "300")))
    except (TypeError, ValueError):
        return 300


def macro_event_risk(now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    configured = os.environ.get("TRADEAI_MACRO_EVENT_WINDOWS", "").strip()
    if not configured:
        return {"risk": "normal", "reason": "Geen macro-event vensters geconfigureerd."}

    for raw_window in configured.split(","):
        try:
            start_raw, end_raw = raw_window.strip().split("/", 1)
            start = datetime.fromisoformat(start_raw).astimezone(timezone.utc)
            end = datetime.fromisoformat(end_raw).astimezone(timezone.utc)
        except Exception:
            continue
        if start <= now <= end:
            return {
                "risk": "high",
                "reason": "Geconfigureerd macro-event venster; geen nieuwe trades rond mogelijk nieuws.",
            }
    return {"risk": "normal", "reason": "Geen actief macro-event venster."}


def crypto_news_risk() -> dict:
    token = os.environ.get("TRADEAI_CRYPTOPANIC_TOKEN", "").strip()
    if not token:
        return {"risk": "unknown", "reason": "CryptoPanic token ontbreekt."}
    try:
        response = requests.get(
            "https://cryptopanic.com/api/v1/posts/",
            params={"auth_token": token, "public": "true", "kind": "news"},
            timeout=8,
        )
        response.raise_for_status()
        posts = response.json().get("results", [])[:20]
        hot = sum(1 for post in posts if post.get("votes", {}).get("negative", 0) >= 3)
        if hot >= 3:
            return {"risk": "high", "reason": f"Veel negatief crypto-nieuws ({hot} headlines)."}
        return {"risk": "normal", "reason": "Geen extreme nieuwsdruk gedetecteerd."}
    except Exception as exc:
        return {"risk": "unknown", "reason": str(exc)}


def rss_news_risk() -> dict:
    configured = os.environ.get("TRADEAI_NEWS_RSS_URLS", "").strip()
    if not configured:
        return {"risk": "unknown", "reason": "Geen RSS nieuwsbronnen geconfigureerd."}

    headlines: list[str] = []
    errors: list[str] = []
    for url in [item.strip() for item in configured.split(",") if item.strip()]:
        try:
            response = requests.get(url, timeout=8)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            for item in root.findall(".//item")[:10]:
                title = (item.findtext("title") or "").strip()
                if title:
                    headlines.append(title)
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    if not headlines and errors:
        return {"risk": "unknown", "reason": "; ".join(errors)[:300], "headlines": []}
    if not headlines:
        return {"risk": "unknown", "reason": "Geen RSS headlines gevonden.", "headlines": []}

    # Dedup (case-insensitief, volgorde behouden): dezelfde kop uit meerdere feeds
    # mag de negatieve telling niet kunstmatig opblazen.
    seen: set[str] = set()
    unieke_headlines: list[str] = []
    for headline in headlines:
        sleutel = headline.lower().strip()
        if sleutel and sleutel not in seen:
            seen.add(sleutel)
            unieke_headlines.append(headline)
    headlines = unieke_headlines

    negative = []
    for headline in headlines[:30]:
        lower = headline.lower()
        if any(keyword in lower for keyword in NEGATIVE_NEWS_KEYWORDS):
            negative.append(headline)

    if len(negative) >= 3:
        return {
            "risk": "high",
            "reason": f"Veel negatieve RSS-headlines ({len(negative)}).",
            "headlines": negative[:5],
        }
    if negative:
        return {
            "risk": "caution",
            "reason": f"Enkele negatieve RSS-headlines ({len(negative)}).",
            "headlines": negative[:5],
        }
    return {
        "risk": "normal",
        "reason": "Geen negatieve RSS-nieuwsdruk gedetecteerd.",
        "headlines": headlines[:5],
    }


def news_watch_snapshot(force_refresh: bool = False) -> dict:
    now = time.time()
    if not force_refresh and _NEWS_CACHE["snapshot"] and now < _NEWS_CACHE["expires_at"]:
        return _NEWS_CACHE["snapshot"]

    macro = macro_event_risk()
    cryptopanic = crypto_news_risk()
    rss = rss_news_risk()
    block = macro["risk"] == "high" or cryptopanic["risk"] == "high" or rss["risk"] == "high"
    snapshot = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "block": block,
        "macro": macro,
        "cryptopanic": cryptopanic,
        "rss": rss,
    }
    _NEWS_CACHE["snapshot"] = snapshot
    _NEWS_CACHE["expires_at"] = now + _cache_ttl_seconds()
    return snapshot


def should_block_new_trades() -> tuple[bool, dict]:
    snapshot = news_watch_snapshot()
    return bool(snapshot["block"]), snapshot
