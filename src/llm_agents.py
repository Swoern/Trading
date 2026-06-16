"""
llm_agents.py — optionele hybride LLM-adviseslaag (Fase 3).

Een klein panel van LLM-"analisten" (technisch, sentiment/nieuws, risico) plus een
coördinator dat hun visies samenvat tot één begrensd advies per asset. Dit draait
NAAST de snelle regel-agents, op trage cadans (bv. elk uur), en is volledig optioneel.

Veiligheidsprincipes:
  * Standaard UIT (env TRADEAI_LLM_ENABLED=1 om aan te zetten) + Anthropic API-key nodig.
  * Het advies kan de bot alleen BEGRENSD bijsturen: de min_confidence-drempel een
    paar punten verschuiven (vooral strenger). Het kan NOOIT een trade forceren of
    de discipline-regels overrulen.
  * Faalt altijd zacht: zonder SDK/key/flag of bij een API-fout → neutraal advies.

Kosten: met Haiku en ~1 panel/uur per asset is dit een paar euro per maand.
"""
from __future__ import annotations

import json
import os
import re

# Standaard goedkoop model; override via env.
DEFAULT_MODEL = os.environ.get("TRADEAI_LLM_MODEL", "claude-haiku-4-5-20251001")

# Grenzen waarbinnen het advies de min_confidence-drempel mag verschuiven.
MIN_CONF_DELTA_FLOOR = -2.0   # mag de drempel maar minimaal verlagen (voorzichtig)
MIN_CONF_DELTA_CAP = 8.0      # mag de drempel flink verhogen (strenger in rommel)

ANALYST_ROLES = ("technical", "sentiment", "risk")


def is_enabled() -> bool:
    """LLM-laag is alleen actief met expliciete flag én een API-key."""
    flag = os.environ.get("TRADEAI_LLM_ENABLED", "0").strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return False
    return bool(_api_key())


def _api_key() -> str | None:
    return (
        os.environ.get("TRADEAI_ANTHROPIC_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or None
    )


def _build_client():
    """Maak een Anthropic-client; None als de SDK of key ontbreekt."""
    key = _api_key()
    if not key:
        return None
    try:
        import anthropic  # lazy import — geen harde dependency
    except Exception:
        return None
    try:
        return anthropic.Anthropic(api_key=key)
    except Exception:
        return None


def _snapshot_text(asset: str, snapshot: dict) -> str:
    """Compacte, deterministische tekstweergave van de marktcontext voor de prompt."""
    regels = [f"Asset: {asset}"]
    for key in sorted(snapshot):
        value = snapshot[key]
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)[:300]
        regels.append(f"{key}: {value}")
    return "\n".join(regels)


def build_analyst_prompt(role: str, asset: str, snapshot: dict) -> str:
    """Bouw de prompt voor één analist-rol."""
    focus = {
        "technical": "Beoordeel de technische conditie (trend, momentum, regime, niveaus).",
        "sentiment": "Beoordeel sentiment & nieuws (Fear&Greed, nieuwsdruk, marktbreedte).",
        "risk": "Beoordeel het risico (volatiliteit, drawdown-gevaar, correlatie, timing).",
    }.get(role, "Beoordeel de markt.")
    return (
        f"Je bent de {role}-analist van een crypto paper-trading bot. {focus}\n"
        f"Geef een korte analyse (max 3 zinnen) en sluit af met een regel:\n"
        f"VERDICT: bias=<-1..1>, vertrouwen=<0..1>\n\n"
        f"Marktcontext:\n{_snapshot_text(asset, snapshot)}"
    )


def build_coordinator_prompt(asset: str, analyst_views: dict) -> str:
    """Bouw de coördinator-prompt die de analisten samenvat tot één advies."""
    views = "\n\n".join(f"[{rol}]\n{tekst}" for rol, tekst in analyst_views.items())
    return (
        "Je bent de coördinator van een crypto paper-trading bot. Vat de analisten "
        "hieronder samen tot één advies. De bot blijft volledig regel-gebaseerd; jouw "
        "advies mag de strengheid licht bijsturen maar nooit een trade forceren.\n"
        "Antwoord UITSLUITEND met JSON:\n"
        '{"bias": <-1..1>, "confidence": <0..1>, '
        '"min_confidence_delta": <-2..8, hoger = strenger>, "rationale": "<1 zin>"}\n\n'
        f"Asset: {asset}\n\nAnalisten:\n{views}"
    )


def _complete(client, model: str, prompt: str, max_tokens: int = 400) -> str:
    """Roep het model aan en haal de tekst eruit. Werkt met de Anthropic SDK-vorm."""
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    parts = getattr(resp, "content", None) or []
    teksten = []
    for part in parts:
        tekst = getattr(part, "text", None)
        if tekst is None and isinstance(part, dict):
            tekst = part.get("text")
        if tekst:
            teksten.append(tekst)
    return "\n".join(teksten).strip()


def parse_advice(text: str) -> dict:
    """
    Parse het coördinator-antwoord naar een begrensd advies. Faalt zacht naar neutraal.
    Clamps: bias∈[-1,1], confidence∈[0,1], min_confidence_delta∈[FLOOR, CAP].
    """
    neutraal = {"bias": 0.0, "confidence": 0.0, "min_confidence_delta": 0.0, "rationale": ""}
    if not text:
        return neutraal
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return neutraal
    try:
        data = json.loads(match.group(0))
    except Exception:
        return neutraal

    def _num(value, default=0.0):
        try:
            return float(value)
        except Exception:
            return default

    bias = max(-1.0, min(1.0, _num(data.get("bias"))))
    confidence = max(0.0, min(1.0, _num(data.get("confidence"))))
    delta = max(MIN_CONF_DELTA_FLOOR, min(MIN_CONF_DELTA_CAP, _num(data.get("min_confidence_delta"))))
    rationale = str(data.get("rationale") or "")[:500]
    return {
        "bias": round(bias, 3),
        "confidence": round(confidence, 3),
        "min_confidence_delta": round(delta, 2),
        "rationale": rationale,
    }


def parse_analyst_verdict(text: str) -> dict | None:
    """Haal de 'VERDICT: bias=.., vertrouwen=..'-regel uit een analist-antwoord."""
    if not text:
        return None
    match = re.search(
        r"bias\s*=\s*(-?\d+(?:\.\d+)?).*?(?:vertrouwen|confidence)\s*=\s*(\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    try:
        bias = max(-1.0, min(1.0, float(match.group(1))))
        confidence = max(0.0, min(1.0, float(match.group(2))))
    except Exception:
        return None
    return {"bias": bias, "confidence": confidence}


def detect_panel_conflict(analyst_views: dict, advice: dict) -> list[str]:
    """
    'Liever weigeren dan gokken': geef reden-codes als het panel tegenstrijdig is.

    - analyst_direction_split: minstens één analist sterk bullish én één sterk bearish.
    - coordinator_contradicts_analysts: de coördinator wijst de andere kant op dan
      het duidelijke gemiddelde van de analisten.
    Leeg = geen conflict.
    """
    redenen: list[str] = []
    biases = [
        v["bias"]
        for v in (parse_analyst_verdict(t) for t in analyst_views.values())
        if v is not None
    ]
    if len(biases) >= 2:
        if any(b >= 0.4 for b in biases) and any(b <= -0.4 for b in biases):
            redenen.append("analyst_direction_split")
        gemiddelde = sum(biases) / len(biases)
        coord_bias = float(advice.get("bias") or 0.0)
        if (
            abs(gemiddelde) >= 0.3
            and abs(coord_bias) >= 0.3
            and (gemiddelde > 0) != (coord_bias > 0)
        ):
            redenen.append("coordinator_contradicts_analysts")
    return redenen


def _neutraliseer_bij_conflict(advice: dict, conflict: list[str]) -> dict:
    """
    Bij een tegenstrijdig panel: advies veilig neutraliseren. De bias gaat naar 0 en
    de drempel-verschuiving mag alleen STRENGER worden (nooit losser) — een verward
    panel mag de bot nooit agressiever maken.
    """
    veilig = dict(advice)
    veilig["bias"] = 0.0
    veilig["confidence"] = round(min(float(advice.get("confidence") or 0.0), 0.30), 3)
    veilig["min_confidence_delta"] = round(max(0.0, float(advice.get("min_confidence_delta") or 0.0)), 2)
    veilig["rationale"] = (
        f"[panel-conflict: {', '.join(conflict)}] geneutraliseerd. "
        + str(advice.get("rationale") or "")
    )[:500]
    return veilig


def run_panel(asset: str, snapshot: dict, client=None, persist: bool = True) -> dict:
    """
    Draai het analisten-panel voor één asset en geef een begrensd advies terug.

    Returns altijd een dict met minstens {"available": bool, "bias", "confidence",
    "min_confidence_delta", "rationale"}. Bij uitgeschakeld/fout → available=False
    en neutrale waarden, zodat de bot ongehinderd doordraait.
    """
    neutraal = {
        "available": False, "bias": 0.0, "confidence": 0.0,
        "min_confidence_delta": 0.0, "rationale": "LLM-laag uit of niet beschikbaar.",
        "analysts": {},
    }
    if client is None:
        if not is_enabled():
            return neutraal
        client = _build_client()
        if client is None:
            return neutraal

    model = DEFAULT_MODEL
    try:
        analyst_views = {}
        for rol in ANALYST_ROLES:
            analyst_views[rol] = _complete(client, model, build_analyst_prompt(rol, asset, snapshot))
        coord_text = _complete(client, model, build_coordinator_prompt(asset, analyst_views))
        advice = parse_advice(coord_text)
    except Exception as exc:
        return neutraal | {"rationale": f"LLM-fout: {exc}"[:200]}

    # 'Liever weigeren dan gokken': bij een tegenstrijdig panel het advies neutraliseren.
    conflict = detect_panel_conflict(analyst_views, advice)
    if conflict:
        advice = _neutraliseer_bij_conflict(advice, conflict)

    result = {
        "available": True,
        "bias": advice["bias"],
        "confidence": advice["confidence"],
        "min_confidence_delta": advice["min_confidence_delta"],
        "rationale": advice["rationale"],
        "conflict": conflict,
        "analysts": analyst_views,
        "model": model,
    }

    if persist:
        try:
            from src.database import save_llm_advice
            save_llm_advice(
                asset=asset,
                bias=result["bias"],
                confidence=result["confidence"],
                min_conf_delta=result["min_confidence_delta"],
                rationale=result["rationale"],
                analysts=analyst_views,
                model=model,
            )
        except Exception:
            pass
    return result


def bounded_min_confidence_delta(advice: dict | None) -> float:
    """Veilige, begrensde drempel-verschuiving uit een (opgeslagen) advies."""
    if not advice:
        return 0.0
    raw = advice.get("min_confidence_delta", advice.get("min_conf_delta", 0.0))
    try:
        raw = float(raw)
    except Exception:
        return 0.0
    return max(MIN_CONF_DELTA_FLOOR, min(MIN_CONF_DELTA_CAP, raw))
