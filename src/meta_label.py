"""
meta_label.py — lichte meta-labeling filter (numpy-only, geen externe ML-dependency).

Idee (Lopez de Prado meta-labeling): een tweede model leert vanuit de eigen
trade-historie wanneer een setup-context historisch wínt. Dat vult het bestaande
scorecard-systeem aan met een geleerd, multi-dimensionaal gewicht.

Bewust GEEN scikit-learn: op een Raspberry Pi is dat een zware build. We trainen
een gewogen logistische regressie met L2-regularisatie in pure numpy. Valt netjes
terug op "neutraal" (prob 0.5, available=False) als er te weinig data is.
"""
from __future__ import annotations

import json

import numpy as np

# Contextsleutels die we NIET als feature gebruiken (vrij-tekst of lek-gevoelig).
_EXCLUDE_KEYS = frozenset({
    "context_key", "missed_horizon", "missed_variant", "missed_exit_type",
    "timeframe",
})

MIN_TRAIN_SAMPLES = 80      # Onder dit aantal samples geen model (te weinig signaal)
_MAX_ITERS = 300
_LEARNING_RATE = 0.3
_L2 = 1.0

# In-memory cache zodat we niet elke beslissing opnieuw trainen.
_CACHE: dict = {"model": None, "trained_at_count": -1}


def extract_features(context: dict) -> list[str]:
    """Zet een context-dict om in een lijst categorische 'sleutel=waarde'-features."""
    feats = []
    for key in sorted(context):
        if key in _EXCLUDE_KEYS:
            continue
        value = context[key]
        if isinstance(value, bool):
            feats.append(f"{key}={int(value)}")
        elif isinstance(value, str) and value:
            feats.append(f"{key}={value}")
    return feats


class MetaLabelModel:
    """Gewogen logistische regressie over one-hot categorische features."""

    def __init__(self, vocab: dict[str, int], weights: np.ndarray, bias: float, n_samples: int):
        self.vocab = vocab
        self.weights = weights
        self.bias = bias
        self.n_samples = n_samples

    def _vectorize(self, feats: list[str]) -> np.ndarray:
        x = np.zeros(len(self.vocab), dtype=float)
        for f in feats:
            idx = self.vocab.get(f)
            if idx is not None:
                x[idx] = 1.0
        return x

    def predict_proba(self, context: dict) -> float:
        x = self._vectorize(extract_features(context))
        z = float(np.dot(self.weights, x) + self.bias)
        return 1.0 / (1.0 + np.exp(-z))

    def to_dict(self) -> dict:
        return {
            "vocab": self.vocab,
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "n_samples": self.n_samples,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MetaLabelModel":
        return cls(
            vocab=data["vocab"],
            weights=np.asarray(data["weights"], dtype=float),
            bias=float(data["bias"]),
            n_samples=int(data["n_samples"]),
        )


def train_model(samples: list[dict], min_samples: int = MIN_TRAIN_SAMPLES) -> MetaLabelModel | None:
    """
    Train op een lijst samples: [{"context": {...}, "outcome": 0/1, "weight": float}, ...].
    Geeft None als er te weinig data is.
    """
    rows = [s for s in samples if s.get("context") and s.get("outcome") is not None]
    if len(rows) < min_samples:
        return None

    # Bouw vocabulaire.
    vocab: dict[str, int] = {}
    feat_lists = []
    for s in rows:
        feats = extract_features(s["context"])
        feat_lists.append(feats)
        for f in feats:
            if f not in vocab:
                vocab[f] = len(vocab)
    if not vocab:
        return None

    n, d = len(rows), len(vocab)
    X = np.zeros((n, d), dtype=float)
    y = np.zeros(n, dtype=float)
    w = np.zeros(n, dtype=float)
    for i, (s, feats) in enumerate(zip(rows, feat_lists)):
        for f in feats:
            X[i, vocab[f]] = 1.0
        y[i] = 1.0 if float(s["outcome"]) > 0 else 0.0
        w[i] = max(0.0, float(s.get("weight", 1.0)))

    if w.sum() <= 0:
        w = np.ones(n)
    w = w / w.mean()  # normaliseer rond 1

    weights = np.zeros(d, dtype=float)
    bias = 0.0
    for _ in range(_MAX_ITERS):
        z = X @ weights + bias
        pred = 1.0 / (1.0 + np.exp(-z))
        error = (pred - y) * w
        grad_w = X.T @ error / n + _L2 * weights / n
        grad_b = float(error.sum() / n)
        weights -= _LEARNING_RATE * grad_w
        bias -= _LEARNING_RATE * grad_b

    return MetaLabelModel(vocab, weights, bias, n)


def _load_samples_from_db() -> list[dict]:
    """Haal leer-samples uit de DB als trainingsdata (live wegen het zwaarst)."""
    try:
        from src.database import get_connection
        conn = get_connection()
        rows = conn.execute(
            "SELECT context_json, outcome, weight FROM bot_learning_samples "
            "WHERE outcome IS NOT NULL ORDER BY id DESC LIMIT 4000"
        ).fetchall()
        conn.close()
    except Exception:
        return []

    samples = []
    for row in rows:
        try:
            context = json.loads(row["context_json"] or "{}")
        except Exception:
            continue
        if not context:
            continue
        samples.append({
            "context": context,
            "outcome": row["outcome"],
            "weight": row["weight"] if row["weight"] is not None else 1.0,
        })
    return samples


def get_model(force_retrain: bool = False) -> MetaLabelModel | None:
    """Geef een (gecached) getraind model terug; traint lui vanuit de DB."""
    samples = _load_samples_from_db()
    count = len(samples)
    if not force_retrain and _CACHE["model"] is not None and _CACHE["trained_at_count"] == count:
        return _CACHE["model"]
    model = train_model(samples)
    _CACHE["model"] = model
    _CACHE["trained_at_count"] = count
    return model


def meta_label_score(context: dict, min_samples: int = MIN_TRAIN_SAMPLES) -> dict:
    """
    Beoordeel een setup-context. Returns:
      {"available": bool, "prob_win": float, "n_samples": int, "reason": str}
    available=False ⇒ neutraal, de bot negeert het filter (degradatie).
    """
    model = get_model()
    if model is None or model.n_samples < min_samples:
        return {
            "available": False,
            "prob_win": 0.5,
            "n_samples": model.n_samples if model else 0,
            "reason": "te weinig data voor meta-label",
        }
    prob = round(model.predict_proba(context), 4)
    return {
        "available": True,
        "prob_win": prob,
        "n_samples": model.n_samples,
        "reason": f"meta-label winkans {prob:.0%} (n={model.n_samples})",
    }


def reset_cache() -> None:
    """Leeg de in-memory modelcache (gebruikt door tests/na reset_learning)."""
    _CACHE["model"] = None
    _CACHE["trained_at_count"] = -1
