"""
cost_sensitivity.py — bij welk kostenniveau ontstaat edge? (route A: maker-orders)

Verzamelt de BRUTO out-of-sample returns van het gecombineerde ML-model (één keer),
en trekt er verschillende kostenniveaus van af (0% t/m 0,35%). Zo zie je of het
signaal bruto positief is — en hoeveel kosten je dan maximaal mag betalen.

LET OP: 0% = onrealistische bovengrens (negeert dat limit-orders niet altijd vullen
en adverse selection). Het beantwoordt alleen: is het bruto-signaal positief?

Gebruik:  python tools/cost_sensitivity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import ml_meta as mm  # noqa: E402
from stats import edge_stats  # noqa: E402

COSTS = [0.0, 0.02, 0.05, 0.10, 0.20, 0.35]  # round-trip kosten in %


def collect_gross():
    """Bruto OOS-returns (geen kosten) van het ML-model, walk-forward."""
    gross = []
    for name, sym in mm.MARKETS.items():
        try:
            df = mm.build(sym)
        except Exception as exc:
            print(f"  {name}: mislukt — {exc}")
            continue
        start = 0
        while start + mm.TRAIN + mm.TEST <= len(df):
            tr = df.iloc[start:start + mm.TRAIN]
            te = df.iloc[start + mm.TRAIN:start + mm.TRAIN + mm.TEST]
            mu, sd = tr[mm.FEATURES].mean(), tr[mm.FEATURES].std().replace(0, 1)
            Xtr = ((tr[mm.FEATURES] - mu) / sd).values
            Xte = ((te[mm.FEATURES] - mu) / sd).values
            w, b = mm.train_logreg(Xtr, tr["y"].values)
            prob = 1 / (1 + np.exp(-(Xte @ w + b)))
            fwd = te["fwd_ret"].values
            for k in range(len(te)):
                d = 1 if prob[k] > 0.5 else -1
                gross.append(d * fwd[k])   # GEEN kosten
            start += mm.STEP
    return gross


def main():
    print("KOSTEN-GEVOELIGHEID — gecombineerd ML-model, bruto OOS-returns minus kosten\n")
    gross = collect_gross()
    print(f"  {len(gross)} out-of-sample trades\n")
    print(f"  {'kosten':>8}{'expectancy/trade':>20}{'t-stat':>9}   oordeel")
    for c in COSTS:
        net = [g - c / 100 for g in gross]
        st = edge_stats(net)
        vlag = "  <<< EDGE" if st["t_stat"] > 1.96 else ""
        print(f"  {c:>7.2f}%{st['expectancy_pct']*100:>18.3f}%{st['t_stat']:>9.1f}   {st['verdict']}{vlag}")
    print("\n  (0% = onrealistische bovengrens; maker-orders kosten in werkelijkheid méér door "
          "niet-vullen + adverse selection.)")


if __name__ == "__main__":
    main()
