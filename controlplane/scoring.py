"""Aggregate detector outputs into a single, policy-ready risk view.

We combine the *blocking* dimensions (performance, responsibility) with a
noisy-OR, which is deliberately safety-biased: risk rises as soon as ANY
dimension is concerning, and compounds when several are. Cost is tracked
separately because it informs efficiency reporting, not gating.
"""
from __future__ import annotations
from .schemas import DetectorResult

BLOCKING_DIMENSIONS = ("performance", "responsibility")

def combine(results: list[DetectorResult], weights: dict | None = None) -> tuple[float, dict]:
    weights = weights or {}
    per_dim = {r.name: r.risk for r in results}
    # noisy-OR over weighted blocking dimensions
    prod = 1.0
    for name in BLOCKING_DIMENSIONS:
        w = float(weights.get(name, 1.0))
        r = per_dim.get(name, 0.0) * w
        r = min(1.0, max(0.0, r))
        prod *= (1.0 - r)
    overall = round(1.0 - prod, 3)
    return overall, per_dim
