"""Core data structures passed through the ControlPlane pipeline.

Everything the system reasons about is a small, explicit dataclass so the
audit trail and evaluation harness have a stable, inspectable contract.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
import json

# Ordered severity of actions. Index = severity; higher = more restrictive.
ACTIONS = ["allow", "edit", "review", "block"]
def action_rank(a: str) -> int:
    return ACTIONS.index(a)
def escalate(a: str, b: str) -> str:
    """Return the more restrictive of two actions."""
    return a if action_rank(a) >= action_rank(b) else b


@dataclass
class Interaction:
    """One AI response to be checked, plus everything needed to check it."""
    id: str
    use_case: str                 # customer_facing | internal_copilot | regulated_decision
    query: str
    response: str
    context: str = ""             # retrieved source text the answer should be grounded in
    samples: list[str] = field(default_factory=list)  # alt generations for self-consistency
    model_used: str = "large"     # which model produced it (for cost/routing signal)
    regenerations: int = 0        # how many times it was re-generated (rework signal)
    # ---- ground-truth labels (used ONLY by the evaluation harness, never by detectors) ----
    label_hallucination: Optional[bool] = None
    label_pii: Optional[bool] = None
    label_toxic: Optional[bool] = None
    gold_action: Optional[str] = None
    category: Optional[str] = None  # clean|hallucination|pii_leak|toxic|cost_waste


@dataclass
class DetectorResult:
    """Output of a single detector."""
    name: str
    risk: float                       # 0..1 risk contributed by this detector
    flags: dict = field(default_factory=dict)   # boolean/scalar signals for policy rules
    detail: dict = field(default_factory=dict)  # human-readable evidence
    speed: str = "async"              # "inline" (fast, blocks stream) | "async" (parallel)
    latency_ms: float = 0.0


@dataclass
class Decision:
    """Final decision for one interaction."""
    interaction_id: str
    use_case: str
    action: str                       # allow|edit|review|block
    risk_scores: dict                 # per-dimension risk
    overall_risk: float
    p_harm: float = 0.0               # calibrated P(response is harmful if served)
    expected_loss: dict = field(default_factory=dict)     # E[loss] per action
    thresholds_used: dict = field(default_factory=dict)   # bands derived from the cost model
    reasons: list[str] = field(default_factory=list)
    fired_rules: list[str] = field(default_factory=list)
    detector_results: list[dict] = field(default_factory=list)
    total_latency_ms: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)
