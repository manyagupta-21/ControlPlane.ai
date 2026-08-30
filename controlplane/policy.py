"""Governance / decision layer.

A single YAML file (config/policies.yaml) defines, PER USE-CASE:
  * a cost model (`costs:`)   -> thresholds DERIVED by expected-loss minimisation
    (or, for legacy configs, hard-coded `thresholds:`)
  * hard_rules on detector flags -> minimum action, regardless of score

This is what lets one engine behave differently for a customer-facing chatbot
(low risk appetite) vs an internal copilot (higher tolerance) vs a regulated
decision tool (strictest), without changing code. Risk appetite is expressed as
what the four outcomes cost the business; the cut-points follow from that.
Every fired rule is recorded, giving a per-decision audit trail.
"""
from __future__ import annotations
import yaml
from .schemas import DetectorResult, Decision, escalate
from .scoring import combine
from .decision_theory import from_config, harm_probability


class PolicyEngine:
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.use_cases = self.cfg["use_cases"]
        # Derive thresholds once, at load time, for every use-case that ships a
        # cost model. Legacy use-cases keep their hard-coded thresholds.
        self.loss_models, self.thresholds = {}, {}
        for name, uc in self.use_cases.items():
            lm = from_config(uc.get("costs"))
            self.loss_models[name] = lm
            self.thresholds[name] = lm.derive_thresholds() if lm else uc["thresholds"]

    def _band_action(self, use_case: str, p: float) -> str:
        th = self.thresholds[use_case]
        # thresholds give the LOWER edge of each escalating band
        if p >= th["block"]:
            return "block"
        if p >= th["review"]:
            return "review"
        if p >= th["edit"]:
            return "edit"
        return "allow"

    def decide(self, interaction_id: str, use_case: str,
               results: list[DetectorResult], total_latency_ms: float = 0.0) -> Decision:
        if use_case not in self.use_cases:
            use_case = self.cfg.get("default_use_case", "internal_copilot")
        uc = self.use_cases[use_case]

        overall, per_dim = combine(results, uc.get("weights"))

        # Prefer the calibrated hallucination probability when the isotonic
        # model is loaded: expected-loss reasoning is only meaningful on a
        # genuine probability scale, not on a raw similarity score.
        calibrated = None
        for r in results:
            if r.name == "performance":
                calibrated = r.detail.get("calibrated_hallucination_prob")
        p_harm = harm_probability(per_dim, calibrated)

        lm = self.loss_models.get(use_case)
        action = self._band_action(use_case, p_harm)
        reasons, fired = [], []

        if lm is not None:
            reasons.append(lm.explain(p_harm))
            reasons.append("derived bands (from cost model): "
                           + ", ".join(f"{k}>={v}" for k, v in self.thresholds[use_case].items()))
        else:
            reasons.append(f"p_harm={p_harm} -> band '{action}' (static thresholds)")
        if calibrated is None:
            reasons.append("note: uncalibrated performance score used as P(harm) proxy")

        # gather all flags from detectors into one namespace for rule matching
        flags = {}
        for r in results:
            flags.update({k: v for k, v in r.flags.items()})

        for rule in uc.get("hard_rules", []):
            cond = rule["if"]
            if flags.get(cond):
                new_action = escalate(action, rule["action"])
                if new_action != action:
                    reasons.append(f"rule '{cond}' -> escalate to '{rule['action']}'")
                    action = new_action
                fired.append(cond)

        return Decision(
            interaction_id=interaction_id,
            use_case=use_case,
            action=action,
            risk_scores={k: round(v, 3) for k, v in per_dim.items()},
            overall_risk=overall,
            p_harm=p_harm,
            expected_loss=(lm.optimal_action(p_harm)[1] if lm else {}),
            thresholds_used=dict(self.thresholds[use_case]),
            reasons=reasons,
            fired_rules=fired,
            detector_results=[r.__dict__ for r in results],
            total_latency_ms=round(total_latency_ms, 2),
        )
