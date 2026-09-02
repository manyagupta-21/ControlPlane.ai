"""Governance / decision layer.

A single YAML file (config/policies.yaml) defines, PER USE-CASE:
  * a cost model (`costs:`)   -> thresholds DERIVED by expected-loss minimisation
    (or, for legacy configs, hard-coded `thresholds:`)
  * hard_rules on detector flags -> minimum action, regardless of score

It also defines TWO further, independent axes that COMPOSE with use_case
rather than duplicating it, each scoped by `applies_to`:
  * jurisdictions: -> geography-specific obligations (EU AI Act, India DPDP, ...)
  * sectors:       -> industry-specific obligations (healthcare, finance, ...)

This is what lets one engine behave differently for a customer-facing chatbot
vs. an internal copilot vs. a regulated decision tool (use_case), AND
differently again by regulatory regime (jurisdiction), AND differently again
by industry (sector) -- without duplicating use cases per country or per
vertical. Risk appetite is expressed as what the four outcomes cost the
business; the cut-points follow from that. Every fired rule is recorded,
tagged with which axis fired it, giving a per-decision audit trail across all
three dimensions.
"""
from __future__ import annotations
import yaml
from .schemas import DetectorResult, Decision, escalate
from .scoring import combine
from .decision_theory import from_config, harm_probability
from .calibration import widen


class PolicyEngine:
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.use_cases = self.cfg["use_cases"]
        self.jurisdictions = self.cfg.get("jurisdictions", {})
        self.sectors = self.cfg.get("sectors", {})
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

    def _scoped_rules(self, axis_cfg: dict, axis_value: str, use_case: str) -> list[dict]:
        """Shared lookup for both jurisdictions: and sectors: -- an axis
        entry only contributes rules if it's configured AND scoped to this
        use_case via applies_to (or has no applies_to, i.e. applies to all)."""
        entry = axis_cfg.get(axis_value)
        if not entry:
            return []
        applies_to = entry.get("applies_to")
        if applies_to is not None and use_case not in applies_to:
            return []
        return entry.get("hard_rules", [])

    def decide(self, interaction_id: str, use_case: str,
               results: list[DetectorResult], jurisdiction: str = "US",
               sector: str = "general", total_latency_ms: float = 0.0) -> Decision:
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

        # --- STEP 1C: out-of-domain widening ------------------------------
        # Anomaly detection reaches the decision HERE, and only here. It does
        # not add risk (unusual is not harmful); it widens the confidence
        # interval on P(harm) and the band is read off the upper end. A response
        # scored outside the envelope the calibrator was validated on gets a
        # conservative add-on, exactly as a model used outside its validated
        # domain would under any model-risk framework. Normal traffic has
        # severity ~0, so its p_decision ~ p_harm and its action is unchanged.
        ood = 0.0
        for r in results:
            if r.name == "performance":
                a = r.detail.get("anomaly") or {}
                if a.get("fitted"):
                    ood = float(a.get("ood_severity", 0.0))
                    if (a.get("window") or {}).get("population_shifted"):
                        ood = 1.0     # a confirmed population shift is full OOD
        band = widen(p_harm, ood)
        p_decision = band["p_decision"]

        lm = self.loss_models.get(use_case)
        action = self._band_action(use_case, p_decision)
        reasons, fired = [], []

        if band["widened_by"] > 0.02:
            reasons.append(
                f"OOD widening: p_point={band['p_point']} -> p_decision={p_decision} "
                f"(severity {band['ood_severity']}, effective n {band['n_effective']}); "
                f"deciding on the upper bound because this response sits outside "
                f"the validated envelope")
        if lm is not None:
            reasons.append(lm.explain(p_decision))
            reasons.append("derived bands (from cost model): "
                           + ", ".join(f"{k}>={v}" for k, v in self.thresholds[use_case].items()))
        else:
            reasons.append(f"p_harm={p_decision} -> band '{action}' (static thresholds)")
        if calibrated is None:
            reasons.append("note: uncalibrated performance score used as P(harm) proxy")

        # gather all flags from detectors into one namespace for rule matching
        flags = {}
        for r in results:
            flags.update({k: v for k, v in r.flags.items()})

        def apply_rules(rules: list[dict], tag: str):
            nonlocal action
            for rule in rules:
                cond = rule["if"]
                if flags.get(cond):
                    new_action = escalate(action, rule["action"])
                    if new_action != action:
                        reasons.append(f"[{tag}] rule '{cond}' -> escalate to '{rule['action']}'")
                        action = new_action
                    fired.append(f"{tag}:{cond}")

        # --- axis 1: use_case hard_rules -----------------------------------
        apply_rules(uc.get("hard_rules", []), f"use_case:{use_case}")

        # --- axis 2: jurisdiction hard_rules (composes with axis 1) --------
        apply_rules(self._scoped_rules(self.jurisdictions, jurisdiction, use_case),
                    f"jurisdiction:{jurisdiction}")

        # --- axis 3: sector hard_rules (composes with axes 1 and 2) --------
        apply_rules(self._scoped_rules(self.sectors, sector, use_case),
                    f"sector:{sector}")

        return Decision(
            interaction_id=interaction_id,
            use_case=use_case,
            action=action,
            risk_scores={k: round(v, 3) for k, v in per_dim.items()},
            overall_risk=overall,
            p_harm=p_decision,
            expected_loss=(lm.optimal_action(p_decision)[1] if lm else {}),
            thresholds_used=dict(self.thresholds[use_case]),
            reasons=reasons,
            fired_rules=fired,
            detector_results=[r.__dict__ for r in results],
            total_latency_ms=round(total_latency_ms, 2),
        )