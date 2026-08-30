"""Bayes decision layer: the action is chosen by minimising expected loss.

Most guardrail systems hard-code cut-points ("block above 0.8"). Those numbers
are unexplainable and unauditable: nobody can say why 0.8 rather than 0.7.

We do it the way a risk function does it. The business states what the four
outcomes *cost*; the cut-points are then derived, not chosen:

    a* (p) = argmin_a  E[ L(a, S) ]      where  p = P(response is harmful)

Every loss is linear in p, so the optimal action is the lower envelope of four
straight lines and the thresholds are exactly the crossing points. Consequences:

  * "risk appetite" becomes a cost ratio, not a magic number;
  * changing appetite is a business decision, and the new thresholds follow
    automatically and reproducibly;
  * the audit trail can state *why* a response was blocked in currency terms.

Loss model  (S = harmful / benign)
-----------------------------------------------------------------------------
  allow   harmful -> C_serve_bad          benign -> 0
  edit    harmful -> r_edit * C_serve_bad benign -> C_caveat
          (a caveat reduces but does not remove the harm of a bad answer, and
           costs a little credibility even when the answer was fine)
  review  harmful -> r_review * C_serve_bad + C_review
          benign  -> C_review
          (a human catches most, not all, of what reaches them)
  block   harmful -> 0                    benign -> C_block_good
-----------------------------------------------------------------------------
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .schemas import ACTIONS


@dataclass
class LossMatrix:
    """Costs, in the business's own units (₹, $, or arbitrary utility)."""
    serve_bad: float = 500.0     # a harmful response reaches the user and is acted on
    block_good: float = 20.0     # a good response is withheld (lost value, friction)
    review: float = 8.0          # one reviewer's time on one item
    caveat: float = 1.0          # friction of an unnecessary warning label
    resid_edit: float = 0.40     # fraction of harm surviving a caveat
    resid_review: float = 0.05   # fraction of harm a human reviewer still misses

    # ---- expected loss of each action at harm-probability p ----
    def expected_loss(self, p: float | np.ndarray) -> dict[str, np.ndarray]:
        p = np.asarray(p, dtype=float)
        return {
            "allow":  p * self.serve_bad,
            "edit":   p * self.resid_edit * self.serve_bad + (1.0 - p) * self.caveat,
            "review": self.review + p * self.resid_review * self.serve_bad,
            "block":  (1.0 - p) * self.block_good,
        }

    def optimal_action(self, p: float) -> tuple[str, dict]:
        el = self.expected_loss(p)
        best = min(ACTIONS, key=lambda a: float(el[a]))
        return best, {a: round(float(el[a]), 4) for a in ACTIONS}

    def realised_loss(self, action: str, harmful: bool) -> float:
        """Loss actually incurred once the true state is known.

        This is what the back-test scores against — no reference to any
        hand-written 'gold action', only to what the outcome cost.
        """
        if action == "allow":
            return self.serve_bad if harmful else 0.0
        if action == "edit":
            return self.resid_edit * self.serve_bad if harmful else self.caveat
        if action == "review":
            return self.review + (self.resid_review * self.serve_bad if harmful else 0.0)
        if action == "block":
            return 0.0 if harmful else self.block_good
        raise ValueError(action)

    def derive_thresholds(self, grid: int = 20001) -> dict[str, float]:
        """Lower edge of each escalating band, read off the lower envelope.

        Returned in the same shape the YAML used to hard-code, so the rest of
        the policy engine is unchanged — only the provenance of the numbers is.
        A band that is never optimal under these costs is reported as 1.0
        (unreachable), which is itself informative: e.g. if review is cheap
        enough, 'edit' may never be the best action at any probability.
        """
        ps = np.linspace(0.0, 1.0, grid)
        el = self.expected_loss(ps)
        stacked = np.vstack([el[a] for a in ACTIONS])       # 4 x grid
        best_idx = stacked.argmin(axis=0)
        out = {}
        for rank, name in enumerate(ACTIONS):
            if rank == 0:
                continue                                     # 'allow' is the floor
            hit = np.where(best_idx >= rank)[0]
            out[name] = float(round(ps[hit[0]], 4)) if len(hit) else 1.0
        return out

    def explain(self, p: float) -> str:
        a, el = self.optimal_action(p)
        second = sorted(el.items(), key=lambda kv: kv[1])[1]
        return (f"P(harm)={p:.3f} -> '{a}' minimises expected loss "
                f"({el[a]:.2f} vs next-best '{second[0]}' {second[1]:.2f})")


def harm_probability(per_dim: dict, calibrated: float | None = None) -> float:
    """P(this response is harmful if served), on a probability scale.

    Responsibility risk (PII / toxicity) is already close to a probability: the
    detectors fire on observed entities. The performance dimension is a raw
    similarity-derived score, so we prefer its *calibrated* value when the
    isotonic model is available, and fall back to the raw score otherwise.

    The two are combined with a noisy-OR: a response is harmful if it is
    factually wrong OR unsafe, and the two failure modes are treated as
    conditionally independent. That is deliberately conservative — it is an
    upper bound whenever the modes are positively correlated, which is the safe
    direction for a control layer.
    """
    p_perf = calibrated if calibrated is not None else float(per_dim.get("performance", 0.0))
    p_resp = float(per_dim.get("responsibility", 0.0))
    p_perf = min(max(p_perf, 0.0), 1.0)
    p_resp = min(max(p_resp, 0.0), 1.0)
    return round(1.0 - (1.0 - p_perf) * (1.0 - p_resp), 4)


def adjust_prior(p: float, train_prevalence: float, deploy_prevalence: float) -> float:
    """Correct a probability for prior shift (label shift).

    Our detectors and the isotonic calibrator are fitted on evaluation sets where
    harmful responses are roughly half the data. Live traffic is nothing like
    that — a few percent at most. A probability calibrated at one prevalence is
    mis-calibrated at another, and the Bayes decision rule inherits the error:
    the system would block far more benign traffic than the cost model intends.

    The standard label-shift correction reweights the likelihood ratio by the
    ratio of priors:

        odds_deploy = odds_train * [pi_d / (1 - pi_d)] * [(1 - pi_t) / pi_t]

    With this in place the same loss matrix stays optimal at ANY base rate, and
    `base_rate:` in the policy config becomes an explicit, auditable assumption
    rather than a hidden one.
    """
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    pt = min(max(float(train_prevalence), 1e-6), 1 - 1e-6)
    pd_ = min(max(float(deploy_prevalence), 1e-6), 1 - 1e-6)
    odds = (p / (1 - p)) * (pd_ / (1 - pd_)) * ((1 - pt) / pt)
    return round(odds / (1 + odds), 6)


def from_config(cfg: dict | None) -> LossMatrix | None:
    """Build a LossMatrix from a use-case's `costs:` block, if present."""
    if not cfg:
        return None
    return LossMatrix(**{k: float(v) for k, v in cfg.items()})
