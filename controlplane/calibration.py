"""Turn raw detector scores into honest probabilities — and know when to abstain.

A raw grounding-risk of 0.74 is just a number. Calibration maps it to a real
probability: "0.74 -> 71% of responses that score this high are actually
hallucinated." We fit that mapping on labelled data (isotonic regression), then
measure honesty with the Brier score and Expected Calibration Error (ECE).

Abstention (the "reject option"): when the calibrated probability sits in an
uncertain middle band, the system does not guess — it escalates to a human.
This trades coverage for accuracy, exactly as a risk desk would.

Serialised as plain JSON (interpolation knots), so inference needs only NumPy.
"""
from __future__ import annotations
import json, os
import numpy as np


class Calibrator:
    def __init__(self, x=None, y=None):
        # monotonic interpolation knots: raw score -> calibrated probability
        self.x = np.asarray(x, float) if x is not None else None
        self.y = np.asarray(y, float) if y is not None else None

    def fit(self, scores, labels):
        from sklearn.isotonic import IsotonicRegression
        s = np.asarray(scores, float)
        l = np.asarray(labels, float)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(s, l)
        # store a compact set of knots for JSON-only inference
        grid = np.linspace(s.min(), s.max(), 50)
        self.x = grid
        self.y = np.clip(iso.predict(grid), 1e-4, 1 - 1e-4)
        return self

    def predict(self, scores):
        s = np.asarray(scores, float)
        if self.x is None:
            return s  # identity if unfitted
        return np.interp(s, self.x, self.y)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump({"x": self.x.tolist(), "y": self.y.tolist()},
                  open(path, "w"))

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        return cls(d["x"], d["y"])


# ---- honesty metrics ----
def brier(labels, probs):
    labels, probs = np.asarray(labels, float), np.asarray(probs, float)
    return float(np.mean((probs - labels) ** 2))

def ece(labels, probs, bins=10):
    """Expected Calibration Error: avg gap between confidence and accuracy."""
    labels, probs = np.asarray(labels, float), np.asarray(probs, float)
    edges = np.linspace(0, 1, bins + 1)
    e, n = 0.0, len(labels)
    for i in range(bins):
        m = (probs >= edges[i]) & (probs < edges[i + 1] if i < bins - 1 else probs <= 1.0)
        if m.sum() == 0:
            continue
        conf = probs[m].mean()
        acc = labels[m].mean()
        e += (m.sum() / n) * abs(conf - acc)
    return float(e)

def reliability_curve(labels, probs, bins=10):
    labels, probs = np.asarray(labels, float), np.asarray(probs, float)
    edges = np.linspace(0, 1, bins + 1)
    xs, ys = [], []
    for i in range(bins):
        m = (probs >= edges[i]) & (probs < edges[i + 1] if i < bins - 1 else probs <= 1.0)
        if m.sum() == 0:
            continue
        xs.append(probs[m].mean())
        ys.append(labels[m].mean())
    return np.array(xs), np.array(ys)


# ---- abstention (reject option) ----
def decide_with_abstention(p, low=0.40, high=0.60):
    """Return 'flag' / 'pass' / 'abstain' from a calibrated probability p.
    The [low, high] band is the uncertain zone -> escalate to a human."""
    if low <= p <= high:
        return "abstain"
    return "flag" if p > high else "pass"

def widen(p: float, ood_severity: float = 0.0, n_effective: int = 200,
          z: float = 1.96) -> dict:
    """Return a conservative P(harm) for the decision layer, given how far the
    response sits outside the envelope the calibrator was validated on.

    THE PRINCIPLE. Anomaly detection contributes nothing to the point estimate
    of P(harm) — an unusual response is not a more harmful one, and adding
    surprise to a harm probability would corrupt the expected-loss arithmetic
    the decision rests on. But an observation outside the validated domain does
    mean something specific: the calibrated number is LESS TRUSTWORTHY than
    usual. So OOD does not move the estimate; it widens the interval around it,
    and the policy decides on the upper end. This is ordinary model-risk
    practice — a model used outside its validated domain gets a conservative
    add-on, not a different point estimate.

    THE MECHANICS. The isotonic calibrator estimates each probability from a bin
    of finite size, so it already carries sampling error: a Wilson interval on
    n_effective observations. Out-of-domain severity shrinks the effective
    sample size, because observations from a distribution the calibrator never
    saw are worth less than ones it was fitted on. At severity 1.0 the effective
    n collapses to a handful and the interval is wide enough that the decision
    layer escalates on uncertainty alone — the correct behaviour when the system
    genuinely does not know.

    At ood_severity == 0 the widening is exactly ZERO — p_decision == p_point —
    so NORMAL traffic passes through the policy unchanged and the pre-1C
    decisions are preserved. The interval opens only as OOD severity rises, so
    only genuinely anomalous responses are pushed upward. We take the Wilson
    upper half-width at the shrunk effective n and scale it by severity, which
    keeps the widening 0 at s=0 and monotone in s. Returns the point estimate,
    the widened decision value, and the effective n, so all three land in the
    audit trail.
    """
    import numpy as _np
    p = min(max(float(p), 0.0), 1.0)
    s = min(max(float(ood_severity), 0.0), 1.0)
    if s <= 0.0:                                      # normal traffic: untouched
        return {"p_point": round(p, 4), "p_lower": round(p, 4),
                "p_decision": round(p, 4), "ood_severity": 0.0,
                "n_effective": float(n_effective), "widened_by": 0.0}
    n_eff = max(5.0, float(n_effective) * (1.0 - 0.95 * s))
    denom = 1.0 + z * z / n_eff
    centre = (p + z * z / (2 * n_eff)) / denom
    half = (z / denom) * _np.sqrt(p * (1 - p) / n_eff + z * z / (4 * n_eff * n_eff))
    # scale the add-on by severity so it grows from 0 (at s=0) to the full
    # Wilson upper bound (at s=1); never let widening reduce the estimate.
    hi = min(1.0, p + s * max(0.0, (centre + half) - p))
    lo = max(0.0, p - s * max(0.0, p - (centre - half)))
    return {"p_point": round(p, 4),
            "p_lower": round(float(lo), 4),
            "p_decision": round(float(hi), 4),
            "ood_severity": round(s, 3),
            "n_effective": round(n_eff, 1),
            "widened_by": round(float(hi) - p, 4)}


def coverage_accuracy(labels, probs, thr=0.5):
    """Risk-coverage curve: as we abstain on the least-confident cases, how does
    accuracy on the ones we DO answer improve? Returns (coverage, accuracy)."""
    labels, probs = np.asarray(labels, float), np.asarray(probs, float)
    conf = np.abs(probs - thr)                      # distance from the boundary
    order = np.argsort(-conf)                        # most confident first
    labels, probs = labels[order], probs[order]
    preds = (probs >= thr).astype(float)
    cov, acc = [], []
    for k in range(10, len(labels) + 1, max(1, len(labels) // 25)):
        cov.append(k / len(labels))
        acc.append(float((preds[:k] == labels[:k]).mean()))
    return np.array(cov), np.array(acc)
