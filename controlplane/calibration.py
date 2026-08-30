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
