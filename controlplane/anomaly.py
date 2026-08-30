"""Statistical anomaly detection — "is this response unusual?"

The Round 2 brief names `embedding/statistical anomaly detection` as a detection
technique. This is that component, and it answers a question none of our other
checks ask.

  * Grounding asks: is this response supported by its source?
  * PSI asks:       has the population of scores shifted since we validated?
  * Anomaly asks:   is THIS ONE response, right now, unlike normal traffic?

The three are genuinely different. A response can be perfectly grounded and
still be anomalous — three times longer than anything we have seen, stuffed with
numbers, or structured unlike any response in the reference window. Those are
real failure signatures (prompt injection, a silent model swap, a broken
template) and neither grounding nor PSI will catch them.

METHOD
------
We build a profile of "normal" from a reference set of known-clean responses,
then score each new response two ways:

1. ROBUST Z-SCORES, per feature.
   Using median and MAD (median absolute deviation) rather than mean and
   standard deviation. This matters: the mean and SD are themselves dragged
   around by the very outliers we are trying to detect, so a contaminated
   reference set quietly hides its own anomalies. The median and MAD are not.
   Scaled by 1.4826 so that for Gaussian data the MAD estimates the SD.

2. MAHALANOBIS DISTANCE, over all features jointly.
   A response can be unremarkable on every feature individually and still be a
   strange COMBINATION — very long but with very few claims, say. Mahalanobis
   distance measures how far a point sits from the centre of the reference
   cloud while accounting for how the features covary, so it catches those.

   Under multivariate normality, squared Mahalanobis distance follows a
   chi-squared distribution with p degrees of freedom. That gives us an actual
   significance test with a stated null hypothesis — "this response is drawn
   from the same distribution as normal traffic" — rather than a hand-set
   cut-off. We reject at alpha = 0.01.

   Covariance is estimated with Ledoit-Wolf shrinkage because the reference set
   is small relative to the number of features, and the sample covariance is
   near-singular in that regime (its inverse, which Mahalanobis needs, would be
   numerically unstable).

DESIGN CHOICE: REPORTING, NOT BLOCKING
--------------------------------------
This detector is `async` and contributes NO risk to p_harm. An unusual response
is not the same as a harmful one, and conflating the two would inflate the harm
probability with something that is merely surprising. It surfaces a flag and a
p-value into the audit trail and the monitoring view, exactly as the cost
detector does.

If a deployment wants anomalies to force review, that is a one-line policy
change — add `{if: statistically_anomalous, action: review}` to a use case's
hard_rules. The decision stays in the governance layer where it belongs.
"""
from __future__ import annotations
import json
import math
import os
import re
import time

import numpy as np

from .schemas import Interaction, DetectorResult

# chi-squared critical values at alpha = 0.01, indexed by degrees of freedom.
# Hard-coded so the detector has no SciPy dependency.
_CHI2_01 = {1: 6.635, 2: 9.210, 3: 11.345, 4: 13.277, 5: 15.086,
            6: 16.812, 7: 18.475, 8: 20.090}
_SENT = re.compile(r"(?<=[.!?])\s+")

FEATURE_NAMES = ["log_length", "n_claims", "type_token_ratio",
                 "digit_density", "entity_density", "mean_sentence_len"]


def features(response: str, context: str = "") -> np.ndarray:
    """Six cheap structural features. No model call, no added latency."""
    r = response or ""
    words = r.split()
    n = max(len(words), 1)
    sentences = [s for s in _SENT.split(r) if s.strip()]
    digits = sum(ch.isdigit() for ch in r)
    entities = sum(1 for w in words[1:] if w[:1].isupper())
    return np.array([
        math.log1p(len(words)),
        len(sentences),
        len(set(w.lower() for w in words)) / n,
        digits / max(len(r), 1),
        entities / n,
        n / max(len(sentences), 1),
    ], dtype=float)


class AnomalyDetector:
    """Flags responses that are statistically unlike the reference window."""

    name = "anomaly"
    speed = "async"          # never on the user-facing critical path

    def __init__(self, reference_path: str = "data/interactions.jsonl",
                 alpha_df: int = len(FEATURE_NAMES)):
        self.median = None
        self.mad = None
        self.mean = None
        self.inv_cov = None
        self.n_reference = 0
        self.critical = _CHI2_01.get(alpha_df, 16.812)
        self._fit_from(reference_path)

    # ------------------------------------------------------------------ fit
    def _fit_from(self, path: str):
        if not os.path.exists(path):
            return
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Reference = known-clean responses only. Fitting "normal" on a
                # set that contains harmful examples would teach the detector
                # that harmful IS normal.
                if r.get("category") == "clean":
                    rows.append(features(r.get("response", ""), r.get("context", "")))
        if len(rows) < 5:
            return
        self.fit(np.vstack(rows))

    def fit(self, X: np.ndarray):
        self.n_reference = len(X)
        self.median = np.median(X, axis=0)
        # 1.4826 makes the MAD a consistent estimator of sigma under normality
        self.mad = np.median(np.abs(X - self.median), axis=0) * 1.4826
        # A degenerate MAD is real and common: if every clean response happens to
        # be one sentence, the MAD of n_claims is exactly 0 and any deviation
        # would divide by zero, producing meaningless z-scores in the billions.
        # Fall back to the standard deviation, then to a floor proportional to
        # the feature's own scale, so the z-score stays interpretable.
        sd = X.std(axis=0, ddof=1) if len(X) > 1 else np.zeros(X.shape[1])
        degenerate = self.mad < 1e-6
        self.mad = np.where(degenerate, sd, self.mad)
        still = self.mad < 1e-6
        self.mad = np.where(still, np.maximum(np.abs(self.median) * 0.10, 0.5), self.mad)
        self.mean = X.mean(axis=0)
        try:
            from sklearn.covariance import LedoitWolf
            cov = LedoitWolf().fit(X).covariance_
        except Exception:
            cov = np.cov(X, rowvar=False) + np.eye(X.shape[1]) * 1e-3
        self.inv_cov = np.linalg.pinv(cov)
        return self

    # ---------------------------------------------------------------- score
    def mahalanobis(self, x: np.ndarray) -> float:
        d = x - self.mean
        return float(d @ self.inv_cov @ d)

    def robust_z(self, x: np.ndarray) -> np.ndarray:
        return (x - self.median) / self.mad

    def run(self, x: Interaction) -> DetectorResult:
        t0 = time.perf_counter()
        if self.inv_cov is None:
            return DetectorResult(self.name, 0.0, {"anomaly_profile_missing": True},
                                  {"reason": "no reference profile fitted"},
                                  self.speed, (time.perf_counter() - t0) * 1000)

        f = features(x.response, x.context)
        d2 = self.mahalanobis(f)
        z = self.robust_z(f)
        worst = int(np.argmax(np.abs(z)))

        # Chi-squared tail probability, computed directly for even/odd df via
        # the regularised upper incomplete gamma function (no SciPy).
        p_value = _chi2_sf(d2, len(f))
        anomalous = d2 > self.critical

        detail = {
            "mahalanobis_d2": round(d2, 3),
            "chi2_critical_alpha_0.01": round(self.critical, 3),
            "p_value": round(p_value, 5),
            "n_reference": self.n_reference,
            "most_extreme_feature": FEATURE_NAMES[worst],
            "robust_z": {k: round(float(v), 2) for k, v in zip(FEATURE_NAMES, z)},
        }
        flags = {
            "statistically_anomalous": bool(anomalous),
            "extreme_single_feature": bool(np.max(np.abs(z)) > 3.5),
        }
        # Risk is reported as 0.0 by design: unusual is not the same as harmful,
        # so this never inflates p_harm. See module docstring.
        return DetectorResult(self.name, 0.0, flags, detail, self.speed,
                              (time.perf_counter() - t0) * 1000)


def _chi2_sf(x: float, df: int) -> float:
    """P(chi2_df > x), via the Wilson-Hilferty transformation.

    The direct series for the incomplete gamma function underflows badly once x
    is large relative to df, which is exactly the regime an anomaly detector
    lives in. Wilson-Hilferty maps the chi-squared to an approximately standard
    normal and is stable across the whole range:

        z = [ (x/df)^(1/3) - (1 - 2/(9df)) ] / sqrt(2/(9df))

    Accurate to about three decimals for df >= 3, which is far finer than any
    decision we make with it.
    """
    if x <= 0:
        return 1.0
    c = 2.0 / (9.0 * df)
    z = ((x / df) ** (1.0 / 3.0) - (1.0 - c)) / math.sqrt(c)
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))
