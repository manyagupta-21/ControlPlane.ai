"""Shared statistical machinery for anomaly detection.

This file provides the statistical primitives once, so that each of the three
risk dimensions (performance, responsibility, cost) can apply them to its own
quantities instead of a single generic detector scoring generic text features.

Nothing here is dimension-specific. It is pure statistics with no SciPy
dependency — the chi-squared and normal tails are computed directly.

Provided:
    ReferenceProfile   a fitted description of "normal" for one block of
                       features, supporting BOTH a per-item test ("is THIS
                       observation unusual?") and a windowed test ("has the
                       POPULATION moved?"). The windowed test is the piece the
                       v1 anomaly detector was missing.
    EWMAControlChart   sequential monitoring of a rate or a mean, so a slow
                       drift is caught before any single item crosses a limit.
    ResidualModel      studentised residual from a conditional mean, for any
                       feature with a strong nuisance covariate (e.g. grounding
                       support depends on response length; token count depends
                       on query length).
    poisson_tail       exact upper-tail probability for a count against a rate
                       (PII entities per response, regenerations).
    chi2_sf, norm_sf   tail probabilities used by the above.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["ReferenceProfile", "EWMAControlChart", "ResidualModel",
           "poisson_tail", "chi2_sf", "norm_sf", "BatchVerdict"]


# --------------------------------------------------------------------- tails
def norm_sf(z: float) -> float:
    """P(Z > z) for a standard normal."""
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


def chi2_sf(x: float, df: int) -> float:
    """P(chi2_df > x) via the Wilson-Hilferty transformation.

    The direct incomplete-gamma series underflows once x is large relative to
    df, which is exactly the regime an anomaly detector lives in. Wilson-
    Hilferty maps chi-squared to an approximate standard normal and is stable
    across the whole range, accurate to about three decimals for df >= 3.
    """
    if x <= 0:
        return 1.0
    c = 2.0 / (9.0 * df)
    z = ((x / df) ** (1.0 / 3.0) - (1.0 - c)) / math.sqrt(c)
    return norm_sf(z)


def poisson_tail(k: int, lam: float) -> float:
    """P(X >= k) for X ~ Poisson(lam). Exact, by summing the lower tail.

    Used for counts: how surprising is it to see 7 PII entities in one response
    when the reference rate is 0.03 per response? A boolean `pii_detected` flag
    cannot express that; a tail probability can, and it is what lets the policy
    layer distinguish an incidental match from a data dump.
    """
    if k <= 0:
        return 1.0
    lam = max(float(lam), 1e-9)
    if lam > 500:                                   # normal approximation
        return norm_sf((k - 0.5 - lam) / math.sqrt(lam))
    term = cum = math.exp(-lam)
    for i in range(1, int(k)):
        term *= lam / i
        cum += term
    return float(min(1.0, max(0.0, 1.0 - cum)))


# ------------------------------------------------------------ reference profile
@dataclass
class BatchVerdict:
    """Result of the windowed two-sample test."""
    n: int
    statistic: float
    critical: float
    p_value: float
    shifted: bool
    note: str = ""


class ReferenceProfile:
    """A fitted description of "normal" for one block of features.

    Each risk dimension owns one of these over its own quantities. The profile
    supports two tests that answer genuinely different questions:

        per_item(x)  is THIS observation unusual?   (chi-2 on Mahalanobis d^2)
        batch(X)     has the POPULATION moved?       (windowed mean d^2 against
                                                      a bootstrapped in-domain
                                                      null)

    Covariance is estimated with Ledoit-Wolf shrinkage: the reference set is
    typically small relative to the feature count, and the sample covariance is
    near-singular in that regime, so its inverse — which Mahalanobis needs — is
    numerically unstable. Location and scale for the marginal z-scores use the
    median and MAD, because the mean and SD are themselves dragged around by the
    outliers we are trying to find.
    """

    def __init__(self, names, alpha: float = 0.01):
        self.names = list(names)
        self.alpha = alpha
        self.median = self.mad = self.mean = self.inv_cov = None
        self.n_reference = 0
        self._ref_d2 = None                          # in-domain null for batch()
        self._null_cache: dict = {}

    # ------------------------------------------------------------------- fit
    def fit(self, X: np.ndarray) -> "ReferenceProfile":
        X = np.atleast_2d(np.asarray(X, float))
        if X.shape[0] < 5:
            return self
        self.n_reference = X.shape[0]
        self.median = np.median(X, axis=0)
        # 1.4826 makes the MAD a consistent estimator of sigma under normality.
        self.mad = np.median(np.abs(X - self.median), axis=0) * 1.4826
        # A degenerate MAD is real and common: if every clean response is one
        # sentence, the MAD of n_claims is exactly 0 and any deviation divides
        # by zero. Fall back to the SD, then to a floor proportional to the
        # feature's own scale.
        sd = X.std(axis=0, ddof=1) if len(X) > 1 else np.zeros(X.shape[1])
        self.mad = np.where(self.mad < 1e-6, sd, self.mad)
        self.mad = np.where(self.mad < 1e-6,
                            np.maximum(np.abs(self.median) * 0.10, 0.5), self.mad)
        self.mean = X.mean(axis=0)
        try:
            from sklearn.covariance import LedoitWolf
            cov = LedoitWolf().fit(X).covariance_
        except Exception:
            cov = np.cov(X, rowvar=False) + np.eye(X.shape[1]) * 1e-3
        self.inv_cov = np.linalg.pinv(np.atleast_2d(cov))
        self._ref_d2 = self._d2(X)
        self._null_cache.clear()
        return self

    @property
    def fitted(self) -> bool:
        return self.inv_cov is not None

    # ---------------------------------------------------------------- scoring
    def _d2(self, X: np.ndarray) -> np.ndarray:
        D = np.atleast_2d(np.asarray(X, float)) - self.mean
        return np.einsum("ij,jk,ik->i", D, self.inv_cov, D)

    def mahalanobis(self, x) -> float:
        return float(self._d2(np.asarray(x, float).reshape(1, -1))[0])

    def robust_z(self, x) -> np.ndarray:
        return (np.asarray(x, float) - self.median) / self.mad

    def per_item(self, x) -> dict:
        """Is this single observation unusual? Chi-squared test on d^2."""
        if not self.fitted:
            return {"fitted": False}
        x = np.asarray(x, float)
        d2 = self.mahalanobis(x)
        z = self.robust_z(x)
        worst = int(np.argmax(np.abs(z)))
        p = chi2_sf(d2, len(self.names))
        return {
            "fitted": True,
            "mahalanobis_d2": round(d2, 3),
            "p_value": round(p, 6),
            "anomalous": bool(p < self.alpha),
            "most_extreme_feature": self.names[worst],
            "max_abs_z": round(float(np.abs(z).max()), 2),
            "robust_z": {k: round(float(v), 2) for k, v in zip(self.names, z)},
            "n_reference": self.n_reference,
        }

    # ------------------------------------------------------------ batch test
    def _null(self, n: int, draws: int = 4000, seed: int = 0):
        """Bootstrap the sampling distribution of the window statistic under the
        in-domain reference. Cached per window size."""
        if n in self._null_cache:
            return self._null_cache[n]
        rng = np.random.default_rng(seed)
        ref = self._ref_d2
        idx = rng.integers(0, len(ref), size=(draws, min(n, len(ref))))
        null = ref[idx].mean(axis=1)
        out = (float(np.quantile(null, 1 - self.alpha)), null)
        self._null_cache[n] = out
        return out

    def batch(self, X: np.ndarray) -> BatchVerdict:
        """Has the population moved? Mean d^2 over a window against the
        bootstrapped in-domain null.

        This is the test with power against domain / population shift. A
        per-item test run one item at a time is a test at n = 1 and has almost
        no power against a distributional change; pooling into a window is what
        makes the change detectable.
        """
        if not self.fitted:
            return BatchVerdict(0, 0.0, 0.0, 1.0, False, "profile not fitted")
        X = np.atleast_2d(np.asarray(X, float))
        n = X.shape[0]
        stat = float(self._d2(X).mean())
        crit, null = self._null(n)
        p = float((null >= stat).mean())
        note = ""
        if n < 25:
            note = ("window is small; this test has little power below n~50 — "
                    "treat a non-detection as uninformative, not as evidence "
                    "of stability")
        return BatchVerdict(n, round(stat, 3), round(crit, 3), round(p, 5),
                            bool(stat > crit), note)

    # ------------------------------------------------------- OOD severity 0-1
    def severity(self, X: np.ndarray) -> float:
        """A bounded 0-1 measure of how far outside the validated envelope the
        traffic sits. Later steps use this to WIDEN the confidence interval on
        P(harm) rather than to add risk: a model scored on inputs unlike
        anything it was validated on should return a less certain number, not a
        different one.
        """
        if not self.fitted:
            return 0.0
        X = np.atleast_2d(np.asarray(X, float))
        stat = float(self._d2(X).mean())
        base = float(np.median(self._ref_d2)) + 1e-9
        return float(min(1.0, max(0.0, (stat / base - 1.0) / 2.0)))

    # --------------------------------------------------------- (de)serialise
    def to_dict(self) -> dict:
        return {"names": self.names, "alpha": self.alpha,
                "median": self.median.tolist(), "mad": self.mad.tolist(),
                "mean": self.mean.tolist(), "inv_cov": self.inv_cov.tolist(),
                "ref_d2": self._ref_d2.tolist(), "n_reference": self.n_reference}

    @classmethod
    def from_dict(cls, d: dict) -> "ReferenceProfile":
        p = cls(d["names"], d.get("alpha", 0.01))
        p.median = np.array(d["median"])
        p.mad = np.array(d["mad"])
        p.mean = np.array(d["mean"])
        p.inv_cov = np.array(d["inv_cov"])
        p._ref_d2 = np.array(d["ref_d2"])
        p.n_reference = d["n_reference"]
        return p


# --------------------------------------------------------------- control chart
class EWMAControlChart:
    """Sequential monitoring of a rate or a mean (Roberts, 1959).

    A threshold on a single observation cannot see a slow drift: a PII leak rate
    creeping from 0.2% to 1.5% never trips a per-response rule, because every
    individual response looks the same as always. An EWMA chart integrates the
    signal over time and fires on the trend.

        z_t = lam * x_t + (1 - lam) * z_{t-1}
        limits = mu0 +/- L * sigma0 * sqrt( lam/(2-lam) * (1 - (1-lam)^(2t)) )

    The time-varying width matters at start-up: using the asymptotic limit from
    t = 1 makes the chart far too tight on its first few observations and
    generates false alarms exactly when an operator is deciding whether to trust
    it.
    """

    def __init__(self, mu0: float, sigma0: float, lam: float = 0.2, L: float = 3.0):
        self.mu0, self.sigma0 = float(mu0), max(float(sigma0), 1e-9)
        self.lam, self.L = lam, L
        self.z = float(mu0)
        self.t = 0

    def update(self, x: float) -> dict:
        self.t += 1
        self.z = self.lam * float(x) + (1 - self.lam) * self.z
        width = self.L * self.sigma0 * math.sqrt(
            self.lam / (2 - self.lam) * (1 - (1 - self.lam) ** (2 * self.t)))
        ucl, lcl = self.mu0 + width, self.mu0 - width
        return {"t": self.t, "ewma": round(self.z, 5),
                "ucl": round(ucl, 5), "lcl": round(lcl, 5),
                "signal": bool(self.z > ucl or self.z < lcl),
                "direction": "up" if self.z > ucl else ("down" if self.z < lcl else None)}

    @classmethod
    def for_rate(cls, p0: float, n_per_period: int, lam: float = 0.2, L: float = 3.0):
        """Chart for a binomial rate — PII flag rate, block rate, override rate."""
        p0 = min(max(float(p0), 1e-6), 1 - 1e-6)
        return cls(p0, math.sqrt(p0 * (1 - p0) / max(n_per_period, 1)), lam, L)


# ------------------------------------------------------------ residual model
class ResidualModel:
    """Studentised residual from a conditional mean: y | x.

    Needed wherever a feature has a strong nuisance covariate. Two cases in this
    system, both real:

      * grounding support falls mechanically with response length, because a
        longer answer has more claims and therefore a lower minimum support;
      * token count rises with query length, so "unusually expensive" only means
        anything conditional on what was asked.

    Ordinary least squares on a small design, with the residual scaled by the
    reference residual SD. Deliberately linear: it is a nuisance adjustment, and
    a flexible fit here would absorb the very signal we want left in.
    """

    def __init__(self):
        self.beta = None
        self.sigma = 1.0

    @staticmethod
    def _design(x) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, float))
        return np.hstack([np.ones((x.shape[0], 1)), x])

    def fit(self, x, y) -> "ResidualModel":
        A = self._design(x)
        y = np.asarray(y, float).ravel()
        if len(y) < max(5, A.shape[1] + 2):
            return self
        self.beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ self.beta
        self.sigma = float(max(np.std(resid, ddof=A.shape[1]), 1e-6))
        return self

    def z(self, x, y) -> float:
        if self.beta is None:
            return 0.0
        pred = float(np.ravel(self._design(x) @ self.beta)[0])
        return float((float(y) - pred) / self.sigma)

    def to_dict(self) -> dict:
        return {"beta": self.beta.tolist() if self.beta is not None else None,
                "sigma": self.sigma}

    @classmethod
    def from_dict(cls, d: dict) -> "ResidualModel":
        m = cls()
        if d.get("beta") is not None:
            m.beta = np.array(d["beta"])
        m.sigma = d.get("sigma", 1.0)
        return m
