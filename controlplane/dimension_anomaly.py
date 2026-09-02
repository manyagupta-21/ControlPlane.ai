"""Statistical anomaly detection, one component per risk dimension.

This is the piece v1 got structurally wrong. It shipped a single `AnomalyDetector`
that scored six generic text features and sat alongside performance,
responsibility and cost as though "unusual" were a fourth kind of risk. It is
not. It is a *method* (the Round 2 brief lists it as a detection technique), and
each of the three dimensions has its own quantity to apply it to:

    performance     is the SUPPORT PROFILE of this answer unusual, once you
                    control for how long the answer is?
    responsibility  is the ENTITY COUNT in this response a tail event against
                    the fitted rate, and is the population rate drifting?
    cost            is this response expensive GIVEN what was asked, and is
                    spend drifting against its control limits?

Each component answers a question the dimension's threshold rules cannot. A
boolean `pii_detected` cannot distinguish one incidental email address from a
dump of forty; a Poisson tail probability can. A fixed `regenerations >= 2` rule
cannot tell a busy hour from a broken prompt template; a control chart can.

Every component exposes two levels, because they catch different failures:

    per-response   is this item unusual?      -> a p-value in the audit trail
    windowed       has the population moved?  -> the control-chart / batch signal

and every component REPORTS rather than blocks. Unusual is not harmful, and
promoting a surprise into a harm probability would corrupt the expected-loss
arithmetic the decision layer depends on. (A later step feeds the anomaly signal
to the decision layer through interval-widening, which is the correct channel:
an observation outside the validated envelope makes the harm estimate less
reliable, not the harm more likely.)

This file is self-contained. It depends only on controlplane.stats and on the
claim/evidence splitters already in controlplane.grounding, so it can be dropped
in and tested without touching detectors.py, policy.py, or pipeline.py.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from .stats import (ReferenceProfile, EWMAControlChart, ResidualModel,
                    poisson_tail, norm_sf)
from .grounding import split_claims, split_evidence

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:                                    # pragma: no cover
    TfidfVectorizer = None

PROFILE_PATH = "data/anomaly_profiles.json"

# The features the performance anomaly monitors. Two families:
#
#   support-shape   describe the SHAPE of the grounding evidence — "unusual
#                   grounding behaviour", separate from "poorly grounded" which
#                   the grounding score already handles.
#   structural      cheap text statistics — length, lexical diversity, digit
#                   density. These are what catch a truncated output, a
#                   repetition loop, or a numeric dump, which the support shape
#                   alone cannot: a one-sentence answer and a 25x repeated
#                   sentence have identical support profiles but very different
#                   structure. (v1's original 6 features were all structural;
#                   dropping them entirely was a mistake, so both families are
#                   kept here.)
SUPPORT_NAMES = ["n_claims", "mean_support", "min_support", "std_support",
                 "frac_unsupported", "support_range",
                 "log_words", "type_token_ratio", "digit_density",
                 "mean_sentence_len", "max_run_repeat"]


# --------------------------------------------------------- feature extraction
def _claim_support(response: str, context: str):
    """Per-claim best support against the evidence (TF-IDF cosine), plus the
    response/context word counts. Reuses the same mechanism the grounding
    detector already uses, so no new modelling is introduced here."""
    claims = split_claims(response)
    evidence = split_evidence(context)
    rw = (response or "").split()
    cw = (context or "").split()
    if not claims or not evidence or TfidfVectorizer is None:
        return np.array([0.5]), len(rw), len(cw)
    try:
        M = TfidfVectorizer(stop_words="english").fit_transform(evidence + claims)
        sims = cosine_similarity(M[len(evidence):], M[:len(evidence)])
        return np.asarray(sims.max(axis=1), float), len(rw), len(cw)
    except ValueError:
        return np.array([0.5]), len(rw), len(cw)


def _max_repeat_run(words) -> int:
    """Longest run of an immediately repeated token — a cheap repetition-loop
    signal. 'a a a b' -> 3."""
    best = run = 1
    for i in range(1, len(words)):
        run = run + 1 if words[i] == words[i - 1] else 1
        best = max(best, run)
    return best


def support_features(response: str, context: str):
    """The feature vector monitored for anomalies (support-shape + structural),
    plus (log_resp_words, log_ctx_words, mean_support) used by the length-
    control regression."""
    import re
    support, n_r, n_c = _claim_support(response, context)
    support = np.asarray(support, float)
    r = response or ""
    words = r.split()
    n = max(len(words), 1)
    sents = [s for s in re.split(r"(?<=[.!?])\s+", r) if s.strip()]
    feat = np.array([
        len(support),                          # n_claims
        float(support.mean()),                 # mean_support
        float(support.min()),                  # min_support
        float(support.std()),                  # std_support
        float((support < 0.20).mean()),        # frac_unsupported
        float(support.max() - support.min()),  # support_range
        math.log1p(len(words)),                # log_words
        len(set(w.lower() for w in words)) / n,  # type_token_ratio
        sum(ch.isdigit() for ch in r) / max(len(r), 1),  # digit_density
        n / max(len(sents), 1),                # mean_sentence_len
        float(_max_repeat_run(words)),         # max_run_repeat
    ], dtype=float)
    aux = (math.log1p(n_r), math.log1p(n_c), float(support.mean()))
    return feat, aux


# ---------------------------------------------------------------- performance
class PerformanceAnomaly:
    """Anomalies in the *shape* of the grounding evidence.

    Grounding already asks "is this answer supported?". This asks a different
    question: "does this answer's support profile look like anything we have
    seen?" A response can be adequately supported on average and still be
    strange — support wildly uneven across claims, far more claims than any
    normal answer, or supported far better than anything in the reference set
    (a signature of a model echoing its context verbatim).

    LENGTH IS REGRESSED OUT FIRST. Long answers make more claims and therefore
    have mechanically lower minimum support whether or not they hallucinate, so
    an unconditional z-score on support is mostly a length detector. We fit
    E[mean_support | log_resp_words, log_ctx_words] and also test the
    studentised residual, so a flag means "poorly supported FOR ITS LENGTH".
    """

    def __init__(self):
        self.profile = ReferenceProfile(SUPPORT_NAMES, alpha=0.01)
        self.resid = ResidualModel()
        self._window: list = []
        self.window_size = 100

    def fit(self, responses, contexts) -> "PerformanceAnomaly":
        feats, aux = [], []
        for r, c in zip(responses, contexts):
            f, a = support_features(r, c)
            feats.append(f)
            aux.append(a)
        F = np.vstack(feats)
        A = np.array(aux)                            # (n, 3): logR, logC, mean_support
        self.profile.fit(F)
        self.resid.fit(A[:, :2], A[:, 2])
        return self

    def run(self, response: str, context: str) -> dict:
        if not self.profile.fitted:
            return {"fitted": False}
        f, a = support_features(response, context)
        out = self.profile.per_item(f)
        z = self.resid.z(np.array([[a[0], a[1]]]), a[2])
        out["support_residual_z"] = round(z, 2)
        out["support_residual_p"] = round(norm_sf(abs(z)) * 2, 5)
        out["underexplained_for_length"] = bool(z < -3.0)

        # Bounded 0-1 OOD severity for THIS response, used by the policy layer
        # (step 1C) to widen the confidence interval on P(harm). Anchored to the
        # chi-squared critical value at alpha: severity 0 at/below the envelope
        # boundary, rising to 1 well outside it. Normal responses -> ~0, so
        # normal decisions are essentially unchanged.
        from .stats import chi2_sf
        df = len(self.profile.names)
        # critical d^2 where chi2_sf == alpha (bisection, cheap, no scipy)
        lo, hi = 0.0, 200.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if chi2_sf(mid, df) > self.profile.alpha:
                lo = mid
            else:
                hi = mid
        crit = hi
        d2 = out["mahalanobis_d2"]
        out["ood_severity"] = round(float(min(1.0, max(0.0, (d2 / crit - 1.0)))), 3)

        self._window.append(f)
        if len(self._window) >= self.window_size:
            v = self.profile.batch(np.vstack(self._window[-self.window_size:]))
            out["window"] = {"n": v.n, "statistic": v.statistic,
                             "critical": v.critical, "p_value": v.p_value,
                             "population_shifted": v.shifted}
            self._window = self._window[-self.window_size:]
        return out


# -------------------------------------------------------------- responsibility
class ResponsibilityAnomaly:
    """Rate anomalies in PII and toxicity, at the item and the population level.

    ITEM: the number of PII entities in a response is a count, so it has a
    distribution. Against a fitted rate of 0.03 entities per response, seeing
    one is unremarkable and seeing eight has a Poisson tail probability around
    1e-14. `pii_detected: true` treats those identically, so the policy layer
    cannot distinguish an incidental match from a data dump — different
    incidents with different escalation paths.

    POPULATION: an EWMA control chart on the flag rate. A leak rate creeping
    from 0.2% to 1.5% trips no per-response rule, because every individual
    response looks exactly as it always did. That is the failure mode behind a
    breach notification, and only a sequential test sees it.
    """

    def __init__(self, pii_rate: float = 0.03, tox_rate: float = 0.01,
                 batch: int = 200):
        self.pii_rate = float(pii_rate)
        self.tox_rate = float(tox_rate)
        self.batch = batch
        self.pii_chart = EWMAControlChart.for_rate(max(pii_rate, 1e-4), batch)
        self.tox_chart = EWMAControlChart.for_rate(max(tox_rate, 1e-4), batch)
        self._pii_buf: list = []
        self._tox_buf: list = []

    def fit(self, pii_flags, tox_flags) -> "ResponsibilityAnomaly":
        p = float(np.mean(np.asarray(pii_flags, float))) if len(pii_flags) else self.pii_rate
        t = float(np.mean(np.asarray(tox_flags, float))) if len(tox_flags) else self.tox_rate
        self.__init__(max(p, 1e-4), max(t, 1e-4), self.batch)
        return self

    def run(self, pii_count: int, tox_score: float) -> dict:
        out = {
            "pii_count": int(pii_count),
            "pii_count_p_value": round(poisson_tail(int(pii_count), self.pii_rate), 8),
            "reference_pii_rate": round(self.pii_rate, 4),
        }
        out["pii_bulk_disclosure"] = bool(pii_count >= 3 and out["pii_count_p_value"] < 1e-4)

        self._pii_buf.append(1 if pii_count else 0)
        self._tox_buf.append(1 if tox_score >= 0.5 else 0)
        if len(self._pii_buf) >= self.batch:
            p = self.pii_chart.update(float(np.mean(self._pii_buf[-self.batch:])))
            t = self.tox_chart.update(float(np.mean(self._tox_buf[-self.batch:])))
            out["window"] = {"pii_rate_chart": p, "toxicity_rate_chart": t}
            out["rate_drift"] = bool(p["signal"] or t["signal"])
            self._pii_buf = self._pii_buf[-self.batch:]
            self._tox_buf = self._tox_buf[-self.batch:]
        return out


# ------------------------------------------------------------------------ cost
class CostAnomaly:
    """Cost anomalies, conditional on the request.

    v1 called a query easy if it was <= 15 words, added 0.5 to a "waste" score
    if a large model answered it, and 0.4 more if it had been regenerated twice.
    Three unestimated constants, none conditional on anything.

    Token counts are positive and heavy-tailed, so we work in logs, and the only
    interesting question is conditional: expensive *given what was asked*. We fit
    E[log response tokens | log query tokens] and studentise the residual, which
    gives a real z-score with a real tail probability instead of a constant.
    Regenerations are a count with a fitted rate, so `regenerations >= 2` becomes
    a Poisson tail probability — two regenerations is routine at a base rate of
    0.5 and a broken template at a base rate of 0.02, and only the second
    deserves a page.
    """

    def __init__(self, batch: int = 200):
        self.resid = ResidualModel()
        self.regen_rate = 0.10
        self.batch = batch

    def fit(self, query_tokens, response_tokens, regenerations) -> "CostAnomaly":
        q = np.log1p(np.asarray(query_tokens, float)).reshape(-1, 1)
        r = np.log1p(np.asarray(response_tokens, float))
        self.resid.fit(q, r)
        self.regen_rate = float(max(np.mean(np.asarray(regenerations, float)), 1e-3))
        return self

    def run(self, query_tokens: int, response_tokens: int, regenerations: int) -> dict:
        z = self.resid.z(np.array([[math.log1p(query_tokens)]]),
                         math.log1p(response_tokens))
        out = {
            "length_residual_z": round(z, 2),
            "verbose_for_query": bool(z > 3.0),
            "regenerations": int(regenerations),
            "regeneration_p_value": round(poisson_tail(int(regenerations), self.regen_rate), 6),
            "reference_regen_rate": round(self.regen_rate, 4),
        }
        out["rework_anomaly"] = bool(out["regeneration_p_value"] < 0.01)
        return out


# ------------------------------------------------------------- persistence
def save_profiles(perf: PerformanceAnomaly, resp: ResponsibilityAnomaly,
                  cost: CostAnomaly, path: str = PROFILE_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump({
        "performance": {"profile": perf.profile.to_dict(),
                        "resid": perf.resid.to_dict(),
                        "window_size": perf.window_size},
        "responsibility": {"pii_rate": resp.pii_rate, "tox_rate": resp.tox_rate,
                           "batch": resp.batch},
        "cost": {"resid": cost.resid.to_dict(), "regen_rate": cost.regen_rate,
                 "batch": cost.batch},
    }, open(path, "w"), indent=1)


def load_profiles(path: str = PROFILE_PATH):
    """Returns (PerformanceAnomaly, ResponsibilityAnomaly, CostAnomaly).

    Unfitted components if the artefact is missing, so a fresh clone still runs;
    each then reports `fitted: false` rather than pretending to a null it never
    estimated.
    """
    perf, resp, cost = PerformanceAnomaly(), ResponsibilityAnomaly(), CostAnomaly()
    if not os.path.exists(path):
        return perf, resp, cost
    try:
        d = json.load(open(path, encoding="utf-8"))
        perf.profile = ReferenceProfile.from_dict(d["performance"]["profile"])
        perf.resid = ResidualModel.from_dict(d["performance"]["resid"])
        perf.window_size = d["performance"].get("window_size", 100)
        rr = d["responsibility"]
        resp = ResponsibilityAnomaly(rr["pii_rate"], rr["tox_rate"], rr["batch"])
        cc = d["cost"]
        cost = CostAnomaly(cc["batch"])
        cost.resid = ResidualModel.from_dict(cc["resid"])
        cost.regen_rate = cc["regen_rate"]
    except Exception:
        pass
    return perf, resp, cost
