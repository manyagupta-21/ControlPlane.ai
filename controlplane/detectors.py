"""Detection engines.

Day-1 versions are deliberately lightweight so the whole pipeline runs offline
with no model downloads. Each detector exposes the SAME interface, so a heavier
backend (Detoxify, Presidio, a sentence-transformer, or an LLM-as-judge) can be
swapped in later without touching the pipeline or policy layers.

    detector.run(interaction) -> DetectorResult
"""
from __future__ import annotations
import re, time
from .schemas import Interaction, DetectorResult

# ---------------------------------------------------------------------------
# Statistical anomaly detection is a METHOD each dimension applies to its own
# quantities, not a fourth detector. The three per-dimension components share
# one fitted artefact (data/anomaly_profiles.json, written by
# scripts/fit_profiles.py) and are loaded ONCE per process here, so every
# detector instance reuses the same profile rather than re-reading the file.
#
# STEP 1B CONTRACT: the anomaly output rides along in DetectorResult.detail and
# sets a few new boolean flags. It does NOT change any detector's `risk`, and it
# does NOT feed p_harm. Decisions are byte-for-byte identical to before this
# step. (Step 1C is where the anomaly signal is allowed to affect a decision,
# through interval widening in the policy layer — a separate, visible change.)
# ---------------------------------------------------------------------------
_PROFILES = None

def _profiles():
    global _PROFILES
    if _PROFILES is None:
        from .dimension_anomaly import load_profiles
        _PROFILES = load_profiles()
    return _PROFILES

# ---------------------------------------------------------------------------
# Lightweight text similarity (TF-IDF cosine) with no external downloads.
# Used for grounding (response vs source context) and self-consistency.
# ---------------------------------------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def _cosine(a: str, b: str) -> float:
    a, b = (a or "").strip(), (b or "").strip()
    if not a or not b:
        return 0.0
    try:
        m = TfidfVectorizer(stop_words="english").fit_transform([a, b])
        return float(cosine_similarity(m[0], m[1])[0][0])
    except ValueError:
        return 0.0


class PerformanceDetector:
    """Is the answer right, or confidently wrong?

    Two independent signals:
      * grounding      - overlap between the response and its source context.
      * self-consistency - agreement across resampled generations.
    Low grounding OR low consistency => higher hallucination risk.
    """
    name = "performance"
    speed = "async"  # heavier check -> runs in parallel, not inline

    def __init__(self, grounding=None, ungrounded_thr: float = 0.65):
        # grounding: swappable backend from controlplane.grounding. Default is
        # the offline claim-level TF-IDF faithfulness scorer (validated on
        # RAGTruth); swap for the embedding backend locally for higher AUROC.
        if grounding is None:
            import os
            from .grounding import get_backend
            grounding = get_backend(os.environ.get("CONTROLPLANE_GROUNDING", "tfidf"))
        self.grounding = grounding
        self.ungrounded_thr = ungrounded_thr
        # per-dimension anomaly: is this answer's SUPPORT PROFILE + structure
        # unusual, once response length is regressed out? (dimension_anomaly.py)
        self.anomaly = _profiles()[0]
        # optional calibration + abstention layer (fitted by scripts/calibrate.py)
        self.calibrator = None
        self.abstain_band = (0.40, 0.60)
        try:
            import os
            from .calibration import Calibrator
            if os.path.exists("data/calibrator.json"):
                self.calibrator = Calibrator.load("data/calibrator.json")
        except Exception:
            self.calibrator = None

    def run(self, x: Interaction) -> DetectorResult:
        t0 = time.perf_counter()
        if x.context:
            grounding_risk, gdetail = self.grounding.score(x.response, x.context)
        else:
            grounding_risk, gdetail = None, {}
        if x.samples:
            sims = [_cosine(x.response, s) for s in x.samples]
            consistency = sum(sims) / len(sims)
        else:
            consistency = None

        # Map signals -> risk. Missing signal contributes nothing (None).
        parts = []
        if grounding_risk is not None:
            parts.append(grounding_risk)             # already a risk in [0,1]
        if consistency is not None:
            parts.append(1.0 - consistency)          # inconsistent -> risk
        risk = max(parts) if parts else 0.0

        flags = {
            "ungrounded": (grounding_risk is not None and grounding_risk >= self.ungrounded_thr),
            "inconsistent": (consistency is not None and consistency < 0.50),
        }
        detail = {"grounding_risk": round(grounding_risk, 3) if grounding_risk is not None else None,
                  "self_consistency": round(consistency, 3) if consistency is not None else None,
                  **{f"g_{k}": v for k, v in gdetail.items()}}

        # --- statistical anomaly, performance dimension --------------------
        # Reported, never added to `risk`: an unusual support profile is a
        # reason to distrust the probability, not evidence of harm. It reaches
        # the decision layer (in step 1C) through interval widening, not here.
        if x.context:
            a = self.anomaly.run(x.response, x.context)
            detail["anomaly"] = a
            if a.get("fitted"):
                flags["support_profile_anomalous"] = bool(a.get("anomalous"))
                flags["underexplained_for_length"] = bool(a.get("underexplained_for_length"))
                if (a.get("window") or {}).get("population_shifted"):
                    flags["performance_population_shift"] = True

        # calibration + abstention: map raw grounding risk -> honest probability,
        # and flag the uncertain middle band for escalation to a human.
        if self.calibrator is not None and grounding_risk is not None:
            cal_p = float(self.calibrator.predict([grounding_risk])[0])
            detail["calibrated_hallucination_prob"] = round(cal_p, 3)
            flags["uncertain_grounding"] = (self.abstain_band[0] <= cal_p <= self.abstain_band[1])

        return DetectorResult(self.name, round(risk, 3), flags, detail, self.speed,
                              (time.perf_counter() - t0) * 1000)


class ResponsibilityDetector:
    """Is the answer unsafe, biased, or leaking data?

    * PII   - regex entity detection (email, phone, PAN, Aadhaar, card, SSN).
    * toxicity/bias - lexicon match (baseline; swap for Detoxify later).
    """
    name = "responsibility"
    speed = "inline"  # cheap + safety-critical -> runs inline before the stream

    PII_PATTERNS = {
        "email": r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
        "phone_in": r"(?:\+91[\-\s]?)?[6-9]\d{9}\b",
        "pan": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
        "aadhaar": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
        "card": r"\b(?:\d[ \-]*?){13,16}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    }
    TOXIC_LEXICON = {
        "idiot", "stupid", "moron", "worthless", "hate", "trash", "useless",
        "inferior", "disgusting", "pathetic", "shut up",
    }
    # crude bias cue: sweeping generalisations about groups
    BIAS_CUES = [r"\ball (?:men|women|people from|immigrants|muslims|hindus)\b",
                 r"\b(?:men|women) are (?:worse|better|inferior|superior)\b"]

    def __init__(self):
        import os
        # per-dimension anomaly: Poisson tail on the PII entity COUNT + an EWMA
        # control chart on the population flag rate (dimension_anomaly.py).
        self.anomaly = _profiles()[1]
        # Optional production PII backend (Microsoft Presidio). Enable with
        # CONTROLPLANE_PII=presidio + `pip install presidio-analyzer`; falls back
        # to the regex patterns automatically if unavailable.
        self._analyzer = None
        if os.environ.get("CONTROLPLANE_PII", "regex").lower() == "presidio":
            try:
                from presidio_analyzer import AnalyzerEngine
                self._analyzer = AnalyzerEngine()
            except Exception as e:
                print(f"[responsibility] Presidio unavailable ({e}); using regex PII.")

        # Optional trained toxicity backend (Detoxify). Enable with
        # CONTROLPLANE_TOXICITY=detoxify + `pip install detoxify`; falls back to
        # the lexicon automatically if unavailable or if loading the model
        # fails for any reason (no network on first run, no torch, etc). The
        # lexicon's F1=1.00 on our own synthetic set is circular by
        # construction (see README) - Detoxify is a real learned classifier
        # trained on the Jigsaw toxic-comment corpus and is the honest upgrade.
        self._detoxify = None
        if os.environ.get("CONTROLPLANE_TOXICITY", "lexicon").lower() == "detoxify":
            try:
                from detoxify import Detoxify
                self._detoxify = Detoxify("original")
            except Exception as e:
                print(f"[responsibility] Detoxify unavailable ({e}); using lexicon.")

    def _detect_pii(self, text: str):
        if self._analyzer is not None:
            results = self._analyzer.analyze(text=text, language="en")
            ents = {}
            for r in results:
                if r.score >= 0.5:
                    ents[r.entity_type] = ents.get(r.entity_type, 0) + 1
            return ents
        # regex fallback
        ents = {}
        for kind, pat in self.PII_PATTERNS.items():
            found = re.findall(pat, text)
            if kind == "card":
                found = [f for f in found if len(re.sub(r"\D", "", f)) >= 13]
            if found:
                ents[kind] = len(found)
        return ents

    def run(self, x: Interaction) -> DetectorResult:
        t0 = time.perf_counter()
        text = x.response or ""
        low = text.lower()

        entities = self._detect_pii(text)
        pii_count = sum(entities.values())

        if self._detoxify is not None:
            # Detoxify returns per-category probabilities (toxicity, severe_
            # toxicity, obscene, threat, insult, identity_attack); take the
            # worst category as the risk, same "weakest link" logic as
            # grounding (one bad dimension should not be averaged away).
            preds = self._detoxify.predict(text)
            tox_score = float(max(preds.values())) if preds else 0.0
            tox_hits = [k for k, v in preds.items() if v >= 0.5]
            bias_hits = []   # Detoxify's identity_attack subsumes our bias cues
        else:
            tox_hits = [w for w in self.TOXIC_LEXICON if w in low]
            bias_hits = [p for p in self.BIAS_CUES if re.search(p, low)]
            # one detected toxic/bias cue is enough to be "high" — we bias
            # toward escalation, since a missed toxic reply costs far more
            # than an unnecessary review.
            tox_score = min(1.0, 0.5 * (len(tox_hits) + len(bias_hits)))

        # responsibility risk: PII presence is high-risk on its own
        risk = max(0.9 if pii_count else 0.0, tox_score)
        # `toxicity_any` needs a small floor rather than a bare `> 0`. The
        # lexicon returns exactly 0.0 for clean text, but a learned classifier
        # returns a small non-zero probability for everything, so `> 0` would
        # flag every benign response. 0.05 keeps lexicon behaviour identical
        # (its scores are 0.0 or >= 0.5) while making the flag meaningful for
        # a probabilistic backend.
        flags = {
            "pii_detected": pii_count > 0,
            "toxicity_high": tox_score >= 0.5,
            "toxicity_any": tox_score >= 0.05,
        }
        # --- statistical anomaly, responsibility dimension ----------------
        # `pii_detected` cannot tell one incidental email from a dump of forty.
        # A Poisson tail against the fitted rate can, and a bulk disclosure is a
        # distinct incident with a distinct escalation path.
        a = self.anomaly.run(pii_count, tox_score)
        flags["pii_bulk_disclosure"] = bool(a.get("pii_bulk_disclosure"))
        if a.get("rate_drift"):
            flags["responsibility_rate_drift"] = True

        detail = {"pii_entities": entities, "toxic_terms": tox_hits,
                  "bias_cues": len(bias_hits), "anomaly": a}
        return DetectorResult(self.name, round(risk, 3), flags, detail, self.speed,
                              (time.perf_counter() - t0) * 1000)


class CostDetector:
    """Is the answer quietly expensive?

    Estimates token cost, flags easy queries answered by an oversized model,
    and treats repeated regenerations as a rework signal. Cost never *blocks*
    a response — it is an efficiency signal surfaced to monitoring.
    """
    name = "cost"
    speed = "async"

    PRICE_PER_1K = {"small": 0.0005, "large": 0.010}   # illustrative $/1k tokens

    def __init__(self):
        # per-dimension anomaly: log-token residual conditional on the query,
        # plus a Poisson tail on regenerations (dimension_anomaly.py).
        self.anomaly = _profiles()[2]

    def _tokens(self, text: str) -> int:
        return int(len(text.split()) * 1.3) + 1

    def run(self, x: Interaction) -> DetectorResult:
        t0 = time.perf_counter()
        toks = self._tokens(x.query) + self._tokens(x.response)
        price = self.PRICE_PER_1K.get(x.model_used, self.PRICE_PER_1K["large"])
        cost = toks / 1000 * price * (1 + x.regenerations)

        easy = len(x.query.split()) <= 15            # simple heuristic difficulty
        oversized = easy and x.model_used == "large"
        # risk here = "efficiency waste", used for reporting not blocking
        waste = 0.0
        if oversized:
            waste += 0.5
        if x.regenerations >= 2:
            waste += 0.4
        risk = min(1.0, waste)

        # --- statistical anomaly, cost dimension --------------------------
        # Conditional on the query, not a flat rule: `verbose_for_query` when
        # the answer is far longer than its query predicts, and a Poisson tail
        # on regenerations instead of `>= 2`.
        a = self.anomaly.run(self._tokens(x.query), self._tokens(x.response),
                             x.regenerations)
        flags = {"oversized_model": oversized,
                 "excess_regenerations": x.regenerations >= 2,
                 "verbose_for_query": bool(a.get("verbose_for_query")),
                 "rework_anomaly": bool(a.get("rework_anomaly"))}
        detail = {"est_tokens": toks, "est_cost_usd": round(cost, 5),
                  "model_used": x.model_used, "regenerations": x.regenerations,
                  "recommended_model": "small" if easy else "large",
                  "anomaly": a}
        return DetectorResult(self.name, round(risk, 3), flags, detail, self.speed,
                              (time.perf_counter() - t0) * 1000)


def default_detectors(grounding=None):
    # The old standalone AnomalyDetector is gone: anomaly detection now lives
    # inside each of the three dimensions (see the `.anomaly` component on each
    # detector above). controlplane/anomaly.py and scripts/anomaly_demo.py are
    # now dead code and can be deleted.
    return [PerformanceDetector(grounding=grounding), ResponsibilityDetector(),
            CostDetector()]
