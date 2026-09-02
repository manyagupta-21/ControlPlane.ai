"""Claim-level grounding / faithfulness scoring — built from scratch.

A response is grounded if each of its claims is supported by the retrieved
evidence. We split the response into claims (sentences), split the context into
evidence units, and score how well each claim is supported by its best-matching
evidence. Low support => hallucination.

Three swappable backends behind one interface:
  TfidfClaimBackend      - lexical, offline, no downloads (runs anywhere)
  EmbeddingClaimBackend  - sentence-transformer semantics (needs internet once)
  NliClaimBackend        - entailment model, bounded by a timeout (see below)

    backend.score(response, context) -> (grounding_risk, detail)
"""
from __future__ import annotations
import re
import concurrent.futures
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

_SENT = re.compile(r"(?<=[.!?])\s+")

def split_claims(text):
    text = (text or "").strip()
    parts = []
    for chunk in text.split("\n"):
        for s in _SENT.split(chunk):
            s = s.strip(" -\u2022\t")
            if len(s.split()) >= 4:
                parts.append(s)
    return parts or ([text] if text else [])

def split_evidence(context):
    context = re.sub(r"passage\s*\d+\s*:", "\n", context or "", flags=re.I)
    units = []
    for chunk in re.split(r"\n+", context):
        for s in _SENT.split(chunk):
            s = s.strip()
            if len(s.split()) >= 3:
                units.append(s)
    return units


class TfidfClaimBackend:
    name = "tfidf"
    def score(self, response, context):
        claims = split_claims(response)
        evidence = split_evidence(context)
        if not claims or not evidence:
            return 0.5, {"reason": "no claims or no evidence", "n_claims": len(claims)}
        vec = TfidfVectorizer(stop_words="english")
        try:
            M = vec.fit_transform(evidence + claims)
        except ValueError:
            return 0.5, {"reason": "empty vocab"}
        ev, cl = M[:len(evidence)], M[len(evidence):]
        sims = cosine_similarity(cl, ev)
        support = sims.max(axis=1)
        return self._risk(support, claims)

    def _risk(self, support, claims):
        support = np.asarray(support, float)
        mean_sup = float(support.mean())
        min_sup = float(support.min())
        frac_unsupported = float((support < 0.20).mean())
        risk = 1.0 - (0.5 * mean_sup + 0.5 * min_sup)
        detail = {"n_claims": len(claims), "mean_support": round(mean_sup, 3),
                  "min_support": round(min_sup, 3),
                  "frac_unsupported": round(frac_unsupported, 3)}
        return round(float(risk), 4), detail

    def warm_up(self):
        pass  # nothing to preload


class EmbeddingClaimBackend(TfidfClaimBackend):
    """Semantic version. Needs sentence-transformers + one model download.
    Falls back to TF-IDF if unavailable."""
    name = "embedding"
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = None
        self.model_name = model_name
    def _load(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        return self.model
    def warm_up(self):
        try:
            self._load()
        except Exception:
            pass
    def score(self, response, context):
        claims = split_claims(response)
        evidence = split_evidence(context)
        if not claims or not evidence:
            return 0.5, {"reason": "no claims or no evidence", "n_claims": len(claims)}
        try:
            m = self._load()
        except Exception:
            return TfidfClaimBackend().score(response, context)
        ce = m.encode(claims, normalize_embeddings=True)
        ee = m.encode(evidence, normalize_embeddings=True)
        sims = ce @ ee.T
        support = sims.max(axis=1)
        return self._risk(support, claims)


class NliClaimBackend(TfidfClaimBackend):
    """Entailment-based grounding — the best-practice faithfulness check.

    For each claim, we ask a Natural Language Inference model whether the
    retrieved evidence *entails* it (supports), is *neutral* (unsupported), or
    *contradicts* it (a hallucination). A claim's support = max entailment
    probability over evidence; a strong contradiction is penalised harder.

    Two guards keep this from ever hanging a request:
      1. top_k_evidence pre-filters each claim down to its most lexically
         relevant evidence sentences (cheap TF-IDF) before the model ever
         runs, so we score claims x top_k pairs instead of claims x all
         evidence.
      2. timeout_s bounds the model call itself. If it runs long, we fall
         back to the fast TF-IDF score instead of blocking the caller. Python
         threads can't be force-killed, so the model call keeps finishing in
         the background — wasted CPU, but the request already returned.

    Needs: pip install transformers torch  (one model download, ~500MB).
    Call warm_up() once at process startup so that download/load doesn't
    happen on a live request.
    """
    name = "nli"

    def __init__(self, model_name="cross-encoder/nli-deberta-v3-base",
                max_evidence=10, max_claims=8, batch_size=16,
                top_k_evidence=2, timeout_s=1.8):
        self.model_name = model_name
        self.max_evidence = max_evidence
        self.max_claims = max_claims
        self.batch_size = batch_size
        self.top_k_evidence = top_k_evidence
        self.timeout_s = timeout_s
        self._pipe = None
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _load(self):
        if self._pipe is None:
            from transformers import pipeline
            self._pipe = pipeline("text-classification", model=self.model_name,
                                  top_k=None)
        return self._pipe

    def warm_up(self):
        """Load the model now, at process startup — not on the first live
        request. Call this from app/api.py right after ControlPlane() is
        constructed."""
        try:
            self._load()
        except Exception:
            pass

    def _prefilter_evidence(self, claims, evidence):
        """Cheap TF-IDF top-k per claim. Cuts pairs from |claims| x |evidence|
        to |claims| x top_k_evidence — the real latency lever, since each
        pair costs one forward pass through the model."""
        if len(evidence) <= self.top_k_evidence:
            return {ci: evidence for ci in range(len(claims))}
        vec = TfidfVectorizer(stop_words="english")
        try:
            M = vec.fit_transform(evidence + claims)
        except ValueError:
            return {ci: evidence[: self.top_k_evidence] for ci in range(len(claims))}
        ev, cl = M[: len(evidence)], M[len(evidence):]
        sims = cosine_similarity(cl, ev)
        picks = {}
        for ci in range(len(claims)):
            order = np.argsort(-sims[ci])[: self.top_k_evidence]
            picks[ci] = [evidence[i] for i in order]
        return picks

    def _score_nli(self, claims, evidence):
        pipe = self._load()
        per_claim_ev = self._prefilter_evidence(claims, evidence)

        pairs, index = [], []
        for ci, claim in enumerate(claims):
            for ev in per_claim_ev[ci]:
                pairs.append({"text": ev, "text_pair": claim})
                index.append(ci)
        outs = pipe(pairs, batch_size=self.batch_size, top_k=None)

        best_entail = [0.0] * len(claims)
        worst_contra = [0.0] * len(claims)
        for ci, scores in zip(index, outs):
            d = {s["label"].lower(): s["score"] for s in scores}
            e, c = d.get("entailment", 0.0), d.get("contradiction", 0.0)
            best_entail[ci] = max(best_entail[ci], e)
            worst_contra[ci] = max(worst_contra[ci], c)

        entail = np.asarray(best_entail, float)
        contra = np.asarray(worst_contra, float)
        claim_risk = np.maximum(1.0 - entail, contra)
        risk = float(max(claim_risk.mean(), contra.max()))
        detail = {"n_claims": len(claims),
                  "mean_entailment": round(float(entail.mean()), 3),
                  "min_entailment": round(float(entail.min()), 3),
                  "max_contradiction": round(float(contra.max()), 3),
                  "n_pairs_scored": len(pairs)}
        return round(min(1.0, risk), 4), detail

    def score(self, response, context):
        claims = split_claims(response)[: self.max_claims]
        evidence = split_evidence(context)[: self.max_evidence]
        if not claims or not evidence:
            return 0.5, {"reason": "no claims or no evidence", "n_claims": len(claims)}

        future = self._executor.submit(self._score_nli, claims, evidence)
        try:
            return future.result(timeout=self.timeout_s)
        except concurrent.futures.TimeoutError:
            risk, detail = TfidfClaimBackend().score(response, context)
            return risk, {**detail, "nli_timeout": True, "nli_timeout_s": self.timeout_s}
        except Exception:
            return TfidfClaimBackend().score(response, context)


def get_backend(name="tfidf"):
    from .judge import LLMJudgeBackend, TfidfPlusJudgeBackend
    return {"tfidf": TfidfClaimBackend,
            "embedding": EmbeddingClaimBackend,
            "nli": NliClaimBackend,
            "judge": LLMJudgeBackend,
            "tfidf+judge": TfidfPlusJudgeBackend}[name]()