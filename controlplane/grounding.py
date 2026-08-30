"""Claim-level grounding / faithfulness scoring — built from scratch.

A response is grounded if each of its claims is supported by the retrieved
evidence. We split the response into claims (sentences), split the context into
evidence units, and score how well each claim is supported by its best-matching
evidence. Low support => hallucination.

Two swappable backends behind one interface:
  TfidfClaimBackend      - lexical, offline, no downloads (runs anywhere)
  EmbeddingClaimBackend  - sentence-transformer semantics (needs internet once)

    backend.score(response, context) -> (grounding_risk, detail)
"""
from __future__ import annotations
import re
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

    This judges MEANING, not word overlap: a correct answer phrased differently
    from the source is recognised as grounded, and a fluent-but-false claim is
    caught even when it reuses the source's vocabulary.

    Needs: pip install transformers torch  (one model download, ~500MB).
    Falls back to TF-IDF automatically if unavailable.
    """
    name = "nli"

    def __init__(self, model_name="cross-encoder/nli-deberta-v3-base",
                 max_evidence=10, max_claims=8, batch_size=16):
        self.model_name = model_name
        self.max_evidence = max_evidence
        self.max_claims = max_claims
        self.batch_size = batch_size
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            from transformers import pipeline
            self._pipe = pipeline("text-classification", model=self.model_name,
                                  top_k=None)
        return self._pipe

    def score(self, response, context):
        claims = split_claims(response)[: self.max_claims]
        evidence = split_evidence(context)[: self.max_evidence]
        if not claims or not evidence:
            return 0.5, {"reason": "no claims or no evidence", "n_claims": len(claims)}
        try:
            pipe = self._load()
        except Exception:
            return TfidfClaimBackend().score(response, context)  # graceful fallback

        # one batched call over all (evidence=premise, claim=hypothesis) pairs
        pairs, index = [], []
        for ci, claim in enumerate(claims):
            for ev in evidence:
                pairs.append({"text": ev, "text_pair": claim})
                index.append(ci)
        outs = pipe(pairs, batch_size=self.batch_size, top_k=None)

        # per claim: best entailment over evidence, and worst contradiction
        best_entail = [0.0] * len(claims)
        worst_contra = [0.0] * len(claims)
        for ci, scores in zip(index, outs):
            d = {s["label"].lower(): s["score"] for s in scores}
            e, c = d.get("entailment", 0.0), d.get("contradiction", 0.0)
            best_entail[ci] = max(best_entail[ci], e)   # is this claim entailed by ANY evidence?
            worst_contra[ci] = max(worst_contra[ci], c)

        entail = np.asarray(best_entail, float)               # 0..1
        contra = np.asarray(worst_contra, float)              # 0..1
        # a claim is risky when no evidence entails it; a contradiction is worse.
        claim_risk = np.maximum(1.0 - entail, contra)
        # overall: average unsupported-ness, but a single clear contradiction dominates
        risk = float(max(claim_risk.mean(), contra.max()))
        detail = {"n_claims": len(claims),
                  "mean_entailment": round(float(entail.mean()), 3),
                  "min_entailment": round(float(entail.min()), 3),
                  "max_contradiction": round(float(contra.max()), 3)}
        return round(min(1.0, risk), 4), detail


def get_backend(name="tfidf"):
    return {"tfidf": TfidfClaimBackend,
            "embedding": EmbeddingClaimBackend,
            "nli": NliClaimBackend}[name]()
