"""Model routing as an extension of the SAME objective, not a separate heuristic.

The v1 rule was `easy = len(query.split()) <= 15`. That is a guess about
difficulty dressed up as a cost policy, and it optimises nothing.

Routing is really just the decision problem from decision_theory.py with one
more term. For each candidate model m we pay a compute cost and then face the
downstream decision, whose expected loss depends on how likely that model is to
be wrong:

    total(m) = compute_cost(m) + min_a E[ L(a, S) | p_harm(m) ]

and we route to argmin_m total(m). One objective, two decisions — which model
answers, and what we then do with its answer. Nothing new to tune.

Two components are needed to make that computable:

  1. A DIFFICULTY MODEL. Cheap lexical features of the query and its retrieved
     context, fitted to predict whether an answer turns out ungrounded. This is
     a stand-in: with production data you would fit it on observed outcomes per
     model. Its cross-validated AUC is reported honestly by cost_frontier.py.

  2. A QUALITY-DEGRADATION MODEL. We do not have paired small/large outputs, so
     we state the assumption explicitly rather than hiding it in a constant:

         p_small = p_large + delta * difficulty * (1 - p_large)

     i.e. the small model's extra error is proportional to how hard the item is,
     and vanishes on easy items. `delta` is the one free parameter; the frontier
     script sweeps it so the conclusion can be judged against its sensitivity
     rather than taken on faith.
"""
from __future__ import annotations
import math
import numpy as np

from .decision_theory import LossMatrix

PRICE_PER_1K = {"small": 0.0005, "large": 0.010}   # illustrative USD / 1k tokens


def _tokens(text: str) -> int:
    return int(len((text or "").split()) * 1.3) + 1


def features(query: str, context: str = "") -> np.ndarray:
    """Cheap, latency-free difficulty signals — no model call required."""
    q, c = (query or ""), (context or "")
    qw, cw = q.split(), c.split()
    digits = sum(ch.isdigit() for ch in q)
    entities = sum(1 for w in qw[1:] if w[:1].isupper())
    multi = q.count("?") + q.lower().count(" and ") + q.count(",")
    return np.array([
        math.log1p(len(qw)),          # longer questions ask for more
        math.log1p(len(cw)),          # long context = more to reconcile
        digits / (len(q) + 1),        # numeric questions invite fabrication
        entities / (len(qw) + 1),     # named entities invite fabrication
        multi,                        # multi-part questions
        len(set(w.lower() for w in qw)) / (len(qw) + 1),   # lexical variety
    ], dtype=float)


class DifficultyModel:
    """Logistic regression on the features above. Falls back to a length rule."""

    def __init__(self, train_prevalence: float | None = None,
                 deploy_prevalence: float | None = None):
        self.clf = None
        self.auc = None
        # Difficulty is a probability, so it inherits the same label-shift problem
        # as P(harm): fitted on a ~65%-positive eval set, its scores are floored
        # far above zero and NO item ever looks easy. Correcting the prior is what
        # makes "easy items are free for the small model" actually true.
        self.train_prevalence = train_prevalence
        self.deploy_prevalence = deploy_prevalence

    def fit(self, queries, contexts, labels):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score

        X = np.vstack([features(q, c) for q, c in zip(queries, contexts)])
        y = np.asarray(labels, dtype=int)
        if len(set(y.tolist())) < 2:
            return self
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=1000, C=1.0))
        try:
            k = min(5, int(min(np.bincount(y))))
            if k >= 2:
                self.auc = float(np.mean(cross_val_score(
                    pipe, X, y, cv=k, scoring="roc_auc")))
        except Exception:
            self.auc = None
        pipe.fit(X, y)
        self.clf = pipe
        return self

    def predict(self, query: str, context: str = "") -> float:
        if self.clf is None:
            return 1.0 if len(query.split()) > 15 else 0.0
        p = float(self.clf.predict_proba(features(query, context).reshape(1, -1))[0, 1])
        if self.train_prevalence and self.deploy_prevalence:
            from .decision_theory import adjust_prior
            p = adjust_prior(p, self.train_prevalence, self.deploy_prevalence)
        return p


class Router:
    """Chooses the model by minimising compute cost plus downstream expected loss."""

    def __init__(self, loss: LossMatrix, difficulty: DifficultyModel,
                 delta: float = 0.35, usd_per_cost_unit: float = 1.0):
        self.loss = loss
        self.difficulty = difficulty
        self.delta = delta
        # The loss matrix is denominated in business cost units; token prices are
        # in USD. This makes the conversion an explicit, auditable assumption
        # rather than an accidental unit mismatch.
        self.usd_per_cost_unit = usd_per_cost_unit

    def compute_cost(self, model: str, query: str, response_len_tokens: int = 180) -> float:
        toks = _tokens(query) + response_len_tokens
        usd = toks / 1000.0 * PRICE_PER_1K[model]
        return usd / self.usd_per_cost_unit

    def degraded(self, p_large: float, difficulty: float) -> float:
        return float(min(1.0, p_large + self.delta * difficulty * (1.0 - p_large)))

    def evaluate(self, query: str, context: str, p_large: float) -> dict:
        d = self.difficulty.predict(query, context)
        out = {}
        for m in ("small", "large"):
            p = p_large if m == "large" else self.degraded(p_large, d)
            decision_loss = min(self.loss.expected_loss(p).values())
            compute = self.compute_cost(m, query)
            out[m] = dict(p_harm=round(p, 4),
                          compute=round(float(compute), 6),
                          decision_loss=round(float(decision_loss), 4),
                          total=round(float(compute + decision_loss), 4))
        chosen = min(out, key=lambda m: out[m]["total"])
        return dict(difficulty=round(d, 4), chosen=chosen, per_model=out,
                    saving_vs_large=round(out["large"]["total"] - out[chosen]["total"], 4))

    def route(self, query: str, context: str, p_large: float) -> str:
        return self.evaluate(query, context, p_large)["chosen"]
