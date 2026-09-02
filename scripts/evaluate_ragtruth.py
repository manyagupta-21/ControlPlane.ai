"""Evaluate the grounding detector on the REAL RAGTruth QA test split.

Reports how well our from-scratch faithfulness score separates human-labelled
hallucinated vs grounded responses: AUROC (threshold-free), plus precision/
recall across thresholds and at the Youden-optimal operating point.

Usage:
  python scripts/evaluate_ragtruth.py                # TF-IDF backend (offline)
  python scripts/evaluate_ragtruth.py embedding      # sentence-transformer (local)
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from controlplane.grounding import get_backend

DATA = "data/ragtruth/ragtruth_qa.jsonl"

def load(split="test"):
    rows = []
    with open(DATA, encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            if r["split"] == split:
                rows.append(r)
    return rows

def prf_at(y, risk, thr):
    pred = (risk >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1, tp, fp, fn, tn

def main():
    backend = get_backend(sys.argv[1] if len(sys.argv) > 1 else "tfidf")
    rows = load("test")
    # optional: python scripts/evaluate_ragtruth.py nli 200  -> stratified subset
    if len(sys.argv) > 2:
        import random
        random.seed(0)
        limit = int(sys.argv[2])
        pos = [r for r in rows if r["label_hallucination"]]
        neg = [r for r in rows if not r["label_hallucination"]]
        k = limit // 2
        rows = random.sample(pos, min(k, len(pos))) + random.sample(neg, min(limit - k, len(neg)))
        random.shuffle(rows)
        print(f"(stratified subset: {len(rows)} of the 900-record test split)")
    print(f"Scoring {len(rows)} real RAGTruth QA test responses with "
          f"'{backend.name}' grounding backend...")

    y, risk = [], []
    n_timeout = 0
    t0 = time.perf_counter()
    for i, r in enumerate(rows):
        rk, d = backend.score(r["response"], r["context"])
        if d.get("nli_timeout"):
            n_timeout += 1
        risk.append(rk); y.append(int(r["label_hallucination"]))
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(rows)}")
    y, risk = np.array(y), np.array(risk)
    dt = time.perf_counter() - t0

    base = y.mean()
    print(f"\nBase rate (hallucinated): {base:.1%}   |   scored in {dt:.1f}s "
          f"({1000*dt/len(rows):.1f} ms/response)")
    if n_timeout:
        print(f"NOTE: {n_timeout}/{len(rows)} ({n_timeout/len(rows):.1%}) fell back to TF-IDF on timeout")

    # AUROC + AUPRC (threshold-free)
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
    auroc = roc_auc_score(y, risk)
    auprc = average_precision_score(y, risk)
    print(f"\nAUROC = {auroc:.3f}   (0.5 = random, 1.0 = perfect)")
    print(f"AUPRC = {auprc:.3f}   (base rate {base:.3f})   lift x{auprc/base:.2f}")

    # Youden-optimal threshold
    fpr, tpr, thr = roc_curve(y, risk)
    j = tpr - fpr
    best = thr[int(np.argmax(j))]
    p, r, f1, tp, fp, fn, tn = prf_at(y, risk, best)
    print(f"\nAt Youden-optimal threshold {best:.3f}:")
    print(f"  precision={p:.2f}  recall={r:.2f}  F1={f1:.2f}   "
          f"(TP={tp} FP={fp} FN={fn} TN={tn})")

    print("\nThreshold sweep:")
    print("  thr   precision recall   F1")
    for t in [0.4, 0.5, 0.6, 0.7, 0.8]:
        p, r, f1, *_ = prf_at(y, risk, t)
        print(f"  {t:.1f}    {p:.2f}     {r:.2f}   {f1:.2f}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(fpr, tpr, color="#A100FF", lw=2, label=f"AUROC={auroc:.3f}")
        ax[0].plot([0, 1], [0, 1], "--", color="#999")
        ax[0].set_title(f"ROC — grounding on real RAGTruth ({backend.name})")
        ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("true positive rate")
        ax[0].legend(loc="lower right")
        ax[1].hist(risk[y == 0], bins=25, alpha=0.6, label="grounded", color="#1FA36B")
        ax[1].hist(risk[y == 1], bins=25, alpha=0.6, label="hallucinated", color="#D24545")
        ax[1].axvline(best, color="#A100FF", ls="--", label=f"threshold={best:.2f}")
        ax[1].set_title("grounding-risk distribution"); ax[1].set_xlabel("grounding risk")
        ax[1].legend()
        plt.tight_layout()
        out = f"data/ragtruth/ragtruth_eval_{backend.name}.png"
        plt.savefig(out, dpi=130)
        print(f"\nSaved figure -> {out}")
    except Exception as e:
        print("  (plot skipped:", e, ")")

if __name__ == "__main__":
    main()
