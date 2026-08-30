"""Statistical validation of ControlPlane — the model-risk-management view.

Treats each detector as a binary classifier and reports how good it is against
ground-truth labels, then evaluates the end-to-end policy for the ONE error
that matters most: under-triage (letting a bad response through with too soft
an action). Also demonstrates a threshold sweep (the FP/FN tradeoff you tune,
not solve) and a PSI drift monitor.

Usage:  python scripts/evaluate.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from controlplane import Interaction
from controlplane.detectors import default_detectors
from controlplane.pipeline import ControlPlane
from controlplane.schemas import action_rank

DATA = "data/interactions.jsonl"
POLICY = "config/policies.yaml"


def load():
    with open(DATA, encoding="utf-8") as f:
        return [Interaction(**json.loads(l)) for l in f]


def prf(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(precision=prec, recall=rec, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn)


def psi(ref, cur, bins=10):
    """Population Stability Index between reference and current samples."""
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    qs = np.quantile(ref, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    r = np.histogram(ref, qs)[0] / max(len(ref), 1)
    c = np.histogram(cur, qs)[0] / max(len(cur), 1)
    r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def main():
    data = load()
    dets = default_detectors()

    # collect per-interaction detector signals
    rows = []
    for x in data:
        res = {d.name: d.run(x) for d in dets}
        rows.append(dict(
            cat=x.category, use_case=x.use_case,
            label_pii=int(bool(x.label_pii)), label_tox=int(bool(x.label_toxic)),
            label_hal=int(bool(x.label_hallucination)),
            pred_pii=int(res["responsibility"].flags["pii_detected"]),
            pred_tox=int(res["responsibility"].flags["toxicity_any"]),
            pred_hal=int(res["performance"].flags["ungrounded"] or
                         res["performance"].flags["inconsistent"]),
            perf_risk=res["performance"].risk,
            grounding=res["performance"].detail.get("grounding_risk"),
        ))

    print("=" * 66)
    print("1) DETECTOR PERFORMANCE  (each detector as a binary classifier)")
    print("=" * 66)
    for name, lt, lp in [("PII", "label_pii", "pred_pii"),
                         ("Toxicity/bias", "label_tox", "pred_tox"),
                         ("Hallucination", "label_hal", "pred_hal")]:
        m = prf([r[lt] for r in rows], [r[lp] for r in rows])
        print(f"  {name:14s} P={m['precision']:.2f}  R={m['recall']:.2f}  "
              f"F1={m['f1']:.2f}   (TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']})")

    print("\n" + "=" * 66)
    print("2) END-TO-END DECISIONS  (predicted action vs gold_action)")
    print("=" * 66)
    cp = ControlPlane(POLICY, audit_path="data/_eval_audit.jsonl")
    under = over = exact = 0
    under_examples = []
    for x in data:
        d = cp.process(x, log=False)
        g = action_rank(x.gold_action)
        p = action_rank(d.action)
        if p == g:
            exact += 1
        elif p < g:                      # softer than it should be = DANGEROUS
            under += 1
            under_examples.append((x.id, x.category, x.use_case, d.action, x.gold_action))
        else:
            over += 1                    # stricter than needed = costly, not unsafe
    n = len(data)
    print(f"  Exact match:   {exact}/{n}  ({exact/n:.0%})")
    print(f"  Over-triage:   {over}/{n}  ({over/n:.0%})   (too strict — cost, not safety)")
    print(f"  UNDER-triage:  {under}/{n}  ({under/n:.0%})   (too soft — the dangerous error)")
    if under_examples:
        print("   under-triage cases:", under_examples[:5])

    print("\n" + "=" * 66)
    print("3) THRESHOLD SWEEP  (hallucination detection: the FP/FN tradeoff)")
    print("=" * 66)
    yhal = np.array([r["label_hal"] for r in rows])
    risk = np.array([r["perf_risk"] for r in rows])
    print("   thr   precision  recall   FP   FN")
    for thr in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
        m = prf(yhal, (risk >= thr).astype(int))
        print(f"   {thr:.1f}     {m['precision']:.2f}      {m['recall']:.2f}    "
              f"{m['fp']:>2}   {m['fn']:>2}")

    print("\n" + "=" * 66)
    print("4) DRIFT MONITORING")
    print("=" * 66)
    print("  Moved to scripts/drift_monitor.py.")
    print("  An earlier version of this section computed PSI between clean and")
    print("  hallucinated responses. That measures class SEPARATION, not stability:")
    print("  the two groups are supposed to look different, so a large value there")
    print("  says nothing. PSI is a temporal statistic — freeze a reference window")
    print("  at validation time, compare each later window against it. The corrected")
    print("  version, paired with a two-sample KS test, is in drift_monitor.py.")

    # ---- save figures for the deck / README ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        thrs = np.linspace(0.05, 0.95, 19)
        fps = [prf(yhal, (risk >= t).astype(int))["fp"] for t in thrs]
        fns = [prf(yhal, (risk >= t).astype(int))["fn"] for t in thrs]
        ax[0].plot(thrs, fps, "-o", label="False positives", color="#A100FF")
        ax[0].plot(thrs, fns, "-s", label="False negatives", color="#D24545")
        ax[0].set_title("Hallucination threshold: FP/FN tradeoff")
        ax[0].set_xlabel("risk threshold"); ax[0].set_ylabel("count"); ax[0].legend()

        cats = ["clean", "hallucination", "pii_leak", "toxic"]
        data_by_cat = [[r["perf_risk"] for r in rows if r["cat"] == c] for c in cats]
        ax[1].boxplot(data_by_cat, tick_labels=cats)
        ax[1].set_title("Performance-risk by category")
        ax[1].set_ylabel("performance risk"); ax[1].tick_params(axis="x", rotation=20)
        plt.tight_layout(); plt.savefig("data/evaluation.png", dpi=130)
        print("\nSaved figure -> data/evaluation.png")
    except Exception as e:
        print("  (plot skipped:", e, ")")


if __name__ == "__main__":
    main()
