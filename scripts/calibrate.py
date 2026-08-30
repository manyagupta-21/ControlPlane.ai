"""Fit and validate the calibration + abstention layer on real RAGTruth.

Fits an isotonic calibrator on the TRAIN split's grounding scores, then on the
held-out TEST split reports:
  * Brier score and ECE, raw vs calibrated  (is the system honest?)
  * a reliability diagram                    (do predicted probs match reality?)
  * a risk-coverage curve                    (does abstaining raise accuracy?)

Saves the fitted calibrator to data/calibrator.json for the live gateway.

Usage:
  python scripts/calibrate.py                 # tfidf grounding (offline)
  python scripts/calibrate.py embedding        # the chosen backend (local)
"""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from controlplane.grounding import get_backend
from controlplane.calibration import (Calibrator, brier, ece, reliability_curve,
                                      coverage_accuracy)

DATA = "data/ragtruth/ragtruth_qa.jsonl"

def load(split):
    return [json.loads(l) for l in open(DATA, encoding="utf-8")
            if json.loads(l)["split"] == split]

def score(rows, backend):
    risk, y = [], []
    for r in rows:
        rk, _ = backend.score(r["response"], r["context"])
        risk.append(rk); y.append(int(r["label_hallucination"]))
    return np.array(risk), np.array(y)

def main():
    backend = get_backend(sys.argv[1] if len(sys.argv) > 1 else "tfidf")
    print(f"backend: {backend.name}")

    train, test = load("train"), load("test")
    # cap train for speed on heavier backends; plenty for a monotonic fit
    if len(train) > 1500:
        import random; random.seed(0); train = random.sample(train, 1500)

    print(f"scoring train ({len(train)}) ...")
    s_tr, y_tr = score(train, backend)
    print(f"scoring test ({len(test)}) ...")
    s_te, y_te = score(test, backend)

    cal = Calibrator().fit(s_tr, y_tr)
    p_raw = np.clip(s_te, 0, 1)          # raw score used as a probability
    p_cal = cal.predict(s_te)            # calibrated probability

    print("\n=== HONESTY (test split) ===")
    print(f"  Brier  raw={brier(y_te, p_raw):.4f}   calibrated={brier(y_te, p_cal):.4f}")
    print(f"  ECE    raw={ece(y_te, p_raw):.4f}   calibrated={ece(y_te, p_cal):.4f}")
    print("  (lower is better; calibration should reduce both)")

    os.makedirs("data", exist_ok=True)
    cal.save("data/calibrator.json")
    print("\nSaved calibrator -> data/calibrator.json")

    # abstention: how much accuracy do we gain by escalating the least-confident?
    cov, acc = coverage_accuracy(y_te, p_cal, thr=0.5)
    full = acc[-1]
    at80 = acc[np.argmin(np.abs(cov - 0.8))]
    at60 = acc[np.argmin(np.abs(cov - 0.6))]
    print("\n=== ABSTENTION (reject option) ===")
    print(f"  answer 100% (no abstain): accuracy {full:.2f}")
    print(f"  answer  80% (abstain 20%): accuracy {at80:.2f}")
    print(f"  answer  60% (abstain 40%): accuracy {at60:.2f}")
    print("  -> escalating the least-confident cases to a human raises accuracy")
    print("     on the ones the system decides automatically.")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        xr, yr = reliability_curve(y_te, p_raw)
        xc, yc = reliability_curve(y_te, p_cal)
        ax[0].plot([0, 1], [0, 1], "--", color="#999", label="perfect")
        ax[0].plot(xr, yr, "-o", color="#D24545", label="raw")
        ax[0].plot(xc, yc, "-o", color="#A100FF", label="calibrated")
        ax[0].set_title("Reliability diagram"); ax[0].set_xlabel("predicted probability")
        ax[0].set_ylabel("observed frequency"); ax[0].legend()
        ax[1].plot(cov, acc, "-o", color="#1FA36B")
        ax[1].set_title("Risk-coverage (abstention)")
        ax[1].set_xlabel("coverage (fraction auto-decided)")
        ax[1].set_ylabel("accuracy on auto-decided"); ax[1].invert_xaxis()
        plt.tight_layout(); plt.savefig(f"data/calibration_{backend.name}.png", dpi=130)
        print(f"\nSaved figure -> data/calibration_{backend.name}.png")
    except Exception as e:
        print("  (plot skipped:", e, ")")

if __name__ == "__main__":
    main()
