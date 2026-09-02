"""Fit the three per-dimension anomaly reference profiles.

Writes data/anomaly_profiles.json, read by controlplane.dimension_anomaly.

  performance     fitted on labelled-CLEAN RAGTruth QA responses (the support
                  profile of answers that are known not to hallucinate)
  responsibility  PII / toxicity base rates from data/interactions.jsonl
  cost            log-token regression + regeneration rate, same local file

WHAT "REFERENCE" MEANS, AND WHY IT IS THE WHOLE GAME
----------------------------------------------------
An anomaly detector has no opinions of its own; it only knows the window it was
fitted on. Two failure modes follow:

  * FIT IT ON CONTAMINATED DATA and it learns that harm is normal. So the
    performance profile is fitted on labelled-clean responses only.
  * FIT IT ON A NARROW WINDOW and everything looks anomalous. v1 fitted on 18
    responses, which is why its README conceded a high false-positive rate.
    Fitting on a few thousand real clean responses widens the envelope to what a
    deployment would actually see, and the realised false-alarm rate falls to
    roughly the nominal alpha — a number this script prints so you can check it.

Prerequisites (your existing scripts):
    python data/ragtruth/fetch_ragtruth.py
    python data/ragtruth/build_ragtruth.py      # -> data/ragtruth/ragtruth_qa.jsonl

Run:
    python scripts/fit_profiles.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from controlplane.dimension_anomaly import (PerformanceAnomaly, ResponsibilityAnomaly,  # noqa: E402
                                            CostAnomaly, save_profiles, PROFILE_PATH)

RAGTRUTH_QA = "data/ragtruth/ragtruth_qa.jsonl"
LOCAL = "data/interactions.jsonl"


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    for line in open(path, encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def main():
    print(f"fitting anomaly profiles -> {PROFILE_PATH}\n")

    # ---- performance ----
    qa = load_jsonl(RAGTRUTH_QA)
    if not qa:
        sys.exit(f"missing {RAGTRUTH_QA}\n  run: python data/ragtruth/fetch_ragtruth.py"
                 f"\n  then: python data/ragtruth/build_ragtruth.py")
    clean_train = [r for r in qa
                   if (not r["label_hallucination"]) and r.get("split") == "train"]
    print(f"  [performance]    fitting on {len(clean_train)} clean QA-train responses")
    perf = PerformanceAnomaly().fit([r["response"] for r in clean_train],
                                    [r["context"] for r in clean_train])

    # ---- responsibility rates (from the local interaction log) ----
    local = load_jsonl(LOCAL)
    if local:
        pii = [1 if r.get("category") == "pii_leak" else 0 for r in local]
        tox = [1 if r.get("category") == "toxic" else 0 for r in local]
        resp = ResponsibilityAnomaly().fit(pii, tox)
        print(f"  [responsibility] base rates from {len(local)} local interactions: "
              f"PII {resp.pii_rate:.3f}, toxicity {resp.tox_rate:.3f}")
    else:
        print(f"  [responsibility] {LOCAL} not found — conservative defaults used.")
        resp = ResponsibilityAnomaly()

    # ---- cost regression: fit E[log response tokens | log query tokens] on
    # RAGTruth, which has thousands of real (question, answer) pairs. 60 local
    # rows would leave the regression near-degenerate and every residual huge.
    qt = [len(str(r.get("query", "")).split()) * 1.3 + 1 for r in clean_train]
    rt = [len(str(r.get("response", "")).split()) * 1.3 + 1 for r in clean_train]
    # regeneration rate stays local — RAGTruth has no regeneration field. If the
    # local log is absent, default to a mild rate rather than zero.
    rg = [r.get("regenerations", 0) for r in local] if local else [0]
    cost = CostAnomaly().fit(qt, rt, rg)
    print(f"  [cost]           log-token regression on {len(qt)} RAGTruth pairs, "
          f"sigma={cost.resid.sigma:.3f}, regeneration rate={cost.regen_rate:.3f}")

    save_profiles(perf, resp, cost)

    # ---- realised false-alarm rate on held-out CLEAN QA ----
    clean_test = [r for r in qa
                  if (not r["label_hallucination"]) and r.get("split") == "test"]
    hits = sum(perf.run(r["response"], r["context"]).get("anomalous", False)
               for r in clean_test)
    rate = hits / max(len(clean_test), 1)
    print(f"\n  sanity check — held-out clean QA: {hits}/{len(clean_test)} flagged "
          f"({rate:.2%}) against a nominal alpha of 1.00%")
    print("  A realised rate near alpha is the point. v1 fitted on 18 responses and")
    print("  had no way to report this number at all.")
    print(f"\nwrote {PROFILE_PATH}")


if __name__ == "__main__":
    main()
