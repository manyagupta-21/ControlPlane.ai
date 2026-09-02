"""Demonstration + smoke test for per-dimension statistical anomaly detection.

Exercises all three components on crafted cases and on real RAGTruth traffic,
WITHOUT touching detectors.py / policy.py / pipeline.py. Run it after
scripts/fit_profiles.py to confirm step 1 works in isolation.

    python scripts/anomaly_dimensions_demo.py

What each section shows:
  1. PERFORMANCE  per-response test on normal vs. structurally strange answers,
     and the length-controlled residual.
  2. PERFORMANCE  the per-response vs. windowed distinction — the core reason
     the v1 single-detector design could not see population shift.
  3. RESPONSIBILITY  Poisson tail on the PII entity COUNT: why one address and a
     dump of forty must not carry the same flag.
  4. COST  cost residual conditional on the query, vs. the flat v1 rule.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from controlplane.dimension_anomaly import load_profiles  # noqa: E402

RULE = "=" * 78


def section(title):
    print("\n" + RULE + "\n" + title + "\n" + RULE)


def main():
    import json
    perf, resp, cost = load_profiles()
    if not perf.profile.fitted:
        sys.exit("no fitted profile at data/anomaly_profiles.json\n"
                 "  run: python scripts/fit_profiles.py")

    qa = [json.loads(l) for l in
          open("data/ragtruth/ragtruth_qa.jsonl", encoding="utf-8")]
    # Use a REAL RAGTruth context and a REAL clean answer as the "normal"
    # baseline, so the comparison is against the same kind of traffic the
    # profile was fitted on — not a synthetic toy string.
    clean_test = [r for r in qa if not r["label_hallucination"]
                  and r.get("split") == "test"]
    ref = clean_test[0]
    CTX = ref["context"]

    # -------------------------------------------------------------- 1. performance
    section("1. PERFORMANCE — per-response anomaly (against a real QA context)")
    cases = [
        ("a real clean answer", ref["response"]),
        ("one-word reply", "Yes."),
        ("repetition loop", "The answer is 42. " * 60),
        ("pure digit dump", " ".join(str(i) for i in range(200))),
    ]
    print(f"{'case':22s}{'d^2':>10}{'p-value':>11}{'anomalous':>11}{'most extreme feature':>22}")
    for name, r in cases:
        o = perf.run(r, CTX)
        print(f"{name:22s}{o['mahalanobis_d2']:>10.2f}{o['p_value']:>11.4f}"
              f"{str(o['anomalous']):>11}{o['most_extreme_feature']:>22}")
    print("\n  The real answer sits deep inside the envelope (d^2 ~ 1). The two")
    print("  structurally broken outputs are rejected at alpha=0.01. A one-word")
    print("  reply sits near the boundary — correctly, since short answers do")
    print("  occur in real traffic. None of these is necessarily HARMFUL; the")
    print("  point is 'unlike normal', which grounding alone would miss (a")
    print("  repetition loop can repeat a perfectly grounded sentence).")

    # ------------------------------------------------ 2. per-response vs windowed
    section("2. PERFORMANCE — per-response vs. windowed test (why both exist)")
    print("  The windowed test detects a shift in the POPULATION that no single")
    print("  response reveals. We build a synthetic shift by mixing in structurally")
    print("  degraded responses at rising rates, and read the batch test at n=100.\n")
    rng = np.random.default_rng(0)
    in_domain = clean_test

    def degrade(text):
        # a mild, realistic degradation: truncate to the first clause
        return text.split(".")[0][:40] or "n/a"

    def window_fire_rate(contaminate, n=100, trials=200):
        hits = 0
        for _ in range(trials):
            perf._window = []
            idx = rng.choice(len(in_domain), n, replace=False)
            v = None
            for i in idx:
                r = in_domain[i]
                resp_text = degrade(r["response"]) if rng.random() < contaminate else r["response"]
                v = perf.run(resp_text, r["context"]).get("window")
            if v and v["population_shifted"]:
                hits += 1
        return hits / trials

    perf.window_size = 100
    print(f"{'contamination rate':>22}{'population_shifted fires':>28}")
    for c in (0.0, 0.1, 0.3, 0.6):
        print(f"{c:>22.0%}{window_fire_rate(c):>27.0%}")
    print("\n  At 0% the batch test is quiet (~alpha, as it should be). As the")
    print("  population degrades it fires more and more — while each individual")
    print("  degraded response, tested alone, may sit under the per-response")
    print("  threshold. A per-turn test at n=1 is blind to this; the same")
    print("  argument the session monitor already makes about per-turn limits.")

    # ---------------------------------------------------------- 3. responsibility
    section("3. RESPONSIBILITY — Poisson tail on the PII entity COUNT")
    print(f"  reference rate = {resp.pii_rate:.3f} PII entities per response\n")
    print(f"{'PII entity count':>18}{'tail P(X>=k)':>16}{'bulk disclosure?':>20}")
    for k in (0, 1, 3, 8, 40):
        o = resp.run(k, 0.0)
        print(f"{k:>18}{o['pii_count_p_value']:>16.2e}{str(o['pii_bulk_disclosure']):>20}")
    print("\n  A boolean `pii_detected` flags k=1 and k=40 identically. The tail")
    print("  probability does not: a bulk disclosure is a distinct incident with a")
    print("  distinct escalation path, and only the count can tell them apart.")

    # -------------------------------------------------------------------- 4. cost
    section("4. COST — residual conditional on the query (vs. the flat v1 rule)")
    print("  The regression learned the normal answer length for a query on")
    print("  RAGTruth. `verbose_for_query` fires when a response is >3 SD longer")
    print("  than that conditional mean — which v1's flat `len(query) <= 15`")
    print("  rule cannot express. (RAGTruth answers are long, so the bar is high.)\n")
    print(f"{'query tokens':>14}{'response tokens':>17}{'residual z':>13}{'verbose?':>11}")
    for qt, rt in [(40, 240), (12, 240), (40, 1200), (8, 3000)]:
        o = cost.run(qt, rt, 0)
        print(f"{qt:>14}{rt:>17}{o['length_residual_z']:>13.2f}{str(o['verbose_for_query']):>11}")
    print("\n  regenerations against the fitted rate "
          f"({cost.regen_rate:.3f} per response):")
    for g in (0, 2, 5):
        o = cost.run(40, 240, g)
        print(f"    {g} regenerations -> tail P={o['regeneration_p_value']:.4f}, "
              f"rework_anomaly={o['rework_anomaly']}")

    print("\n" + RULE)
    print("All three dimensions now carry their own anomaly test. None adds to")
    print("p_harm; each writes a p-value / flag for the audit trail. Wiring these")
    print("into detectors.py and the decision layer is the next step.")
    print(RULE)


if __name__ == "__main__":
    main()
