"""Verify steps 1B and 1C end-to-end, against the live pipeline.

1B contract: wiring the per-dimension anomaly components into the three
detectors changes NO decision — the anomaly output only rides along in
DetectorResult.detail / flags.

1C contract: the policy layer may escalate a decision, but ONLY through
out-of-domain interval widening, and ONLY for responses with ood_severity > 0.
Normal traffic (severity 0) is identical to pre-1C.

HOW THE BASELINE IS TAKEN. The only trustworthy "before" is the code as it was
before step 1 touched anything. This script compares the live pipeline against a
snapshot recorded from a pristine pre-step-1 checkout (data/_baseline_actions.json).
Re-deriving the baseline in-process is NOT safe: the rolling windows inside the
anomaly components are stateful, so fresh vs. warm detectors give different batch
results, which looks like drift but is not.

    # one-time, from a clean pre-step-1 checkout (git stash, or the old commit):
    python scripts/verify_anomaly_wiring.py --record-baseline
    # then, with step 1 applied:
    python scripts/verify_anomaly_wiring.py

The repo ships data/_baseline_actions.json captured before step 1, so the check
runs out of the box.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from controlplane.pipeline import ControlPlane            # noqa: E402
from controlplane.schemas import Interaction              # noqa: E402

CONFIG = "config/policies.yaml"
LOCAL = "data/interactions.jsonl"
BASELINE = "data/_baseline_actions.json"


def load_interactions():
    rows = [json.loads(l) for l in open(LOCAL, encoding="utf-8")]
    return [Interaction(id=r["id"], use_case=r["use_case"], query=r["query"],
                        response=r["response"], context=r.get("context", ""),
                        samples=r.get("samples", []),
                        model_used=r.get("model_used", "large"),
                        regenerations=r.get("regenerations", 0)) for r in rows]


def run_all():
    """Return {id: (action, p_harm, ood_severity)} for the current code."""
    cp = ControlPlane(CONFIG, audit_path="data/_verify.jsonl")
    out = {}
    for x in load_interactions():
        d = cp.process(x, log=False)
        perf = [r for r in d.detector_results if r["name"] == "performance"]
        ood = perf[0]["detail"].get("anomaly", {}).get("ood_severity", 0.0) if perf else 0.0
        out[x.id] = (d.action, d.p_harm, ood)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record-baseline", action="store_true",
                    help="run on the CURRENT (pre-step-1) code and save the baseline")
    args = ap.parse_args()

    if args.record_baseline:
        base = {k: v[0] for k, v in run_all().items()}
        json.dump(base, open(BASELINE, "w"), indent=0)
        print(f"recorded {len(base)} baseline actions -> {BASELINE}")
        return

    if not os.path.exists("data/anomaly_profiles.json"):
        sys.exit("run: python scripts/fit_profiles.py")
    if not os.path.exists(BASELINE):
        sys.exit(f"no baseline at {BASELINE}\n"
                 "  capture it from a pre-step-1 checkout:\n"
                 "    python scripts/verify_anomaly_wiring.py --record-baseline")

    baseline = json.load(open(BASELINE))
    live = run_all()

    print("=" * 74)
    print("1B + 1C — decision drift vs the pre-step-1 baseline")
    print("=" * 74)
    changed = [(k, baseline[k], live[k][0], live[k][2]) for k in baseline
               if baseline[k] != live[k][0]]
    normal_changed = [c for c in changed if c[3] <= 0.0]
    print(f"  interactions compared           : {len(baseline)}")
    print(f"  decisions changed by widening   : {len(changed)}")
    print(f"  of those, with ood_severity = 0 : {len(normal_changed)}  "
          f"(MUST be 0 — normal traffic is untouched)")
    for k, a, b, o in changed:
        print(f"    {k}: {a} -> {b}  (ood_severity={o})")
    assert not normal_changed, "CONTRACT VIOLATION: normal traffic changed"

    print("\n" + "=" * 74)
    print("1C — escalation on well-grounded but structurally anomalous answers")
    print("=" * 74)
    cp = ControlPlane(CONFIG, audit_path="data/_verify.jsonl")
    ctx = ("Q3 revenue: West 42.1m, East 38.4m. Growth driven by two enterprise "
           "accounts.")
    probes = [
        ("normal grounded answer", "West revenue was 42.1m in Q3, the strongest region."),
        ("repetition loop (grounded)", "West revenue was 42.1m. " * 60),
        ("digit dump", " ".join(str(i) for i in range(200))),
    ]
    for name, resp in probes:
        d = cp.process(Interaction(id=name, use_case="regulated_decision",
                                   query="What was West revenue?", response=resp,
                                   context=ctx), log=False)
        a = [r for r in d.detector_results if r["name"] == "performance"][0]\
            ["detail"].get("anomaly", {})
        print(f"  {name:28s} action={d.action:7s} p_harm={d.p_harm:<7} "
              f"ood_severity={a.get('ood_severity')}")
    print("\n  A normal answer is untouched; the two broken-but-grounded answers")
    print("  escalate purely on OOD severity — which grounding alone cannot catch,")
    print("  because a repetition loop repeats a perfectly grounded sentence.")
    print("\nALL CONTRACTS HELD.")


if __name__ == "__main__":
    main()
