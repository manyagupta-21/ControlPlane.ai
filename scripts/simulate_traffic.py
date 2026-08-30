"""Generate a week of traffic so the monitoring view has something to monitor.

Writes real decisions through the real pipeline into the real audit log — no
synthetic rows are fabricated. Overrides are sampled with a deliberate
asymmetry: reviewers overturn cautious calls (block -> allow) far more often
than they overturn permissive ones, which is what actually happens on a review
desk and is the signal the monitoring view is built to surface.

Usage:  python scripts/simulate_traffic.py [--reset] [--n 800]
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from controlplane import ControlPlane, Interaction

POLICY, DATA, LOG = "config/policies.yaml", "data/interactions.jsonl", "data/audit_log.jsonl"
RNG = np.random.default_rng(3)

# Probability a human reviewer overturns each decision, and to what.
# Reviewers push back hardest on blocks they consider unnecessary.
OVERRIDE = {
    "block":  (0.28, ["allow", "edit", "review"], [0.55, 0.30, 0.15]),
    "review": (0.15, ["allow", "edit", "block"],  [0.60, 0.30, 0.10]),
    "edit":   (0.06, ["allow", "review"],         [0.75, 0.25]),
    "allow":  (0.02, ["edit", "review", "block"], [0.50, 0.35, 0.15]),
}
REVIEWERS = ["r_anita", "r_dev", "r_farah", "r_joseph"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    cp = ControlPlane(POLICY, audit_path=LOG)
    if args.reset:
        cp.audit.reset()
        print("audit log reset")

    counts, n_over = {}, 0
    for i in range(args.n):
        r = dict(rows[int(RNG.integers(len(rows)))])
        r["id"] = f"sim-{int(time.time() * 1000) % 10 ** 8}-{i}"
        d = cp.process(Interaction(**r), log=True)
        counts[d.action] = counts.get(d.action, 0) + 1

        p, options, weights = OVERRIDE[d.action]
        if RNG.random() < p:
            to = str(RNG.choice(options, p=weights))
            cp.audit.record_override(r["id"], d.action, to,
                                     reviewer=str(RNG.choice(REVIEWERS)),
                                     note="simulated reviewer decision")
            n_over += 1

    print(f"wrote {args.n} decisions and {n_over} overrides -> {LOG}")
    for a in ("allow", "edit", "review", "block"):
        print(f"  {a:<8} {counts.get(a, 0):>5}  {counts.get(a, 0) / args.n:>6.1%}")
    print(f"  override rate overall: {n_over / args.n:.1%}")
    print("\nNow run:  streamlit run app/monitoring.py")


if __name__ == "__main__":
    main()
