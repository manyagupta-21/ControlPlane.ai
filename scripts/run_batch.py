"""Run the ControlPlane pipeline over the whole dataset and summarise.

Usage:  python scripts/run_batch.py
"""
from __future__ import annotations
import json, os, sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from controlplane import ControlPlane, Interaction

DATA = "data/interactions.jsonl"
POLICY = "config/policies.yaml"


def load(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield Interaction(**json.loads(line))


def main():
    cp = ControlPlane(POLICY)
    cp.audit.reset()
    actions, by_uc = Counter(), Counter()
    inline_lat, wall_lat = [], []
    n = 0
    for x in load(DATA):
        d = cp.process(x)
        actions[d.action] += 1
        by_uc[(x.use_case, d.action)] += 1
        wall_lat.append(d.total_latency_ms)
        n += 1

    print(f"\nProcessed {n} interactions.")
    print("Decisions:", dict(actions))
    print(f"Mean wall latency: {sum(wall_lat)/len(wall_lat):.2f} ms "
          f"(all checks run in parallel)")
    print("\nDecisions by use-case:")
    for uc in ["customer_facing", "internal_copilot", "regulated_decision"]:
        row = {a: by_uc[(uc, a)] for a in ["allow", "edit", "review", "block"]}
        print(f"  {uc:20s} {row}")
    print(f"\nAudit trail written to data/audit_log.jsonl")


if __name__ == "__main__":
    main()
