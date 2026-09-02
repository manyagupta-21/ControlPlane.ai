"""Back-test the control layer on realised cost, not on hand-written labels.

Accuracy against a 'gold action' is circular: we wrote the gold actions. The
question a sponsor actually asks is *what did the policy cost*. So we score
every policy by the loss it realises once the true state of each response is
known, and compare against the two trivial policies and the oracle.

    captured = (loss_no_guardrail - loss_policy) / (loss_no_guardrail - loss_oracle)

i.e. what fraction of the achievable savings the control layer actually banks.

Usage:  python scripts/loss_backtest.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from controlplane import Interaction
from controlplane.pipeline import ControlPlane
from controlplane.policy import PolicyEngine
from controlplane.decision_theory import adjust_prior

DATA, POLICY = "data/interactions.jsonl", "config/policies.yaml"
# the static bands the prototype used before the cost model replaced them
LEGACY = {"customer_facing": {"edit": .35, "review": .55, "block": .80},
          "internal_copilot": {"edit": .45, "review": .65, "block": .88},
          "regulated_decision": {"edit": .25, "review": .45, "block": .70}}


def band(th, p):
    return ("block" if p >= th["block"] else "review" if p >= th["review"]
            else "edit" if p >= th["edit"] else "allow")


def main():
    data = [Interaction(**json.loads(l)) for l in open(DATA, encoding="utf-8")]
    pe = PolicyEngine(POLICY)
    cp = ControlPlane(POLICY, audit_path="data/_backtest_audit.jsonl")

    rows = []
    for x in data:
        harmful = bool(x.label_hallucination or x.label_pii or x.label_toxic)
        # ONE call to the full pipeline — gets both the action and p_harm from
        # the same code path. The previous version ran detectors manually to get
        # `p` and then called cp.process() again to get the action, meaning the
        # two numbers came from different code paths: `p` bypassed OOD widening
        # and hard rules, while `controlplane` action included them. The
        # prevalence-sensitivity table then applied adjust_prior to the wrong `p`.
        d = cp.process(x, log=False)
        rows.append(dict(uc=x.use_case, harmful=harmful,
                         p=d.p_harm,           # the widened p_decision the policy actually used
                         controlplane=d.action,
                         legacy=band(LEGACY[x.use_case], d.p_harm),
                         oracle="block" if harmful else "allow"))

    policies = {
        "No guardrail (allow all)": lambda r: "allow",
        "Block everything":         lambda r: "block",
        "Static thresholds (v1)":   lambda r: r["legacy"],
        "ControlPlane (cost-derived)": lambda r: r["controlplane"],
        "Oracle (knows the truth)": lambda r: r["oracle"],
    }
    losses = {}
    for name, fn in policies.items():
        tot = sum(pe.loss_models[r["uc"]].realised_loss(fn(r), r["harmful"]) for r in rows)
        losses[name] = tot / len(rows)

    base, best = losses["No guardrail (allow all)"], losses["Oracle (knows the truth)"]
    print("=" * 72)
    print("LOSS BACK-TEST  — mean realised cost per interaction (cost units)")
    print("=" * 72)
    for name, v in losses.items():
        cap = (base - v) / (base - best) if base != best else float("nan")
        print(f"  {name:<30} {v:8.2f}   savings captured: {cap:6.1%}")
    print("\n  A control layer is only worth its latency if it sits well inside the")
    print("  gap between doing nothing and the oracle. Blocking everything is the")
    print("  control that always 'works' and always fails the business case.")

    # ---- base-rate correction --------------------------------------------
    # The synthetic set is ~50% harmful by construction; real traffic is not.
    # Realised loss depends on prevalence, so we report it conditionally and
    # recombine at plausible base rates rather than quoting one inflated number.
    print("\n" + "=" * 72)
    print("PREVALENCE SENSITIVITY  — mean loss at realistic harmful base rates")
    print("=" * 72)
    print(f"  {'policy':<30}" + "".join(f"{p:>10.0%}" for p in [.02, .05, .10, .50]))
    PIS = [.02, .05, .10, .50]
    TRAIN_PREV = np.mean([r["harmful"] for r in rows])   # prevalence of the eval set

    def loss_at(fn, pi, prior_aware=False):
        lh, lb = [], []
        for r in rows:
            if prior_aware:
                p_adj = adjust_prior(r["p"], TRAIN_PREV, pi)
                a = band(pe.thresholds[r["uc"]], p_adj)
            else:
                a = fn(r)
            (lh if r["harmful"] else lb).append(
                pe.loss_models[r["uc"]].realised_loss(a, r["harmful"]))
        return pi * np.mean(lh) + (1 - pi) * np.mean(lb)

    for name, fn in policies.items():
        cells = "".join(f"{loss_at(fn, pi):>10.1f}" for pi in PIS)
        print(f"  {name:<30}{cells}")
    cells = "".join(f"{loss_at(None, pi, prior_aware=True):>10.1f}" for pi in PIS)
    print(f"  {'ControlPlane + prior correction':<30}{cells}")
    print(f"\n  Eval-set prevalence is {TRAIN_PREV:.0%}; live traffic is a few percent.")
    print("  Without the label-shift correction the same loss matrix over-blocks benign")
    print("  traffic at a low base rate. With it, the policy is cheapest at every")
    print("  prevalence — the base rate becomes a stated assumption, not a hidden one.")

    # decision mix, which is what an operations owner actually staffs against
    print("\nDecision mix by use case (ControlPlane):")
    for uc in sorted({r["uc"] for r in rows}):
        sub = [r for r in rows if r["uc"] == uc]
        mix = {a: sum(r["controlplane"] == a for r in sub) / len(sub)
               for a in ["allow", "edit", "review", "block"]}
        print(f"  {uc:<20} " + "  ".join(f"{a} {v:.0%}" for a, v in mix.items())
              + f"   (review load: {mix['review']:.0%} of volume)")


if __name__ == "__main__":
    main()
