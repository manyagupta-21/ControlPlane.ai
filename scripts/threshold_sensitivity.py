"""Where the thresholds come from, and how they move.

Two figures:
  (1) Expected loss of each action against P(harm). The optimal policy is the
      lower envelope; the band edges are the crossing points. Nothing is chosen
      by hand.
  (2) Sensitivity of the derived bands to the cost ratio serve_bad / block_good
      — i.e. to risk appetite. The three configured use-cases are marked on it.

Usage:  python scripts/threshold_sensitivity.py
Out:    data/decision_bands.png, data/threshold_sensitivity.png
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from controlplane.decision_theory import LossMatrix
from controlplane.policy import PolicyEngine
from controlplane.schemas import ACTIONS

PURPLE = "#A100FF"
COLORS = {"allow": "#1FA36B", "edit": "#E39A1C", "review": "#3A78C9", "block": "#D24545"}


def fig_bands(pe: PolicyEngine, use_case: str = "customer_facing"):
    lm = pe.loss_models[use_case]
    th = pe.thresholds[use_case]
    ps = np.linspace(0, 1, 500)
    el = lm.expected_loss(ps)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for a in ACTIONS:
        ax.plot(ps, el[a], label=f"E[loss | {a}]", color=COLORS[a], lw=1.6)
    envelope = np.min(np.vstack([el[a] for a in ACTIONS]), axis=0)
    ax.plot(ps, envelope, color="black", lw=2.6, alpha=.85, label="optimal policy (lower envelope)")

    ax.set_ylim(0, float(np.percentile(np.vstack([el[a] for a in ACTIONS]), 92)))
    top = ax.get_ylim()[1]
    edges = [0.0] + [th[k] for k in ["edit", "review", "block"]] + [1.0]
    for a, lo, hi in zip(ACTIONS, edges[:-1], edges[1:]):
        ax.axvspan(lo, hi, color=COLORS[a], alpha=.07)
        ax.text((lo + hi) / 2, top * .95, a.upper(), ha="center", fontsize=8,
                color=COLORS[a], fontweight="bold")
    for name in ["edit", "review", "block"]:
        edge = th[name]
        ax.axvline(edge, ls="--", color="grey", lw=1)
        ax.text(edge, top * .80, f" {edge:.3f}", fontsize=8, color="grey")

    ax.set_xlabel("P(response is harmful if served)")
    ax.set_ylabel("expected loss (cost units)")
    ax.set_title(f"Decision bands are derived, not chosen — {use_case}")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig("data/decision_bands.png", dpi=140)
    print("saved -> data/decision_bands.png")


def fig_sensitivity(pe: PolicyEngine):
    """How risk appetite (the cost ratio) maps to the thresholds."""
    ratios = np.geomspace(1, 500, 120)
    base = LossMatrix()
    rows = {"edit": [], "review": [], "block": []}
    for r in ratios:
        lm = LossMatrix(serve_bad=r * 100.0, block_good=100.0,
                        review=base.review * 5, caveat=base.caveat * 16,
                        resid_edit=base.resid_edit, resid_review=base.resid_review)
        th = lm.derive_thresholds()
        for k in rows:
            rows[k].append(th[k])

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for k in ["edit", "review", "block"]:
        ax.plot(ratios, rows[k], label=f"{k} band edge", color=COLORS[k], lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("risk appetite  =  cost(serve a bad answer) / cost(block a good one)")
    ax.set_ylabel("derived threshold on P(harm)")
    ax.set_title("Risk appetite is a cost ratio; the thresholds follow from it")

    for uc, marker in [("internal_copilot", "o"), ("customer_facing", "s"),
                       ("regulated_decision", "^")]:
        lm = pe.loss_models[uc]
        ratio = lm.serve_bad / lm.block_good
        ax.scatter([ratio], [pe.thresholds[uc]["block"]], marker=marker, s=70,
                   zorder=5, color=PURPLE, edgecolor="white")
        ax.annotate(uc, (ratio, pe.thresholds[uc]["block"]), fontsize=8,
                    textcoords="offset points", xytext=(6, 6))
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("data/threshold_sensitivity.png", dpi=140)
    print("saved -> data/threshold_sensitivity.png")


def table(pe: PolicyEngine):
    print("\nDerived decision bands (lower edge of each band, on P(harm)):")
    print(f"  {'use case':<20} {'appetite ratio':>15} {'edit':>8} {'review':>8} {'block':>8}")
    for uc in pe.use_cases:
        lm, th = pe.loss_models[uc], pe.thresholds[uc]
        ratio = f"{lm.serve_bad / lm.block_good:.1f}x" if lm else "n/a (static)"
        print(f"  {uc:<20} {ratio:>15} {th['edit']:>8.3f} {th['review']:>8.3f} {th['block']:>8.3f}")
    print("\n  Read: the regulated tool blocks from P(harm)=0.30 because a bad answer there")
    print("  costs ~33x a withheld one; the internal copilot tolerates up to 0.90 because an")
    print("  expert reader catches the rest and blocking costs real productivity.")


if __name__ == "__main__":
    pe = PolicyEngine("config/policies.yaml")
    table(pe)
    fig_bands(pe)
    fig_sensitivity(pe)
