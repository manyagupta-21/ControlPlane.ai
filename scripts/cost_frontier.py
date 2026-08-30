"""The cost-quality efficient frontier — and where the money actually goes.

Round 1 promised routing that "maximises quality per compute-rupee". This is
that claim, priced. Sweeping the routing aggressiveness traces a frontier; the
optimum is the point that minimises total cost of ownership, not the point that
minimises the compute bill.

The headline finding is the decomposition. Token spend is the number everyone
optimises and it is not where the money is: a cheaper model raises P(harm),
which pushes more traffic into human review, and reviewer time is orders of
magnitude more expensive per interaction than tokens. Routing decisions that
look like savings on an API invoice can be net-negative once the review queue
they create is priced in.

Reported here:
  * cross-validated AUC of the difficulty model (so it is not taken on trust)
  * the frontier, with the compute-optimal and total-cost-optimal points marked
  * a decomposition into compute / review / residual harm per 1,000 interactions
  * sensitivity to `delta`, the one assumed parameter in the degradation model

Usage:  python scripts/cost_frontier.py
Out:    data/cost_frontier.png, data/cost_frontier.json
"""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from controlplane.policy import PolicyEngine
from controlplane.router import Router, DifficultyModel
from controlplane.decision_theory import adjust_prior
from controlplane.detectors import default_detectors
from controlplane.scoring import combine
from controlplane.schemas import Interaction, ACTIONS
from controlplane.decision_theory import harm_probability

POLICY, DATA = "config/policies.yaml", "data/interactions.jsonl"
USE_CASE = "internal_copilot"     # the only tier where cheaper models are even arguable
BASE_RATE = 0.05                  # assumed live prevalence of harmful responses


def score_corpus(pe):
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    dets = default_detectors()
    train_prev = float(np.mean([bool(r.get("label_hallucination") or r.get("label_pii")
                                     or r.get("label_toxic")) for r in rows]))
    out = []
    for r in rows:
        x = Interaction(**r)
        res = [d.run(x) for d in dets]
        _, per_dim = combine(res, pe.use_cases[USE_CASE].get("weights"))
        cal = next((d.detail.get("calibrated_hallucination_prob")
                    for d in res if d.name == "performance"), None)
        p = adjust_prior(harm_probability(per_dim, cal), train_prev, BASE_RATE)
        out.append(dict(query=r["query"], context=r.get("context", ""),
                        p_large=p, harmful=bool(r.get("label_hallucination")
                                                or r.get("label_pii") or r.get("label_toxic"))))
    return out, train_prev


def decompose(loss, rows, router, threshold):
    """Cost per 1,000 interactions, split by where it is actually incurred."""
    compute = review = harm = caveat = blocked = 0.0
    n_small = 0
    for r in rows:
        d = router.difficulty.predict(r["query"], r["context"])
        use_small = d <= threshold
        n_small += use_small
        m = "small" if use_small else "large"
        p = router.degraded(r["p_large"], d) if use_small else r["p_large"]
        compute += router.compute_cost(m, r["query"])
        el = loss.expected_loss(p)
        action = min(ACTIONS, key=lambda a: float(el[a]))
        if action == "review":
            review += loss.review
            harm += p * loss.resid_review * loss.serve_bad
        elif action == "allow":
            harm += p * loss.serve_bad
        elif action == "edit":
            harm += p * loss.resid_edit * loss.serve_bad
            caveat += (1 - p) * loss.caveat
        else:
            blocked += (1 - p) * loss.block_good
    k = 1000.0 / len(rows)
    return dict(threshold=round(threshold, 3), small_share=n_small / len(rows),
                compute=compute * k, review=review * k, harm=harm * k,
                caveat=caveat * k, blocked=blocked * k,
                total=(compute + review + harm + caveat + blocked) * k)


def main():
    pe = PolicyEngine(POLICY)
    loss = pe.loss_models[USE_CASE]
    rows, train_prev = score_corpus(pe)

    dm = DifficultyModel(train_prevalence=train_prev,
                         deploy_prevalence=BASE_RATE).fit([r["query"] for r in rows],
                               [r["context"] for r in rows],
                               [r["harmful"] for r in rows])
    router = Router(loss, dm, delta=0.35)

    print("=" * 76)
    print(f"COST-QUALITY FRONTIER — {USE_CASE}, assumed live base rate {BASE_RATE:.0%}")
    print("=" * 76)
    print(f"  difficulty model: 6 lexical features, cross-validated AUC = "
          f"{dm.auc:.3f}" if dm.auc else "  difficulty model: fallback length rule")
    print(f"  eval-set prevalence {train_prev:.0%} corrected to {BASE_RATE:.0%} before pricing\n")

    grid = np.linspace(0.0, 1.0, 41)
    curve = [decompose(loss, rows, router, t) for t in grid]
    cheapest_compute = min(curve, key=lambda c: c["compute"])
    best_total = min(curve, key=lambda c: c["total"])

    print(f"  {'route-small if difficulty <=':<30}{'small%':>8}{'compute':>10}"
          f"{'review':>10}{'harm':>10}{'TOTAL':>10}")
    for c in curve[::5]:
        mark = "  <- total-cost optimum" if c["threshold"] == best_total["threshold"] else ""
        print(f"  {c['threshold']:<30.2f}{c['small_share']:>7.0%}{c['compute']:>10.2f}"
              f"{c['review']:>10.2f}{c['harm']:>10.2f}{c['total']:>10.2f}{mark}")

    print(f"\n  Compute-minimising policy  : route {cheapest_compute['small_share']:.0%} to small, "
          f"total cost {cheapest_compute['total']:.2f} per 1k")
    print(f"  Total-cost-minimising policy: route {best_total['small_share']:.0%} to small, "
          f"total cost {best_total['total']:.2f} per 1k")
    saving = cheapest_compute["total"] - best_total["total"]
    print(f"  Chasing the compute bill instead of total cost costs {saving:.2f} per 1k "
          f"({saving / max(best_total['total'], 1e-9):.0%} worse).")

    tot = best_total
    print(f"\n  Where the money goes at the optimum (per 1,000 interactions):")
    for k in ["compute", "review", "harm", "caveat", "blocked"]:
        print(f"     {k:<10} {tot[k]:>9.2f}   {tot[k] / tot['total']:>6.1%}")
    print(f"     {'TOTAL':<10} {tot['total']:>9.2f}")
    print(f"\n  Compute is {tot['compute'] / tot['total']:.2%} of total cost of ownership.")
    print("  Optimising the API invoice optimises the smallest line on the page —")
    print("  a cheaper model buys tokens back and pays for them in review queue.")

    # sensitivity to the one assumed parameter
    print("\n  Sensitivity to delta (assumed quality gap of the small model):")
    print(f"     {'delta':>6}{'optimal small%':>16}{'total cost /1k':>16}")
    sens = []
    for delta in [0.10, 0.20, 0.35, 0.50, 0.75]:
        r2 = Router(loss, dm, delta=delta)
        c2 = min((decompose(loss, rows, r2, t) for t in grid), key=lambda c: c["total"])
        sens.append(dict(delta=delta, small_share=c2["small_share"], total=c2["total"]))
        print(f"     {delta:>6.2f}{c2['small_share']:>15.0%}{c2['total']:>16.2f}")

    json.dump(dict(auc=dm.auc, curve=curve, optimum=best_total, sensitivity=sens),
              open("data/cost_frontier.json", "w"), indent=2, default=float)
    print("\nsaved -> data/cost_frontier.json")
    _plot(curve, best_total, cheapest_compute)


def _plot(curve, best, cheap):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("(plot skipped:", e, ")")
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))

    comp = [c["compute"] for c in curve]
    qual = [c["harm"] + c["review"] for c in curve]
    ax[0].plot(comp, qual, "-", color="#A100FF", lw=2)
    ax[0].scatter(comp, qual, s=14, color="#A100FF", alpha=.5)
    ax[0].scatter([best["compute"]], [best["harm"] + best["review"]], s=130,
                  facecolor="none", edgecolor="#1FA36B", lw=2.5, zorder=5,
                  label="total-cost optimum")
    ax[0].scatter([cheap["compute"]], [cheap["harm"] + cheap["review"]], s=130,
                  facecolor="none", edgecolor="#D24545", lw=2.5, zorder=5,
                  label="compute-cost optimum")
    ax[0].set_xlabel("compute cost per 1,000 interactions (USD)")
    ax[0].set_ylabel("review + residual harm cost per 1,000")
    ax[0].set_title("Efficient frontier: cheaper tokens are not cheaper")
    ax[0].legend(fontsize=8)

    ths = [c["threshold"] for c in curve]
    keys = ["compute", "review", "harm", "caveat", "blocked"]
    cols = ["#A100FF", "#3A78C9", "#D24545", "#E39A1C", "#7A7A7A"]
    ax[1].stackplot(ths, *[[c[k] for c in curve] for k in keys],
                    labels=keys, colors=cols, alpha=.85)
    ax[1].axvline(best["threshold"], ls="--", color="black", lw=1.4)
    ax[1].set_xlabel("route to small model if difficulty <= x")
    ax[1].set_ylabel("cost per 1,000 interactions")
    ax[1].set_title("Total cost of ownership, decomposed")
    ax[1].legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig("data/cost_frontier.png", dpi=140)
    print("saved -> data/cost_frontier.png")


if __name__ == "__main__":
    main()
