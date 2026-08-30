"""Ongoing monitoring: does live traffic still look like what we validated on?

This replaces the earlier PSI calculation, which compared clean responses against
hallucinated ones. That measures class separation, not stability — the two groups
were never meant to look alike, so a large value there means nothing. PSI is a
TEMPORAL statistic: you freeze a reference window at validation time and compare
every later window against it.

    PSI = sum_bins (p_current - p_reference) * ln(p_current / p_reference)

Convention (SR 11-7 practice, borrowed from credit scorecard monitoring):
    < 0.10   stable
    0.10-0.25 investigate
    > 0.25   material shift — revalidate before continuing to rely on the model

We also run a two-sample Kolmogorov-Smirnov test alongside it. PSI has no null
distribution and its cut-offs are convention rather than inference; KS supplies
an actual p-value. Reporting both is the honest thing to do: a large PSI with an
insignificant KS on a small window is a false alarm waiting to happen.

The simulation runs 16 weekly batches. Weeks 1-4 are the frozen reference. From
week 9 we inject a realistic shift — retrieval starts returning shorter, less
relevant context, exactly what happens when an upstream index degrades or a
chunking change ships. Nothing about the control layer changes; the point is
that the monitor notices before anyone files a complaint.

Usage:  python scripts/drift_monitor.py
Out:    data/drift_monitor.png, data/drift_monitor.json
"""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from controlplane import ControlPlane, Interaction

POLICY = "config/policies.yaml"
DATA = "data/interactions.jsonl"
N_WEEKS, REF_WEEKS, SHIFT_WEEK, BATCH = 16, 4, 9, 150
RNG = np.random.default_rng(11)


def psi(reference, current, bins=10):
    """Population Stability Index against a FROZEN reference window.

    Bins are equal-width on [0, 1], not quantiles of the reference. Quantile
    binning is the textbook default, but it is unusable here: P(harm) is lumpy
    (the responsibility detector fires at a fixed level), so reference quantiles
    collapse onto repeated values and PSI then explodes on ordinary sampling
    noise. Equal-width bins on a bounded score are stable and comparable across
    windows, which is what a monitor needs.
    """
    ref, cur = np.asarray(reference, float), np.asarray(current, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    if len(edges) < 3:
        return 0.0
    r = np.histogram(ref, edges)[0] / max(len(ref), 1)
    c = np.histogram(cur, edges)[0] / max(len(cur), 1)
    r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def ks_test(a, b):
    """Two-sample KS statistic and asymptotic p-value (no SciPy dependency)."""
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    ne = len(a) * len(b) / (len(a) + len(b))
    lam = (np.sqrt(ne) + 0.12 + 0.11 / np.sqrt(ne)) * d
    j = np.arange(1, 101)
    p = float(np.clip(2 * np.sum((-1) ** (j - 1) * np.exp(-2 * j ** 2 * lam ** 2)), 0, 1))
    return d, p


def degrade_context(text: str, severity: float) -> str:
    """Simulate a degrading retrieval layer: shorter, partially irrelevant context."""
    words = (text or "").split()
    if not words:
        return text
    keep = max(3, int(len(words) * (1 - 0.55 * severity)))
    return " ".join(words[:keep])


def main():
    base = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    cp = ControlPlane(POLICY, audit_path="data/_drift_audit.jsonl")

    weeks = []
    for w in range(1, N_WEEKS + 1):
        severity = 0.0 if w < SHIFT_WEEK else min(1.0, (w - SHIFT_WEEK + 1) / 5.0)
        rows = RNG.choice(len(base), size=BATCH, replace=True)
        ps, actions = [], []
        for i in rows:
            r = dict(base[i])
            r["context"] = degrade_context(r.get("context", ""), severity)
            x = Interaction(**{k: v for k, v in r.items()})
            d = cp.process(x, log=False)
            ps.append(d.p_harm)
            actions.append(d.action)
        weeks.append(dict(week=w, severity=round(severity, 2), p=ps,
                          block_rate=float(np.mean([a == "block" for a in actions])),
                          review_rate=float(np.mean([a == "review" for a in actions]))))

    reference = [p for wk in weeks[:REF_WEEKS] for p in wk["p"]]

    print("=" * 74)
    print("ONGOING MONITORING — P(harm) distribution vs a frozen reference window")
    print("=" * 74)
    print(f"reference = weeks 1-{REF_WEEKS} (n={len(reference)}); "
          f"retrieval degradation injected from week {SHIFT_WEEK}\n")
    print(f"  {'week':>4} {'PSI':>7} {'KS D':>7} {'KS p':>9} {'block%':>8} {'review%':>8}  verdict")
    out = []
    for wk in weeks:
        val = psi(reference, wk["p"])
        d, p = ks_test(reference, wk["p"])
        verdict = ("stable" if val < 0.10 else
                   "investigate" if val < 0.25 else "MATERIAL SHIFT")
        if val >= 0.10 and p > 0.05:
            verdict += " (KS n.s. — likely small-sample noise)"
        print(f"  {wk['week']:>4} {val:>7.3f} {d:>7.3f} {p:>9.4f} "
              f"{wk['block_rate']:>7.0%} {wk['review_rate']:>7.0%}  {verdict}")
        out.append(dict(week=wk["week"], psi=round(val, 4), ks_d=round(d, 4),
                        ks_p=round(p, 5), block_rate=wk["block_rate"],
                        review_rate=wk["review_rate"], verdict=verdict))

    breach = next((r["week"] for r in out if r["psi"] >= 0.25), None)
    print(f"\n  First material breach: week {breach}" if breach else
          "\n  No material breach detected.")
    print("  The monitor fires on the input distribution, before anyone is harmed —")
    print("  which is the whole point of ongoing monitoring rather than incident review.")

    json.dump(out, open("data/drift_monitor.json", "w"), indent=2)
    print("saved -> data/drift_monitor.json")
    _plot(out, breach)


def _plot(out, breach):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("(plot skipped:", e, ")")
        return
    w = [r["week"] for r in out]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.plot(w, [r["psi"] for r in out], "-o", color="#A100FF", lw=2, label="PSI vs frozen reference")
    ax.set_yscale("symlog", linthresh=0.3)
    ax.set_yticks([0, 0.1, 0.25, 1, 3, 6])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.axhspan(0, 0.10, color="#1FA36B", alpha=.10)
    ax.axhspan(0.10, 0.25, color="#E39A1C", alpha=.12)
    ax.axhline(0.10, ls="--", lw=1, color="#E39A1C", label="investigate (0.10)")
    ax.axhline(0.25, ls="--", lw=1, color="#D24545", label="material shift (0.25)")
    ax.axvline(SHIFT_WEEK, ls=":", color="grey", lw=1.4)
    ax.text(SHIFT_WEEK + .1, ax.get_ylim()[1] * .9, "retrieval degrades", fontsize=8, color="grey")
    if breach:
        ax.scatter([breach], [next(r["psi"] for r in out if r["week"] == breach)],
                   s=110, facecolor="none", edgecolor="#D24545", lw=2, zorder=5)
    ax2 = ax.twinx()
    ax2.plot(w, [r["review_rate"] for r in out], "-s", color="#3A78C9", alpha=.55,
             lw=1.4, ms=4, label="review load")
    ax2.set_ylabel("review load (share of volume)", color="#3A78C9")
    ax2.set_ylim(0, 0.8)
    ax.set_xlabel("week"); ax.set_ylabel("PSI")
    ax.set_title("Ongoing monitoring: the input shift is visible before the outcomes are")
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig("data/drift_monitor.png", dpi=140)
    print("saved -> data/drift_monitor.png")


if __name__ == "__main__":
    main()
