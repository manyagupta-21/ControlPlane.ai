"""Monitoring view — what you show a skeptical stakeholder.

Reads the append-only audit log and answers the four questions an owner of this
system actually gets asked:

  1. What is it doing?          decision mix, by use case
  2. Is it too slow?            inline latency vs the configured budget
  3. Is it trusted?             override rate, and which rules get overturned
  4. What does it cost?         cost of ownership per 1,000 interactions

The override panel is the important one. A rule with a high override rate is not
a rule doing its job — it is a rule the desk has learned to ignore, and it is the
first thing to retire. That number is the closest thing to ground truth this
system ever sees, which is why it feeds the learning loop rather than sitting in
a report.

Run:  streamlit run app/monitoring.py
      (populate the log first: python scripts/simulate_traffic.py --reset)
"""
from __future__ import annotations
import json, os, sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import altair as alt
from controlplane.policy import PolicyEngine
from controlplane.feedback import rule_level_overrides, rolling_trust, recommend

LOG, POLICY = "data/audit_log.jsonl", "config/policies.yaml"
ACTIONS = ["allow", "edit", "review", "block"]

st.set_page_config(page_title="ControlPlane: Monitoring", layout="wide")
st.title("ControlPlane: monitoring")
st.caption("Every number below is computed from the append-only audit trail at "
           f"`{LOG}`. Nothing here is stored separately from the decisions themselves.")


@st.cache_data(show_spinner=False)
def load(path):
    if not os.path.exists(path):
        return pd.DataFrame(), pd.DataFrame()
    dec, ovr = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            (ovr if r.get("type") == "override" else dec).append(r)
    return pd.DataFrame(dec), pd.DataFrame(ovr)


dec, ovr = load(LOG)
if dec.empty:
    st.warning("No decisions in the audit log yet. Run "
               "`python scripts/simulate_traffic.py --reset` first.")
    st.stop()

pe = PolicyEngine(POLICY)

# ---------------------------------------------------------------- headline
inline = []
for reasons in dec.get("reasons", []):
    for s in (reasons or []):
        if "inline_latency_ms=" in s:
            try:
                inline.append(float(s.split("inline_latency_ms=")[1].split()[0]))
            except (ValueError, IndexError):
                pass
inline = np.array(inline) if inline else np.array([0.0])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Decisions logged", f"{len(dec):,}")
c2.metric("Human overrides", f"{len(ovr):,}",
          f"{len(ovr) / len(dec):.1%} of decisions" if len(dec) else None)
c3.metric("Inline latency p50", f"{np.percentile(inline, 50):.1f} ms")
c4.metric("Inline latency p95", f"{np.percentile(inline, 95):.1f} ms")

st.divider()
left, right = st.columns([1.15, 1])

# ---------------------------------------------------------------- decision mix
with left:
    st.subheader("What is it doing?")
    mix = (dec.groupby(["use_case", "action"]).size()
           .unstack(fill_value=0).reindex(columns=ACTIONS, fill_value=0))
    share = mix.div(mix.sum(axis=1), axis=0)
    st.bar_chart(share)
    st.caption("Review load is the staffing number: it is what the operations "
               "team has to resource, and it moves whenever the cost model does.")
    load_tbl = pd.DataFrame({
        "volume": mix.sum(axis=1),
        "review load": share.get("review", 0).map("{:.1%}".format),
        "block rate": share.get("block", 0).map("{:.1%}".format),
        "block band (derived)": [f"P(harm) ≥ {pe.thresholds[u]['block']:.3f}"
                                 if u in pe.thresholds else "—" for u in mix.index],
    })
    st.dataframe(load_tbl, use_container_width=True)

# ---------------------------------------------------------------- latency
with right:
    st.subheader("Is it fast enough?")
    budget = min(uc.get("latency_budget_ms", 1500) for uc in pe.use_cases.values())
    breaches = float((inline > budget).mean())
    st.metric("Inline p99", f"{np.percentile(inline, 99):.1f} ms",
              f"budget {budget} ms · {breaches:.2%} breach")
    st.caption(
        "Only hard rules run inline; grounding and cost checks run asynchronously, "
        "so the user-facing path stays short. The p99 is the number that matters — "
        "an average latency hides exactly the tail a customer notices.")
    st.progress(min(1.0, float(np.percentile(inline, 99)) / budget),
                text=f"p99 uses {np.percentile(inline, 99) / budget:.0%} of the tightest budget")

st.divider()

# ---------------------------------------------------------------- trust
st.subheader("Is it trusted?: override analysis")
if ovr.empty:
    st.info("No overrides logged yet.")
else:
    base = Counter(dec["action"])
    tab = []
    for a in ACTIONS:
        got = ovr[ovr["from_action"] == a] if "from_action" in ovr else ovr.iloc[0:0]
        n = base.get(a, 0)
        tab.append({
            "decision": a,
            "volume": n,
            "overridden": len(got),
            "override rate": len(got) / n if n else 0.0,
            "most often changed to": (Counter(got["to_action"]).most_common(1)[0][0]
                                      if len(got) else "—"),
        })
    tdf = pd.DataFrame(tab)
    st.dataframe(
        tdf.style.format({"override rate": "{:.1%}"})
        .background_gradient(subset=["override rate"], cmap="Reds"),
        use_container_width=True)

    worst = tdf.sort_values("override rate", ascending=False).iloc[0]
    st.warning(
        f"**{worst['decision']}** is overturned {worst['override rate']:.0%} of the time, "
        f"usually to **{worst['most often changed to']}**. Read that as a cost-model "
        f"signal, not reviewer error: the desk is telling us the assumed cost of this "
        f"action is set too high relative to what it is preventing. The fix is to "
        f"re-estimate that cost in `config/policies.yaml`, which moves the derived "
        f"bands automatically — no thresholds to hand-tune.")

    st.caption("Reviewer agreement (low spread = a consistent standard; high spread "
               "= the policy is ambiguous and needs clarifying before it needs retraining).")
    by_rev = (ovr.groupby("reviewer").size() / len(ovr)).sort_values(ascending=False)
    st.bar_chart(by_rev)

    # ------------------------------------------------ rule-level attribution
    st.markdown("**Which specific rule is losing trust?**")
    st.caption(
        "The table above averages every 'review' (or 'block'...) together, but one "
        "action can be produced by several different rules -- a use_case rule, a "
        "jurisdiction rule, a sector rule -- each with its own override rate. This "
        "attributes every override back to the exact fired rule(s) behind the "
        "decision it overturned, so the fix below points at one line in "
        "`config/policies.yaml`, not a guess.")
    rule_tbl = rule_level_overrides(dec, ovr)
    if rule_tbl.empty:
        st.info("No fired_rules recorded on any decision in this log.")
    else:
        st.dataframe(
            rule_tbl.style.format({"override_rate": "{:.1%}"})
            .background_gradient(subset=["override_rate"], cmap="Reds"),
            use_container_width=True, hide_index=True)

        recs = recommend(rule_tbl)
        if recs:
            for r in recs:
                st.warning(r)
        else:
            st.success("No single rule is overturned often enough (>=30%, "
                       "n>=5) to act on yet.")

        worst_rule = rule_tbl.iloc[0]["fired_rule"]
        trend = rolling_trust(dec, ovr, worst_rule)
        if len(trend) >= 3:
            st.caption(f"Rolling override rate for the worst offender, "
                      f"`{worst_rule}`, over the order decisions arrived "
                      f"(not calendar time) -- is trust in this rule stable, "
                      f"improving, or eroding?")
            x_min, x_max = int(trend["decision_seq"].min()), int(trend["decision_seq"].max())
            chart = (
                alt.Chart(trend)
                .mark_line()
                .encode(
                    x=alt.X("decision_seq:Q", title="decision #",
                           scale=alt.Scale(domain=[x_min, x_max])),
                    y=alt.Y("rolling_override_rate:Q", title="rolling override rate",
                           scale=alt.Scale(domain=[0, 1])),
                )
                .properties(height=250)
            )
            st.altair_chart(chart, use_container_width=True)

st.divider()

# ---------------------------------------------------------------- cost
st.subheader("What does it cost?")
per_1k = defaultdict(float)
for _, r in dec.iterrows():
    lm = pe.loss_models.get(r["use_case"])
    if lm is None:
        continue
    p = float(r.get("p_harm", 0.0) or 0.0)
    a = r["action"]
    if a == "review":
        per_1k["human review"] += lm.review
        per_1k["residual harm"] += p * lm.resid_review * lm.serve_bad
    elif a == "allow":
        per_1k["residual harm"] += p * lm.serve_bad
    elif a == "edit":
        per_1k["residual harm"] += p * lm.resid_edit * lm.serve_bad
        per_1k["caveat friction"] += (1 - p) * lm.caveat
    else:
        per_1k["withheld value"] += (1 - p) * lm.block_good

for det in dec.get("detector_results", []):
    for d in (det or []):
        if d.get("name") == "cost":
            per_1k["compute"] += float(d.get("detail", {}).get("est_cost_usd", 0.0))

scale = 1000.0 / len(dec)
cost = pd.DataFrame({"cost per 1,000 interactions":
                     {k: round(v * scale, 2) for k, v in per_1k.items()}})
cost["share"] = (cost.iloc[:, 0] / cost.iloc[:, 0].sum()).map("{:.1%}".format)
st.dataframe(cost.sort_values(cost.columns[0], ascending=False), use_container_width=True)
st.caption(
    "Compute is the line everyone optimises and the smallest one on the page. "
    "The controllable lever is review load, which the cost model sets directly — "
    "which is why risk appetite and operating cost are the same decision here.")

st.divider()
st.caption("Audit trail is append-only. Overrides are stored as their own records, "
           "so any decision can be reconstructed together with what a human did about it.")
