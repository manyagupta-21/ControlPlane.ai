"""Feedback loop: turns human overrides from *logged* into *consumed*.

Existing state before this module: every decision is logged, every override is
logged (controlplane/audit.py), and app/monitoring.py already reads both back
and reports an override rate PER ACTION (allow/edit/review/block), with a
recommendation to re-estimate the cost model behind whichever action gets
overturned most.

That's useful but too coarse. A single action like "review" can be produced
by many different fired rules -- a use_case rule, an EU jurisdiction rule, a
healthcare sector rule -- and averaging them together hides which SPECIFIC
rule the desk has stopped trusting. Since every decision already carries a
fired_rules list tagged by axis (e.g. "jurisdiction:EU:uncertain_grounding",
"sector:healthcare:ungrounded"), this module attributes each override back to
the exact rule(s) that fired on the decision it overturned, not just the
action those rules happened to produce.

This stays a signal for a human to act on in config/policies.yaml, on
purpose. It does NOT feed overrides into recalibrating the isotonic model or
auto-adjusting thresholds -- README already documents that override labels in
this prototype are simulated (no real review desk exists yet), so treating
them as training data would repeat the exact circularity problem already
flagged for the PII/toxicity evaluation harness. Surfacing the signal
precisely is the honest, defensible version of "feedback loops... improve
detection quality over time" for a prototype at this stage.
"""
from __future__ import annotations
import json
import os
from collections import defaultdict

import pandas as pd


def load_audit(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shared loader: split the append-only log into decision rows and
    override rows. (Same split app/monitoring.py already does -- centralised
    here so both consumers agree on what a "decision" vs "override" row is.)"""
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


def rule_level_overrides(dec: pd.DataFrame, ovr: pd.DataFrame) -> pd.DataFrame:
    """For every fired_rule tag that has appeared, what fraction of the
    decisions it touched were later overridden, and to what.

    This is the granular join the action-level panel in app/monitoring.py
    can't do: two "review" decisions with the same action can be driven by
    completely different rules, and only one of those rules might actually
    be losing the desk's trust.
    """
    if dec.empty or "fired_rules" not in dec.columns:
        return pd.DataFrame()

    dec_idx = dec.set_index("interaction_id")
    to_action = (ovr.set_index("interaction_id")["to_action"].to_dict()
                 if not ovr.empty else {})
    overridden_ids = set(to_action.keys())

    touched: dict[str, int] = defaultdict(int)
    hit: dict[str, int] = defaultdict(int)
    changed_to: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for iid, row in dec_idx.iterrows():
        rules = row.get("fired_rules") or []
        for rule in rules:
            touched[rule] += 1
            if iid in overridden_ids:
                hit[rule] += 1
                changed_to[rule][to_action[iid]] += 1

    rows = []
    for rule, n in touched.items():
        n_over = hit.get(rule, 0)
        top_to = (max(changed_to[rule].items(), key=lambda kv: kv[1])[0]
                  if changed_to[rule] else "-")
        rows.append({
            "fired_rule": rule,
            "volume": n,
            "overridden": n_over,
            "override_rate": (n_over / n) if n else 0.0,
            "most_often_changed_to": top_to,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("override_rate", ascending=False).reset_index(drop=True)


def rolling_trust(dec: pd.DataFrame, ovr: pd.DataFrame, rule: str,
                   window: int = 25) -> pd.DataFrame:
    """Rolling override rate for one rule, in decision order (not calendar
    time -- simulate_traffic.py generates a batch in rapid succession, so a
    calendar-week bucketing like drift_monitor.py uses would be misleading
    here; decision order is the honest stand-in for "as more of the desk's
    judgement accumulates").

    Lets a stakeholder see whether trust in a specific rule is stable,
    improving, or actively eroding, rather than a single lifetime average.
    """
    if dec.empty or "fired_rules" not in dec.columns:
        return pd.DataFrame()
    overridden_ids = set(ovr["interaction_id"]) if not ovr.empty else set()

    hits = []
    for _, row in dec.iterrows():
        rules = row.get("fired_rules") or []
        if rule in rules:
            hits.append(1 if row["interaction_id"] in overridden_ids else 0)
    if not hits:
        return pd.DataFrame()

    s = pd.Series(hits, name="overridden")
    roll = s.rolling(window=min(window, len(s)), min_periods=1).mean()
    return pd.DataFrame({"decision_seq": range(1, len(s) + 1),
                         "rolling_override_rate": roll})


def recommend(rule_table: pd.DataFrame, min_volume: int = 5,
              high_rate: float = 0.30) -> list[str]:
    """Plain-English, per-rule version of the recommendation
    app/monitoring.py already gives at the action level -- pointed at the
    exact axis (use_case / jurisdiction / sector) and condition losing trust,
    so it's directly actionable as a specific edit in config/policies.yaml
    rather than a guess about which of several rules to touch."""
    if rule_table.empty:
        return []
    flagged = rule_table[(rule_table["volume"] >= min_volume) &
                         (rule_table["override_rate"] >= high_rate)]
    out = []
    for _, r in flagged.iterrows():
        axis = r["fired_rule"].split(":", 1)[0]
        out.append(
            f"`{r['fired_rule']}` fired {r['volume']}x, overridden "
            f"{r['override_rate']:.0%} of the time (mostly changed to "
            f"'{r['most_often_changed_to']}'). It's a {axis}-scoped rule -- "
            f"re-check its cost assumption or `applies_to` scope in "
            f"config/policies.yaml, rather than adjusting the underlying "
            f"detector, which other rules still rely on.")
    return out