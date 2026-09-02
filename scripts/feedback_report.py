"""Rule-level feedback attribution -- closes the loop the README's issue list
flagged as "logged but not consumed."

app/monitoring.py already reports an override rate PER ACTION. This report
goes one level deeper: which SPECIFIC fired rule (use_case, jurisdiction, or
sector scoped) is losing the desk's trust, since one action can be produced
by several different rules with very different override rates.

Usage:  python scripts/feedback_report.py
        (populate the log first: python scripts/simulate_traffic.py --reset)
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from controlplane.feedback import load_audit, rule_level_overrides, recommend

LOG = "data/audit_log.jsonl"


def main():
    dec, ovr = load_audit(LOG)
    if dec.empty:
        print(f"No decisions in {LOG} yet. Run "
              "`python scripts/simulate_traffic.py --reset` first.")
        return

    print(f"{len(dec)} decisions, {len(ovr)} overrides "
          f"({len(ovr) / len(dec):.1%} overall override rate)\n")

    table = rule_level_overrides(dec, ovr)
    if table.empty:
        print("No fired_rules recorded on any decision -- nothing to attribute.")
        return

    print("Override rate BY FIRED RULE (not just by action):")
    print(table.to_string(index=False, formatters={
        "override_rate": "{:.1%}".format}))

    print("\nRecommendations (volume >= 5, override rate >= 30%):")
    recs = recommend(table)
    if not recs:
        print("  None. No single rule is being overturned often enough to act on.")
    else:
        for r in recs:
            print(f"  - {r}")


if __name__ == "__main__":
    main()