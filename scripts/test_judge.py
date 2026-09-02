"""Demo: AI-as-judge grounding verdict alongside TF-IDF primary score.

Prerequisites:
    pip install groq
    $env:GROQ_API_KEY = "your_free_key"   # https://console.groq.com

Run:
    python scripts/test_judge.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from controlplane.pipeline import ControlPlane
from controlplane.schemas import Interaction

RULE = "=" * 74
cp = ControlPlane("config/policies.yaml", audit_path="data/_judge_demo.jsonl")

CTX = (
    "Q3 revenue: West region 42.1m, East region 38.4m. "
    "Growth was driven by two enterprise accounts signed in August. "
    "North region declined 3% due to supply chain disruption."
)

CASES = [
    ("grounded",
     "West region revenue was 42.1m in Q3, driven by two enterprise accounts.",
     "Response matches the context directly."),
    ("ungrounded — fabricated figure",
     "West revenue was 42.1m and South region hit 99.9m, a record high.",
     "South region does not appear in the context at all."),
    ("contradicted — wrong direction",
     "The North region grew strongly in Q3, outperforming all other regions.",
     "Context says North declined 3%."),
]

print(RULE)
print("AI-AS-JUDGE DEMO — structured verdict alongside TF-IDF primary score")
print(RULE)

groq_installed = True
try:
    import groq  # noqa: F401
except ImportError:
    groq_installed = False

if not groq_installed:
    print("\nSetup needed:")
    print("  1. pip install groq")
    print("  2. Get a free key: https://console.groq.com")
    print("  3. $env:GROQ_API_KEY = 'your_key'\n")
elif not os.environ.get("GROQ_API_KEY"):
    print("\nGROQ_API_KEY not set — judge will return api_key_missing.\n")

for name, response, note in CASES:
    d = cp.process(Interaction(
        id=f"judge_{name.replace(' ','_').replace('—','').replace(' ','')}",
        use_case="regulated_decision",
        query="Summarise Q3 revenue.", response=response,
        context=CTX, jurisdiction="US", sector="finance",
    ), log=False)

    perf = next(r for r in d.detector_results if r["name"] == "performance")
    det = perf["detail"]
    judge = det.get("judge", {})

    print(f"\nCase: {name}")
    print(f"  Note: {note}")
    print(f"  Response:  {response}")
    print(f"  TF-IDF risk:   {det.get('grounding_risk')}  "
          f"(ungrounded={perf['flags'].get('ungrounded')})")
    print(f"  Judge verdict: {judge.get('verdict', 'n/a')}")
    if judge.get("unsupported_claims"):
        print(f"  Unsupported:   {judge['unsupported_claims']}")
    if judge.get("contradicted_claims"):
        print(f"  Contradicted:  {judge['contradicted_claims']}")
    if judge.get("reasoning"):
        print(f"  Reasoning:     {judge['reasoning']}")
    if judge.get("confidence") is not None:
        print(f"  Confidence:    {judge['confidence']}")
    if judge.get("judge_latency_ms"):
        print(f"  Judge latency: {judge['judge_latency_ms']} ms")
    # Always show raw output on parse_error so failures are diagnosable
    if judge.get("verdict") == "parse_error":
        print(f"  RAW OUTPUT (first 400 chars):\n    {judge.get('raw_preview','(empty)')}")
    if judge.get("verdict") == "reasoning_exhausted":
        print(f"  (model spent its full token budget on internal reasoning "
              f"and never wrote an answer — a known gpt-oss behaviour, retried once)")
    if perf["flags"].get("judge_overrides_tfidf"):
        print("  *** judge_overrides_tfidf=True: TF-IDF missed this, judge caught it ***")
    print(f"  Final action:  {d.action}  (p_harm={d.p_harm})")

print(f"\n{RULE}")
print("The judge verdict is always in the audit trail (detail['judge']).")
print("It never overrides the calibrated TF-IDF risk used for the decision.")
print("judge_overrides_tfidf=True flags cases a human reviewer should check.")
print(RULE)
