import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from controlplane.policy import PolicyEngine
from controlplane.schemas import DetectorResult

policy = PolicyEngine("config/policies.yaml")

def show(use_case, jurisdiction, resp_risk, flags):
    results = [DetectorResult("performance", 0.05, {}, {}, "async", 1.0),
               DetectorResult("responsibility", resp_risk, flags, {}, "inline", 0.1)]
    d = policy.decide("t1", use_case, results, jurisdiction=jurisdiction)
    print(f"{use_case:18} {jurisdiction:3}  action={d.action:6}  "
          f"p_harm={d.p_harm}  fired={d.fired_rules}")

print("-- low-risk PII case: does the RULE do the escalating, not the score? --")
for j in ["US", "EU", "IN"]:
    show("internal_copilot", j, resp_risk=0.05, flags={"pii_detected": True})

print("\n-- EU-scoped use case, uncertain grounding, no PII at all --")
for j in ["US", "EU", "IN"]:
    show("customer_facing", j, resp_risk=0.0, flags={})

def show2(use_case, jurisdiction):
    results = [DetectorResult("performance", 0.1, {"uncertain_grounding": True}, {}, "async", 1.0),
               DetectorResult("responsibility", 0.0, {}, {}, "inline", 0.1)]
    d = policy.decide("t1", use_case, results, jurisdiction=jurisdiction)
    print(f"{use_case:18} {jurisdiction:3}  action={d.action:6}  fired={d.fired_rules}")

print("\n-- EU-scoped use case, uncertain grounding, no PII at all --")
for j in ["US", "EU", "IN"]:
    show2("customer_facing", j)

def show3(use_case, jurisdiction):
    results = [DetectorResult("performance", 0.05, {}, {}, "async", 1.0),
               DetectorResult("responsibility", 0.1, {"toxicity_any": True, "toxicity_high": False}, {}, "inline", 0.1)]
    d = policy.decide("t1", use_case, results, jurisdiction=jurisdiction)
    print(f"{use_case:18} {jurisdiction:3}  action={d.action:6}  fired={d.fired_rules}")

print("\n-- borderline toxicity only, nothing else wrong --")
for j in ["US", "EU", "IN"]:
    show3("customer_facing", j)