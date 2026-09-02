import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from controlplane.policy import PolicyEngine
from controlplane.schemas import DetectorResult

policy = PolicyEngine("config/policies.yaml")

def show(use_case, jurisdiction, sector, resp_flags=None, perf_flags=None):
    results = [DetectorResult("performance", 0.1, perf_flags or {}, {}, "async", 1.0),
               DetectorResult("responsibility", 0.05, resp_flags or {}, {}, "inline", 0.1)]
    d = policy.decide("t1", use_case, results, jurisdiction=jurisdiction, sector=sector)
    print(f"{use_case:18} {jurisdiction:3} {sector:11} action={d.action:6} fired={d.fired_rules}")

print("-- sector alone escalates: ungrounded claim, general vs healthcare --")
for s in ["general", "healthcare", "finance"]:
    show("customer_facing", "US", s, perf_flags={"ungrounded": True})