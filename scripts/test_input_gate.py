import os, sys, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from controlplane import ControlPlane

LOG = "data/test_input_gate_audit.jsonl"
if os.path.exists(LOG):
    os.remove(LOG)
cp = ControlPlane("config/policies.yaml", audit_path=LOG)

print("-- same PII prompt, different use cases: is the gate use-case-aware? --")
prompt_pii = "My email is manya.test@example.com, can you draft a reply for me?"
for uc in ["customer_facing", "internal_copilot", "regulated_decision"]:
    d = cp.check_input(prompt_pii, use_case=uc)
    print(f"{uc:20s} action={d.action:8s} fired={d.fired_rules}")

print("\n-- same PII prompt, India (DPDP) vs US, for a use case that only reviews by default --")
for j in ["US", "IN"]:
    d = cp.check_input(prompt_pii, use_case="internal_copilot", jurisdiction=j)
    print(f"internal_copilot / {j}   action={d.action:8s} fired={d.fired_rules}")

print("\n-- clean prompt: allowed, still logged --")
d = cp.check_input("What's the weather like in Kanpur today?", use_case="customer_facing")
print(f"customer_facing      action={d.action:8s} fired={d.fired_rules}")

print("\n-- toxic prompt --")
d = cp.check_input("You're such an idiot, this system is useless.", use_case="customer_facing")
print(f"customer_facing      action={d.action:8s} fired={d.fired_rules}")

print(f"\n-- audit trail check: every gate decision above should be in {LOG} --")
with open(LOG, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]
gate_rows = [r for r in rows if r.get("stage") == "input_gate"]
print(f"{len(gate_rows)} input_gate rows logged (expected 7 -- one per check_input call above)")
for r in gate_rows:
    print(f"  id={r['interaction_id']:<30s} use_case={r['use_case']:<18s} action={r['action']:8s} fired={r['fired_rules']}")