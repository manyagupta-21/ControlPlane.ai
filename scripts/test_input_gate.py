import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from controlplane import ControlPlane

cp = ControlPlane("config/policies.yaml", audit_path="data/audit_log.jsonl")

tests = [
    "What's the weather like in Kanpur today?",
    "My email is manya.test@example.com, can you draft a reply for me?",
    "You're such an idiot, this system is useless.",
]

for t in tests:
    gate = cp.check_input(t)
    print(f"prompt: {t!r}\n  -> action={gate['action']}  flags={gate['flags']}\n")