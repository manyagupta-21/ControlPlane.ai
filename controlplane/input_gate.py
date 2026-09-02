"""Input-side gate: checks the PROMPT before it ever reaches the LLM.

Everything else in this system is output-only -- it governs what the model
said. This is the one place that looks at what the USER asked, before a
single token is generated. Two things justify blocking here rather than
waiting for the response:
  1. cost -- a blocked prompt never triggers a (paid) generation call at all.
  2. some obligations are about what was ASKED, not just what was answered --
     PII in the prompt has already left the user's hands the moment it's
     sent, regardless of what the model does with it.

Reuses ResponsibilityDetector's PII/toxicity scan -- the same detector the
output side already uses -- so there is one definition of PII/toxicity in
this system, not two.
"""
from __future__ import annotations
from .detectors import ResponsibilityDetector
from .schemas import Interaction


class InputGate:
    def __init__(self):
        self._detector = ResponsibilityDetector()

    def check(self, prompt: str) -> dict:
        """Scan the raw prompt. Returns action ('allow' | 'block') plus the
        same-shaped detail the output side produces, so it can be reasoned
        about and logged identically."""
        # ResponsibilityDetector.run() reads x.response -- it doesn't care
        # where the text came from, only what's in it, so we pass the prompt
        # in as if it were the "response" being scanned.
        stub = Interaction(id="pre-gate", use_case="_gate", query="", response=prompt)
        result = self._detector.run(stub)
        action = "block" if (result.flags.get("pii_detected") or
                             result.flags.get("toxicity_high")) else "allow"
        return {"action": action, "risk": result.risk,
                "flags": result.flags, "detail": result.detail}