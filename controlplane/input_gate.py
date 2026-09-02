"""Input-side gate: checks the PROMPT before it ever reaches the LLM.

Everything else in this system is output-only -- it governs what the model
said. This is the one place that looks at what the USER asked, before a
single token is generated. Two things justify blocking here rather than
waiting for the response:
  1. cost -- a blocked prompt never triggers a (paid) generation call at all.
  2. some obligations are about what was ASKED, not just what was answered --
     PII in the prompt has already left the user's hands the moment it's
     sent, regardless of what the model does with it.

Two things this gate does NOT do, on purpose, unlike an earlier version:
  1. It is not one-size-fits-all. A single hardcoded rule ("block if PII or
     high toxicity") is exactly the failure mode the brief opens with --
     "a single, one-size-fits-all checking approach rarely works well
     everywhere." A customer_facing prompt and an internal_copilot prompt
     carrying the same PII should not be governed identically, so this gate
     hands its flags to PolicyEngine.decide_gate(), which runs them through
     the SAME use_case -> jurisdiction -> sector hard_rule composition the
     output side already uses. Same appetite, same config file, no second
     policy mechanism to keep in sync.
  2. It is not silent. Every gate decision -- allow, review, or block -- is
     returned as a real Decision and logged through the same audit trail as
     post-generation decisions (tagged stage="input_gate"), because a prompt
     blocked before generation is still a decision, and "a clear audit trail
     behind every decision" doesn't have an exception for the cheap ones.

Reuses ResponsibilityDetector's PII/toxicity scan -- the same detector the
output side already uses -- so there is one definition of PII/toxicity in
this system, not two.
"""
from __future__ import annotations
import time
from .detectors import ResponsibilityDetector
from .schemas import Interaction, Decision
from .policy import PolicyEngine


class InputGate:
    def __init__(self, policy: PolicyEngine):
        self._detector = ResponsibilityDetector()
        self._policy = policy

    def check(self, prompt: str, interaction_id: str,
              use_case: str = "customer_facing", jurisdiction: str = "US",
              sector: str = "general") -> Decision:
        """Scan the raw prompt and decide allow/review/block through the same
        three-axis policy the output side uses. Returns a full Decision,
        stage="input_gate", ready to be logged and reasoned about identically
        to a post-generation one."""
        t0 = time.perf_counter()
        # ResponsibilityDetector.run() reads x.response -- it doesn't care
        # where the text came from, only what's in it, so we pass the prompt
        # in as if it were the "response" being scanned.
        stub = Interaction(id=interaction_id, use_case=use_case, query="",
                           response=prompt, jurisdiction=jurisdiction, sector=sector)
        result = self._detector.run(stub)
        latency_ms = (time.perf_counter() - t0) * 1000

        return self._policy.decide_gate(
            interaction_id, use_case, result.flags, result.detail, result.risk,
            jurisdiction=jurisdiction, sector=sector, total_latency_ms=latency_ms)