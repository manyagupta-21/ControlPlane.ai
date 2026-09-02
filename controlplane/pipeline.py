"""Orchestrator: interaction -> detectors (in parallel) -> policy -> audit.

Demonstrates the two-speed architecture: detectors are run concurrently, and
each detector is tagged inline vs async. The 'inline' latency (what the user
waits for) is measured separately from total wall-clock, so we can show that
heavy checks run in parallel and don't sit on the critical path.
"""
from __future__ import annotations
import asyncio, time
from .schemas import Interaction, Decision
from .detectors import default_detectors
from .policy import PolicyEngine
from .audit import AuditLog
from .input_gate import InputGate


class ControlPlane:
    def __init__(self, policy_path: str, audit_path: str = "data/audit_log.jsonl",
                 detectors=None):
        self.detectors = detectors or default_detectors()
        self.policy = PolicyEngine(policy_path)
        self.audit = AuditLog(audit_path)
        self.input_gate = InputGate()

    def check_input(self, prompt: str) -> dict:
        """Pre-generation gate. Call this BEFORE the LLM call; if it returns
        action='block', skip generation entirely."""
        return self.input_gate.check(prompt)

    async def _run_all(self, x: Interaction):
        # each detector runs in a thread so they truly overlap
        tasks = [asyncio.to_thread(d.run, x) for d in self.detectors]
        return await asyncio.gather(*tasks)

    def process(self, x: Interaction, log: bool = True) -> Decision:
        t0 = time.perf_counter()
        results = asyncio.run(self._run_all(x))
        return self._finish(x, results, t0, log)

    async def aprocess(self, x: Interaction, log: bool = True) -> Decision:
        """Async variant for use inside a running event loop (the API gateway)."""
        t0 = time.perf_counter()
        results = await self._run_all(x)
        return self._finish(x, results, t0, log)

    def _finish(self, x, results, t0, log) -> Decision:
        wall_ms = (time.perf_counter() - t0) * 1000
        # inline latency = the slowest INLINE detector (what the user waits on)
        inline = [r.latency_ms for r in results if r.speed == "inline"]
        inline_ms = max(inline) if inline else 0.0
        decision = self.policy.decide(x.id, x.use_case, results,
                                      jurisdiction=x.jurisdiction, sector=x.sector,
                                      total_latency_ms=wall_ms)
        decision.reasons.append(
            f"inline_latency_ms={round(inline_ms,2)} (user-facing), "
            f"wall_ms={round(wall_ms,2)} (all checks, parallel)")
        if log:
            self.audit.record(decision)
        return decision
