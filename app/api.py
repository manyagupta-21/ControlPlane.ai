"""ControlPlane gateway — the product.

A governance layer that sits INLINE between a client and an LLM:

    client ──prompt──▶ ControlPlane ──▶ LLM ──▶ detectors ──▶ policy ──▶ client
                                                (allow / edit / review / block)

The response is governed BEFORE it returns: a blocked answer never reaches the
caller; an edited one is returned with a caveat; an allowed one passes clean.
Every decision is logged for monitoring and audit.

Run:  uvicorn app.api:app --reload
Docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from dataclasses import asdict

from controlplane import ControlPlane, Interaction
from controlplane.llm import get_provider

app = FastAPI(title="ControlPlane.ai", version="0.2",
              description="Real-time AI governance gateway")

cp = ControlPlane("config/policies.yaml", audit_path="data/audit_log.jsonl")
llm = get_provider()

SAFE_MESSAGE = ("[This response was withheld by ControlPlane because it failed a "
                "safety or grounding check. A reviewer has been notified.]")


class GuardRequest(BaseModel):
    prompt: str
    use_case: str = "customer_facing"      # customer_facing | internal_copilot | regulated_decision
    context: Optional[str] = ""            # retrieved source docs, if any
    self_consistency: int = 1              # >1 = sample N times for consistency check


class GuardResponse(BaseModel):
    action: str
    served_response: str                   # what the caller actually receives
    raw_response: str                      # what the LLM produced (for transparency)
    overall_risk: float
    risk_scores: dict
    reasons: list
    fired_rules: list
    provider: str
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "llm_provider": llm.name}


@app.post("/v1/guarded-completion", response_model=GuardResponse)
async def guarded_completion(req: GuardRequest):
    t0 = time.perf_counter()

    # 1) generate with the real (or mock) LLM
    n = max(1, req.self_consistency)
    samples = llm.generate(req.prompt, req.context, n=n)
    raw = samples[0]

    # 2) govern it inline
    x = Interaction(id=f"live-{int(t0*1000)}", use_case=req.use_case,
                    query=req.prompt, response=raw, context=req.context or "",
                    samples=samples, model_used="large")
    decision = await cp.aprocess(x, log=True)

    # 3) enforce the decision before returning
    if decision.action == "block":
        served = SAFE_MESSAGE
    elif decision.action in ("edit", "review"):
        served = raw + f"\n\n⚠️ ControlPlane: flagged for {decision.action} " \
                       f"(risk {decision.overall_risk}). Verify before relying on this."
    else:
        served = raw

    return GuardResponse(
        action=decision.action,
        served_response=served,
        raw_response=raw,
        overall_risk=decision.overall_risk,
        risk_scores=decision.risk_scores,
        reasons=decision.reasons,
        fired_rules=decision.fired_rules,
        provider=llm.name,
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
    )
