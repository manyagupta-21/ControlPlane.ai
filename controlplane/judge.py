"""AI-as-judge grounding backend.

Provides a structured LLM verdict on grounding, run as a second opinion
alongside the primary TF-IDF faithfulness scorer.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import time

from .grounding import TfidfClaimBackend

_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="judge")

_SYSTEM = (
    "You are a factual grounding auditor for an enterprise AI control layer. "
    "Decide whether each claim in the AI response is supported by the provided context. "
    "SUPPORTED = follows directly from the context text. "
    "UNSUPPORTED = not mentioned in the context (even if generally true). "
    "CONTRADICTED = directly conflicts with something the context states. "
    "Output ONLY a JSON object with exactly these keys: "
    "verdict (supported|unsupported|contradicted), "
    "unsupported_claims (list of strings), "
    "contradicted_claims (list of strings), "
    "reasoning (one sentence), "
    "confidence (float 0-1). "
    "No markdown. No explanation outside the JSON."
)

_USER_TMPL = (
    "CONTEXT:\n{context}\n\n"
    "AI RESPONSE TO AUDIT:\n{response}\n\n"
    "JSON verdict:"
)

_VERDICT_RISK = {
    "supported": 0.10,
    "unsupported": 0.65,
    "contradicted": 0.90,
}


def _extract_json(text: str) -> str:
    """Extract JSON object from any model output format robustly.

    Handles: <think>...</think> blocks, markdown fences, preamble text,
    trailing text, and nested braces inside string values (brace-depth
    tracking rather than a naive first-{/last-} match).
    """
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if start is None:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:i + 1]
    return text


class LLMJudgeBackend(TfidfClaimBackend):
    """AI-as-judge grounding backend. Falls back to TF-IDF on any failure."""

    name = "judge"

    def __init__(self,
                 model: str = "openai/gpt-oss-20b",
                 timeout_s: float = 6.0,
                 max_context_chars: int = 2000,
                 max_response_chars: int = 1500):
        self.model = model
        self.timeout_s = timeout_s
        self.max_context_chars = max_context_chars
        self.max_response_chars = max_response_chars
        self._client = None

    def _get_client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=os.environ["GROQ_API_KEY"])
        return self._client

    def _call_judge(self, response: str, context: str) -> dict:
        """Returns {"raw": str, "latency_ms": float}.

        Tries Groq's JSON mode first (response_format=json_object), which
        produces clean output most of the time. Groq's own generation
        validator can reject a request with a 400 json_validate_failed error
        even for straightforward inputs — an observed failure mode of this
        model, not a bug in our parsing. If the JSON-mode call fails for any
        reason, this retries once WITHOUT the constraint and relies on
        _extract_json() to pull the JSON object out of free-form text, so a
        single flaky JSON-mode call doesn't take down the judge.
        """
        client = self._get_client()
        user = _USER_TMPL.format(
            context=context[:self.max_context_chars],
            response=response[:self.max_response_chars],
        )
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ]

        t0 = time.perf_counter()
        try:
            completion = client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.0,
                max_tokens=1024,                       # room for reasoning + JSON
                reasoning_effort="low",                # gpt-oss supports this; minimises
                                                        # reasoning tokens so more of the
                                                        # budget reaches the actual answer
                response_format={"type": "json_object"},
            )
        except Exception:
            # JSON mode rejected the generation, or errored for any other
            # reason -- retry unconstrained. _extract_json() handles think
            # blocks, fences, and preamble text, so this is a different
            # source of raw text, not a weaker parse path.
            completion = client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.0,
                max_tokens=1024, reasoning_effort="low",
            )
        latency_ms = (time.perf_counter() - t0) * 1000
        choice = completion.choices[0]
        raw = choice.message.content or ""
        finish_reason = getattr(choice, "finish_reason", None)
        # Known Groq/gpt-oss failure mode: the model spends its ENTIRE token
        # budget on hidden reasoning and never writes the answer. The API call
        # succeeds and returns content="" with finish_reason="length" -- not
        # an exception, so it must be detected here rather than caught above.
        # See: https://community.groq.com and the Groq reasoning docs.
        reasoning_exhausted = (not raw.strip()) and finish_reason == "length"
        return {"raw": raw, "latency_ms": round(latency_ms, 1),
                "finish_reason": finish_reason,
                "reasoning_exhausted": reasoning_exhausted}

    def score(self, response: str, context: str):
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            return None, {"judge": {
                "verdict": "api_key_missing",
                "reasoning": "GROQ_API_KEY not set — set it to enable the judge",
                "unsupported_claims": [], "contradicted_claims": [],
                "confidence": 0.0}}

        future = _EXECUTOR.submit(self._call_judge, response, context)
        try:
            result = future.result(timeout=self.timeout_s)
        except concurrent.futures.TimeoutError:
            return None, {"judge": {
                "verdict": "timeout",
                "timeout_s": self.timeout_s,
                "reasoning": f"judge did not respond within {self.timeout_s}s",
                "unsupported_claims": [], "contradicted_claims": [],
                "confidence": 0.0}}
        except Exception as e:
            err = str(e)
            if "groq" in err.lower() and "module" in err.lower():
                err = "groq package not installed — run: pip install groq"
            return None, {"judge": {
                "verdict": "unavailable",
                "error": err[:200],
                "reasoning": err[:200],
                "unsupported_claims": [], "contradicted_claims": [],
                "confidence": 0.0}}

        raw = result["raw"]
        latency_ms = result["latency_ms"]

        if result.get("reasoning_exhausted"):
            # One retry with a much larger budget before giving up. Cheap
            # insurance: this case is rare (observed on ~1 in 3 calls in
            # testing) and the retry costs at most a few hundred extra tokens.
            try:
                retry_future = _EXECUTOR.submit(
                    self._call_judge_raw_only, response, context, max_tokens=2048)
                retry = retry_future.result(timeout=self.timeout_s)
                if retry["raw"].strip():
                    raw, latency_ms = retry["raw"], latency_ms + retry["latency_ms"]
                else:
                    return None, {"judge": {
                        "verdict": "reasoning_exhausted",
                        "reasoning": ("model spent its full token budget on internal "
                                     "reasoning and never wrote an answer, even after "
                                     "a retry at a larger budget — a known gpt-oss "
                                     "failure mode, not an application error"),
                        "unsupported_claims": [], "contradicted_claims": [],
                        "confidence": 0.0}}
            except Exception:
                return None, {"judge": {
                    "verdict": "reasoning_exhausted",
                    "reasoning": "model exhausted its token budget on reasoning; retry failed",
                    "unsupported_claims": [], "contradicted_claims": [],
                    "confidence": 0.0}}

        try:
            clean = _extract_json(raw)
            parsed = json.loads(clean)
            verdict = parsed.get("verdict", "unsupported")
            risk = _VERDICT_RISK.get(verdict, 0.65)
            parsed["judge_latency_ms"] = latency_ms
            parsed["model"] = self.model
            return risk, {"judge": parsed}
        except json.JSONDecodeError as e:
            return None, {"judge": {
                "verdict": "parse_error",
                "error": str(e)[:120],
                "reasoning": "judge returned non-JSON output",
                "raw_preview": raw[:400] if raw else "(empty)",
                "unsupported_claims": [], "contradicted_claims": [],
                "confidence": 0.0}}

    def _call_judge_raw_only(self, response: str, context: str, max_tokens: int) -> dict:
        """Same as _call_judge but with a caller-specified token budget and no
        JSON-mode attempt (used only for the reasoning-exhaustion retry)."""
        client = self._get_client()
        user = _USER_TMPL.format(
            context=context[:self.max_context_chars],
            response=response[:self.max_response_chars],
        )
        t0 = time.perf_counter()
        completion = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": _SYSTEM},
                     {"role": "user", "content": user}],
            temperature=0.0, max_tokens=max_tokens, reasoning_effort="low",
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        raw = completion.choices[0].message.content or ""
        return {"raw": raw, "latency_ms": round(latency_ms, 1)}

    def warm_up(self):
        pass


class TfidfPlusJudgeBackend:
    """TF-IDF primary + LLM judge second opinion.

    Select with: CONTROLPLANE_GROUNDING=tfidf+judge
    """
    name = "tfidf+judge"

    def __init__(self, **judge_kwargs):
        self._tfidf = TfidfClaimBackend()
        self._judge = LLMJudgeBackend(**judge_kwargs)

    def warm_up(self):
        self._judge.warm_up()

    def score(self, response: str, context: str):
        tfidf_risk, tfidf_detail = self._tfidf.score(response, context)
        judge_risk, judge_detail = self._judge.score(response, context)
        detail = {**tfidf_detail, **judge_detail, "backend": "tfidf+judge",
                  "tfidf_risk": round(float(tfidf_risk), 4)}
        if judge_risk is not None:
            detail["judge_risk"] = round(float(judge_risk), 4)
            gap = abs(float(tfidf_risk) - float(judge_risk))
            detail["judge_disagrees"] = bool(gap > 0.35)
            detail["judge_tfidf_gap"] = round(gap, 3)
        return tfidf_risk, detail
