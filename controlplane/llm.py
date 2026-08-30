"""LLM provider layer — the thing ControlPlane governs.

Two providers behind one interface so the service runs everywhere:
  * MockProvider  - deterministic, offline. Lets the API + tests run with no key.
                    Rigged to produce a grounded OR an ungrounded answer so the
                    control layer can be demonstrated without a real model.
  * GroqProvider  - real LLM via Groq's free API (needs GROQ_API_KEY + `pip
                    install groq`). This is what you demo live.

Select with env var CONTROLPLANE_LLM = mock | groq   (default: mock).

    provider.generate(prompt, context="", n=1) -> list[str]   # n samples
"""
from __future__ import annotations
import os, hashlib


class MockProvider:
    name = "mock"

    def generate(self, prompt: str, context: str = "", n: int = 1) -> list[str]:
        # deterministic pseudo-response; if the prompt asks for something not in
        # context, emit an unsupported claim so grounding has something to catch.
        h = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
        grounded = bool(context) and (h % 2 == 0)
        if grounded:
            base = f"Based on the provided context, {context.strip()[:160]}"
        else:
            base = ("Yes — absolutely. The service is free, available worldwide, "
                    "and there are no restrictions or fees of any kind.")
        outs = [base]
        for i in range(1, n):
            outs.append(base if grounded else base + f" (variant {i}: unlimited too.)")
        return outs


class GroqProvider:
    name = "groq"
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        from groq import Groq  # pip install groq
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.model = model

    def generate(self, prompt: str, context: str = "", n: int = 1) -> list[str]:
        sys_msg = "You are a helpful enterprise assistant."
        user = prompt if not context else f"Context:\n{context}\n\nQuestion: {prompt}"
        outs = []
        for _ in range(n):
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": sys_msg},
                          {"role": "user", "content": user}],
                temperature=0.7 if n > 1 else 0.2,
            )
            outs.append(r.choices[0].message.content)
        return outs


def get_provider():
    choice = os.environ.get("CONTROLPLANE_LLM", "mock").lower()
    if choice == "groq":
        try:
            return GroqProvider()
        except Exception as e:
            print(f"[llm] Groq unavailable ({e}); falling back to mock.")
    return MockProvider()
