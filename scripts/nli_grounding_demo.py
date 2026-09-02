"""Minimal standalone demo of the NLI cross-encoder grounding model.

CORRECTION: the previous version of this script named a model repo,
`microsoft/deberta-v3-base-mnli`, that does not exist on Hugging Face — I had
not verified it before writing the script, and your 401/RepositoryNotFound
error was the correct outcome for a bad repo id. I'm sorry for that; it should
not have shipped without a check.

The real, verified checkpoint is `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
(fetched and cross-checked against its live model card at
https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli just now —
144k downloads/month, MIT licensed, DeBERTa-v3-base fine-tuned on MultiNLI +
Fever-NLI + ANLI, which is the standard, well-documented choice for this task).
The tokenizer/model loading calls and the softmax-over-logits pattern below are
copied from that model card's own "NLI use-case" code sample, not written from
memory.

ONE THING I COULD NOT DO: my own sandbox cannot reach huggingface.co (it isn't
on this environment's allowed network list — I confirmed this by trying, not
by assuming), so I could not execute this script myself before sending it, the
way I ran every other file in this conversation. Please treat this run as the
first real test of it, and paste me the full output including any traceback.

Run:
    python scripts/nli_grounding_demo.py
"""
import sys
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
LABELS = ["entailment", "neutral", "contradiction"]   # this model's label order,
                                                       # per its own model card
CLAIM = "The Eiffel Tower is located in Paris."
CONTEXT = ("The Eiffel Tower is a wrought-iron tower in Paris, France. It was "
          "constructed in 1889 and is named after its engineer, Gustave Eiffel.")


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"device: {device}")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device).eval()
    print(f"model loaded in {time.time()-t0:.2f} sec "
          f"(first run downloads ~370MB; cached after that)")

    t0 = time.time()
    with torch.no_grad():
        # premise = context (the evidence), hypothesis = claim (what we're
        # checking) — this order matches the model card's NLI example exactly.
        inputs = tokenizer(CONTEXT, CLAIM, truncation=True, return_tensors="pt").to(device)
        logits = model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1).tolist()
    latency_ms = (time.time() - t0) * 1000

    scored = {name: round(p, 4) for name, p in zip(LABELS, probs)}
    print(f"\nclaim:   {CLAIM}")
    print(f"context: {CONTEXT}")
    print(f"probabilities: {scored}")
    print(f"entailment probability (the grounding score): {scored['entailment']}")
    print(f"latency: {int(latency_ms)} ms")

    if device.type == "cpu":
        print("\nnote: timed on CPU, which is the worst case and the honest number "
              "to design the latency budget around.")


if __name__ == "__main__":
    main()
