"""Build a clean grounding-evaluation set from the raw RAGTruth corpus.

RAGTruth ships two files:
  source_info.jsonl : {source_id, task_type, source_info:{question, passages}, ...}
  response.jsonl    : {source_id, response, labels:[...spans...], split, ...}

We take the QA subset and join them into records ControlPlane can check:
  {id, query, context, response, label_hallucination, split}

label_hallucination = True if the response has ANY human-annotated hallucination
span (the standard response-level binarisation used in the literature).

Run:  python data/ragtruth/build_ragtruth.py
Out:  data/ragtruth/ragtruth_qa.jsonl
"""
import json, os
from collections import Counter

HERE = os.path.dirname(__file__)

def main():
    src = {}
    with open(os.path.join(HERE, "source_info.jsonl"), encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            src[s["source_id"]] = s

    rows = []
    with open(os.path.join(HERE, "response.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            s = src.get(r["source_id"])
            if not s or s["task_type"] != "QA":
                continue
            info = s["source_info"]                      # {question, passages}
            rows.append({
                "id": f"rt{r['id']}",
                "query": info.get("question", ""),
                "context": info.get("passages", ""),      # the retrieved documents
                "response": r["response"],
                "label_hallucination": len(r["labels"]) > 0,
                "split": r["split"],
                "model": r["model"],
            })

    out = os.path.join(HERE, "ragtruth_qa.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} QA records -> {out}")
    print("split:", dict(Counter(r["split"] for r in rows)))
    print("hallucinated:", dict(Counter(r["label_hallucination"] for r in rows)))
    tst = [r for r in rows if r["split"] == "test"]
    print(f"test split: {len(tst)} responses, "
          f"{sum(r['label_hallucination'] for r in tst)} hallucinated "
          f"({sum(r['label_hallucination'] for r in tst)/len(tst):.1%})")

if __name__ == "__main__":
    main()
