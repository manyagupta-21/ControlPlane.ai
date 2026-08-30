# RAGTruth grounding benchmark

Real human-annotated RAG hallucination corpus (Niu et al., ACL 2024; MIT License),
used to validate ControlPlane's grounding detector on the QA subset.

```
python data/ragtruth/fetch_ragtruth.py     # download from source (~49 MB)
python data/ragtruth/build_ragtruth.py      # -> ragtruth_qa.jsonl (5,934 records)
python scripts/evaluate_ragtruth.py         # AUROC on the 900-record test split
python scripts/evaluate_ragtruth.py embedding   # semantic backend (local, higher AUROC)
```

Source: https://github.com/ParticleMedia/RAGTruth
