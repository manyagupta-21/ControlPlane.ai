"""Download the RAGTruth corpus from its official GitHub repo (MIT License).

We fetch rather than redistribute, so the repo stays lean and reproducible.

Run:  python data/ragtruth/fetch_ragtruth.py
Then: python data/ragtruth/build_ragtruth.py
"""
import os, urllib.request

BASE = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset"
FILES = ["source_info.jsonl", "response.jsonl"]
HERE = os.path.dirname(__file__)

def main():
    for name in FILES:
        dst = os.path.join(HERE, name)
        if os.path.exists(dst):
            print(f"already have {name}")
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(f"{BASE}/{name}", dst)
        print(f"  saved -> {dst}  ({os.path.getsize(dst)//1_000_000} MB)")
    print("Done. Next: python data/ragtruth/build_ragtruth.py")

if __name__ == "__main__":
    main()
