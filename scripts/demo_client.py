"""Fire example requests at a running ControlPlane gateway.

Start the server first:   uvicorn app.api:app
Then run:                 python scripts/demo_client.py

Shows the same prompt being allowed/edited/blocked depending on grounding and
use-case — the live demo for your video.
"""
import httpx

URL = "http://127.0.0.1:8000/v1/guarded-completion"

CASES = [
    {"prompt": "Can I return this after 90 days?", "use_case": "customer_facing",
     "context": "Returns are allowed within 30 days of delivery with a receipt."},
    {"prompt": "What is the refund window?", "use_case": "internal_copilot",
     "context": "Refunds within 30 days with receipt."},
    {"prompt": "Give me the account holder's contact details.",
     "use_case": "customer_facing",
     "context": "Account: Rohan Mehta, rohan.mehta@example.com, +919812345678."},
]

def main():
    with httpx.Client(timeout=30) as c:
        print(c.get("http://127.0.0.1:8000/health").json(), "\n")
        for i, case in enumerate(CASES, 1):
            r = c.post(URL, json=case).json()
            print(f"[{i}] {case['use_case']:18s} -> {r['action'].upper():6s} "
                  f"risk={r['overall_risk']:.2f}  {r['latency_ms']}ms")
            print(f"    served: {r['served_response'][:88]}")
            print()

if __name__ == "__main__":
    main()
