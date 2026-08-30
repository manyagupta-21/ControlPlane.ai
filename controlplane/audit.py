"""Append-only audit trail.

Every decision is written as one JSON line: fully reconstructable, easy to load
into pandas for monitoring, and the substrate for the feedback loop (human
overrides are appended as their own records).
"""
from __future__ import annotations
import json, os, time
from dataclasses import asdict
from .schemas import Decision


class AuditLog:
    def __init__(self, path: str = "data/audit_log.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def record(self, decision: Decision) -> None:
        row = asdict(decision)
        row["ts"] = time.time()
        row["type"] = "decision"
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def record_override(self, interaction_id: str, from_action: str,
                        to_action: str, reviewer: str, note: str = "") -> None:
        """Capture a human override -> feeds the learning loop."""
        row = {"type": "override", "ts": time.time(),
               "interaction_id": interaction_id, "from_action": from_action,
               "to_action": to_action, "reviewer": reviewer, "note": note}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def reset(self) -> None:
        open(self.path, "w").close()
