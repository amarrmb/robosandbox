"""Append-only JSONL store for eval-log rows. One file per entity type."""
from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock

from robosandbox.eval_log.schema import (
    DemoRow, EvalRow, EvalRunRow, PolicyRow,
)

POLICIES = "policies.jsonl"
DEMOS = "demos.jsonl"
EVALS = "evals.jsonl"
EVAL_RUNS = "eval_runs.jsonl"


class EvalLogStore:
    """File-based append-only store. Creates files lazily on first append.

    Idempotency: ``append_policy`` is no-op if ``policy_id`` already in file.
    Other appends are unconditionally additive (RL emits 1024 trial rows per
    eval; trial_id makes them distinct).

    Concurrency: appends use OS-level open(..., "a") + flushed writes;
    POSIX guarantees atomic appends for writes <= PIPE_BUF (4096 bytes).
    Our rows are ~600 bytes; safe.
    """

    def __init__(self, root: str | Path):
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()  # in-process serialization for idempotency reads

    def append_policy(self, row: PolicyRow) -> None:
        with self._lock:
            if self._policy_exists(row.policy_id):
                return
            self._append(POLICIES, row.to_dict())

    def append_demo(self, row: DemoRow) -> None:
        self._append(DEMOS, row.to_dict())

    def append_eval(self, row: EvalRow) -> None:
        self._append(EVALS, row.to_dict())

    def append_eval_run(self, row: EvalRunRow) -> None:
        self._append(EVAL_RUNS, row.to_dict())

    def _append(self, fname: str, obj: dict) -> None:
        path = self._root / fname
        line = json.dumps(obj, separators=(",", ":")) + "\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def _policy_exists(self, policy_id: str) -> bool:
        path = self._root / POLICIES
        if not path.exists():
            return False
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    if json.loads(line).get("policy_id") == policy_id:
                        return True
                except json.JSONDecodeError:
                    continue
        return False
