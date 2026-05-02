"""One-shot import of legacy runs/<timestamp>/result.json into eval_log."""
from __future__ import annotations

import json
from pathlib import Path

from robosandbox.eval_log.schema import EvalRow, EvalRunRow, PolicyRow
from robosandbox.eval_log.store import EVAL_RUNS, EvalLogStore


def _existing_eval_ids(log_root: Path) -> set[str]:
    p = log_root / EVAL_RUNS
    if not p.exists():
        return set()
    out = set()
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.add(json.loads(line)["eval_id"])
        except Exception:
            continue
    return out


def migrate_legacy_runs(runs_dir: str | Path, log_root: str | Path) -> int:
    """Walk ``runs_dir/*/result.json``; for each legacy result that isn't in
    the log, append synthetic policy + eval_run + per-trial rows. Returns
    the number of newly-imported runs.
    """
    runs_dir = Path(runs_dir)
    log_root = Path(log_root)
    store = EvalLogStore(log_root)
    seen = _existing_eval_ids(log_root)
    imported = 0
    if not runs_dir.exists():
        return 0
    for sub in sorted(runs_dir.iterdir()):
        if not sub.is_dir():
            continue
        rp = sub / "result.json"
        if not rp.exists():
            continue
        try:
            res = json.loads(rp.read_text())
        except Exception:
            continue
        eval_id = f"legacy_{sub.name}"
        if eval_id in seen:
            continue
        policy_path = res.get("policy", str(sub))
        pid = Path(policy_path).resolve().name or sub.name
        store.append_policy(PolicyRow(
            policy_id=pid, kind="replay_trajectory",
            parent_policy_id=None, lineage_op=None,
            training_demo_set_id=None, training_steps=None,
            training_dist_summary=None,
            trained_at=res.get("created_at", "1970-01-01T00:00:00Z"),
            checkpoint_path=str(Path(policy_path).resolve()),
            metadata={"migrated_from": str(rp)},
        ))
        n_trials = int(res.get("n_trials", 1))
        successes = int(res.get(
            "successes",
            round(float(res.get("rate", 0.0)) * n_trials),
        ))
        store.append_eval_run(EvalRunRow(
            eval_id=eval_id, policy_id=pid,
            task_id=str(res.get("task", "")),
            n_trials=n_trials,
            started_at=res.get("started_at", res.get("created_at", "1970-01-01T00:00:00Z")),
            completed_at=res.get("completed_at", res.get("created_at", "1970-01-01T00:00:00Z")),
            git_sha=res.get("git_sha", "unknown"),
            config_hash="", compute_resource="",
            command_line=res.get("command", ""),
        ))
        for i in range(n_trials):
            store.append_eval(EvalRow(
                eval_id=eval_id, trial_id=f"{i:05d}", policy_id=pid,
                task_id=str(res.get("task", "")),
                task_variant_hash="",
                sim_backend=str(res.get("sim_backend", "unknown")),
                trial_seed=i,
                slice_axes={},
                outcome={"success": i < successes},
                wall_seconds=0.0,
                compute_resource="",
                started_at=res.get("started_at", "1970-01-01T00:00:00Z"),
            ))
        imported += 1
    return imported
