import json
from pathlib import Path

from robosandbox.eval_log.migrate import migrate_legacy_runs
from robosandbox.eval_log.store import EvalLogStore


def _make_legacy_run(root: Path, name: str, success_rate: float):
    d = root / name
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({
        "task": "task1", "policy": "out/legacy_p1",
        "sim_backend": "newton",
        "n_trials": 4, "successes": int(round(success_rate * 4)),
        "rate": success_rate,
    }))


def test_migrate_creates_log_rows(tmp_path):
    runs = tmp_path / "runs"
    _make_legacy_run(runs, "20260301-001", 0.5)
    _make_legacy_run(runs, "20260301-002", 1.0)
    log = tmp_path / "runs" / "eval_log"
    n = migrate_legacy_runs(runs, log)
    assert n == 2
    assert (log / "eval_runs.jsonl").exists()
    assert (log / "policies.jsonl").exists()


def test_migrate_idempotent(tmp_path):
    runs = tmp_path / "runs"
    _make_legacy_run(runs, "20260301-001", 0.5)
    log = tmp_path / "runs" / "eval_log"
    migrate_legacy_runs(runs, log)
    n_again = migrate_legacy_runs(runs, log)
    # second call sees the eval_id already there → 0 imports
    assert n_again == 0
