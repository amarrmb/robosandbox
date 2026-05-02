import json
from pathlib import Path
from robosandbox.eval_log.store import EvalLogStore
from robosandbox.eval_log.schema import PolicyRow, EvalRow, DemoRow, EvalRunRow


def _policy(pid="p1"):
    return PolicyRow(pid, "ppo_neural", None, None, None, 0, None, "2026-05-01T00:00:00Z", "out/p1")


def _eval(eid="e1", tid="00000"):
    return EvalRow(eid, tid, "p1", "task1", "h", "newton", 0,
                   {}, {"success": True}, 0.1, "cpu", "2026-05-01T00:00:00Z")


def test_append_round_trip(tmp_path):
    store = EvalLogStore(tmp_path)
    store.append_policy(_policy())
    store.append_eval(_eval())
    rows = list(_jsonl_rows(tmp_path / "policies.jsonl"))
    assert len(rows) == 1
    assert rows[0]["policy_id"] == "p1"
    rows = list(_jsonl_rows(tmp_path / "evals.jsonl"))
    assert len(rows) == 1


def test_append_policy_idempotent(tmp_path):
    """Appending the same policy_id twice writes one row."""
    store = EvalLogStore(tmp_path)
    store.append_policy(_policy("dup"))
    store.append_policy(_policy("dup"))
    rows = list(_jsonl_rows(tmp_path / "policies.jsonl"))
    assert len(rows) == 1


def test_files_created_on_first_append(tmp_path):
    store = EvalLogStore(tmp_path)
    assert not (tmp_path / "evals.jsonl").exists()
    store.append_eval(_eval())
    assert (tmp_path / "evals.jsonl").exists()


def test_concurrent_appends_dont_corrupt(tmp_path):
    """Two stores writing to the same path produce a valid JSONL file."""
    s1 = EvalLogStore(tmp_path)
    s2 = EvalLogStore(tmp_path)
    for i in range(50):
        s1.append_eval(_eval(eid="e_a", tid=f"{i:05d}"))
        s2.append_eval(_eval(eid="e_b", tid=f"{i:05d}"))
    rows = list(_jsonl_rows(tmp_path / "evals.jsonl"))
    assert len(rows) == 100
    for r in rows:
        assert "trial_id" in r


def _jsonl_rows(path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if line.strip():
            yield json.loads(line)
