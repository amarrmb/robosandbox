import pytest
from robosandbox.eval_log.query import EvalLogQuery
from robosandbox.eval_log.schema import EvalRow, EvalRunRow, PolicyRow
from robosandbox.eval_log.store import EvalLogStore


def _seed(tmp_path):
    store = EvalLogStore(tmp_path)
    store.append_policy(PolicyRow(
        "p_a", "ppo_neural", None, None, None, 0, None,
        "2026-05-01T00:00:00Z", "out/p_a",
    ))
    store.append_policy(PolicyRow(
        "p_b", "ppo_neural", "p_a", "warm_start_ppo", None, 0, None,
        "2026-05-01T01:00:00Z", "out/p_b",
    ))
    # 10 trials per policy on the same task: p_a 6/10 success, p_b 8/10 success
    for pid, n_succ in [("p_a", 6), ("p_b", 8)]:
        store.append_eval_run(EvalRunRow(
            f"eval_{pid}", pid, "task1", 10,
            "2026-05-01T02:00:00Z", "2026-05-01T02:00:30Z",
            "abc123", "cfg_h", "cpu",
            f"robo-sandbox eval --policy {pid}",
        ))
        for i in range(10):
            store.append_eval(EvalRow(
                f"eval_{pid}", f"{i:05d}", pid, "task1", "th", "newton", i,
                {"target_xy_bucket": "left" if i < 5 else "right"},
                {"success": i < n_succ, "ever_within_threshold": i < n_succ,
                 "end_within_threshold": i < n_succ, "min_dist_m": 0.01,
                 "final_dist_m": 0.05, "steps_used": 100, "step_budget": 100},
                0.1, "cpu", "2026-05-01T02:00:00Z",
            ))
    return tmp_path


def test_list_policies(tmp_path):
    _seed(tmp_path)
    q = EvalLogQuery(tmp_path)
    rows = q.list_policies()
    pids = {r["policy_id"] for r in rows}
    assert pids == {"p_a", "p_b"}


def test_slice_breakdown(tmp_path):
    _seed(tmp_path)
    q = EvalLogQuery(tmp_path)
    rows = q.slice_breakdown("p_b", axes=["target_xy_bucket"])
    rates = {r["target_xy_bucket"]: r["success_rate"] for r in rows}
    # p_b: 8/10 success — 5 left (all success), 3 of 5 right success
    assert rates["left"] == 1.0
    assert rates["right"] == pytest.approx(0.6)


def test_compare_two_policies(tmp_path):
    _seed(tmp_path)
    q = EvalLogQuery(tmp_path)
    cmp = q.compare_policies("p_a", "p_b", task_id="task1")
    assert cmp["a_success_rate"] == 0.6
    assert cmp["b_success_rate"] == 0.8
    assert cmp["delta_pp"] == pytest.approx(20.0)


def test_friendly_empty_message(tmp_path):
    """Querying an empty / nonexistent eval log doesn't crash."""
    q = EvalLogQuery(tmp_path)
    rows = q.list_policies()
    assert rows == []
