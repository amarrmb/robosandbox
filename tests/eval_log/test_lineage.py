import pytest
from robosandbox.eval_log.lineage import walk_lineage, LineageError, MAX_DEPTH
from robosandbox.eval_log.schema import PolicyRow
from robosandbox.eval_log.store import EvalLogStore


def _p(pid, parent=None):
    return PolicyRow(pid, "ppo_neural", parent, "warm_start_ppo" if parent else None,
                     None, 0, None, "2026-05-01T00:00:00Z", f"out/{pid}")


def test_walk_chain(tmp_path):
    store = EvalLogStore(tmp_path)
    for r in [_p("root"), _p("v1", "root"), _p("v2", "v1"), _p("v3", "v2")]:
        store.append_policy(r)
    chain = walk_lineage(tmp_path, "v3")
    assert [p.policy_id for p in chain] == ["v3", "v2", "v1", "root"]


def test_walk_root(tmp_path):
    store = EvalLogStore(tmp_path)
    store.append_policy(_p("solo"))
    chain = walk_lineage(tmp_path, "solo")
    assert [p.policy_id for p in chain] == ["solo"]


def test_walk_unknown_policy(tmp_path):
    chain = walk_lineage(tmp_path, "ghost")
    assert chain == []


def test_walk_orphan_parent(tmp_path):
    """A policy whose parent_policy_id doesn't exist in the log -> chain stops."""
    store = EvalLogStore(tmp_path)
    store.append_policy(_p("v1", "missing_parent"))
    chain = walk_lineage(tmp_path, "v1")
    assert [p.policy_id for p in chain] == ["v1"]


def test_cycle_detection(tmp_path):
    """Two policies that name each other as parent -> LineageError after MAX_DEPTH."""
    store = EvalLogStore(tmp_path)
    store.append_policy(_p("a", "b"))
    store.append_policy(_p("b", "a"))
    with pytest.raises(LineageError):
        walk_lineage(tmp_path, "a")
