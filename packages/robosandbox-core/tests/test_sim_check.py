"""sim-check structural tests.

The end-to-end MuJoCo↔Newton agreement assertion lives on DGX (Newton
needs CUDA + warp). These tests cover the module's structural
contracts that we can validate on any machine: report dataclass
shape, JSON serialization, error handling when a backend is
unavailable.
"""

from __future__ import annotations

import json

import pytest

from robosandbox.eval.sim_check import (
    SimCheckReport,
    report_to_dict,
    run_sim_check,
)


def test_report_dict_has_stable_top_level_keys() -> None:
    rep = SimCheckReport(
        task="pick_cube_franka",
        policy="runs/x",
        tolerances={"joint_rad": 0.087, "xy_m": 0.005},
    )
    out = report_to_dict(rep)
    for key in (
        "schema_version", "task", "policy", "tolerances",
        "mujoco", "newton", "agreement", "verdict",
    ):
        assert key in out
    for key in (
        "outcome_match", "joint_max_abs_diff_rad",
        "object_max_xy_diff_m", "object_diffs",
    ):
        assert key in out["agreement"]
    # Schema version locked so external consumers can rely on it.
    assert out["schema_version"] == 1
    # JSON-serializable as written.
    json.dumps(out)


def test_run_sim_check_reports_run_failed_when_newton_missing() -> None:
    """Without warp installed, Newton creation must surface as RUN_FAILED
    (exit 2 in the CLI) rather than crashing the run or pretending to PASS.

    This test runs only when a real episode exists in runs/. Using the
    bundled scripted-Pick episode is overkill for a structural test, so
    skip cleanly if there's nothing to point at.
    """
    from pathlib import Path

    runs = sorted(Path("runs").glob("20*"), reverse=True)
    if not runs:
        pytest.skip("no episode in runs/ to use as policy")
    try:
        import warp  # noqa: F401
        pytest.skip("warp present — this test needs a no-warp environment")
    except ImportError:
        pass

    rep = run_sim_check(
        task_name="pick_cube_franka",
        policy_path=str(runs[0]),
        max_steps=20,  # short — we just want to reach the Newton step
        settle_steps=10,
    )
    # MuJoCo should have run cleanly; Newton failed at backend creation.
    assert "error" not in rep.mujoco
    assert "error" in rep.newton
    assert rep.verdict == "RUN_FAILED"
