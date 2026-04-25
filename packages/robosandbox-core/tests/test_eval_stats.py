"""Stats correctness for robosandbox.eval.

The Wilson CI and z-test math is load-bearing for any future "policy A vs B"
claim, so a few targeted regression tests beat trusting that the formulas
were transcribed correctly.
"""

from __future__ import annotations

import math

from robosandbox.eval import proportion_z_test, summarise_eval, wilson_ci


def test_wilson_ci_matches_known_values() -> None:
    # Reference values from R: prop.test(8, 10, correct=FALSE)$conf.int
    # Wilson is conf.int when correct=FALSE.
    lo, hi = wilson_ci(8, 10)
    assert math.isclose(lo, 0.4901624, abs_tol=1e-4)
    assert math.isclose(hi, 0.9433178, abs_tol=1e-4)


def test_wilson_ci_handles_extremes() -> None:
    # 0/n must produce a useful upper bound (not collapse to 0..0)
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0
    assert 0.20 < hi < 0.35
    # n/n symmetric
    lo, hi = wilson_ci(10, 10)
    assert hi == 1.0 or math.isclose(hi, 1.0, abs_tol=1e-9)
    assert 0.65 < lo < 0.80
    # Degenerate n == 0 returns trivial interval, not crash
    lo, hi = wilson_ci(0, 0)
    assert (lo, hi) == (0.0, 1.0)


def test_proportion_z_test_significant_difference() -> None:
    # 80/100 vs 50/100 is a textbook "highly significant" comparison.
    z, p = proportion_z_test(80, 100, 50, 100)
    assert z > 4.0
    assert p < 1e-4


def test_proportion_z_test_no_difference() -> None:
    z, p = proportion_z_test(50, 100, 50, 100)
    assert z == 0.0
    # Two-sided p-value at z=0 is 1.0 by definition.
    assert math.isclose(p, 1.0, abs_tol=1e-12)


def test_proportion_z_test_handles_empty_n() -> None:
    # No silent crash; report "no evidence of difference" so callers
    # can keep their JSON schema stable when one side has 0 trials.
    assert proportion_z_test(0, 0, 5, 10) == (0.0, 1.0)


def test_summarise_eval_includes_required_fields() -> None:
    out = summarise_eval(
        task="pick_cube_franka",
        policy="runs/abc",
        sim_backend="newton",
        successes=42,
        n_trials=64,
        success_per_trial=[True] * 42 + [False] * 22,
        steps=38400,
        wall_seconds=12.3,
        throughput=3120.0,
    )
    # Schema lock: anyone parsing eval JSON downstream relies on these keys.
    for key in (
        "schema_version", "task", "policy", "sim_backend", "n_trials",
        "successes", "rate", "ci_low", "ci_high", "ci_level",
        "steps", "wall_seconds", "throughput_env_steps_per_s",
        "success_per_trial",
    ):
        assert key in out, f"missing key: {key}"
    # Schema bumped to 2 when per_trial_details + spatial_breakdown were
    # added — both fields are optional, so callers that don't pass details
    # get the same shape as v1 minus the new keys (older parsers ignore
    # the version bump and still find every field they relied on).
    assert out["schema_version"] == 2
    assert math.isclose(out["rate"], 42 / 64)
    assert 0.0 <= out["ci_low"] <= out["rate"] <= out["ci_high"] <= 1.0


def test_summarise_eval_emits_spatial_breakdown_when_details_provided() -> None:
    """When per_trial_details carries object_initial_xyz, the summary should
    derive a per-bin success-rate breakdown so spatial failure analysis
    works without re-running anything."""
    # 4 trials: 2 successes at high x, 2 failures at low x — extreme spatial
    # gradient that the breakdown should surface clearly.
    details = [
        {"trial": 1, "seed": 1, "success": False, "steps": 500,
         "peak_lift_mm": 5.0, "min_ee_object_dist_mm": 80.0,
         "object_id": "red_cube", "object_initial_xyz": [0.36, 0.0, 0.05],
         "object_initial_quat_xyzw": [0, 0, 0, 1]},
        {"trial": 2, "seed": 2, "success": False, "steps": 500,
         "peak_lift_mm": 4.0, "min_ee_object_dist_mm": 75.0,
         "object_id": "red_cube", "object_initial_xyz": [0.37, 0.0, 0.05],
         "object_initial_quat_xyzw": [0, 0, 0, 1]},
        {"trial": 3, "seed": 3, "success": True, "steps": 400,
         "peak_lift_mm": 120.0, "min_ee_object_dist_mm": 12.0,
         "object_id": "red_cube", "object_initial_xyz": [0.44, 0.0, 0.05],
         "object_initial_quat_xyzw": [0, 0, 0, 1]},
        {"trial": 4, "seed": 4, "success": True, "steps": 400,
         "peak_lift_mm": 130.0, "min_ee_object_dist_mm": 10.0,
         "object_id": "red_cube", "object_initial_xyz": [0.45, 0.0, 0.05],
         "object_initial_quat_xyzw": [0, 0, 0, 1]},
    ]
    out = summarise_eval(
        task="pick_cube_franka_random", policy="ckpt", sim_backend="mujoco",
        successes=2, n_trials=4, success_per_trial=[False, False, True, True],
        per_trial_details=details, steps=1800, wall_seconds=20.0, throughput=90.0,
    )
    assert "per_trial_details" in out and len(out["per_trial_details"]) == 4
    assert "spatial_breakdown" in out
    bins_x = out["spatial_breakdown"]["by_object_x"]
    # Lowest x bin should be 0% successful, highest 100% — the gradient
    # signature we explicitly want callers to be able to read off.
    assert bins_x[0]["rate"] == 0.0
    assert bins_x[-1]["rate"] == 1.0


def test_summarise_eval_includes_provenance_when_provided() -> None:
    """Provenance pins down what produced the result. Field shape is open
    (callers may add their own keys), but if a caller passes one it must
    flow through unchanged so downstream comparison tools can rely on it."""
    prov = {
        "policy_path": "/tmp/ckpt",
        "checkpoint_sha256": "deadbeef" * 8,
        "robosandbox_git_rev": "abc123-dirty",
        "lerobot_version": "0.4.4",
    }
    out = summarise_eval(
        task="x", policy="/tmp/ckpt", sim_backend="mujoco",
        successes=1, n_trials=1, provenance=prov,
        steps=10, wall_seconds=1.0, throughput=10.0,
    )
    assert out["provenance"] == prov


def test_summarise_eval_omits_spatial_breakdown_when_no_pose_info() -> None:
    """Tasks whose success criterion has no .object (e.g. pure joint-state
    targets) shouldn't produce a misleading empty/zero breakdown."""
    out = summarise_eval(
        task="reach_pose", policy="ckpt", sim_backend="mujoco",
        successes=1, n_trials=2, success_per_trial=[True, False],
        per_trial_details=[
            {"trial": 1, "seed": 1, "success": True, "steps": 100,
             "peak_lift_mm": 0.0, "min_ee_object_dist_mm": None},
            {"trial": 2, "seed": 2, "success": False, "steps": 500,
             "peak_lift_mm": 0.0, "min_ee_object_dist_mm": None},
        ],
        steps=600, wall_seconds=10.0, throughput=60.0,
    )
    # spatial_breakdown is present but empty when no trial carries pose info.
    assert out["spatial_breakdown"] == {}
