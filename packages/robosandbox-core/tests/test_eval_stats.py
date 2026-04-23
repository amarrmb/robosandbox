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
    assert out["schema_version"] == 1
    assert math.isclose(out["rate"], 42 / 64)
    assert 0.0 <= out["ci_low"] <= out["rate"] <= out["ci_high"] <= 1.0
