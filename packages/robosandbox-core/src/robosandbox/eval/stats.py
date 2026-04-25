"""Binomial success-rate stats for policy eval.

Stdlib only — no scipy. The Wilson score interval is the textbook
binomial CI for small ``n`` (Newton typically runs 32-128 worlds, where
the normal-approximation Wald CI breaks down near 0% and 100%). The
two-proportion z-test is the standard significance test for comparing
two policies' success rates.
"""

from __future__ import annotations

import math
from typing import TypedDict

# 95% two-sided z-quantile. Hardcoded so callers don't need scipy/numpy
# just to ask for the default. Override via ``z`` for other confidence
# levels (e.g. z=2.576 for 99%).
_Z_95 = 1.959963984540054


class EvalSummary(TypedDict, total=False):
    """Structured eval result. Stable JSON schema for ``robo-sandbox eval --output``."""

    schema_version: int
    task: str
    policy: str
    sim_backend: str
    n_trials: int
    successes: int
    rate: float
    ci_low: float
    ci_high: float
    ci_level: float
    success_per_trial: list[bool]
    steps: int
    wall_seconds: float
    throughput_env_steps_per_s: float


def wilson_ci(successes: int, n: int, z: float = _Z_95) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    Returns ``(lo, hi)`` as fractions in [0, 1]. With ``n == 0`` the
    rate is undefined; we return the trivial ``(0.0, 1.0)`` interval
    rather than raise — callers can branch on ``n`` upstream if they
    want to emit ``null``.

    The Wilson interval is preferred over the textbook Wald interval
    (``p ± z·√(p(1-p)/n)``) because Wald collapses to a zero-width
    bound at ``p == 0`` or ``p == 1`` and undercovers for small ``n``.
    Wilson stays well-defined and conservative across the whole range.
    """
    if n <= 0:
        return (0.0, 1.0)
    if successes < 0 or successes > n:
        raise ValueError(f"successes={successes} out of range for n={n}")
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (lo, hi)


def proportion_z_test(
    s_a: int, n_a: int, s_b: int, n_b: int
) -> tuple[float, float]:
    """Two-proportion z-test (pooled). Returns ``(z, p_value_two_sided)``.

    Tests H0: rate_A == rate_B. Use the returned p-value with the usual
    α = 0.05 cutoff to decide significance. With either ``n`` zero the
    test is undefined; we return ``(0.0, 1.0)`` (no evidence of any
    difference) so callers can keep their JSON schema stable.

    The pooled-variance form is appropriate here because we are testing
    equality, not estimating the difference — that's the standard choice
    when the null hypothesis is ``p_A = p_B``.
    """
    if n_a <= 0 or n_b <= 0:
        return (0.0, 1.0)
    p_a = s_a / n_a
    p_b = s_b / n_b
    pooled = (s_a + s_b) / (n_a + n_b)
    var = pooled * (1.0 - pooled) * (1.0 / n_a + 1.0 / n_b)
    if var <= 0.0:
        # Both 0% or both 100%; no difference detectable.
        return (0.0, 1.0)
    z = (p_a - p_b) / math.sqrt(var)
    p_value = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return (z, p_value)


def summarise_eval(
    *,
    task: str,
    policy: str,
    sim_backend: str,
    successes: int,
    n_trials: int,
    success_per_trial: list[bool] | None = None,
    per_trial_details: list[dict] | None = None,
    provenance: dict | None = None,
    steps: int = 0,
    wall_seconds: float = 0.0,
    throughput: float = 0.0,
    ci_level: float = 0.95,
) -> EvalSummary:
    """Build the canonical EvalSummary dict from a finished eval run.

    Wraps ``wilson_ci`` and locks the JSON schema (``schema_version=2``)
    so downstream tools (``robo-sandbox compare``, plotting scripts)
    can rely on field names + presence. v2 adds ``per_trial_details``
    (per-trial cube pose, peak lift, EE-object min distance) and a
    derived ``spatial_breakdown`` so spatial failure analysis is
    available without re-running anything.
    """
    z = _Z_95 if abs(ci_level - 0.95) < 1e-9 else _z_from_two_sided_level(ci_level)
    lo, hi = wilson_ci(successes, n_trials, z=z)
    out: EvalSummary = {
        "schema_version": 2,
        "task": task,
        "policy": policy,
        "sim_backend": sim_backend,
        "n_trials": n_trials,
        "successes": successes,
        "rate": (successes / n_trials) if n_trials > 0 else 0.0,
        "ci_low": lo,
        "ci_high": hi,
        "ci_level": ci_level,
        "steps": steps,
        "wall_seconds": wall_seconds,
        "throughput_env_steps_per_s": throughput,
    }
    if success_per_trial is not None:
        out["success_per_trial"] = list(success_per_trial)
    if per_trial_details is not None:
        out["per_trial_details"] = list(per_trial_details)
        out["spatial_breakdown"] = _spatial_breakdown(per_trial_details)
    if provenance is not None:
        # Provenance is "what produced this result, exactly" — checkpoint
        # content hash, robosandbox/lerobot/mujoco versions, git rev. Two
        # eval JSONs with matching provenance are guaranteed-comparable;
        # mismatching provenance means at least one of code/checkpoint/sim
        # changed and any apples-to-apples comparison is suspect.
        out["provenance"] = dict(provenance)
    return out


def _spatial_breakdown(details: list[dict], n_bins: int = 5) -> dict:
    """Group trials into x/y bins of the tracked object's initial position
    and report success rate per bin. Skips trials without ``object_initial_xyz``
    (e.g. tasks whose success criterion has no .object). Returns an empty
    dict when no trials carry pose info.

    The breakdown is the cheapest way to see *where* the policy fails — a
    per-bin success-rate gradient is the signature of a generalization gap
    (model trained uniformly but only succeeds in one region).
    """
    valid = [d for d in details if d.get("object_initial_xyz") is not None]
    if not valid:
        return {}
    xs = [d["object_initial_xyz"][0] for d in valid]
    ys = [d["object_initial_xyz"][1] for d in valid]
    def _bin(values: list[float], n: int) -> list[tuple[float, float]]:
        lo, hi = min(values), max(values)
        if hi - lo < 1e-9:
            return [(lo, hi + 1e-9)]
        step = (hi - lo) / n
        return [(lo + i * step, lo + (i + 1) * step + (1e-9 if i == n - 1 else 0.0))
                for i in range(n)]
    def _by(axis_index: int) -> list[dict]:
        bins = _bin([d["object_initial_xyz"][axis_index] for d in valid], n_bins)
        rows = []
        for blo, bhi in bins:
            in_bin = [d for d in valid
                      if blo <= d["object_initial_xyz"][axis_index] < bhi]
            n_in = len(in_bin)
            n_succ = sum(1 for d in in_bin if d.get("success"))
            rows.append({
                "lo": blo,
                "hi": bhi,
                "n": n_in,
                "successes": n_succ,
                "rate": (n_succ / n_in) if n_in else 0.0,
            })
        return rows
    return {
        "by_object_x": _by(0),
        "by_object_y": _by(1),
    }


def _normal_cdf(x: float) -> float:
    """Standard normal CDF Φ(x). Stdlib via ``math.erf``."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _z_from_two_sided_level(level: float) -> float:
    """Inverse normal for arbitrary two-sided confidence levels.

    Supports the common 0.90 / 0.95 / 0.99 levels without scipy via a
    small lookup. Anything else falls back to a numeric inversion.
    """
    table = {0.90: 1.6448536269514722, 0.95: _Z_95, 0.99: 2.5758293035489004}
    if level in table:
        return table[level]
    # Bisect on Φ⁻¹((1+level)/2). Fine for any reasonable level.
    target = (1.0 + level) / 2.0
    lo, hi = 0.0, 10.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _normal_cdf(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
