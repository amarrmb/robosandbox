"""Honest eval matrix for the insert_connector_franka policies.

Runs each (policy, distribution, world_count) combo and reports
successes / total + Wilson 95% CI. Distributions:
  * fixed: zero out group_xy_jitter (port at exactly the YAML pose)
  * random_xy: as defined in the YAML (group_xy_jitter=0.10)

World counts: 64 (deployment), 256 (training-class), 1024 (full scale).
The 1024 column shows what mujoco_warp's contact-buffer overflow
costs us in practice.

Designed for DGX (needs Newton + warp).
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, "/home/amar/robosandbox/packages/robosandbox-core/src")


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    margin = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, centre - margin) * 100, min(1.0, centre + margin) * 100


@dataclass
class EvalCell:
    policy_id: str
    distribution: str
    world_count: int
    successes: int      # sustained ≥ hold_steps (the "real" metric)
    total: int
    wall_s: float
    n_peak: int = 0      # ever inserted ≥1 step (capability)
    n_final: int = 0     # inserted at the final step (steady-state)


def run_one(policy_path: str, world_count: int, distribution: str,
             max_steps: int = 400, settle: int = 50,
             hold_steps: int = 50) -> EvalCell:
    import time
    from robosandbox.tasks.loader import load_builtin_task
    from robosandbox.tasks.randomize import jitter_scene
    from robosandbox.tasks.runner import _eval_check
    from robosandbox.sim.newton_backend import NewtonBackend
    from robosandbox.policy import load_policy
    from robosandbox.rl.ppo import NeuralPolicy

    task = load_builtin_task("insert_connector_franka")

    if distribution == "fixed":
        randomize_spec = None
        per_world_scenes = None
    elif distribution == "random_xy":
        # Use the YAML's randomize block, group jitter only
        randomize_spec = task.randomize
        per_world_scenes = [
            jitter_scene(task.scene, randomize_spec, seed=i + 1)
            for i in range(world_count)
        ]
    else:
        raise ValueError(distribution)

    sim = NewtonBackend(world_count=world_count, render_size=(120, 160))
    if per_world_scenes is not None:
        sim.load(task.scene, per_world_scenes=per_world_scenes)
    else:
        sim.load(task.scene)

    # Load policy with scene attached for ee_xyz_rpy
    policy = load_policy(Path(policy_path), scene=task.scene)

    for _ in range(settle):
        sim.step()
    initial_obs = sim.observe_all()
    n_dof = sim.n_dof

    # Track per-world rolling window of inserted? results across steps.
    # Success = inserted at the LAST `hold_steps` consecutive steps. This
    # measures sustained insertion (real capability), not peak-detection.
    inserted_streak = np.zeros(world_count, dtype=np.int64)
    max_streak = np.zeros(world_count, dtype=np.int64)

    t0 = time.time()
    for step in range(max_steps):
        obs_all = sim.observe_all()
        targets = np.zeros((world_count, n_dof), dtype=np.float64)
        grippers = np.zeros(world_count, dtype=np.float64)
        for w in range(world_count):
            a = np.asarray(policy.act(obs_all[w]), dtype=np.float64).ravel()
            targets[w] = a[:n_dof]
            grippers[w] = float(a[n_dof])
        sim.step_all(targets, grippers)

        # Per-step inserted check (vectorized for speed)
        post_obs = sim.observe_all()
        for w in range(world_count):
            ok, _ = _eval_check(task.success.data, initial_obs[w], post_obs[w])
            if ok:
                inserted_streak[w] += 1
                if inserted_streak[w] > max_streak[w]:
                    max_streak[w] = inserted_streak[w]
            else:
                inserted_streak[w] = 0

    final_obs = sim.observe_all()
    wall = time.time() - t0

    # Honest success = sustained insertion ≥ hold_steps consecutive steps
    # somewhere in the rollout. Stricter than "ever inserted" (peak), more
    # forgiving than "inserted at final step" (post-drift).
    sustained = max_streak >= hold_steps
    n_success = int(sustained.sum())

    # Diagnostics: also count peak (ever inserted ≥1 step) and final
    n_peak = int((max_streak >= 1).sum())
    n_final = sum(
        1 for w in range(world_count)
        if _eval_check(task.success.data, initial_obs[w], final_obs[w])[0]
    )
    print(f"    [diag] peak (ever 1+ step): {n_peak}/{world_count}  "
          f"sustained ≥{hold_steps}: {n_success}/{world_count}  "
          f"final-step: {n_final}/{world_count}")

    sim.close()
    return EvalCell(
        policy_id=Path(policy_path).name,
        distribution=distribution,
        world_count=world_count,
        successes=n_success,
        total=world_count,
        wall_s=wall,
        n_peak=n_peak,
        n_final=n_final,
    )


def main() -> int:
    policies = [
        "/home/amar/robosandbox/outputs/insert_v4c_hold",
        "/home/amar/robosandbox/outputs/insert_v6a_yaw10",
        "/home/amar/robosandbox/outputs/insert_v6b_yaw25",
        "/home/amar/robosandbox/outputs/insert_v6c_yaw45",
    ]
    distributions = ["fixed", "random_xy"]
    world_counts = [64, 256]   # 1024 separately if buffer fits; skip for now

    cells: list[EvalCell] = []
    for policy in policies:
        for dist in distributions:
            for n in world_counts:
                print(f"[run] policy={Path(policy).name} dist={dist} N={n}")
                try:
                    c = run_one(policy, n, dist)
                    cells.append(c)
                    pct = 100.0 * c.successes / c.total
                    lo, hi = wilson_ci(c.successes, c.total)
                    print(f"    → {c.successes}/{c.total} = {pct:.1f}%  CI [{lo:.1f}, {hi:.1f}]  ({c.wall_s:.1f}s)")
                except Exception as e:
                    print(f"    ! FAILED: {type(e).__name__}: {e}")

    print()
    print("=" * 92)
    print("Honest eval matrix (insert_connector_franka, Newton)")
    print("Metric meanings:")
    print("  peak      = inserted at ANY step during rollout (peak capability)")
    print("  sustained = inserted for ≥50 consecutive steps (real, holds the pose)")
    print("  final     = inserted at the FINAL step (steady-state)")
    print("=" * 92)
    print(f"{'policy':<22}  {'dist':<10}  {'N':>5}  "
          f"{'peak%':>7}  {'sustained%':>11}  {'final%':>8}")
    for c in cells:
        pct_peak = 100.0 * c.n_peak / c.total
        pct_sust = 100.0 * c.successes / c.total
        pct_final = 100.0 * c.n_final / c.total
        print(f"{c.policy_id:<22}  {c.distribution:<10}  {c.world_count:>5}  "
              f"{pct_peak:>6.1f}%  {pct_sust:>10.1f}%  {pct_final:>7.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
