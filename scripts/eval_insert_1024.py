"""1024-world eval column for the insert_connector_franka policies.

Runs v4c / v7a / v7b at world_count=1024 in random_xy distribution
with auto-scaled max_triangle_pairs. Closes the 1024 column the
original matrix didn't have.

Each policy runs at its trained clearance:
  v4c: 4mm    (YAML default)
  v7a: 2mm    (--clearance-m 0.002)
  v7b: 1.5mm  (--clearance-m 0.0015 — USB-A spec)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
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


def run_one(policy_path: str, clearance_m: float | None, world_count: int = 1024,
            max_steps: int = 400, settle: int = 50, hold_steps: int = 50) -> dict:
    from robosandbox.tasks.loader import load_builtin_task
    from robosandbox.tasks.randomize import jitter_scene
    from robosandbox.tasks.runner import _eval_check
    from robosandbox.sim.newton_backend import NewtonBackend
    from robosandbox.policy import load_policy
    from robosandbox.tasks.insertion_geometry import rescale_port_clearance

    task = load_builtin_task("insert_connector_franka")
    if clearance_m is not None:
        task.scene = rescale_port_clearance(task.scene, clearance_m)

    per_world = [
        jitter_scene(task.scene, task.randomize, seed=i + 1)
        for i in range(world_count)
    ]
    sim = NewtonBackend(world_count=world_count, render_size=(60, 80))
    sim.load(task.scene, per_world_scenes=per_world)
    policy = load_policy(Path(policy_path), scene=task.scene)

    for _ in range(settle):
        sim.step()
    initial_obs = sim.observe_all()
    n_dof = sim.n_dof

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

        post_obs = sim.observe_all()
        for w in range(world_count):
            ok, _ = _eval_check(task.success.data, initial_obs[w], post_obs[w])
            if ok:
                inserted_streak[w] += 1
                if inserted_streak[w] > max_streak[w]:
                    max_streak[w] = inserted_streak[w]
            else:
                inserted_streak[w] = 0
        if (step + 1) % 50 == 0:
            print(f"  [step {step+1}/{max_steps}] sustained ≥{hold_steps}: "
                  f"{int((max_streak >= hold_steps).sum())}/{world_count}")

    final_obs = sim.observe_all()
    wall = time.time() - t0

    sustained = max_streak >= hold_steps
    n_sustained = int(sustained.sum())
    n_peak = int((max_streak >= 1).sum())
    n_final = sum(
        1 for w in range(world_count)
        if _eval_check(task.success.data, initial_obs[w], final_obs[w])[0]
    )
    sim.close()

    return {
        "policy": Path(policy_path).name,
        "clearance_mm": clearance_m * 1000 if clearance_m else 4.0,
        "world_count": world_count,
        "n_peak": n_peak,
        "n_sustained": n_sustained,
        "n_final": n_final,
        "wall_s": wall,
    }


def main() -> int:
    cells = [
        ("/home/amar/robosandbox/outputs/insert_v4c_hold", None),
        ("/home/amar/robosandbox/outputs/insert_v7a_2mm", 0.002),
        ("/home/amar/robosandbox/outputs/insert_v7b_1p5mm", 0.0015),
    ]

    results = []
    for policy, clearance in cells:
        print(f"\n[run] {Path(policy).name} clearance={clearance}")
        try:
            r = run_one(policy, clearance, world_count=1024)
            results.append(r)
            n = r["world_count"]
            sus = r["n_sustained"]
            lo, hi = wilson_ci(sus, n)
            print(f"  → peak {r['n_peak']}/{n}  sustained {sus}/{n} "
                  f"({100*sus/n:.1f}%, CI [{lo:.1f}, {hi:.1f}])  "
                  f"final {r['n_final']}/{n}  ({r['wall_s']:.1f}s)")
        except Exception as e:
            print(f"  ! FAILED: {type(e).__name__}: {e}")
            results.append({"policy": Path(policy).name, "error": str(e)})

    out = Path("/home/amar/robosandbox/outputs/eval_matrix_1024.json")
    out.write_text(json.dumps(results, indent=2))

    print()
    print("=" * 80)
    print("Insert connector — Newton 1024 worlds, random ±10cm port")
    print("=" * 80)
    print(f"{'policy':<22} {'clr_mm':>7} {'peak%':>7} {'sustained%':>11} {'final%':>8} {'CI':>22}")
    for r in results:
        if "error" in r:
            print(f"{r['policy']:<22} ERROR: {r['error']}")
            continue
        n = r["world_count"]
        lo, hi = wilson_ci(r["n_sustained"], n)
        print(f"{r['policy']:<22} {r['clearance_mm']:>7.2f} "
              f"{100*r['n_peak']/n:>6.1f}% "
              f"{100*r['n_sustained']/n:>10.1f}% "
              f"{100*r['n_final']/n:>7.1f}%  "
              f"[{lo:>5.1f},{hi:>5.1f}]")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
