"""Scripted IK oracle for reach_target_franka.

For each of N seeds:
  1. jitter the scene
  2. plan a Cartesian trajectory from home to target xyz (palm-down)
  3. execute open-loop
  4. measure final EE-target distance

If oracle hits ~100% success, the task setup is solvable and any plateau
in PPO is an architecture/PPO problem (likely action space).
If oracle fails, the threshold/workspace itself is the problem.

Usage:
    MUJOCO_GL=egl python scripts/oracle_reach_sanity.py --n 32
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

sys.path.insert(
    0, str(Path(__file__).parent.parent / "packages" / "robosandbox-core" / "src")
)

import numpy as np

from robosandbox.motion.ik import UnreachableError, plan_linear_cartesian
from robosandbox.sim.mujoco_backend import MuJoCoBackend
from robosandbox.tasks.loader import load_builtin_task
from robosandbox.tasks.randomize import jitter_scene
from robosandbox.types import Pose


_PALM_DOWN_XYZW = (1.0, 0.0, 0.0, 0.0)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="reach_target_franka")
    p.add_argument("--n", type=int, default=32, help="Number of randomized trials")
    p.add_argument("--threshold", type=float, default=0.04, help="Success threshold (m)")
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--settle-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    task = load_builtin_task(args.task)

    print(f"[oracle] task={args.task} threshold={args.threshold:.3f}m n={args.n}")
    distances = []
    n_success = 0
    n_unreachable = 0
    for trial in range(args.n):
        scene = jitter_scene(task.scene, task.randomize, seed=args.seed + trial)
        sim = MuJoCoBackend(render_size=(80, 100))
        sim.load(scene)
        for _ in range(args.settle_steps):
            sim.step()
        obs0 = sim.observe()
        target = obs0.scene_objects["target_marker"]
        target_pose = Pose(xyz=target.xyz, quat_xyzw=_PALM_DOWN_XYZW)
        try:
            traj = plan_linear_cartesian(
                sim,
                start_joints=obs0.robot_joints,
                target_pose=target_pose,
                n_waypoints=120,
                dt=0.005,
                orientation="z_down",
            )
        except UnreachableError as e:
            n_unreachable += 1
            print(f"  trial {trial}: UNREACHABLE  ({e})")
            sim.close()
            continue
        # Execute open-loop: track waypoints with PD position control
        held_q = traj.waypoints[0]
        for step_i in range(min(args.max_steps, len(traj.waypoints))):
            held_q = traj.waypoints[step_i]
            sim.step(target_joints=held_q, gripper=1.0)  # gripper open (1.0)
        # Continue holding final pose for a few extra steps to settle
        for _ in range(50):
            sim.step(target_joints=traj.waypoints[-1], gripper=1.0)
        obs_final = sim.observe()
        ee = np.asarray(obs_final.ee_pose.xyz)
        tgt = np.asarray(target.xyz)
        d = float(np.linalg.norm(ee - tgt))
        distances.append(d)
        ok = d <= args.threshold
        if ok:
            n_success += 1
        marker = "OK" if ok else "miss"
        print(f"  trial {trial}: {marker:<4}  dist {d*100:5.2f}cm  target_xyz {tuple(round(v,3) for v in tgt)}")
        sim.close()

    distances_arr = np.array(distances) if distances else np.array([])
    print()
    print(f"[oracle] success: {n_success}/{args.n} ({100.0*n_success/args.n:.1f}%)  "
          f"unreachable: {n_unreachable}")
    if distances_arr.size > 0:
        print(f"[oracle] dist (cm): mean {distances_arr.mean()*100:.2f}  "
              f"median {np.median(distances_arr)*100:.2f}  "
              f"min {distances_arr.min()*100:.2f}  "
              f"max {distances_arr.max()*100:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
