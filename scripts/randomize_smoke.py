"""Sanity check: scripted Pick still succeeds at >=80% on randomized scenes.

This is the dataset-quality bar from the journey discussion: if scripted
Pick can't track xy/yaw randomization, the resulting demos won't span a
useful distribution and learned policies will inherit the failure mode.
Defaults to 16 trials of xy=±5 cm, yaw=±0.5 rad against pick_cube_franka_random.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "robosandbox-core" / "src"))

from robosandbox.agent.context import AgentContext
from robosandbox.grasp.analytic import AnalyticTopDown
from robosandbox.motion.ik import DLSMotionPlanner
from robosandbox.perception.ground_truth import GroundTruthPerception
from robosandbox.sim.mujoco_backend import MuJoCoBackend
from robosandbox.skills.pick import Pick
from robosandbox.tasks.loader import load_builtin_task
from robosandbox.tasks.randomize import jitter_scene


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="pick_cube_franka_random")
    p.add_argument("--n-trials", type=int, default=16)
    p.add_argument("--bar", type=float, default=0.80, help="Pass/fail success-rate bar.")
    args = p.parse_args()

    task = load_builtin_task(args.task)
    if not task.randomize:
        print(f"task {args.task!r} has no randomize block — nothing to test", file=sys.stderr)
        return 2

    successes = 0
    t0 = time.time()
    for trial in range(args.n_trials):
        scene = jitter_scene(task.scene, task.randomize, seed=trial + 1)
        sim = MuJoCoBackend(render_size=(240, 320), camera="scene")
        sim.load(scene)
        try:
            for _ in range(60):  # settle
                sim.step()
            ctx = AgentContext(
                sim=sim,
                perception=GroundTruthPerception(),
                grasp=AnalyticTopDown(),
                motion=DLSMotionPlanner(n_waypoints=160, dt=0.005),
            )
            result = Pick()(ctx, object="red cube")
            ok = bool(result.success)
            successes += int(ok)
            print(f"  trial {trial + 1:2d}/{args.n_trials}: {'success' if ok else 'failure':7s}  ({result.reason})")
        finally:
            sim.close()

    wall = time.time() - t0
    rate = successes / args.n_trials if args.n_trials else 0.0
    print()
    print(f"  successes: {successes} / {args.n_trials}  ({rate * 100:.1f}%)")
    print(f"  wall:      {wall:.1f}s")
    print(f"  bar:       {args.bar * 100:.0f}%")
    if rate >= args.bar:
        print(f"  verdict:   PASS")
        return 0
    print(f"  verdict:   FAIL (under bar)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
