"""Generate scripted Pick demos in Newton (with MuJoCo as kinematics oracle).

Newton runs the physics; MuJoCo is loaded once, side-by-side, purely as
the IK oracle (analytic Jacobian via mj_jacSite). The scripted Pick skill
plans against MuJoCo's kinematics and executes via Newton's actuators.

This unblocks the native Newton record→export→train→eval loop:
  1. generate_newton_demos.py   → events.jsonl per episode
  2. robo-sandbox export-lerobot runs/  data/                  → LeRobot dataset
  3. lerobot-train ...                                          → ACT checkpoint
  4. robo-sandbox eval --sim-backend newton --policy <ckpt>     → success rate

Mirrors generate_demos.py (MuJoCo-only version) — same args, same output
layout, drop-in replacement for the Newton-side dataset.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

# Headless EGL by default — required on most multi-core machines.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "robosandbox-core" / "src"))


def _generate_one(seed: int, task_name: str, runs_root: Path, *, keep_failures: bool = False) -> tuple[int, bool, float, str, str | None]:
    """One demo: scene jitter → Newton + MuJoCo (kinematics) → scripted Pick → record."""
    from robosandbox.agent.context import AgentContext
    from robosandbox.grasp.analytic import AnalyticTopDown
    from robosandbox.motion.ik import DLSMotionPlanner
    from robosandbox.perception.ground_truth import GroundTruthPerception
    from robosandbox.recorder.local import LocalRecorder
    from robosandbox.sim.mujoco_backend import MuJoCoBackend
    from robosandbox.sim.newton_backend import NewtonBackend
    from robosandbox.skills.pick import Pick
    from robosandbox.tasks.loader import load_builtin_task
    from robosandbox.tasks.randomize import jitter_scene

    task = load_builtin_task(task_name)
    if not task.randomize:
        raise RuntimeError(
            f"task {task_name!r} has no randomize block — generated demos would all be identical"
        )

    scene = jitter_scene(task.scene, task.randomize, seed=seed)
    # Newton = physics; MuJoCo = IK oracle (loaded with same scene so the
    # robot URDF + workspace + object positions match exactly).
    # dt=0.005 matches MuJoCo's panda.xml timestep — keeps integration
    # behaviour close enough that joint targets land in the same place.
    sim = NewtonBackend(world_count=1, render_size=(240, 320), dt=0.005)
    sim.load(scene)
    kin = MuJoCoBackend(render_size=(240, 320), camera="scene")
    kin.load(scene)
    # Gravity-feedforward via inverse dynamics on the kinematics oracle.
    # Cancels Newton's PD steady-state sag so the arm tracks the IK
    # trajectory accurately instead of settling 10-20 mm under target
    # at stretched-out configurations.
    # Scale FF slightly above unity: MuJoCo's inertial parameters are
    # close but not identical to Newton's MJCF parser, so unity FF leaves
    # a small residual gravity bias. 1.05 over-compensates by ~5% which
    # eliminates the residual on the panda chain (empirical).
    sim.set_gravity_compensation(lambda j, g: 1.05 * kin.compute_gravity_torque(j, g))

    recorder = LocalRecorder(root=runs_root, video_fps=30)

    def _on_step() -> None:
        recorder.write_frame(sim.observe(), action=sim.last_action() if hasattr(sim, "last_action") else None)

    ctx = AgentContext(
        sim=sim,
        perception=GroundTruthPerception(),
        grasp=AnalyticTopDown(),
        motion=DLSMotionPlanner(n_waypoints=160, dt=0.005),
        recorder=recorder,
        on_step=_on_step,
        kinematics_sim=kin,
    )
    recorder.start_episode(
        task=task.prompt,
        metadata={"source": "generate_newton_demos", "seed": seed, "sim_backend": "newton"},
    )

    # Settle physics in Newton before recording. Don't write_frame —
    # last_action() is None during settle and the exporter rejects None.
    for _ in range(60):
        sim.step()

    t0 = time.time()
    try:
        result = Pick()(ctx, object="red cube")
        wall = time.time() - t0
        ok = bool(result.success)
        reason = result.reason_detail or result.reason or ("ok" if ok else "unknown")
    except Exception as e:
        wall = time.time() - t0
        reason = f"exception: {e}"
        recorder.end_episode(success=False, result={"reason": reason})
        sim.close()
        kin.close()
        return (seed, False, wall, reason, None)
    recorder.end_episode(success=ok, result={"reason": reason})
    sim.close()
    kin.close()

    dirs = sorted(Path(runs_root).glob("20*"), reverse=True)
    ep = str(dirs[0]) if dirs else None
    if not ok and ep is not None and not keep_failures:
        shutil.rmtree(ep, ignore_errors=True)
        ep = None
    return (seed, ok, wall, reason, ep)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="pick_cube_franka_random",
                   help="Task name (must have a randomize block).")
    p.add_argument("--n", type=int, default=10,
                   help="Total demos to attempt.")
    p.add_argument("--runs-root", type=Path, default=Path("runs/newton_demos_franka_pick"),
                   help="Output directory; one subdir per successful demo.")
    p.add_argument("--seed-start", type=int, default=1,
                   help="First seed (incremented per demo).")
    p.add_argument("--keep-failures", action="store_true",
                   help="Keep failed-episode dirs for debugging (default: delete).")
    args = p.parse_args()

    args.runs_root.mkdir(parents=True, exist_ok=True)
    print(f"[newton-demos] task: {args.task}  n: {args.n}  runs_root: {args.runs_root}")
    print(f"[newton-demos] physics: Newton; IK oracle: MuJoCo")

    t_start = time.time()
    successes = 0
    failures = 0
    seeds = list(range(args.seed_start, args.seed_start + args.n))

    for i, seed in enumerate(seeds):
        seed_, ok, wall, reason, ep = _generate_one(
            seed, args.task, args.runs_root, keep_failures=args.keep_failures
        )
        if ok:
            successes += 1
        else:
            failures += 1
        elapsed = time.time() - t_start
        rate = (i + 1) / elapsed if elapsed > 0 else 0.0
        eta = (args.n - (i + 1)) / rate if rate > 0 else 0.0
        print(
            f"[newton-demos] {i + 1}/{args.n}  seed={seed}  ok={ok}  wall={wall:.1f}s  "
            f"reason={reason!r}  successes={successes}  failures={failures}  "
            f"elapsed={elapsed:.0f}s  eta={eta:.0f}s"
        )

    wall = time.time() - t_start
    print()
    print(f"[newton-demos] FINISHED")
    print(f"[newton-demos]   total:     {args.n}")
    print(f"[newton-demos]   successes: {successes}  ({successes / max(args.n, 1) * 100:.1f}%)")
    print(f"[newton-demos]   failures:  {failures}")
    print(f"[newton-demos]   wall:      {wall:.1f}s ({wall / 60:.1f} min)")
    print()
    print(f"[newton-demos] Next: roll into a LeRobot dataset:")
    print(
        f"  robo-sandbox export-lerobot {args.runs_root} "
        f"datasets/{args.runs_root.name}_lerobot --task {args.task} --fps 30"
    )
    return 0 if successes > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
