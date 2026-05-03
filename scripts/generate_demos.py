"""Generate N randomized scripted demonstrations for ACT/Diffusion training.

Per demo:
  - jitter the cube pose with a per-trial seed
  - run scripted Pick to grasp + lift the cube
  - record events.jsonl (with action: {joints, gripper}) + video.mp4

Saves each successful demo to runs/<timestamp>-<id>/. Failed picks are
discarded so the resulting dataset is purely positive demonstrations.
After this script finishes, point `robo-sandbox export-lerobot` at the
runs/ directory to roll them all into one LeRobot dataset.

Parallelism: --jobs J fans out across J worker processes (defaults to
4). Each worker runs episodes sequentially with its own seed range.
On a multi-core laptop, J=4 cuts 200 sequential demos (~67 min) down
to ~17 min wall time. The MuJoCo backend is CPU-only with EGL for
rendering, so workers don't contend on the GPU.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

# Headless EGL by default — required on most multi-core machines.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "robosandbox-core" / "src"))


def _generate_one(args: tuple[int, str, Path]) -> tuple[int, bool, float, str | None]:
    """One worker call: run a single seeded demo. Returns (seed, ok, wall, episode_dir)."""
    seed, task_name, runs_root = args
    # Late imports inside the worker so multiprocessing fork doesn't pay
    # the import cost in the parent process before fan-out.
    from robosandbox.agent.context import AgentContext
    from robosandbox.grasp.analytic import AnalyticTopDown
    from robosandbox.motion.ik import DLSMotionPlanner
    from robosandbox.perception.ground_truth import GroundTruthPerception
    from robosandbox.recorder.local import LocalRecorder
    from robosandbox.sim.mujoco_backend import MuJoCoBackend
    from robosandbox.skills.pick import Pick
    from robosandbox.tasks.loader import load_builtin_task
    from robosandbox.tasks.randomize import jitter_scene

    task = load_builtin_task(task_name)
    if not task.randomize:
        raise RuntimeError(
            f"task {task_name!r} has no randomize block — generated demos would all be identical"
        )

    scene = jitter_scene(task.scene, task.randomize, seed=seed)
    sim = MuJoCoBackend(render_size=(240, 320), camera="scene")
    sim.load(scene)
    recorder = LocalRecorder(root=runs_root, video_fps=30)

    def _on_step() -> None:
        recorder.write_frame(sim.observe(), action=sim.last_action())

    ctx = AgentContext(
        sim=sim,
        perception=GroundTruthPerception(),
        grasp=AnalyticTopDown(),
        motion=DLSMotionPlanner(n_waypoints=160, dt=0.005),
        recorder=recorder,
        on_step=_on_step,
    )
    recorder.start_episode(
        task=task.prompt,
        metadata={"source": "generate_demos", "seed": seed, "sim_dt": 0.005},
    )

    # Settle physics before recording. write_frame is intentionally NOT
    # called here — settle has no commanded action (sim.last_action() is
    # None) and the exporter rejects action=None frames.
    for _ in range(60):
        sim.step()

    t0 = time.time()
    try:
        result = Pick()(ctx, object="red cube")
        wall = time.time() - t0
        ok = bool(result.success)
    except Exception as e:
        wall = time.time() - t0
        recorder.end_episode(success=False, result={"reason": f"exception: {e}"})
        sim.close()
        return (seed, False, wall, None)
    recorder.end_episode(success=ok, result={"reason": result.reason})
    sim.close()

    # Find the episode dir — recorder consumes it after end_episode().
    dirs = sorted(Path(runs_root).glob("20*"), reverse=True)
    ep = str(dirs[0]) if dirs else None
    # Discard failed picks so the dataset is purely positive demonstrations.
    if not ok and ep is not None:
        shutil.rmtree(ep, ignore_errors=True)
        ep = None
    return (seed, ok, wall, ep)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="pick_cube_franka_random",
                   help="Task name (must have a randomize block).")
    p.add_argument("--n", type=int, default=200,
                   help="Total demos to attempt.")
    p.add_argument("--jobs", type=int, default=4,
                   help="Worker processes. Each runs episodes sequentially.")
    p.add_argument("--runs-root", type=Path, default=Path("runs/demos_franka_pick"),
                   help="Output directory; one subdir per successful demo.")
    p.add_argument("--seed-start", type=int, default=1,
                   help="First seed (incremented per demo). Skip 0 — that is the identity scene.")
    args = p.parse_args()

    args.runs_root.mkdir(parents=True, exist_ok=True)
    print(f"[demos] task: {args.task}  n: {args.n}  jobs: {args.jobs}  runs_root: {args.runs_root}")

    seeds = list(range(args.seed_start, args.seed_start + args.n))
    payload = [(s, args.task, args.runs_root) for s in seeds]

    t_start = time.time()
    successes = 0
    failures = 0
    if args.jobs <= 1:
        results = [_generate_one(item) for item in payload]
    else:
        # Use spawn explicitly so each worker inherits a clean MuJoCo state.
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.jobs) as pool:
            results = []
            for i, r in enumerate(pool.imap_unordered(_generate_one, payload)):
                results.append(r)
                seed, ok, wall, ep = r
                if ok:
                    successes += 1
                else:
                    failures += 1
                if (i + 1) % 10 == 0 or (i + 1) == args.n:
                    elapsed = time.time() - t_start
                    rate = (i + 1) / elapsed if elapsed > 0 else 0.0
                    eta = (args.n - (i + 1)) / rate if rate > 0 else 0.0
                    print(
                        f"[demos] progress {i + 1}/{args.n}  "
                        f"successes={successes}  failures={failures}  "
                        f"elapsed={elapsed:.0f}s  rate={rate:.2f}/s  eta={eta:.0f}s"
                    )

    if args.jobs <= 1:
        for seed, ok, wall, ep in results:
            if ok:
                successes += 1
            else:
                failures += 1

    wall = time.time() - t_start
    print()
    print(f"[demos] FINISHED")
    print(f"[demos]   total:     {args.n}")
    print(f"[demos]   successes: {successes}  ({successes / max(args.n, 1) * 100:.1f}%)")
    print(f"[demos]   failures:  {failures}")
    print(f"[demos]   wall:      {wall:.1f}s ({wall / 60:.1f} min)")
    print()
    print(f"[demos] Next: roll into a LeRobot dataset:")
    print(f"  robo-sandbox export-lerobot {args.runs_root} datasets/{args.runs_root.name}_lerobot --task {args.task} --fps 30")
    return 0 if successes > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
