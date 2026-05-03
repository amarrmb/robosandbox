"""Audit which scripted skills run cleanly end-to-end in MuJoCo.

For each (skill, task) pair we care about, instantiates the skill, runs it
once against the task scene, and reports pass/fail + wall time. Used to
gate the "Beat 1: six tasks one stack" video — we only ship skills that
work today.

Usage:
    MUJOCO_GL=egl python scripts/audit_skills.py
    MUJOCO_GL=egl python scripts/audit_skills.py --only pick stack
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

sys.path.insert(
    0, str(Path(__file__).parent.parent / "packages" / "robosandbox-core" / "src")
)

from robosandbox.agent.context import AgentContext
from robosandbox.grasp.analytic import AnalyticTopDown
from robosandbox.motion.ik import DLSMotionPlanner
from robosandbox.perception.ground_truth import GroundTruthPerception
from robosandbox.recorder.local import LocalRecorder
from robosandbox.sim.mujoco_backend import MuJoCoBackend
from robosandbox.tasks.loader import load_builtin_task


@dataclass
class Trial:
    name: str
    task: str
    skill_factory: callable  # () -> skill instance
    skill_args: dict


def make_ctx(sim, recorder):
    return AgentContext(
        sim=sim,
        perception=GroundTruthPerception(),
        grasp=AnalyticTopDown(),
        motion=DLSMotionPlanner(n_waypoints=160, dt=0.005),
        recorder=recorder,
    )


def run_one(trial: Trial) -> dict:
    from robosandbox import skills as _ensure_pkg  # noqa
    out = {"trial": trial.name, "task": trial.task, "ok": False, "reason": "", "wall": 0.0}
    try:
        task = load_builtin_task(trial.task)
        sim = MuJoCoBackend(render_size=(240, 320), camera="scene")
        sim.load(task.scene)
        recorder = LocalRecorder(root=Path("/tmp/audit_runs"), video_fps=30)
        recorder.start_episode(task=task.prompt, metadata={"audit": trial.name})
        # Settle physics
        for _ in range(100):
            sim.step()
        ctx = make_ctx(sim, recorder)
        skill = trial.skill_factory()
        t0 = time.time()
        result = skill(ctx, **trial.skill_args)
        wall = time.time() - t0
        recorder.end_episode(success=getattr(result, "success", False),
                             result={"reason": getattr(result, "reason", "?")})
        sim.close()
        out["ok"] = bool(getattr(result, "success", False))
        out["reason"] = str(getattr(result, "reason", "?"))
        out["wall"] = wall
    except Exception as e:
        out["reason"] = f"EXCEPTION: {type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", default=None,
                        help="Only run trials whose name contains any of these substrings")
    args = parser.parse_args()

    # Skill-task pairs to audit. Each is one trial.
    from robosandbox.skills.pick import Pick
    from robosandbox.skills.place import PlaceOn
    from robosandbox.skills.push import Push
    from robosandbox.skills.stack import Stack
    from robosandbox.skills.drawer import OpenDrawer, CloseDrawer
    from robosandbox.skills.pour import Pour
    from robosandbox.skills.tap import Tap

    trials = [
        Trial("pick_cube_franka_random", "pick_cube_franka_random",
              Pick, dict(object="red cube")),
        Trial("pick_ycb_mug", "pick_ycb_mug",
              Pick, dict(object="mug")),
        Trial("pick_from_three", "pick_from_three",
              Pick, dict(object="green cube")),
        Trial("push_forward", "push_forward",
              Push, dict(object="red cube", direction="forward", distance=0.10)),
        Trial("stack_two", "_experimental_stack_two",
              Stack, dict(sources=["red cube"], base="green cube")),
        Trial("open_drawer", "open_drawer",
              OpenDrawer, dict(drawer="drawer_a")),
        Trial("pour_can_into_bowl", "pour_can_into_bowl",
              Pour, dict(target="bowl")),
        Trial("tap_red_cube", "pick_cube_franka_random",
              Tap, dict(object="red cube")),
    ]

    if args.only:
        trials = [t for t in trials if any(s in t.name for s in args.only)]

    print(f"[audit] running {len(trials)} trials")
    print()
    print(f"{'TRIAL':<32} {'TASK':<30} {'OK':<5} {'WALL':<7} REASON")
    print("-" * 100)

    results = []
    for tr in trials:
        r = run_one(tr)
        results.append(r)
        ok = "✅" if r["ok"] else "❌"
        wall = f"{r['wall']:.1f}s"
        print(f"{tr.name:<32} {tr.task:<30} {ok:<5} {wall:<7} {r['reason'][:50]}")

    print()
    n_ok = sum(1 for r in results if r["ok"])
    print(f"[audit] {n_ok}/{len(results)} skills succeeded")

    # Detailed exception traces
    for r in results:
        if "traceback" in r:
            print()
            print(f"--- {r['trial']} ---")
            print(r["traceback"])

    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
