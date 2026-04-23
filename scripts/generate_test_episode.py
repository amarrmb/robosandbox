"""Generate a scripted pick episode for pick_cube_franka using the Franka arm.

Saves to runs/<timestamp>-franka-test/ in the standard events.jsonl format
so it can be used directly as a policy with `robo-sandbox eval --policy`.

Usage:
    MUJOCO_GL=egl python scripts/generate_test_episode.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Headless default
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "robosandbox-core" / "src"))

from robosandbox.agent.context import AgentContext
from robosandbox.grasp.analytic import AnalyticTopDown
from robosandbox.motion.ik import DLSMotionPlanner
from robosandbox.perception.ground_truth import GroundTruthPerception
from robosandbox.recorder.local import LocalRecorder
from robosandbox.sim.mujoco_backend import MuJoCoBackend
from robosandbox.skills.pick import Pick
from robosandbox.tasks.loader import load_builtin_task


def main() -> int:
    task = load_builtin_task("pick_cube_franka")
    sim = MuJoCoBackend(render_size=(240, 320), camera="scene")
    sim.load(task.scene)
    print(f"[generate] n_dof={sim.n_dof}  joints={sim.joint_names}")

    recorder = LocalRecorder(root=Path("runs"), video_fps=30)

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

    episode_id = recorder.start_episode(
        task=task.prompt,
        metadata={"source": "generate_test_episode", "sim_dt": 0.005},
    )

    for _ in range(100):
        sim.step()
        recorder.write_frame(sim.observe(), action=sim.last_action())

    t0 = time.time()
    result = Pick()(ctx, object="red cube")
    elapsed = time.time() - t0

    recorder.end_episode(success=result.success, result={"reason": result.reason})
    sim.close()

    episode_dir = recorder.current_episode_dir
    # current_episode_dir is None after end_episode — find it
    dirs = sorted(Path("runs").glob("20*"), reverse=True)
    episode_dir = dirs[0] if dirs else None

    print(f"[generate] Pick -> {result!r}  ({elapsed:.1f}s)")
    if episode_dir:
        events = episode_dir / "events.jsonl"
        rows = len(events.read_text().splitlines()) if events.exists() else 0
        print(f"[generate] saved: {episode_dir}  ({rows} rows)")
        print(episode_dir)  # last line = path for shell capture
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
