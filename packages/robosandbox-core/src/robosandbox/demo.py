"""`python -m robosandbox.demo` — the hello-pick vertical slice.

Builds a scene (table + red cube), runs Pick with ground-truth
perception + analytic top-down grasp + DLS Jacobian IK, and writes an
MP4 video + JSON trace to ``runs/``.

No VLM, no network. If this script works on a fresh machine, the
plumbing is intact.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from robosandbox.agent.context import AgentContext
from robosandbox.grasp.analytic import AnalyticTopDown
from robosandbox.motion.ik import DLSMotionPlanner
from robosandbox.perception.ground_truth import GroundTruthPerception
from robosandbox.recorder.local import LocalRecorder
from robosandbox.scene.mjcf_builder import build_mjcf
from robosandbox.sim.mujoco_backend import MuJoCoBackend
from robosandbox.skills.pick import Pick
from robosandbox.types import Pose, Scene, SceneObject


def build_demo_scene() -> Scene:
    cube = SceneObject(
        id="red_cube",
        kind="box",
        size=(0.012, 0.012, 0.012),  # 2.4cm cube — fits the 7cm-open gripper with room
        pose=Pose(xyz=(0.05, 0.0, 0.07), quat_xyzw=(0, 0, 0, 1)),
        mass=0.05,
        rgba=(0.85, 0.2, 0.2, 1.0),
    )
    return Scene(
        robot_urdf=None,
        objects=(cube,),
        table_height=0.04,
    )


def main(argv: list[str] | None = None) -> int:
    _ = argv  # v0.1 demo takes no args
    scene = build_demo_scene()

    # Smoke-check that MJCF parses before we spin up the full backend.
    mjcf_preview = build_mjcf(scene)
    print(f"[demo] MJCF built: {len(mjcf_preview)} chars, {mjcf_preview.count(chr(10))} lines")

    sim = MuJoCoBackend(render_size=(480, 640), camera="scene")
    sim.load(scene)
    print(f"[demo] sim loaded: n_dof={sim.n_dof}, joints={sim.joint_names}")

    recorder = LocalRecorder(root=Path("runs"), video_fps=30)
    ctx = AgentContext(
        sim=sim,
        perception=GroundTruthPerception(),
        grasp=AnalyticTopDown(default_object_width=0.04),
        motion=DLSMotionPlanner(n_waypoints=160, dt=0.005),
        recorder=recorder,
        on_step=None,  # wired below
    )

    # Hook recorder to every sim step.
    def _on_step() -> None:
        obs = sim.observe()
        recorder.write_frame(obs)

    ctx.on_step = _on_step

    task = "pick up the red cube"
    episode_id = recorder.start_episode(
        task=task,
        metadata={
            "source": "robosandbox.demo",
            "source_version": "0.1.0",
            "sim_dt": 0.005,
            "scene": {
                "objects": [
                    {"id": o.id, "kind": o.kind, "xyz": o.pose.xyz} for o in scene.objects
                ]
            },
        },
    )
    print(f"[demo] episode_id={episode_id}")

    # Let the sim settle so the cube rests on the table before we start.
    for _ in range(100):
        sim.step()
        recorder.write_frame(sim.observe())

    t0 = time.time()
    pick = Pick()
    result = pick(ctx, object="red cube")
    elapsed = time.time() - t0

    print(f"[demo] Pick -> {result!r}  (wall {elapsed:.2f}s)")
    print(f"[demo] artifacts: {result.artifacts}")

    recorder.end_episode(
        success=result.success,
        result={
            "reason": result.reason,
            "reason_detail": result.reason_detail,
            "artifacts": result.artifacts,
            "wall_seconds": elapsed,
        },
    )
    sim.close()
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
