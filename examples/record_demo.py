"""Record one pick episode with LocalRecorder (MP4 + JSONL + result.json).

The recorder plugs into AgentContext. It gets frame events through the
``on_step`` callback and lifecycle events (start / frame / end) through
the standard RecordSink protocol.

Run:
    uv run python examples/record_demo.py --out-dir runs
"""

from __future__ import annotations

import argparse
from importlib.resources import files
from pathlib import Path

from robosandbox.agent.context import AgentContext
from robosandbox.grasp.analytic import AnalyticTopDown
from robosandbox.motion.ik import DLSMotionPlanner
from robosandbox.perception.ground_truth import GroundTruthPerception
from robosandbox.recorder.local import LocalRecorder
from robosandbox.sim.mujoco_backend import MuJoCoBackend
from robosandbox.skills.pick import Pick
from robosandbox.types import Pose, Scene, SceneObject


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("runs"))
    args = ap.parse_args()

    urdf = Path(str(files("robosandbox").joinpath("assets/robots/franka_panda/panda.xml")))
    cfg = Path(
        str(files("robosandbox").joinpath("assets/robots/franka_panda/panda.robosandbox.yaml"))
    )
    sidecar = Path(
        str(files("robosandbox").joinpath("assets/objects/ycb/013_apple/apple.robosandbox.yaml"))
    )
    scene = Scene(
        robot_urdf=urdf,
        robot_config=cfg,
        objects=(
            SceneObject(
                id="apple",
                kind="mesh",
                size=(0.0,),
                pose=Pose(xyz=(0.42, 0.0, 0.05)),
                mass=0.0,
                mesh_sidecar=sidecar,
            ),
        ),
    )

    sim = MuJoCoBackend(render_size=(240, 320))
    sim.load(scene)
    recorder = LocalRecorder(root=args.out_dir)
    episode_id = recorder.start_episode(
        task="pick up the apple",
        metadata={"sim_dt": sim.model.opt.timestep, "example": "record_demo.py"},
    )
    result = None
    try:
        ctx = AgentContext(
            sim=sim,
            perception=GroundTruthPerception(),
            grasp=AnalyticTopDown(),
            motion=DLSMotionPlanner(n_waypoints=160, dt=0.005),
            recorder=recorder,
            # Stream frames to the recorder on every sim step.
            on_step=lambda: recorder.write_frame(sim.observe()),
        )
        for _ in range(60):  # settle
            sim.step()
        result = Pick()(ctx, object="apple")
    finally:
        recorder.end_episode(
            success=bool(result and result.success),
            result={"reason": result.reason if result else "aborted"},
        )
        sim.close()

    print(f"recorded episode {episode_id}")
    ep_dir = next(args.out_dir.glob(f"*-{episode_id}"))
    print(f"  {ep_dir}")
    for f in sorted(ep_dir.iterdir()):
        print(f"    {f.name} ({f.stat().st_size} B)")


if __name__ == "__main__":
    main()
