"""Implement a new skill and drop it into the agent loop.

Skills are any object with ``name``, ``description``, ``parameters_schema``,
and a ``__call__(ctx, **kwargs) -> SkillResult``. No base class, no
registry required — just match the protocol and pass instances to
``Agent(skills=[...])``.

This example defines a ``tap`` skill that moves the end-effector down
onto a named object and back up, then plugs it into the agent loop.

Run:
    uv run python examples/custom_skill.py
"""

from __future__ import annotations

import numpy as np

from robosandbox.agent.agent import Agent
from robosandbox.agent.context import AgentContext
from robosandbox.agent.planner import SkillCall, StubPlanner
from robosandbox.grasp.analytic import AnalyticTopDown
from robosandbox.motion.ik import DLSMotionPlanner
from robosandbox.perception.ground_truth import GroundTruthPerception
from robosandbox.sim.mujoco_backend import MuJoCoBackend
from robosandbox.skills.home import Home
from robosandbox.types import Pose, Scene, SceneObject, SkillResult


class Tap:
    """Move the end-effector onto the named object and back up.

    Doesn't close the gripper. Useful as a button-press primitive and as
    a demonstration of the Skill protocol shape.
    """

    name = "tap"
    description = "Tap the named object by touching its top with the end-effector."
    parameters_schema = {
        "type": "object",
        "properties": {
            "object": {"type": "string", "description": "Scene object id to tap."}
        },
        "required": ["object"],
    }

    def __call__(self, ctx: AgentContext, object: str) -> SkillResult:
        from robosandbox.motion.ik import UnreachableError
        from robosandbox.skills._common import execute_trajectory

        obs = ctx.sim.observe()
        target_pose = obs.scene_objects.get(object)
        if target_pose is None:
            return SkillResult(success=False, reason="not_found", reason_detail=object)

        tx, ty, tz = target_pose.xyz
        above = Pose(xyz=(tx, ty, tz + 0.12), quat_xyzw=(1.0, 0.0, 0.0, 0.0))
        contact = Pose(xyz=(tx, ty, tz + 0.025), quat_xyzw=(1.0, 0.0, 0.0, 0.0))

        try:
            for pose in (above, contact, above):
                now = ctx.sim.observe()
                traj = ctx.motion.plan(
                    ctx.sim,
                    start_joints=now.robot_joints,
                    target_pose=pose,
                    constraints={"orientation": "z_down"},
                )
                execute_trajectory(ctx, traj, gripper=0.0)
        except UnreachableError as e:
            return SkillResult(success=False, reason="unreachable", reason_detail=str(e))
        return SkillResult(success=True, reason="tapped", artifacts={"object": object})


class TapStub:
    """Tiny planner: always emits one tap on a fixed object. Enough to
    drive the agent loop without needing natural-language parsing.
    """

    def plan(self, task, obs, prior_attempts):
        return ([SkillCall(name="tap", arguments={"object": "apple"})], 0)


def main() -> None:
    from importlib.resources import files
    from pathlib import Path

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
                pose=Pose(xyz=(0.4, 0.0, 0.06)),
                mass=0.0,
                mesh_sidecar=sidecar,
            ),
        ),
    )

    sim = MuJoCoBackend(render_size=(240, 320))
    sim.load(scene)
    try:
        ctx = AgentContext(
            sim=sim,
            perception=GroundTruthPerception(),
            grasp=AnalyticTopDown(),
            motion=DLSMotionPlanner(n_waypoints=160, dt=0.005),
        )
        result = Tap()(ctx, object="apple")
        print(f"tap result: {result}")

        # Same skill via the Agent loop + TapStub planner:
        agent = Agent(ctx=ctx, skills=[Tap(), Home()], planner=TapStub())
        ep = agent.run("tap the apple")
        print(f"agent episode success={ep.success}, steps={len(ep.steps)}")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
