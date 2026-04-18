"""PlaceOn skill: move held object above a target, descend, release."""

from __future__ import annotations

from robosandbox.agent.context import AgentContext
from robosandbox.motion.ik import UnreachableError
from robosandbox.skills._common import execute_trajectory, pose_offset_z, set_gripper
from robosandbox.types import Pose, SkillResult


_PALM_DOWN = (1.0, 0.0, 0.0, 0.0)


class PlaceOn:
    name = "place_on"
    description = (
        "Place the currently-held object on top of a target surface or object. "
        "Use after a successful Pick."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "Natural-language name of the target to place on, e.g. 'red plate'.",
            },
            "height_offset": {
                "type": "number",
                "description": "Release height above the target, in meters.",
                "default": 0.04,
            },
        },
        "required": ["target"],
    }

    def __call__(
        self, ctx: AgentContext, *, target: str, height_offset: float = 0.04
    ) -> SkillResult:
        obs = ctx.sim.observe()
        detected = ctx.perception.locate(target, obs)
        if not detected:
            return SkillResult(
                success=False,
                reason="object_not_found",
                reason_detail=f"no match for target '{target}'",
            )
        t = max(detected, key=lambda d: d.confidence)
        if t.pose_3d is None:
            return SkillResult(success=False, reason="object_not_found")

        x, y, z = t.pose_3d.xyz
        release_pose = Pose(
            xyz=(x, y, z + height_offset + 0.04), quat_xyzw=_PALM_DOWN
        )
        approach_pose = pose_offset_z(release_pose, 0.1)

        try:
            traj = ctx.motion.plan(
                ctx.sim,
                start_joints=obs.robot_joints,
                target_pose=approach_pose,
                constraints={"orientation": "z_down"},
            )
        except UnreachableError as e:
            return SkillResult(success=False, reason="unreachable", reason_detail=str(e))
        execute_trajectory(ctx, traj, gripper=1.0)

        obs_now = ctx.sim.observe()
        try:
            traj = ctx.motion.plan(
                ctx.sim,
                start_joints=obs_now.robot_joints,
                target_pose=release_pose,
                constraints={"orientation": "z_down"},
            )
        except UnreachableError as e:
            return SkillResult(success=False, reason="unreachable", reason_detail=str(e))
        execute_trajectory(ctx, traj, gripper=1.0)

        set_gripper(ctx, closed=0.0, hold_steps=40)

        obs_now = ctx.sim.observe()
        try:
            traj = ctx.motion.plan(
                ctx.sim,
                start_joints=obs_now.robot_joints,
                target_pose=pose_offset_z(release_pose, 0.1),
                constraints={"orientation": "z_down"},
            )
        except UnreachableError as e:
            return SkillResult(success=False, reason="unreachable", reason_detail=str(e))
        execute_trajectory(ctx, traj, gripper=0.0)

        return SkillResult(success=True, reason="placed")
