"""PlaceOn skill: move held object above a target, descend, release."""

from __future__ import annotations

from robosandbox.agent.context import AgentContext
from robosandbox.motion.ik import UnreachableError, plan_linear_cartesian
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
                "description": "Release gap above the target, in meters.",
                "default": 0.015,
            },
        },
        "required": ["target"],
    }

    def __call__(
        self, ctx: AgentContext, *, target: str, height_offset: float = 0.015
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
        target_half = 0.015
        held_half = 0.012
        release_z = z + target_half + held_half + height_offset
        release_pose = Pose(xyz=(x, y, release_z), quat_xyzw=_PALM_DOWN)
        approach_pose = pose_offset_z(release_pose, 0.1)

        # Traverse to above-target (Cartesian — cube is held, joint-space
        # interpolation swings the held cube loose).
        try:
            traj = plan_linear_cartesian(
                ctx.sim,
                start_joints=obs.robot_joints,
                target_pose=approach_pose,
                n_waypoints=100,
                dt=0.005,
                orientation="z_down",
            )
        except UnreachableError as e:
            return SkillResult(success=False, reason="unreachable", reason_detail=str(e))
        execute_trajectory(ctx, traj, gripper=1.0)

        # Descend to release point (Cartesian).
        obs_now = ctx.sim.observe()
        try:
            traj = plan_linear_cartesian(
                ctx.sim,
                start_joints=obs_now.robot_joints,
                target_pose=release_pose,
                n_waypoints=60,
                dt=0.005,
                orientation="z_down",
            )
        except UnreachableError as e:
            return SkillResult(success=False, reason="unreachable", reason_detail=str(e))
        execute_trajectory(ctx, traj, gripper=1.0)

        # Release.
        set_gripper(ctx, closed=0.0, hold_steps=40, ramp_steps=80)

        # Retract up — Cartesian so we pull straight up, not arc.
        obs_now = ctx.sim.observe()
        try:
            traj = plan_linear_cartesian(
                ctx.sim,
                start_joints=obs_now.robot_joints,
                target_pose=pose_offset_z(release_pose, 0.08),
                n_waypoints=50,
                dt=0.005,
                orientation="z_down",
            )
        except UnreachableError as e:
            return SkillResult(success=False, reason="unreachable", reason_detail=str(e))
        execute_trajectory(ctx, traj, gripper=0.0)

        return SkillResult(success=True, reason="placed")
