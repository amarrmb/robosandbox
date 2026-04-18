"""Pour skill: move the held object above a target container and hold.

Physics for fluid dynamics isn't on the v0.1 roadmap, so "pouring" in
this sandbox is a motion pantomime — the end-effector carries the held
object above the target, holds briefly (so downstream perception/VLM
sees a distinct "pouring" moment), and retreats. Adding a real wrist
tilt requires quaternion-interpolating Cartesian IK, deferred to v0.2.

This motion is still useful as a planner decomposition target:

    "pour the mustard into the bowl"
        -> [Pick(mustard), Pour(target=bowl), Home]

and the pour skill is the distinct symbol the planner emits for that
step, independent of what exact motion it makes.

Precondition: the gripper is already closed on something. Callers
should chain pick -> pour. The skill doesn't verify the grip; if
nothing is held, the motion still runs (cosmetically silly, but safe).
"""

from __future__ import annotations

from robosandbox.agent.context import AgentContext
from robosandbox.motion.ik import UnreachableError, plan_linear_cartesian
from robosandbox.skills._common import execute_trajectory, pose_offset_z
from robosandbox.types import Pose, SkillResult


_PALM_DOWN_XYZW = (1.0, 0.0, 0.0, 0.0)


class Pour:
    name = "pour"
    description = (
        "Pour the currently-held object into a target container. Position "
        "the end-effector above the target and hold. Use after picking up "
        "a pourable object."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "Natural-language name of the container to pour into, e.g. 'bowl'.",
            },
            "height_above": {
                "type": "number",
                "description": "Height above the target to hold at, in meters.",
                "default": 0.15,
            },
            "hold_steps": {
                "type": "integer",
                "description": "Physics ticks to dwell at the pour pose.",
                "default": 60,
            },
        },
        "required": ["target"],
    }

    def __call__(
        self,
        ctx: AgentContext,
        *,
        target: str,
        height_above: float = 0.15,
        hold_steps: int = 60,
    ) -> SkillResult:
        obs = ctx.sim.observe()
        detected = ctx.perception.locate(target, obs)
        if not detected:
            return SkillResult(
                success=False,
                reason="object_not_found",
                reason_detail=f"no match for target {target!r}",
            )
        t = max(detected, key=lambda d: d.confidence)
        if t.pose_3d is None:
            return SkillResult(success=False, reason="object_not_found")

        x, y, z = t.pose_3d.xyz
        pour_pose = Pose(xyz=(x, y, z + height_above), quat_xyzw=_PALM_DOWN_XYZW)

        # Traverse to above-target, palm down, held object in gripper.
        try:
            traj = plan_linear_cartesian(
                ctx.sim,
                start_joints=obs.robot_joints,
                target_pose=pour_pose,
                n_waypoints=100,
                dt=0.005,
                orientation="z_down",
            )
        except UnreachableError as e:
            return SkillResult(success=False, reason="unreachable", reason_detail=str(e))
        execute_trajectory(ctx, traj, gripper=1.0)

        # Dwell — the "pour" moment. Distinguishes the skill's pose from a
        # transient pass-through for downstream perception/recorders.
        for _ in range(hold_steps):
            ctx.sim.step(target_joints=traj.waypoints[-1], gripper=1.0)
            if ctx.on_step is not None:
                ctx.on_step()

        # Retreat up so the next skill starts from a clean pose.
        obs_now = ctx.sim.observe()
        try:
            traj = plan_linear_cartesian(
                ctx.sim,
                start_joints=obs_now.robot_joints,
                target_pose=pose_offset_z(pour_pose, 0.08),
                n_waypoints=50,
                dt=0.005,
                orientation="z_down",
            )
        except UnreachableError as e:
            return SkillResult(
                success=False, reason="unreachable", reason_detail=f"retreat: {e}"
            )
        execute_trajectory(ctx, traj, gripper=1.0)

        return SkillResult(
            success=True,
            reason="poured",
            artifacts={"target": target},
        )
