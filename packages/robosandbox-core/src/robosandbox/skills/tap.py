"""Tap skill: touch the top of a named object with a closed gripper, retreat.

Covers the "button_press" primitive from the task-diversity roadmap
without requiring a dedicated button scene primitive: any object with a
ground-truth 3D pose can be tapped. The gripper closes during the
downstroke so fingers don't splay on contact.

Useful as a building block for button-press tasks, UI simulations, and
as the simplest possible trajectory skill for teaching the protocol.
"""

from __future__ import annotations

from robosandbox.agent.context import AgentContext
from robosandbox.motion.ik import UnreachableError
from robosandbox.skills._common import execute_trajectory
from robosandbox.types import Pose, SkillResult


_PALM_DOWN = (1.0, 0.0, 0.0, 0.0)


class Tap:
    name = "tap"
    description = (
        "Tap the named object by touching its top with the end-effector and "
        "retreating. The gripper stays closed during contact."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "object": {
                "type": "string",
                "description": "Natural-language name of the object to tap, e.g. 'red button'.",
            },
            "hover_above": {
                "type": "number",
                "description": "Hover height before contact, in meters.",
                "default": 0.12,
            },
            "contact_above": {
                "type": "number",
                "description": "Residual clearance above object top at contact, in meters.",
                "default": 0.025,
            },
        },
        "required": ["object"],
    }

    def __call__(
        self,
        ctx: AgentContext,
        *,
        object: str,
        hover_above: float = 0.12,
        contact_above: float = 0.025,
    ) -> SkillResult:
        obs = ctx.sim.observe()
        detected = ctx.perception.locate(object, obs)
        if not detected:
            return SkillResult(
                success=False,
                reason="object_not_found",
                reason_detail=f"no match for target {object!r}",
            )
        t = max(detected, key=lambda d: d.confidence)
        if t.pose_3d is None:
            return SkillResult(success=False, reason="object_not_found")

        tx, ty, tz = t.pose_3d.xyz
        hover = Pose(xyz=(tx, ty, tz + hover_above), quat_xyzw=_PALM_DOWN)
        contact = Pose(xyz=(tx, ty, tz + contact_above), quat_xyzw=_PALM_DOWN)

        try:
            for pose in (hover, contact, hover):
                now = ctx.sim.observe()
                traj = ctx.motion.plan(
                    ctx.sim,
                    start_joints=now.robot_joints,
                    target_pose=pose,
                    constraints={"orientation": "z_down"},
                )
                execute_trajectory(ctx, traj, gripper=1.0)
        except UnreachableError as e:
            return SkillResult(success=False, reason="unreachable", reason_detail=str(e))

        return SkillResult(success=True, reason="tapped", artifacts={"object": object})
