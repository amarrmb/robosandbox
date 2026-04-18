"""Push skill: nudge an object on the table in a compass direction.

Top-down strategy: the gripper closes to a "paddle" (fingers together),
descends behind the object, then translates in the target direction to
drag the object along.
"""

from __future__ import annotations

import numpy as np

from robosandbox.agent.context import AgentContext
from robosandbox.motion.ik import UnreachableError, plan_linear_cartesian
from robosandbox.skills._common import execute_trajectory, set_gripper
from robosandbox.types import Pose, SkillResult

_PALM_DOWN = (1.0, 0.0, 0.0, 0.0)

_DIR_VECS = {
    "forward": (1.0, 0.0),
    "back": (-1.0, 0.0),
    "backward": (-1.0, 0.0),
    "left": (0.0, -1.0),
    "right": (0.0, 1.0),
    "north": (1.0, 0.0),
    "south": (-1.0, 0.0),
    "east": (0.0, 1.0),
    "west": (0.0, -1.0),
}

_DEFAULT_DISTANCE = 0.08      # meters
_STANDOFF = 0.035             # how far behind the object to land gripper
_PUSH_HEIGHT_ABOVE_TABLE = 0.01


class Push:
    name = "push"
    description = (
        "Slide an object on the table in a compass direction. "
        "Use for clearing a path or nudging objects without lifting."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "object": {"type": "string", "description": "Object to push."},
            "direction": {
                "type": "string",
                "enum": list(_DIR_VECS),
                "description": "Direction to push (compass term).",
            },
            "distance": {
                "type": "number",
                "description": "Push distance in meters.",
                "default": _DEFAULT_DISTANCE,
            },
        },
        "required": ["object", "direction"],
    }

    def __call__(
        self,
        ctx: AgentContext,
        *,
        object: str,
        direction: str,
        distance: float = _DEFAULT_DISTANCE,
    ) -> SkillResult:
        d = direction.strip().lower()
        if d not in _DIR_VECS:
            return SkillResult(
                success=False,
                reason="bad_arguments",
                reason_detail=f"unknown direction {direction!r}; must be one of {list(_DIR_VECS)}",
            )
        dx, dy = _DIR_VECS[d]

        obs = ctx.sim.observe()
        detected = ctx.perception.locate(object, obs)
        if not detected:
            return SkillResult(
                success=False,
                reason="object_not_found",
                reason_detail=f"no match for {object!r}",
            )
        target = max(detected, key=lambda t: t.confidence)
        if target.pose_3d is None:
            return SkillResult(success=False, reason="object_not_found")
        x0, y0, z0 = target.pose_3d.xyz

        # Stand-off pose: behind the object along -direction, at the
        # level of the object's centre so the paddle strikes the side.
        table_top = 0.04
        push_z = table_top + _PUSH_HEIGHT_ABOVE_TABLE + 0.04  # fingertip clearance
        behind = Pose(
            xyz=(x0 - dx * _STANDOFF, y0 - dy * _STANDOFF, push_z),
            quat_xyzw=_PALM_DOWN,
        )
        approach = Pose(
            xyz=(behind.xyz[0], behind.xyz[1], behind.xyz[2] + 0.08),
            quat_xyzw=_PALM_DOWN,
        )

        # Close the gripper first so fingers form a paddle.
        set_gripper(ctx, closed=1.0, hold_steps=40, ramp_steps=60)

        # Move above the stand-off.
        try:
            traj = ctx.motion.plan(
                ctx.sim,
                start_joints=obs.robot_joints,
                target_pose=approach,
                constraints={"orientation": "z_down"},
            )
        except UnreachableError as e:
            return SkillResult(success=False, reason="unreachable", reason_detail=str(e))
        execute_trajectory(ctx, traj, gripper=1.0)

        # Descend to stand-off height.
        obs_now = ctx.sim.observe()
        try:
            traj = plan_linear_cartesian(
                ctx.sim,
                start_joints=obs_now.robot_joints,
                target_pose=behind,
                n_waypoints=50,
                dt=0.005,
                orientation="z_down",
            )
        except UnreachableError as e:
            return SkillResult(success=False, reason="unreachable", reason_detail=str(e))
        execute_trajectory(ctx, traj, gripper=1.0)

        # Push: translate horizontally by `distance`.
        end_pose = Pose(
            xyz=(x0 + dx * distance, y0 + dy * distance, push_z),
            quat_xyzw=_PALM_DOWN,
        )
        obs_now = ctx.sim.observe()
        try:
            traj = plan_linear_cartesian(
                ctx.sim,
                start_joints=obs_now.robot_joints,
                target_pose=end_pose,
                n_waypoints=80,
                dt=0.005,
                orientation="z_down",
            )
        except UnreachableError as e:
            return SkillResult(success=False, reason="unreachable", reason_detail=str(e))
        execute_trajectory(ctx, traj, gripper=1.0)

        # Verify: object moved in the expected direction by some amount.
        obs_final = ctx.sim.observe()
        obj_final = obs_final.scene_objects.get(target.label)
        if obj_final is None:
            return SkillResult(success=False, reason="verification_failed")
        displacement = np.array(obj_final.xyz[:2]) - np.array([x0, y0])
        dot = float(displacement @ np.array([dx, dy]))
        if dot < 0.01:
            return SkillResult(
                success=False,
                reason="verification_failed",
                reason_detail=f"object only moved {dot*1000:.1f}mm in the requested direction",
                artifacts={"displacement": displacement.tolist()},
            )
        return SkillResult(
            success=True,
            reason="pushed",
            artifacts={
                "displacement": displacement.tolist(),
                "distance_m": float(np.linalg.norm(displacement)),
            },
        )
