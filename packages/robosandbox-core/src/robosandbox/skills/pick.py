"""Pick skill: locate → approach → descend → close → lift.

Single-attempt. On transient grasp failure the agent's replan loop
calls us again from a fresh state — that's the correct layer for
recovery (each replan retries with updated observations, not a blind
retry from a potentially-wedged sim state).
"""

from __future__ import annotations

from robosandbox.agent.context import AgentContext
from robosandbox.motion.ik import UnreachableError, plan_linear_cartesian
from robosandbox.skills._common import execute_trajectory, pose_offset_z, set_gripper
from robosandbox.types import SkillResult

_LIFT_HEIGHT = 0.18
_VERIFY_LIFT_MM = 50.0


class Pick:
    name = "pick"
    description = (
        "Pick up an object from the table. Use when the agent needs to "
        "grasp and lift a specific object by name."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "object": {
                "type": "string",
                "description": "Natural-language name of the object to pick, e.g. 'blue cube'.",
            },
        },
        "required": ["object"],
    }

    def __call__(self, ctx: AgentContext, *, object: str) -> SkillResult:
        obs = ctx.sim.observe()
        detected = ctx.perception.locate(object, obs)
        if not detected:
            return SkillResult(
                success=False,
                reason="object_not_found",
                reason_detail=f"perception returned zero matches for query '{object}'",
            )
        target = max(detected, key=lambda d: d.confidence)
        if target.pose_3d is None:
            return SkillResult(
                success=False,
                reason="object_not_found",
                reason_detail="detected object has no 3D pose",
            )
        z0 = target.pose_3d.xyz[2]

        grasps = ctx.grasp.plan(obs, target)
        if not grasps:
            return SkillResult(
                success=False,
                reason="no_grasp_found",
                reason_detail="grasp planner returned zero candidates",
            )
        grasp = grasps[0]
        approach_pose = pose_offset_z(grasp.pose, grasp.approach_offset)

        # Approach (joint-space is fine for long traversals).
        try:
            traj = ctx.motion.plan(
                ctx.sim,
                start_joints=obs.robot_joints,
                target_pose=approach_pose,
                constraints={"orientation": "z_down"},
            )
        except UnreachableError as e:
            return SkillResult(success=False, reason="unreachable", reason_detail=str(e))
        execute_trajectory(ctx, traj, gripper=0.0)

        # Descend (straight-line Cartesian — avoids sideways swing from
        # arbitrary joint-space interpolation into a top-down grasp).
        obs_now = ctx.sim.observe()
        try:
            traj = plan_linear_cartesian(
                ctx.sim,
                start_joints=obs_now.robot_joints,
                target_pose=grasp.pose,
                n_waypoints=60,
                dt=0.005,
                orientation="z_down",
            )
        except UnreachableError as e:
            return SkillResult(
                success=False, reason="unreachable", reason_detail=f"descend: {e}"
            )
        execute_trajectory(ctx, traj, gripper=0.0)

        # Close gripper.
        set_gripper(ctx, closed=1.0, hold_steps=60)

        # Lift straight up.
        obs_now = ctx.sim.observe()
        try:
            traj = plan_linear_cartesian(
                ctx.sim,
                start_joints=obs_now.robot_joints,
                target_pose=pose_offset_z(grasp.pose, _LIFT_HEIGHT),
                n_waypoints=80,
                dt=0.005,
                orientation="z_down",
            )
        except UnreachableError as e:
            return SkillResult(
                success=False, reason="unreachable", reason_detail=f"lift: {e}"
            )
        execute_trajectory(ctx, traj, gripper=1.0)

        # Verify.
        obs_final = ctx.sim.observe()
        obj_final = obs_final.scene_objects.get(target.label)
        if obj_final is None:
            return SkillResult(
                success=False,
                reason="verification_failed",
                reason_detail="target vanished",
            )
        z_final = obj_final.xyz[2]
        lifted_by = z_final - z0
        if lifted_by * 1000 < _VERIFY_LIFT_MM:
            return SkillResult(
                success=False,
                reason="verification_failed",
                reason_detail=f"rose {lifted_by*1000:.1f}mm; need >{_VERIFY_LIFT_MM:.0f}mm",
                artifacts={"z_initial": z0, "z_final": z_final},
            )
        return SkillResult(
            success=True,
            reason="picked",
            artifacts={"z_initial": z0, "z_final": z_final, "lifted_m": lifted_by},
        )
