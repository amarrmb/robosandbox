"""Pick skill: locate → grasp → approach → descend → close → lift.

Returns a SkillResult that the agent's EVALUATE state consumes.
"""

from __future__ import annotations

from robosandbox.agent.context import AgentContext
from robosandbox.motion.ik import UnreachableError
from robosandbox.skills._common import execute_trajectory, pose_offset_z, set_gripper
from robosandbox.types import SkillResult


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
        # ---- 1. Perceive ------------------------------------------------
        obs = ctx.sim.observe()
        detected = ctx.perception.locate(object, obs)
        if not detected:
            return SkillResult(
                success=False,
                reason="object_not_found",
                reason_detail=f"perception returned zero matches for query '{object}'",
            )
        target = max(detected, key=lambda d: d.confidence)

        # Record the starting object z so we can verify it was lifted.
        target_pose_0 = target.pose_3d
        if target_pose_0 is None:
            return SkillResult(
                success=False,
                reason="object_not_found",
                reason_detail="detected object has no 3D pose",
            )
        z0 = target_pose_0.xyz[2]

        # ---- 2. Propose a grasp ----------------------------------------
        grasps = ctx.grasp.plan(obs, target)
        if not grasps:
            return SkillResult(
                success=False,
                reason="no_grasp_found",
                reason_detail="grasp planner returned zero candidates",
            )
        grasp = grasps[0]
        approach_pose = pose_offset_z(grasp.pose, grasp.approach_offset)

        # ---- 3. Approach pre-grasp (gripper open) ----------------------
        try:
            traj = ctx.motion.plan(
                ctx.sim,
                start_joints=obs.robot_joints,
                target_pose=approach_pose,
                constraints={"orientation": "z_down"},
            )
        except UnreachableError as e:
            return SkillResult(
                success=False, reason="unreachable", reason_detail=str(e)
            )
        execute_trajectory(ctx, traj, gripper=0.0)  # 0.0 = open

        # ---- 4. Descend to grasp pose (still open) ---------------------
        obs_now = ctx.sim.observe()
        try:
            traj = ctx.motion.plan(
                ctx.sim,
                start_joints=obs_now.robot_joints,
                target_pose=grasp.pose,
                constraints={"orientation": "z_down"},
            )
        except UnreachableError as e:
            return SkillResult(
                success=False, reason="unreachable", reason_detail=f"descend: {e}"
            )
        execute_trajectory(ctx, traj, gripper=0.0)

        # ---- 5. Close gripper and hold ---------------------------------
        set_gripper(ctx, closed=1.0, hold_steps=60)

        # ---- 6. Lift ----------------------------------------------------
        obs_now = ctx.sim.observe()
        try:
            traj = ctx.motion.plan(
                ctx.sim,
                start_joints=obs_now.robot_joints,
                target_pose=pose_offset_z(grasp.pose, 0.18),
                constraints={"orientation": "z_down"},
            )
        except UnreachableError as e:
            return SkillResult(
                success=False, reason="unreachable", reason_detail=f"lift: {e}"
            )
        execute_trajectory(ctx, traj, gripper=1.0)

        # ---- 7. Verify --------------------------------------------------
        obs_final = ctx.sim.observe()
        obj_pose_final = obs_final.scene_objects.get(target.label)
        if obj_pose_final is None:
            return SkillResult(
                success=False,
                reason="verification_failed",
                reason_detail="target object vanished from sim state",
            )
        z_final = obj_pose_final.xyz[2]
        lifted_by = z_final - z0
        if lifted_by < 0.05:
            return SkillResult(
                success=False,
                reason="verification_failed",
                reason_detail=f"object only rose {lifted_by*1000:.1f}mm, expected >50mm",
                artifacts={"z_initial": z0, "z_final": z_final},
            )
        return SkillResult(
            success=True,
            reason="picked",
            artifacts={"z_initial": z0, "z_final": z_final, "lifted_m": lifted_by},
        )
