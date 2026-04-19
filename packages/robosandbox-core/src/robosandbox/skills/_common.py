"""Shared helpers for skills."""

from __future__ import annotations

from robosandbox.agent.context import AgentContext
from robosandbox.types import JointTrajectory


def execute_trajectory(
    ctx: AgentContext,
    traj: JointTrajectory,
    gripper: float | None = None,
    steps_per_waypoint: int = 4,
    settle_steps: int = 40,
    tol: float = 2e-2,
) -> None:
    """Step the sim through a JointTrajectory, holding gripper at the given value.

    Each waypoint is held for ``steps_per_waypoint`` physics ticks so that
    the position-controlled actuators can actually track. After the final
    waypoint, we hold for ``settle_steps`` more ticks (or until the joint
    error falls below ``tol``) so callers observe a converged arm.
    """
    import numpy as _np

    for row in traj.waypoints:
        target = _np.asarray(row, dtype=_np.float64)
        for _ in range(steps_per_waypoint):
            ctx.sim.step(target_joints=target, gripper=gripper)
            if ctx.on_step is not None:
                ctx.on_step()

    # Final settle: hold the goal until the arm converges or we time out.
    goal = _np.asarray(traj.waypoints[-1], dtype=_np.float64)
    for _ in range(settle_steps):
        ctx.sim.step(target_joints=goal, gripper=gripper)
        if ctx.on_step is not None:
            ctx.on_step()
        err = float(_np.linalg.norm(ctx.sim.observe().robot_joints - goal))
        if err < tol:
            break


def set_gripper(
    ctx: AgentContext,
    closed: float,
    hold_steps: int = 80,
    ramp_steps: int = 120,
) -> None:
    """Command the gripper to `closed` (0=open, 1=closed) and hold.

    The command ramps linearly from the current value to the target over
    ``ramp_steps`` so position-controlled fingers don't snap shut and
    bat the object aside. After the ramp, holds for ``hold_steps`` more
    steps so friction builds before the caller tries to lift.
    """
    import numpy as _np

    joints_now = ctx.sim.observe().robot_joints.copy()
    current = ctx.sim.observe().gripper_width
    # Convert observed width back to [0,1] scale: 0 == max open (70mm), 1 == closed.
    start = float(_np.clip(1.0 - current / 0.07, 0.0, 1.0))
    for i in range(ramp_steps):
        alpha = (i + 1) / ramp_steps
        g = start + (closed - start) * alpha
        ctx.sim.step(target_joints=joints_now, gripper=g)
        if ctx.on_step is not None:
            ctx.on_step()
    for _ in range(hold_steps):
        ctx.sim.step(target_joints=joints_now, gripper=closed)
        if ctx.on_step is not None:
            ctx.on_step()


def pose_offset_z(p, dz: float):
    """Return a copy of Pose `p` with `dz` added to its z coordinate."""
    from robosandbox.types import Pose

    x, y, z = p.xyz
    return Pose(xyz=(x, y, z + dz), quat_xyzw=p.quat_xyzw)
