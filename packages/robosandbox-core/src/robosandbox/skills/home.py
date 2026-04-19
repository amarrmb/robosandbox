"""Home skill: return to a known safe joint pose."""

from __future__ import annotations

import numpy as np

from robosandbox.agent.context import AgentContext
from robosandbox.types import JointTrajectory, SkillResult

# Last-resort fallback when the backend can't report its own home.
# Shape: the built-in 6-DOF arm's neutral pose. Used only as a zero-
# element fallback so the skill's broadcast math stays valid.
_FALLBACK_HOME = np.array([0.0, -0.4, 1.2, -0.8, 0.0, 0.0])


def _resolve_home(ctx: AgentContext, current_dof: int) -> np.ndarray:
    """Pick a home pose that matches the backend's DoF count.

    Order of preference:
      1. ``ctx.sim.home_qpos`` (public property on both MuJoCoBackend
         and RealRobotBackend subclasses).
      2. ``ctx.sim._robot.home_qpos`` / ``ctx.sim.config.home_qpos``
         (older / internal access paths — defensive).
      3. The 6-DOF built-in-arm fallback above. If that length doesn't
         match the backend, zeros sized to ``current_dof`` as a final
         safe default (stays in place instead of flinging the arm).
    """
    for attr in ("home_qpos",):
        val = getattr(ctx.sim, attr, None)
        if val is not None:
            return np.asarray(val, dtype=np.float64).ravel()
    robot = getattr(ctx.sim, "_robot", None)
    if robot is not None and getattr(robot, "home_qpos", None) is not None:
        return np.asarray(robot.home_qpos, dtype=np.float64).ravel()
    config = getattr(ctx.sim, "config", None)
    if config is not None and getattr(config, "home_qpos", None):
        return np.asarray(config.home_qpos, dtype=np.float64).ravel()
    if current_dof == len(_FALLBACK_HOME):
        return _FALLBACK_HOME.copy()
    return np.zeros(current_dof, dtype=np.float64)


class Home:
    name = "home"
    description = "Return the arm to a neutral home pose."
    parameters_schema = {"type": "object", "properties": {}, "required": []}

    def __call__(self, ctx: AgentContext) -> SkillResult:
        obs = ctx.sim.observe()
        start = np.asarray(obs.robot_joints, dtype=np.float64).ravel()
        home = _resolve_home(ctx, current_dof=start.shape[0])
        if home.shape != start.shape:
            return SkillResult(
                success=False,
                reason="home_dim_mismatch",
                reason_detail=(
                    f"backend reports n_dof={start.shape[0]} but home_qpos "
                    f"has length {home.shape[0]}; sidecar / config is out of sync"
                ),
            )
        t = np.linspace(0.0, 1.0, 150).reshape(-1, 1)
        waypoints = (1.0 - t) * start.reshape(1, -1) + t * home.reshape(1, -1)
        traj = JointTrajectory(waypoints=waypoints, dt=0.005)
        for row in traj.waypoints:
            ctx.sim.step(target_joints=row)
            if ctx.on_step is not None:
                ctx.on_step()
        return SkillResult(success=True, reason="homed")
