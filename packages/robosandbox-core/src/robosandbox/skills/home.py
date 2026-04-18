"""Home skill: return to a known safe joint pose."""

from __future__ import annotations

import numpy as np

from robosandbox.agent.context import AgentContext
from robosandbox.types import JointTrajectory, SkillResult

_HOME_JOINTS = np.array([0.0, -0.4, 1.2, -0.8, 0.0, 0.0])


class Home:
    name = "home"
    description = "Return the arm to a neutral home pose."
    parameters_schema = {"type": "object", "properties": {}, "required": []}

    def __call__(self, ctx: AgentContext) -> SkillResult:
        obs = ctx.sim.observe()
        t = np.linspace(0.0, 1.0, 150).reshape(-1, 1)
        waypoints = (1.0 - t) * obs.robot_joints.reshape(1, -1) + t * _HOME_JOINTS.reshape(1, -1)
        traj = JointTrajectory(waypoints=waypoints, dt=0.005)
        for row in traj.waypoints:
            ctx.sim.step(target_joints=row)
            if ctx.on_step is not None:
                ctx.on_step()
        return SkillResult(success=True, reason="homed")
