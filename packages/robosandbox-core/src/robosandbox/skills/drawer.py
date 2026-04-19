"""Drawer skills — OpenDrawer + CloseDrawer.

The drawer scene primitive (``SceneObject(kind="drawer")``) creates a
static cabinet + a sliding inner drawer + a handle body. Drawer opens
along world -x. The handle is short in x, wider in y, short in z — so
a top-down palm-down grip closes fingers around the x-extent of the
handle. This approach is far more reachable for 7-DoF arms than a
palm-forward grasp requiring a 90° wrist rotation.

Handle body is named ``<drawer_id>_handle`` and is observable through
``Observation.scene_objects``. The drawer body (sliding part) is named
``<drawer_id>``; its xy pose reports the drawer's open position.
"""

from __future__ import annotations

from robosandbox.agent.context import AgentContext
from robosandbox.motion.ik import UnreachableError, plan_linear_cartesian
from robosandbox.skills._common import execute_trajectory, set_gripper
from robosandbox.types import Pose, SkillResult

_PALM_DOWN = (1.0, 0.0, 0.0, 0.0)

_PULL_DISTANCE_OPEN = 0.10     # how far to pull when opening
_PUSH_DISTANCE_CLOSE = 0.10    # how far to push when closing
_HOVER_ABOVE = 0.13            # approach height above handle top (clears cabinet roof)


def _handle_id(drawer: str) -> str:
    return drawer if drawer.endswith("_handle") else f"{drawer}_handle"


def _locate_handle(ctx: AgentContext, drawer: str) -> Pose | None:
    obs = ctx.sim.observe()
    direct = obs.scene_objects.get(_handle_id(drawer))
    if direct is not None:
        return direct
    detected = ctx.perception.locate(_handle_id(drawer), obs)
    if detected:
        return max(detected, key=lambda d: d.confidence).pose_3d
    return None


def _pull_sequence(ctx: AgentContext, drawer: str, signed_dx: float) -> SkillResult:
    """Top-down grip on handle + xy-plane pull/push.

    Positive ``signed_dx`` pushes along +x (closes), negative pulls along -x
    (opens).
    """
    pose = _locate_handle(ctx, drawer)
    if pose is None:
        return SkillResult(
            success=False,
            reason="object_not_found",
            reason_detail=f"no handle body for drawer {drawer!r}",
        )
    hx, hy, hz = pose.xyz

    hover = Pose(xyz=(hx, hy, hz + _HOVER_ABOVE), quat_xyzw=_PALM_DOWN)
    grip = Pose(xyz=(hx, hy, hz + 0.005), quat_xyzw=_PALM_DOWN)
    pulled = Pose(xyz=(hx + signed_dx, hy, hz + 0.005), quat_xyzw=_PALM_DOWN)
    retreat = Pose(xyz=(pulled.xyz[0], pulled.xyz[1], pulled.xyz[2] + _HOVER_ABOVE),
                   quat_xyzw=_PALM_DOWN)

    obs = ctx.sim.observe()

    # 1. Hover above handle.
    try:
        traj = ctx.motion.plan(
            ctx.sim,
            start_joints=obs.robot_joints,
            target_pose=hover,
            constraints={"orientation": "z_down"},
        )
    except UnreachableError as e:
        return SkillResult(success=False, reason="unreachable", reason_detail=f"hover: {e}")
    execute_trajectory(ctx, traj, gripper=0.0)

    # 2. Descend straight down onto the handle.
    now = ctx.sim.observe()
    try:
        traj = plan_linear_cartesian(
            ctx.sim,
            start_joints=now.robot_joints,
            target_pose=grip,
            n_waypoints=40,
            dt=0.005,
            orientation="z_down",
        )
    except UnreachableError as e:
        return SkillResult(success=False, reason="unreachable", reason_detail=f"descend: {e}")
    execute_trajectory(ctx, traj, gripper=0.0)

    # 3. Close gripper on handle.
    set_gripper(ctx, closed=1.0, hold_steps=30, ramp_steps=60)

    # 4. Translate along x (pull or push), keeping height constant.
    now = ctx.sim.observe()
    try:
        traj = plan_linear_cartesian(
            ctx.sim,
            start_joints=now.robot_joints,
            target_pose=pulled,
            n_waypoints=60,
            dt=0.005,
            orientation="z_down",
        )
    except UnreachableError as e:
        return SkillResult(success=False, reason="unreachable", reason_detail=f"pull: {e}")
    execute_trajectory(ctx, traj, gripper=1.0)

    # 5. Release + retreat upward.
    set_gripper(ctx, closed=0.0, hold_steps=20, ramp_steps=40)
    now = ctx.sim.observe()
    try:
        traj = plan_linear_cartesian(
            ctx.sim,
            start_joints=now.robot_joints,
            target_pose=retreat,
            n_waypoints=30,
            dt=0.005,
            orientation="z_down",
        )
    except UnreachableError as e:
        return SkillResult(success=False, reason="unreachable", reason_detail=f"retreat: {e}")
    execute_trajectory(ctx, traj, gripper=0.0)

    return SkillResult(success=True, reason="pulled", artifacts={"drawer": drawer, "dx": signed_dx})


class OpenDrawer:
    name = "open_drawer"
    description = "Open a named drawer by pulling its handle toward the robot."
    parameters_schema = {
        "type": "object",
        "properties": {
            "drawer": {"type": "string", "description": "Scene id of the drawer to open."},
            "distance": {"type": "number", "default": _PULL_DISTANCE_OPEN},
        },
        "required": ["drawer"],
    }

    def __call__(
        self, ctx: AgentContext, *, drawer: str, distance: float = _PULL_DISTANCE_OPEN
    ) -> SkillResult:
        r = _pull_sequence(ctx, drawer, signed_dx=-float(distance))
        return SkillResult(
            success=r.success,
            reason="opened" if r.success else r.reason,
            reason_detail=r.reason_detail,
            artifacts=r.artifacts,
        )


class CloseDrawer:
    name = "close_drawer"
    description = "Close a named drawer by pushing its handle away from the robot."
    parameters_schema = {
        "type": "object",
        "properties": {
            "drawer": {"type": "string", "description": "Scene id of the drawer to close."},
            "distance": {"type": "number", "default": _PUSH_DISTANCE_CLOSE},
        },
        "required": ["drawer"],
    }

    def __call__(
        self, ctx: AgentContext, *, drawer: str, distance: float = _PUSH_DISTANCE_CLOSE
    ) -> SkillResult:
        r = _pull_sequence(ctx, drawer, signed_dx=+float(distance))
        return SkillResult(
            success=r.success,
            reason="closed" if r.success else r.reason,
            reason_detail=r.reason_detail,
            artifacts=r.artifacts,
        )
