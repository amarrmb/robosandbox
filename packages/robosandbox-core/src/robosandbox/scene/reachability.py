"""Authoring-time reachability check for scene objects.

Given a ``Scene``, verify that each object's **approach**, **grasp**, and
**lift** waypoints are IK-feasible from the configured robot's home
pose. Returns a list of ``UnreachableObject``s — objects the v0.1
``Pick`` skill would not be able to handle.

The check matches the Pick pipeline's default geometry:

- ``approach`` = object_xyz + (0, 0, grasp_height_offset + approach_offset)
- ``grasp``    = object_xyz + (0, 0, grasp_height_offset)
- ``lift``     = object_xyz + (0, 0, grasp_height_offset + lift_height)

All tested with ``z_down`` orientation — the standard top-down grasp
the v0.1 ``AnalyticTopDown`` grasp planner emits.

This is a fast, no-execution pre-flight check. It catches **kinematic**
infeasibility — reach envelope, joint limits, singularities near the
workspace edge. It does **not** catch dynamic failures (object slipped,
finger collided with neighbour). Those are the replan loop's job.

Typical use::

    from robosandbox.scene.reachability import check_scene_reachability
    from robosandbox.tasks.loader import load_builtin_task

    task = load_builtin_task("pick_cube_franka")
    warnings = check_scene_reachability(task.scene)
    for w in warnings:
        print(f"  - {w.id}: {w.reason} ({w.detail})")
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robosandbox.motion.ik import DLSMotionPlanner, UnreachableError
from robosandbox.types import Pose, Scene

# Defaults mirror the v0.1 Pick skill pipeline exactly. Keep in sync if
# AnalyticTopDown or skills.pick._LIFT_HEIGHT change.
_GRASP_HEIGHT_OFFSET = 0.013
_APPROACH_OFFSET = 0.1
_LIFT_HEIGHT = 0.18
# Quaternion for palm-down (rotate 180° about +X). Matches AnalyticTopDown.
_PALM_DOWN = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class UnreachableObject:
    """One object whose Pick waypoints are kinematically infeasible."""

    id: str                                  # scene-object id
    first_failed_phase: str                  # 'approach' | 'grasp' | 'lift'
    target_xyz: tuple[float, float, float]   # world-frame target the IK failed on
    detail: str                              # human-readable reason from the IK solver


def check_scene_reachability(
    scene: Scene,
    *,
    grasp_height_offset: float = _GRASP_HEIGHT_OFFSET,
    approach_offset: float = _APPROACH_OFFSET,
    lift_height: float = _LIFT_HEIGHT,
) -> list[UnreachableObject]:
    """Return the objects whose Pick pipeline would fail due to IK infeasibility.

    Empty list means every object looks reachable from the robot's home
    pose. A non-empty list is a warning: the Pick skill will fail on
    those objects at the listed phase before any physics runs.

    Silently skips non-pickable objects (currently: drawers, whose
    `Pick` isn't the right skill anyway).
    """
    # Deferred import: the MuJoCo backend pulls in a compiled extension,
    # callers who don't need reachability shouldn't pay for it.
    from robosandbox.sim.mujoco_backend import MuJoCoBackend

    # Uses the same multi-seed retry logic Pick uses at runtime so the
    # check's failure surface matches actual execution.
    planner = DLSMotionPlanner()
    # Small offscreen framebuffer — reachability doesn't need pretty renders.
    sim = MuJoCoBackend(render_size=(240, 320), camera="scene")
    try:
        sim.load(scene)
        home_joints = np.asarray(sim._robot.home_qpos, dtype=np.float64)  # noqa: SLF001

        out: list[UnreachableObject] = []
        for obj in scene.objects:
            if obj.kind == "drawer":
                # Drawers use a different skill; reachability of a drawer
                # body isn't the same as pickability.
                continue

            x, y, z = obj.pose.xyz
            phases: list[tuple[str, Pose]] = [
                ("approach", Pose(
                    xyz=(x, y, z + grasp_height_offset + approach_offset),
                    quat_xyzw=_PALM_DOWN,
                )),
                ("grasp", Pose(
                    xyz=(x, y, z + grasp_height_offset),
                    quat_xyzw=_PALM_DOWN,
                )),
                ("lift", Pose(
                    xyz=(x, y, z + grasp_height_offset + lift_height),
                    quat_xyzw=_PALM_DOWN,
                )),
            ]

            for phase_name, target in phases:
                try:
                    planner.plan(
                        sim, start_joints=home_joints,
                        target_pose=target,
                        constraints={"orientation": "z_down"},
                    )
                except UnreachableError as e:
                    out.append(UnreachableObject(
                        id=obj.id,
                        first_failed_phase=phase_name,
                        target_xyz=target.xyz,
                        detail=str(e),
                    ))
                    break  # first failure is enough — report the earliest problem
        return out
    finally:
        sim.close()


def format_warnings(warnings: list[UnreachableObject]) -> str:
    """Pretty-print reachability warnings for CLI / log output."""
    if not warnings:
        return "reachability: all objects look reachable"
    lines = [f"reachability: {len(warnings)} unreachable object(s):"]
    for w in warnings:
        x, y, z = w.target_xyz
        lines.append(
            f"  - {w.id}: {w.first_failed_phase} target ({x:.3f}, {y:.3f}, {z:.3f}) "
            f"— {w.detail}"
        )
    return "\n".join(lines)
