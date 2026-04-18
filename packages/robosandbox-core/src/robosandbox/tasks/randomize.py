"""Per-seed scene jittering for randomized benchmarks.

A task YAML may carry a ``randomize:`` block::

    randomize:
      xy_jitter: 0.05     # ± metres, uniform, applied to every object
      yaw_jitter: 0.5     # ± radians around z, applied to every object

``jitter_scene(scene, spec, seed)`` returns a new Scene with each
SceneObject's pose perturbed deterministically by ``seed``. Seed 0 is
the identity — the base scene is returned unchanged so existing
single-seed benchmarks are bit-exact preserved.

This is v0.1 of scene randomization: only object xy translation + yaw.
v0.2 can add rgba / size / mass / rotation jitter without changing the
call shape.
"""

from __future__ import annotations

import math
import random
from dataclasses import replace
from typing import Any

from robosandbox.types import Pose, Scene


def _yaw_from_quat_xyzw(q: tuple[float, float, float, float]) -> float:
    """Extract yaw (rotation about world z) from an (x, y, z, w) quaternion."""
    x, y, z, w = q
    # yaw = atan2(2 (wz + xy), 1 - 2 (y^2 + z^2))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _quat_xyzw_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    half = yaw * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _quat_mul_xyzw(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def jitter_scene(scene: Scene, spec: dict[str, Any] | None, seed: int) -> Scene:
    """Apply position/yaw jitter to every SceneObject's pose.

    seed == 0 returns ``scene`` unchanged — identity, for bit-exactness
    with pre-randomize benchmarks.
    """
    if not spec or seed == 0 or not scene.objects:
        return scene

    xy = float(spec.get("xy_jitter", 0.0))
    yaw = float(spec.get("yaw_jitter", 0.0))
    if xy <= 0.0 and yaw <= 0.0:
        return scene

    rng = random.Random(seed)
    new_objects = []
    for obj in scene.objects:
        dx = rng.uniform(-xy, xy) if xy > 0.0 else 0.0
        dy = rng.uniform(-xy, xy) if xy > 0.0 else 0.0
        dyaw = rng.uniform(-yaw, yaw) if yaw > 0.0 else 0.0
        x, y, z = obj.pose.xyz
        new_xyz = (x + dx, y + dy, z)
        if dyaw != 0.0:
            new_quat = _quat_mul_xyzw(_quat_xyzw_from_yaw(dyaw), obj.pose.quat_xyzw)
        else:
            new_quat = obj.pose.quat_xyzw
        new_pose = Pose(xyz=new_xyz, quat_xyzw=new_quat)
        new_objects.append(replace(obj, pose=new_pose))
    return replace(scene, objects=tuple(new_objects))
