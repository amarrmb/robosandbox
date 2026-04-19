"""Per-seed scene jittering for randomized benchmarks.

A task YAML may carry a ``randomize:`` block::

    randomize:
      xy_jitter: 0.05     # ± metres, uniform, per object
      yaw_jitter: 0.5     # ± radians about z, per object
      rgba_jitter: 0.1    # max per-channel delta on (r,g,b); alpha unchanged; clamp [0,1]
      size_jitter: 0.15   # relative: each size component *= (1 + U(-s, +s)); clamp [0.3x, 3.0x]
      mass_jitter: 0.3    # relative: mass *= (1 + U(-m, +m)); clamp >= 1e-4

``jitter_scene(scene, spec, seed)`` returns a new Scene with each
SceneObject perturbed deterministically by ``seed``. Seed 0 is the
identity — the base scene is returned unchanged so existing
single-seed benchmarks are bit-exact preserved.

Per-kind semantics (documented here so callers can reason about what a
randomized benchmark will actually vary):

- ``box`` / ``sphere`` / ``cylinder`` (primitives): all knobs apply.
- ``mesh``: pose (xy, yaw), rgba, mass apply. size_jitter is SKIPPED
  because the geom size lives in the mesh asset (MuJoCo mesh ``scale``)
  and randomized scaling is not wired through the mesh injection path —
  silently scaling would desync collision hulls. A one-line warning is
  logged the first time per process.
- ``drawer``: ALL jitters beyond pose-level xy/yaw are SKIPPED. The
  drawer is articulated (cabinet + sliding inner body + handle); rgba
  jitter only touches one of the three geoms, and size/mass changes
  would break joint limits / handle offsets. Pose (xy + yaw) is still
  applied because it's a rigid transform of the whole assembly.

RNG draw order is stable: for every object, we consume uniforms in the
order [dx, dy, dyaw, dr, dg, db, d_size_0..N-1, d_mass]. Knobs that are
disabled still consume their draws when the per-kind policy would
otherwise apply them — so flipping a single knob on/off doesn't
re-key every downstream object's jitter.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import replace
from typing import Any

from robosandbox.types import Pose, Scene

_log = logging.getLogger(__name__)
_warned_mesh_size = False


def _yaw_from_quat_xyzw(q: tuple[float, float, float, float]) -> float:
    """Extract yaw (rotation about world z) from an (x, y, z, w) quaternion."""
    x, y, z, w = q
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


_SIZE_MIN_SCALE = 0.3
_SIZE_MAX_SCALE = 3.0
_MASS_FLOOR = 1e-4


def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def jitter_scene(scene: Scene, spec: dict[str, Any] | None, seed: int) -> Scene:
    """Apply deterministic per-object jitter (pose, rgba, size, mass).

    seed == 0 returns ``scene`` unchanged — identity, for bit-exactness
    with pre-randomize benchmarks.

    See module docstring for per-kind skip rules (mesh size / drawer
    rgba+size+mass).
    """
    if not spec or seed == 0 or not scene.objects:
        return scene

    xy = float(spec.get("xy_jitter", 0.0))
    yaw = float(spec.get("yaw_jitter", 0.0))
    rgba_amp = float(spec.get("rgba_jitter", 0.0))
    size_amp = float(spec.get("size_jitter", 0.0))
    mass_amp = float(spec.get("mass_jitter", 0.0))

    if xy <= 0.0 and yaw <= 0.0 and rgba_amp <= 0.0 and size_amp <= 0.0 and mass_amp <= 0.0:
        return scene

    global _warned_mesh_size
    rng = random.Random(seed)
    new_objects = []
    for obj in scene.objects:
        is_drawer = obj.kind == "drawer"
        is_mesh = obj.kind == "mesh"

        # --- Pose: xy + yaw jitter ---
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

        # --- RGBA jitter (skip on drawer) ---
        if rgba_amp > 0.0:
            dr = rng.uniform(-rgba_amp, rgba_amp)
            dg = rng.uniform(-rgba_amp, rgba_amp)
            db = rng.uniform(-rgba_amp, rgba_amp)
        else:
            dr = dg = db = 0.0
        if rgba_amp > 0.0 and not is_drawer:
            r, g, b, a = obj.rgba
            new_rgba = (
                _clamp(r + dr, 0.0, 1.0),
                _clamp(g + dg, 0.0, 1.0),
                _clamp(b + db, 0.0, 1.0),
                a,
            )
        else:
            new_rgba = obj.rgba

        # --- Size jitter (skip on drawer; skip on mesh with warn-once) ---
        if size_amp > 0.0:
            size_deltas = [rng.uniform(-size_amp, size_amp) for _ in obj.size]
        else:
            size_deltas = [0.0 for _ in obj.size]
        if size_amp > 0.0 and not is_drawer and not is_mesh:
            new_size = tuple(
                _clamp(
                    s * (1.0 + d),
                    _SIZE_MIN_SCALE * s,
                    _SIZE_MAX_SCALE * s,
                )
                for s, d in zip(obj.size, size_deltas)
            )
        else:
            new_size = obj.size
            if size_amp > 0.0 and is_mesh and not _warned_mesh_size:
                _log.warning(
                    "size_jitter is not supported for kind='mesh' (object "
                    "%r); skipping size jitter for all mesh objects.",
                    obj.id,
                )
                _warned_mesh_size = True

        # --- Mass jitter (skip on drawer) ---
        if mass_amp > 0.0:
            dmass = rng.uniform(-mass_amp, mass_amp)
        else:
            dmass = 0.0
        if mass_amp > 0.0 and not is_drawer:
            new_mass = max(obj.mass * (1.0 + dmass), _MASS_FLOOR)
        else:
            new_mass = obj.mass

        new_objects.append(
            replace(obj, pose=new_pose, rgba=new_rgba, size=new_size, mass=new_mass)
        )
    return replace(scene, objects=tuple(new_objects))
