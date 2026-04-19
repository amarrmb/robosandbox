"""Procedural scene presets.

A preset function takes a seed and returns a ``Scene`` — the same Scene
type the rest of the stack consumes. Scenes built this way can be run
through the bench runner, recorded, fed to VLMs, etc. without any
special-case plumbing.

v0.1 ships one preset: :func:`tabletop_clutter`. Others (``kitchen_drawer``,
``desk_push``) can follow the same shape.
"""

from __future__ import annotations

import math
import random
from importlib.resources import files
from pathlib import Path

from robosandbox.tasks.loader import _ycb_short_name, list_builtin_ycb_objects
from robosandbox.types import Pose, Scene, SceneObject

_FRANKA_URDF = Path(str(files("robosandbox").joinpath("assets/robots/franka_panda/panda.xml")))
_FRANKA_CONFIG = Path(
    str(files("robosandbox").joinpath("assets/robots/franka_panda/panda.robosandbox.yaml"))
)


def _ycb_sidecar(ycb_id: str) -> Path:
    short = _ycb_short_name(ycb_id)
    return Path(
        str(
            files("robosandbox").joinpath(
                "assets", "objects", "ycb", ycb_id, f"{short}.robosandbox.yaml"
            )
        )
    )


def _sample_positions(
    n: int,
    rng: random.Random,
    center_xy: tuple[float, float],
    radius: float,
    min_spacing: float,
    max_tries: int = 200,
) -> list[tuple[float, float]]:
    """Rejection-sample ``n`` xy positions in a disk of ``radius`` around
    ``center_xy`` such that no two are within ``min_spacing`` of each other.
    """
    cx, cy = center_xy
    out: list[tuple[float, float]] = []
    for _ in range(n):
        for _ in range(max_tries):
            # Uniform sampling in a disk.
            r = radius * math.sqrt(rng.random())
            theta = 2 * math.pi * rng.random()
            x = cx + r * math.cos(theta)
            y = cy + r * math.sin(theta)
            if all((x - px) ** 2 + (y - py) ** 2 >= min_spacing**2 for px, py in out):
                out.append((x, y))
                break
        else:
            raise RuntimeError(
                f"Failed to sample {n} non-overlapping positions "
                f"(radius={radius}, min_spacing={min_spacing}). "
                f"Relax min_spacing or reduce n."
            )
    return out


def tabletop_clutter(
    n_objects: int = 5,
    *,
    seed: int = 0,
    ycb_pool: tuple[str, ...] | None = None,
    table_center_xy: tuple[float, float] = (0.40, 0.0),
    table_radius: float = 0.12,
    min_spacing: float = 0.08,
    object_drop_z: float = 0.07,
) -> Scene:
    """Return a Scene with the bundled Franka + N random YCB objects.

    Each seed produces a distinct, reproducible layout. Objects are
    sampled without replacement from ``ycb_pool`` (default: every bundled
    YCB object) and placed in a disk around ``table_center_xy`` such that
    their xy footprints stay ``min_spacing`` apart.

    The objects spawn slightly above the table (``object_drop_z``) and
    settle onto it during the usual benchmark-runner settling phase —
    guarantees they arrive at a valid resting pose even when the sampled
    pose lightly clips the table.
    """
    if n_objects < 0:
        raise ValueError(f"n_objects must be >= 0, got {n_objects}")

    pool = list(ycb_pool) if ycb_pool is not None else list_builtin_ycb_objects()
    if len(pool) < n_objects:
        raise ValueError(
            f"tabletop_clutter(n_objects={n_objects}) needs >= {n_objects} objects "
            f"in the pool; got {len(pool)}: {pool}"
        )

    rng = random.Random(seed)
    picked = rng.sample(pool, n_objects)
    positions = _sample_positions(
        n_objects,
        rng=rng,
        center_xy=table_center_xy,
        radius=table_radius,
        min_spacing=min_spacing,
    )

    objects = tuple(
        SceneObject(
            id=_ycb_short_name(ycb_id),
            kind="mesh",
            size=(0.0,),
            pose=Pose(xyz=(x, y, object_drop_z)),
            mass=0.0,  # use sidecar default
            mesh_sidecar=_ycb_sidecar(ycb_id),
        )
        for ycb_id, (x, y) in zip(picked, positions, strict=True)
    )

    return Scene(robot_urdf=_FRANKA_URDF, robot_config=_FRANKA_CONFIG, objects=objects)
