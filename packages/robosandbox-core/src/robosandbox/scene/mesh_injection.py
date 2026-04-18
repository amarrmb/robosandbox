"""Inject a mesh SceneObject into an ``mujoco.MjSpec`` worldbody.

Shared by the URDF path (``scene.robot_loader.inject_scene_objects``) and
the unified built-in-arm path (once ``mjcf_builder.build_model`` learns to
route through MjSpec when a mesh object is present).

One ``MeshAsset`` becomes:

- N ``<asset><mesh>`` entries (visual + one per collision hull), prefixed
  with the object id to stay unique.
- One free-body at the object's pose.
- One visual geom (``contype=0, conaffinity=0``), if a visual mesh exists.
- One or more collision geoms (default contype/conaffinity), colour'd by
  ``asset.rgba`` so the viewer still shows the chunky hull shape when the
  visual mesh is absent.
- Mass split evenly across the collision geoms (MuJoCo sums geom mass
  into body mass; even split is a reasonable uniform-density approximation
  for rigid free-bodies at grasp scales).
"""

from __future__ import annotations

from typing import Any

import mujoco

from robosandbox.scene.mesh_conversion import MeshAsset
from robosandbox.types import SceneObject


def _xyzw_to_wxyz(q: tuple[float, float, float, float]) -> list[float]:
    x, y, z, w = q
    return [w, x, y, z]


def _mesh_name(obj_id: str, kind: str, idx: int | None = None) -> str:
    if idx is None:
        return f"{obj_id}__{kind}"
    return f"{obj_id}__{kind}_{idx}"


def inject_mesh_object(
    spec: mujoco.MjSpec, obj: SceneObject, asset: MeshAsset
) -> Any:
    """Add a mesh SceneObject (assets + body + geoms) to ``spec.worldbody``.

    Returns the created body for tests that want to poke at it.
    """
    if obj.id != asset.obj_id:
        raise ValueError(
            f"SceneObject id {obj.id!r} != MeshAsset obj_id {asset.obj_id!r}"
        )

    scale = list(asset.scale)

    # Register mesh assets. Names are prefixed with obj_id to stay unique
    # across multiple mesh objects in the same scene.
    visual_mesh_names: list[str] = []
    for i, p in enumerate(asset.visual_files):
        name = _mesh_name(obj.id, "visual", i if len(asset.visual_files) > 1 else None)
        spec.add_mesh(name=name, file=str(p), scale=scale)
        visual_mesh_names.append(name)

    collision_mesh_names: list[str] = []
    for i, p in enumerate(asset.collision_files):
        name = _mesh_name(obj.id, "col", i)
        spec.add_mesh(name=name, file=str(p), scale=scale)
        collision_mesh_names.append(name)

    # Create the free-body at the object's pose.
    body = spec.worldbody.add_body(
        name=obj.id,
        pos=list(obj.pose.xyz),
        quat=_xyzw_to_wxyz(obj.pose.quat_xyzw),
    )
    body.add_freejoint()

    # Visual geoms: no collision, no mass contribution.
    for vname in visual_mesh_names:
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=vname,
            rgba=list(asset.rgba),
            contype=0,
            conaffinity=0,
            group=2,  # MuJoCo convention: visual-only in group 2
            mass=0.0,
        )

    # Collision geoms: carry the mass and friction.
    # Distribute mass evenly across hulls. Use SceneObject.mass if the task
    # YAML overrode it (default SceneObject.mass = 0.1); otherwise fall back
    # to the asset's sidecar mass. The loader decides which wins — here we
    # just use whatever ``obj.mass`` arrives at.
    total_mass = float(obj.mass if obj.mass and obj.mass > 0 else asset.mass)
    n = len(collision_mesh_names)
    per_hull_mass = total_mass / n if n > 0 else 0.0

    # If the visual mesh is the same file as the single collision mesh (BYO
    # default), the visual geom above is redundant — but harmless (contype=0).
    # We keep the separate render/physics geoms for consistency with the
    # bundled path.
    for cname in collision_mesh_names:
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=cname,
            rgba=[asset.rgba[0], asset.rgba[1], asset.rgba[2], 0.0]
            if visual_mesh_names
            else list(asset.rgba),
            friction=list(asset.friction),
            mass=per_hull_mass,
            group=3,  # MuJoCo convention: collision-only in group 3
        )

    return body
