"""Procedural port-geometry helpers for the connector-insertion task.

The insert_connector_franka.yaml hardcodes wall positions for a single
clearance (4 mm USB-C class). To support a tightness curriculum
(4 mm → 2 mm → 1.5 mm USB-A spec), we need to rebuild the wall + chamfer
positions from a target clearance value at scene-load time, without
forking the YAML for every clearance.

`rescale_port_clearance(scene, clearance_m)` returns a *new* Scene where
the 4 walls + 4 chamfer plates have been moved so the inner hole matches
the requested clearance. Plug dimensions are taken from the panda_with_plug
MJCF (hardcoded constants below).

Wall thickness, wall height, and chamfer length are preserved — only the
inward/outward offset of each wall and chamfer plate changes.

This is what the curriculum CLI flag `--clearance-m` calls under the hood.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import math

from robosandbox.types import Pose, Scene, SceneObject


# Plug dimensions (half-extents) — matches assets/robots/franka_panda/panda_with_plug.xml
PLUG_HX = 0.005     # 1 cm wide  (long-axis-perpendicular short side)
PLUG_HY = 0.0025    # 5 mm wide  (long-axis-perpendicular even shorter side)
# Wall thickness and height — matches the canonical YAML
WALL_T = 0.005      # 5 mm wall thickness
WALL_H = 0.025      # 2.5 cm wall half-height (so wall spans 5 cm tall in mjcf box)
CHAMFER_LEN = 0.006 # 6 mm chamfer plate
CHAMFER_T = 0.00075 # 0.75 mm thick chamfer (half-extent)


def rescale_port_clearance(scene: Scene, clearance_m: float) -> Scene:
    """Return a new Scene whose port walls + chamfer plates fit ``clearance_m``.

    Reads the existing port_target's xy as the port center; rebuilds the 9
    port pieces (base + 4 walls + 4 chamfer plates + the target marker is
    untouched) at the new clearance. Other scene objects (table, robot, etc.)
    pass through unchanged.

    No-op if no port_target is present.
    """
    port_center = None
    plane_z = None
    for obj in scene.objects:
        if obj.id == "port_target":
            port_center = (float(obj.pose.xyz[0]), float(obj.pose.xyz[1]))
            plane_z = float(obj.pose.xyz[2])
            break
    if port_center is None or plane_z is None:
        return scene  # task doesn't have a port_target → leave alone

    cx, cy = port_center
    base_z = plane_z - 2 * WALL_H        # base sits below walls; walls span [base_z, plane_z]
    hole_hx = PLUG_HX + clearance_m / 2.0
    hole_hy = PLUG_HY + clearance_m / 2.0
    outer_hx = hole_hx + WALL_T
    outer_hy = hole_hy + WALL_T

    # Chamfer plate centers: anchored at the wall top inner corner, offset
    # outward+up by (CHAMFER_LEN/2)*cos(45°) ≈ 0.00424 m each direction.
    qcos = math.cos(math.pi / 8)
    qsin = math.sin(math.pi / 8)
    chamf_off = (CHAMFER_LEN / 2.0) * math.cos(math.pi / 4)

    def _make(id_: str, size, xyz, quat=(0.0, 0.0, 0.0, 1.0), rgba=(0.5, 0.5, 0.5, 1.0)):
        return SceneObject(
            id=id_,
            kind="box",
            size=tuple(float(s) for s in size),
            pose=Pose(xyz=xyz, quat_xyzw=quat),
            mass=0.001,
            rgba=rgba,
            static=True,
        )

    new_port_pieces = [
        _make("port_base",
              size=(outer_hx, outer_hy, WALL_T / 2),
              xyz=(cx, cy, base_z + WALL_T / 2),
              rgba=(0.45, 0.45, 0.45, 1.0)),
        _make("port_wall_xp",
              size=(WALL_T / 2, outer_hy, WALL_H),
              xyz=(cx + hole_hx + WALL_T / 2, cy, base_z + WALL_H),
              rgba=(0.55, 0.55, 0.55, 1.0)),
        _make("port_wall_xn",
              size=(WALL_T / 2, outer_hy, WALL_H),
              xyz=(cx - hole_hx - WALL_T / 2, cy, base_z + WALL_H),
              rgba=(0.55, 0.55, 0.55, 1.0)),
        _make("port_wall_yp",
              size=(hole_hx, WALL_T / 2, WALL_H),
              xyz=(cx, cy + hole_hy + WALL_T / 2, base_z + WALL_H),
              rgba=(0.55, 0.55, 0.55, 1.0)),
        _make("port_wall_yn",
              size=(hole_hx, WALL_T / 2, WALL_H),
              xyz=(cx, cy - hole_hy - WALL_T / 2, base_z + WALL_H),
              rgba=(0.55, 0.55, 0.55, 1.0)),
        _make("chamfer_xp",
              size=(CHAMFER_LEN / 2, outer_hy, CHAMFER_T),
              xyz=(cx + hole_hx + chamf_off, cy, plane_z + chamf_off),
              quat=(0.0, -qsin, 0.0, qcos),
              rgba=(0.62, 0.62, 0.62, 1.0)),
        _make("chamfer_xn",
              size=(CHAMFER_LEN / 2, outer_hy, CHAMFER_T),
              xyz=(cx - hole_hx - chamf_off, cy, plane_z + chamf_off),
              quat=(0.0, qsin, 0.0, qcos),
              rgba=(0.62, 0.62, 0.62, 1.0)),
        _make("chamfer_yp",
              size=(outer_hx, CHAMFER_LEN / 2, CHAMFER_T),
              xyz=(cx, cy + hole_hy + chamf_off, plane_z + chamf_off),
              quat=(qsin, 0.0, 0.0, qcos),
              rgba=(0.62, 0.62, 0.62, 1.0)),
        _make("chamfer_yn",
              size=(outer_hx, CHAMFER_LEN / 2, CHAMFER_T),
              xyz=(cx, cy - hole_hy - chamf_off, plane_z + chamf_off),
              quat=(-qsin, 0.0, 0.0, qcos),
              rgba=(0.62, 0.62, 0.62, 1.0)),
    ]

    # Replace existing port_* and chamfer_* objects with the rescaled ones;
    # keep everything else (port_target, table, robot etc.) untouched.
    rescaled_ids = {p.id for p in new_port_pieces}
    new_objects = []
    for obj in scene.objects:
        if obj.id in rescaled_ids:
            continue  # drop the old version
        new_objects.append(obj)
    new_objects.extend(new_port_pieces)
    return replace(scene, objects=tuple(new_objects))
