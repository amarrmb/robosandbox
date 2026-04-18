"""Mesh asset loading + (future) convex decomposition for SceneObject(kind="mesh").

A ``MeshAsset`` is the resolved, ready-to-inject description of one mesh
object: absolute paths to visual/collision mesh files, scale, mass,
friction, colour. Produced by:

- ``load_bundled_mesh(sidecar_path)`` — reads a per-object sidecar YAML
  next to pre-decomposed meshes (the shipped YCB path).
- ``load_byo_mesh(mesh_path, collision_mode, cache_dir)`` — reads a
  user-provided OBJ/STL; runs CoACD or convex-hull decomposition; caches
  the output hulls keyed by the mesh's sha256. (Step 6 in the mesh slice;
  stub raises NotImplementedError for now.)

Errors are subclasses of ``MeshConfigError`` so the task loader can catch
a single exception type and surface the cause cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


# -- exceptions --------------------------------------------------------------


class MeshConfigError(Exception):
    """Base for mesh-object loading errors."""


class MeshConfigNotFoundError(MeshConfigError):
    def __init__(self, tried: Path, what: str = "mesh sidecar") -> None:
        super().__init__(f"{what} not found: {tried}")
        self.tried = tried


class MeshConfigValidationError(MeshConfigError):
    def __init__(self, field: str, reason: str) -> None:
        super().__init__(f"Invalid mesh sidecar at {field!r}: {reason}")
        self.field = field
        self.reason = reason


# -- types -------------------------------------------------------------------


@dataclass(frozen=True)
class MeshAsset:
    """Everything the MjSpec injector needs to spawn one mesh object.

    Paths are absolute. All meshes in ``collision_files`` must be convex —
    the decomposer is responsible for that property before we get here.
    """

    obj_id: str
    visual_files: tuple[Path, ...]
    collision_files: tuple[Path, ...]
    scale: tuple[float, float, float]
    mass: float
    friction: tuple[float, float, float]
    rgba: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if not self.collision_files:
            raise MeshConfigValidationError(
                "collision_files", "must contain at least one convex hull mesh"
            )
        for p in self.visual_files + self.collision_files:
            if not p.is_absolute():
                raise MeshConfigValidationError(
                    "mesh path", f"expected absolute path, got {p!r}"
                )


# -- helpers -----------------------------------------------------------------


def _as_scale3(value: Any, field: str) -> tuple[float, float, float]:
    """Accept scalar or 3-list; return a 3-tuple."""
    if isinstance(value, (int, float)):
        s = float(value)
        return (s, s, s)
    if isinstance(value, list) and len(value) == 3 and all(
        isinstance(v, (int, float)) for v in value
    ):
        return (float(value[0]), float(value[1]), float(value[2]))
    raise MeshConfigValidationError(field, "must be a number or list of 3 numbers")


def _as_fixed_list(value: Any, field: str, n: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != n or not all(
        isinstance(v, (int, float)) for v in value
    ):
        raise MeshConfigValidationError(field, f"must be a list of {n} numbers")
    return tuple(float(v) for v in value)


def _require(d: dict[str, Any], key: str, parent: str) -> Any:
    if key not in d:
        raise MeshConfigValidationError(f"{parent}.{key}", "missing required field")
    return d[key]


def _resolve_mesh_file(raw: str, base_dir: Path, field: str) -> Path:
    """Resolve a mesh filename relative to the sidecar dir; must exist."""
    if not isinstance(raw, str):
        raise MeshConfigValidationError(field, "must be a string path")
    p = Path(raw)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    if not p.exists():
        raise MeshConfigNotFoundError(p, what=f"mesh file referenced by {field}")
    return p


# -- public API --------------------------------------------------------------


def load_bundled_mesh(sidecar_path: Path, obj_id: str) -> MeshAsset:
    """Load a per-object sidecar YAML that ships alongside pre-decomposed hulls.

    Sidecar schema::

        visual_mesh: mug_visual.obj            # relative to sidecar dir
        collision_meshes:                       # list of convex hulls
          - mug_hull_0.obj
          - mug_hull_1.obj
        scale: 1.0                              # scalar or [sx, sy, sz]
        mass: 0.15                              # kg
        friction: [1.5, 0.1, 0.01]              # sliding, torsional, rolling
        rgba: [0.9, 0.9, 0.9, 1.0]
    """
    sidecar_path = Path(sidecar_path)
    if not sidecar_path.exists():
        raise MeshConfigNotFoundError(sidecar_path, what="mesh sidecar")

    with sidecar_path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise MeshConfigValidationError("root", "sidecar must be a YAML mapping")

    base_dir = sidecar_path.parent

    visual_raw = _require(raw, "visual_mesh", "root")
    collision_raw = _require(raw, "collision_meshes", "root")
    if not isinstance(collision_raw, list) or not collision_raw:
        raise MeshConfigValidationError(
            "collision_meshes", "must be a non-empty list of mesh filenames"
        )

    visual_files = (_resolve_mesh_file(visual_raw, base_dir, "visual_mesh"),)
    collision_files = tuple(
        _resolve_mesh_file(c, base_dir, f"collision_meshes[{i}]")
        for i, c in enumerate(collision_raw)
    )

    scale = _as_scale3(raw.get("scale", 1.0), "scale")
    mass = float(raw.get("mass", 0.1))
    friction = _as_fixed_list(raw.get("friction", [1.5, 0.1, 0.01]), "friction", 3)
    rgba = _as_fixed_list(raw.get("rgba", [0.7, 0.7, 0.7, 1.0]), "rgba", 4)

    return MeshAsset(
        obj_id=obj_id,
        visual_files=visual_files,
        collision_files=collision_files,
        scale=scale,
        mass=mass,
        friction=friction,
        rgba=rgba,
    )


def load_byo_mesh(
    mesh_path: Path,
    obj_id: str,
    collision_mode: str = "coacd",
    cache_dir: Path | None = None,
) -> MeshAsset:
    """Load a user-provided OBJ/STL and decompose its collision geometry.

    Step 6 in the mesh slice — not yet implemented. Placeholder raises so
    the task loader has a symbol to call and a clear error until this lands.
    """
    raise NotImplementedError(
        "BYO mesh loading (collision_mode=%r) lands in step 6 of the mesh slice; "
        "use a bundled sidecar via mesh: '@builtin:objects/...' for now."
        % collision_mode
    )


def resolve_mesh_asset(obj: "SceneObject") -> MeshAsset:  # type: ignore[name-defined]
    """Dispatch a ``SceneObject(kind="mesh")`` to the right loader.

    Exactly one of ``obj.mesh_sidecar`` / ``obj.mesh_path`` must be set;
    raises ``MeshConfigError`` otherwise.
    """
    from robosandbox.types import SceneObject  # avoid circular at import time

    if not isinstance(obj, SceneObject):
        raise TypeError(f"expected SceneObject, got {type(obj).__name__}")
    if obj.kind != "mesh":
        raise MeshConfigError(
            f"resolve_mesh_asset called on SceneObject(kind={obj.kind!r}), expected 'mesh'"
        )
    has_sidecar = obj.mesh_sidecar is not None
    has_path = obj.mesh_path is not None
    if has_sidecar == has_path:
        raise MeshConfigError(
            f"mesh object {obj.id!r} must set exactly one of mesh_sidecar (bundled) "
            f"or mesh_path (bring-your-own); got "
            + ("both" if has_sidecar else "neither")
        )
    if has_sidecar:
        assert obj.mesh_sidecar is not None
        return load_bundled_mesh(obj.mesh_sidecar, obj_id=obj.id)
    assert obj.mesh_path is not None
    return load_byo_mesh(obj.mesh_path, obj_id=obj.id, collision_mode=obj.collision)
