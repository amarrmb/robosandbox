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

import hashlib
import os
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


_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "robosandbox" / "mesh_hulls"

_BYO_VALID_MODES = ("coacd", "hull")


def _byo_cache_key(mesh_path: Path, mode: str) -> str:
    """Deterministic cache key: sha256(mesh bytes) + mode."""
    h = hashlib.sha256()
    with mesh_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return f"{h.hexdigest()[:32]}_{mode}"


def _read_hulls_from_cache(cache_subdir: Path) -> list[Path] | None:
    """If ``cache_subdir`` holds a complete cached decomposition, return hull
    paths in order. Otherwise return None (caller should recompute).
    """
    manifest = cache_subdir / "manifest.yaml"
    if not manifest.exists():
        return None
    try:
        data = yaml.safe_load(manifest.read_text()) or {}
    except yaml.YAMLError:
        return None
    names = data.get("collision_meshes")
    if not isinstance(names, list) or not names:
        return None
    paths = [cache_subdir / n for n in names]
    if not all(p.exists() for p in paths):
        return None
    return paths


def _write_hulls_to_cache(cache_subdir: Path, hull_paths: list[Path]) -> None:
    manifest = cache_subdir / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {"collision_meshes": [p.name for p in hull_paths]},
            sort_keys=False,
        )
    )


def _decompose_coacd(mesh_path: Path, out_dir: Path) -> list[Path]:
    """Run CoACD and write per-hull OBJs into ``out_dir``. Requires ``coacd``."""
    try:
        import coacd  # type: ignore
        import numpy as np
        import trimesh
    except ImportError as e:  # pragma: no cover — covered by installed-deps check
        raise MeshConfigError(
            "BYO mesh collision='coacd' requires the 'robosandbox[meshes]' extra. "
            f"Install with: pip install 'robosandbox[meshes]'  ({e})"
        ) from e

    mesh = trimesh.load(str(mesh_path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise MeshConfigError(f"mesh file did not load as a single mesh: {mesh_path}")

    co = coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    parts = coacd.run_coacd(co, threshold=0.2)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, (verts, faces) in enumerate(parts):
        hull = trimesh.Trimesh(
            vertices=np.asarray(verts), faces=np.asarray(faces), process=False
        )
        p = out_dir / f"hull_{i}.obj"
        hull.export(str(p), file_type="obj")
        paths.append(p)
    if not paths:
        raise MeshConfigError(f"CoACD produced no hulls for {mesh_path}")
    return paths


def _decompose_hull_passthrough(mesh_path: Path, out_dir: Path) -> list[Path]:
    """Single-hull passthrough: copy the input mesh as the one collision mesh.

    Honest contract: the user asserts the mesh is already convex. No hull
    computation happens here — MuJoCo will treat the mesh as-is for contact.
    This is the no-CoACD-needed fallback for convex BYO meshes.
    """
    import shutil

    out_dir.mkdir(parents=True, exist_ok=True)
    # Preserve the original file's suffix so MuJoCo's loader picks the right
    # parser (.obj / .stl / .ply).
    dest = out_dir / f"hull_0{mesh_path.suffix.lower()}"
    shutil.copyfile(mesh_path, dest)
    return [dest]


def load_byo_mesh(
    mesh_path: Path,
    obj_id: str,
    collision_mode: str = "coacd",
    cache_dir: Path | None = None,
    mass: float = 0.1,
    friction: tuple[float, float, float] = (1.5, 0.1, 0.01),
    rgba: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0),
    scale: float | tuple[float, float, float] = 1.0,
) -> MeshAsset:
    """Load a user-provided OBJ/STL and resolve its collision meshes.

    ``collision_mode``:
      - ``"coacd"``: run CoACD for convex decomposition (requires the
        ``robosandbox[meshes]`` extra). Multi-hull; correct for concave objects.
      - ``"hull"``: passthrough — treat the input mesh as already convex.
        Fast, no extra dep, but silently broken for concave meshes. Use only
        when you know the mesh is convex.

    Decomposition result is cached at
    ``~/.cache/robosandbox/mesh_hulls/<sha32>_<mode>/`` keyed by the mesh
    file's sha256 so re-runs skip the expensive CoACD call. Set
    ``cache_dir`` to override (tests use ``tmp_path``).
    """
    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        raise MeshConfigNotFoundError(mesh_path, what="mesh file")
    if collision_mode not in _BYO_VALID_MODES:
        raise MeshConfigValidationError(
            "collision", f"must be one of {_BYO_VALID_MODES}, got {collision_mode!r}"
        )

    cache_root = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR
    key = _byo_cache_key(mesh_path, collision_mode)
    cache_subdir = cache_root / key

    cached = _read_hulls_from_cache(cache_subdir)
    if cached is None:
        cache_subdir.mkdir(parents=True, exist_ok=True)
        if collision_mode == "coacd":
            hull_paths = _decompose_coacd(mesh_path, cache_subdir)
        else:
            hull_paths = _decompose_hull_passthrough(mesh_path, cache_subdir)
        _write_hulls_to_cache(cache_subdir, hull_paths)
    else:
        hull_paths = cached

    scale3 = _as_scale3(scale if not isinstance(scale, (int, float)) else float(scale), "scale")

    return MeshAsset(
        obj_id=obj_id,
        visual_files=(mesh_path.resolve(),),
        collision_files=tuple(p.resolve() for p in hull_paths),
        scale=scale3,
        mass=float(mass),
        friction=tuple(float(v) for v in friction),
        rgba=tuple(float(v) for v in rgba),
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
    # BYO: use the SceneObject's colour/mass as the asset defaults. mass=0
    # means "unspecified", so fall back to a sensible default — inject_mesh_
    # object applies obj.mass later anyway when > 0.
    byo_mass = obj.mass if obj.mass and obj.mass > 0 else 0.1
    return load_byo_mesh(
        obj.mesh_path,
        obj_id=obj.id,
        collision_mode=obj.collision,
        mass=byo_mass,
        rgba=obj.rgba,
    )
