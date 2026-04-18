"""Load benchmark tasks from YAML definitions.

A Task bundles a Scene, a natural-language prompt, and a success
criterion. The criterion is evaluated against the final Observation:
it's a dict shape the runner understands, not executable code (so
tasks can be defined declaratively and versioned safely).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from robosandbox.types import Pose, Scene, SceneObject


_BUILTIN_DIR = Path(__file__).parent / "definitions"


@dataclass
class SuccessCriterion:
    """Declarative success check.

    Supported shapes:
      - {"kind": "lifted", "object": id, "min_mm": 50}
      - {"kind": "moved_above", "object": id, "target": id, "xy_tol": 0.05, "min_dz": 0.015}
      - {"kind": "displaced", "object": id, "direction": "forward|back|left|right", "min_mm": 30}
      - {"kind": "all", "checks": [...]}      (every sub-criterion must hold)
      - {"kind": "any", "checks": [...]}      (at least one must hold)
    """

    data: dict[str, Any]


@dataclass
class Task:
    name: str
    scene: Scene
    prompt: str
    success: SuccessCriterion
    seed_note: str = ""
    randomize: dict[str, Any] | None = None


def _pose_from_dict(d: dict[str, Any]) -> Pose:
    xyz = tuple(float(v) for v in d.get("xyz", [0.0, 0.0, 0.0]))
    q = d.get("quat_xyzw", [0.0, 0.0, 0.0, 1.0])
    qx, qy, qz, qw = [float(v) for v in q]
    return Pose(xyz=xyz, quat_xyzw=(qx, qy, qz, qw))


def _object_from_dict(d: dict[str, Any], base_dir: Path) -> SceneObject:
    kind = str(d.get("kind", "box"))
    pose = _pose_from_dict(d.get("pose", {}))

    if kind == "mesh":
        # Mesh objects set exactly one of mesh (bundled sidecar ref) or
        # mesh_path (bring-your-own). mass=0 means "use sidecar default";
        # callers can override per-task with an explicit mass.
        mesh_ref = d.get("mesh")
        mesh_path_ref = d.get("mesh_path")
        if bool(mesh_ref) == bool(mesh_path_ref):
            raise ValueError(
                f"Mesh SceneObject {d.get('id')!r} must set exactly one of "
                f"'mesh' (bundled sidecar) or 'mesh_path' (BYO); got "
                + ("both" if mesh_ref else "neither")
            )
        mesh_sidecar: Path | None = None
        mesh_path: Path | None = None
        if mesh_ref:
            mesh_sidecar = _resolve_asset_path(str(mesh_ref), base_dir)
        else:
            mesh_path = _resolve_asset_path(str(mesh_path_ref), base_dir)
        return SceneObject(
            id=str(d["id"]),
            kind="mesh",
            size=(0.0,),  # unused for mesh; present so the dataclass is valid
            pose=pose,
            mass=float(d["mass"]) if "mass" in d else 0.0,
            rgba=tuple(float(v) for v in d.get("rgba", [0.7, 0.7, 0.7, 1.0])),
            mesh_sidecar=mesh_sidecar,
            mesh_path=mesh_path,
            collision=str(d.get("collision", "coacd")),
        )

    if kind == "drawer":
        return SceneObject(
            id=str(d["id"]),
            kind="drawer",
            size=tuple(float(v) for v in d.get("size", [0.15, 0.12, 0.05])),
            pose=pose,
            rgba=tuple(float(v) for v in d.get("rgba", [0.55, 0.35, 0.2, 1.0])),
            drawer_max_open=float(d.get("drawer_max_open", 0.12)),
        )

    # Primitive objects (unchanged behaviour).
    return SceneObject(
        id=str(d["id"]),
        kind=kind,
        size=tuple(float(v) for v in d.get("size", [0.012, 0.012, 0.012])),
        pose=pose,
        mass=float(d.get("mass", 0.05)),
        rgba=tuple(float(v) for v in d.get("rgba", [0.7, 0.7, 0.7, 1.0])),
    )


_BUILTIN_PREFIX = "@builtin:"
_YCB_PREFIX = "@ycb:"


def _ycb_short_name(ycb_id: str) -> str:
    """Convention: YCB ids are ``NNN_<short_name>`` (e.g. ``025_mug``).

    The sidecar for each bundled YCB object lives at
    ``assets/objects/ycb/<ycb_id>/<short_name>.robosandbox.yaml``. The
    short name is everything after the first underscore.
    """
    parts = ycb_id.split("_", 1)
    if len(parts) != 2 or not parts[0].isdigit():
        raise ValueError(
            f"YCB id must look like 'NNN_short_name' (e.g. '025_mug'), got {ycb_id!r}"
        )
    return parts[1]


def _resolve_asset_path(raw: str, base_dir: Path) -> Path:
    """Resolve a task-YAML asset reference.

    Order:
      1. ``@ycb:<ycb_id>`` -> bundled YCB sidecar
         (``assets/objects/ycb/<ycb_id>/<short_name>.robosandbox.yaml``)
      2. ``@builtin:<rel>``  -> packaged assets (``robosandbox/assets/<rel>``)
      3. absolute path       -> ``Path(raw)`` unchanged
      4. relative path       -> ``(base_dir / raw).resolve()``

    Raises FileNotFoundError if the resolved path doesn't exist, so broken
    task YAMLs fail loudly at load time rather than later during compile.
    """
    from importlib.resources import files

    if raw.startswith(_YCB_PREFIX):
        ycb_id = raw[len(_YCB_PREFIX) :].lstrip("/")
        short = _ycb_short_name(ycb_id)
        resolved = Path(
            str(
                files("robosandbox").joinpath(
                    "assets", "objects", "ycb", ycb_id, f"{short}.robosandbox.yaml"
                )
            )
        )
    elif raw.startswith(_BUILTIN_PREFIX):
        rel = raw[len(_BUILTIN_PREFIX) :].lstrip("/")
        resolved = Path(str(files("robosandbox").joinpath("assets", rel)))
    else:
        p = Path(raw)
        resolved = p if p.is_absolute() else (base_dir / p).resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Asset reference {raw!r} resolved to {resolved} which does not exist"
        )
    return resolved


def list_builtin_ycb_objects() -> list[str]:
    """List all bundled YCB object ids (``NNN_short_name``).

    Scans the packaged assets directory for per-object sidecars. Useful
    for documentation, demos, and "what can I grasp?" discovery from
    notebooks.
    """
    from importlib.resources import files

    ycb_root = Path(str(files("robosandbox").joinpath("assets", "objects", "ycb")))
    if not ycb_root.exists():
        return []
    out: list[str] = []
    for child in ycb_root.iterdir():
        if not child.is_dir():
            continue
        try:
            short = _ycb_short_name(child.name)
        except ValueError:
            continue
        if (child / f"{short}.robosandbox.yaml").exists():
            out.append(child.name)
    return sorted(out)


def _scene_from_dict(d: dict[str, Any], base_dir: Path) -> Scene:
    objs = tuple(_object_from_dict(o, base_dir) for o in d.get("objects", []))
    robot_urdf = _resolve_asset_path(d["robot_urdf"], base_dir) if d.get("robot_urdf") else None
    robot_config = (
        _resolve_asset_path(d["robot_config"], base_dir) if d.get("robot_config") else None
    )
    return Scene(
        robot_urdf=robot_urdf,
        robot_config=robot_config,
        objects=objs,
        table_height=float(d.get("table_height", 0.04)),
    )


def load_task(path: Path) -> Task:
    with path.open() as fh:
        raw = yaml.safe_load(fh)
    scene = _scene_from_dict(raw["scene"], base_dir=path.parent)
    randomize = raw.get("randomize")
    if randomize is not None and not isinstance(randomize, dict):
        raise ValueError(f"task {path}: 'randomize:' must be a mapping")
    return Task(
        name=str(raw.get("name", path.stem)),
        scene=scene,
        prompt=str(raw["prompt"]),
        success=SuccessCriterion(data=raw["success"]),
        seed_note=str(raw.get("seed_note", "")),
        randomize=randomize,
    )


def load_builtin_task(name: str) -> Task:
    path = _BUILTIN_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Built-in task {name!r} not found under {_BUILTIN_DIR}"
        )
    return load_task(path)


def list_builtin_tasks() -> list[str]:
    """List non-experimental builtin tasks (those not prefixed ``_``)."""
    return sorted(
        p.stem for p in _BUILTIN_DIR.glob("*.yaml") if not p.stem.startswith("_")
    )
