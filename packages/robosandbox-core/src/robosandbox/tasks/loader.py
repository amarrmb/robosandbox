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


def _pose_from_dict(d: dict[str, Any]) -> Pose:
    xyz = tuple(float(v) for v in d.get("xyz", [0.0, 0.0, 0.0]))
    q = d.get("quat_xyzw", [0.0, 0.0, 0.0, 1.0])
    qx, qy, qz, qw = [float(v) for v in q]
    return Pose(xyz=xyz, quat_xyzw=(qx, qy, qz, qw))


def _object_from_dict(d: dict[str, Any]) -> SceneObject:
    return SceneObject(
        id=str(d["id"]),
        kind=str(d.get("kind", "box")),
        size=tuple(float(v) for v in d.get("size", [0.012, 0.012, 0.012])),
        pose=_pose_from_dict(d.get("pose", {})),
        mass=float(d.get("mass", 0.05)),
        rgba=tuple(float(v) for v in d.get("rgba", [0.7, 0.7, 0.7, 1.0])),
    )


_BUILTIN_PREFIX = "@builtin:"


def _resolve_asset_path(raw: str, base_dir: Path) -> Path:
    """Resolve a task-YAML asset reference.

    Order:
      1. "@builtin:<rel>"  -> packaged assets (robosandbox/assets/<rel>)
      2. absolute path     -> Path(raw) unchanged
      3. relative path     -> (base_dir / raw).resolve()

    Raises FileNotFoundError if the resolved path doesn't exist, so broken
    task YAMLs fail loudly at load time rather than later during compile.
    """
    if raw.startswith(_BUILTIN_PREFIX):
        from importlib.resources import files

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


def _scene_from_dict(d: dict[str, Any], base_dir: Path) -> Scene:
    objs = tuple(_object_from_dict(o) for o in d.get("objects", []))
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
    return Task(
        name=str(raw.get("name", path.stem)),
        scene=scene,
        prompt=str(raw["prompt"]),
        success=SuccessCriterion(data=raw["success"]),
        seed_note=str(raw.get("seed_note", "")),
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
