"""Load a URDF/MJCF + sidecar YAML into (MjSpec, RobotSpec).

MuJoCo's MjSpec lets us load a robot file into an editable in-memory
representation, inject objects/decor/sites/actuators programmatically,
and compile. We use this to compose a complete scene from:
  - the robot file (URDF or MJCF), absorbed via MjSpec.from_file
  - a sidecar YAML that names arm/gripper/site/actuator elements
  - the Scene's objects, added as free-body children of worldbody
  - RoboSandbox's standard scene decor (floor, table, cameras, lights)

Errors raised by this module are subclasses of RobotConfigError so callers
can catch a single type. The intent is to surface MuJoCo's cryptic compile
failures with enough context (file paths, which field / which name is at
fault) that the user can fix their sidecar without reading MjSpec docs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import yaml

from robosandbox.scene.robot_spec import RobotSpec
from robosandbox.types import Scene, SceneObject


# -- exceptions --------------------------------------------------------------


class RobotConfigError(Exception):
    """Base for all sidecar/URDF loading errors."""


class RobotConfigNotFoundError(RobotConfigError):
    def __init__(self, urdf_path: Path, tried: list[Path]) -> None:
        super().__init__(
            f"No sidecar config found for {urdf_path}. Tried:\n"
            + "\n".join(f"  - {p}" for p in tried)
            + "\nProvide one via Scene.robot_config or alongside the URDF as "
            f"{urdf_path.stem}.robosandbox.yaml."
        )
        self.urdf_path = urdf_path
        self.tried = tried


class RobotConfigValidationError(RobotConfigError):
    def __init__(self, field: str, reason: str) -> None:
        super().__init__(f"Invalid sidecar at '{field}': {reason}")
        self.field = field
        self.reason = reason


class RobotConfigMismatchError(RobotConfigError):
    def __init__(self, kind: str, name: str, available: list[str]) -> None:
        preview = ", ".join(available[:10]) + (", ..." if len(available) > 10 else "")
        super().__init__(
            f"Sidecar references {kind} {name!r} which is not in the compiled model. "
            f"Available {kind}s: [{preview}]"
        )
        self.kind = kind
        self.name = name
        self.available = available


class RobotModelCompileError(RobotConfigError):
    def __init__(self, urdf_path: Path, cause: BaseException) -> None:
        super().__init__(f"MuJoCo failed to compile {urdf_path}: {cause}")
        self.urdf_path = urdf_path
        self.__cause__ = cause


# -- sidecar resolution + validation -----------------------------------------


def resolve_sidecar(urdf_path: Path, config_path: Path | None) -> Path:
    urdf_path = Path(urdf_path)
    if config_path is not None:
        p = Path(config_path)
        if not p.exists():
            raise RobotConfigNotFoundError(urdf_path, [p])
        return p
    sibling = urdf_path.with_name(f"{urdf_path.stem}.robosandbox.yaml")
    if sibling.exists():
        return sibling
    raise RobotConfigNotFoundError(urdf_path, [sibling])


def _require(d: dict[str, Any], key: str, parent: str) -> Any:
    if key not in d:
        raise RobotConfigValidationError(f"{parent}.{key}", "missing required field")
    return d[key]


def _as_str_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise RobotConfigValidationError(field, "must be a list of strings")
    return list(value)


def _as_float_list(value: Any, field: str) -> list[float]:
    if not isinstance(value, list) or not all(isinstance(v, (int, float)) for v in value):
        raise RobotConfigValidationError(field, "must be a list of numbers")
    return [float(v) for v in value]


@dataclass
class _ParsedSidecar:
    arm_joints: tuple[str, ...]
    arm_actuators: tuple[str, ...]
    home_qpos: tuple[float, ...]
    gripper_joints: tuple[str, ...]
    gripper_primary_joint: str
    gripper_actuator: str
    gripper_open_qpos: float
    gripper_closed_qpos: float
    ee_site_mode: str  # "name" or "inject"
    ee_site_name: str | None
    ee_site_attach_body: str | None
    ee_site_xyz: tuple[float, float, float]
    ee_site_quat_xyzw: tuple[float, float, float, float]
    base_xyz: tuple[float, float, float]
    base_quat_xyzw: tuple[float, float, float, float]


def _parse_sidecar(raw: dict[str, Any]) -> _ParsedSidecar:
    arm = _require(raw, "arm", "root")
    if not isinstance(arm, dict):
        raise RobotConfigValidationError("arm", "must be a mapping")
    arm_joints = _as_str_list(_require(arm, "joints", "arm"), "arm.joints")
    arm_actuators = _as_str_list(_require(arm, "actuators", "arm"), "arm.actuators")
    home_qpos = _as_float_list(_require(arm, "home_qpos", "arm"), "arm.home_qpos")

    if len(arm_joints) != len(arm_actuators):
        raise RobotConfigValidationError(
            "arm", f"joints len {len(arm_joints)} != actuators len {len(arm_actuators)}"
        )
    if len(arm_joints) != len(home_qpos):
        raise RobotConfigValidationError(
            "arm.home_qpos",
            f"length {len(home_qpos)} != arm.joints length {len(arm_joints)}",
        )

    gripper = _require(raw, "gripper", "root")
    if not isinstance(gripper, dict):
        raise RobotConfigValidationError("gripper", "must be a mapping")
    g_joints = _as_str_list(_require(gripper, "joints", "gripper"), "gripper.joints")
    g_primary = str(_require(gripper, "primary_joint", "gripper"))
    g_actuator = str(_require(gripper, "actuator", "gripper"))
    g_open = float(_require(gripper, "open_qpos", "gripper"))
    g_closed = float(_require(gripper, "closed_qpos", "gripper"))
    if g_primary not in g_joints:
        raise RobotConfigValidationError(
            "gripper.primary_joint",
            f"{g_primary!r} not in gripper.joints {g_joints}",
        )

    ee = _require(raw, "ee_site", "root")
    if not isinstance(ee, dict):
        raise RobotConfigValidationError("ee_site", "must be a mapping")
    has_name = "name" in ee
    has_inject = "inject" in ee
    if has_name == has_inject:
        raise RobotConfigValidationError(
            "ee_site",
            "must specify exactly one of ee_site.name (reference existing) or "
            "ee_site.inject (create new) — found "
            + ("both" if has_name else "neither"),
        )
    ee_name: str | None = None
    attach_body: str | None = None
    ee_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ee_quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    if has_name:
        mode = "name"
        ee_name = str(ee["name"])
    else:
        mode = "inject"
        inj = ee["inject"]
        if not isinstance(inj, dict):
            raise RobotConfigValidationError("ee_site.inject", "must be a mapping")
        attach_body = str(_require(inj, "attach_body", "ee_site.inject"))
        xyz = _as_float_list(_require(inj, "xyz", "ee_site.inject"), "ee_site.inject.xyz")
        if len(xyz) != 3:
            raise RobotConfigValidationError("ee_site.inject.xyz", "must have 3 components")
        ee_xyz = (xyz[0], xyz[1], xyz[2])
        quat_raw = inj.get("quat_xyzw", [0.0, 0.0, 0.0, 1.0])
        quat = _as_float_list(quat_raw, "ee_site.inject.quat_xyzw")
        if len(quat) != 4:
            raise RobotConfigValidationError(
                "ee_site.inject.quat_xyzw", "must have 4 components"
            )
        ee_quat = (quat[0], quat[1], quat[2], quat[3])

    base = _require(raw, "base_pose", "root")
    if not isinstance(base, dict):
        raise RobotConfigValidationError("base_pose", "must be a mapping")
    base_xyz_l = _as_float_list(_require(base, "xyz", "base_pose"), "base_pose.xyz")
    if len(base_xyz_l) != 3:
        raise RobotConfigValidationError("base_pose.xyz", "must have 3 components")
    base_xyz = (base_xyz_l[0], base_xyz_l[1], base_xyz_l[2])
    base_quat_l = _as_float_list(
        base.get("quat_xyzw", [0.0, 0.0, 0.0, 1.0]), "base_pose.quat_xyzw"
    )
    if len(base_quat_l) != 4:
        raise RobotConfigValidationError("base_pose.quat_xyzw", "must have 4 components")
    base_quat = (base_quat_l[0], base_quat_l[1], base_quat_l[2], base_quat_l[3])

    return _ParsedSidecar(
        arm_joints=tuple(arm_joints),
        arm_actuators=tuple(arm_actuators),
        home_qpos=tuple(home_qpos),
        gripper_joints=tuple(g_joints),
        gripper_primary_joint=g_primary,
        gripper_actuator=g_actuator,
        gripper_open_qpos=g_open,
        gripper_closed_qpos=g_closed,
        ee_site_mode=mode,
        ee_site_name=ee_name,
        ee_site_attach_body=attach_body,
        ee_site_xyz=ee_xyz,
        ee_site_quat_xyzw=ee_quat,
        base_xyz=base_xyz,
        base_quat_xyzw=base_quat,
    )


# -- helpers -----------------------------------------------------------------


def _quat_wxyz_from_xyzw(q: tuple[float, float, float, float]) -> list[float]:
    x, y, z, w = q
    return [w, x, y, z]


def _list_bodies(spec: mujoco.MjSpec) -> list[str]:
    return [b.name for b in spec.bodies if b.name]


def _find_body(spec: mujoco.MjSpec, name: str) -> Any:
    for b in spec.bodies:
        if b.name == name:
            return b
    raise RobotConfigMismatchError("body", name, _list_bodies(spec))


def _find_robot_root_body(spec: mujoco.MjSpec) -> Any:
    """The first direct child of worldbody — that's the robot's base link."""
    world = spec.worldbody
    children = list(world.bodies)
    if not children:
        raise RobotConfigError(
            "robot file has no top-level body under worldbody; cannot apply base_pose"
        )
    return children[0]


def _validate_names_in_model(model: mujoco.MjModel, spec: RobotSpec) -> None:
    def _all_names(count: int, getter) -> list[str]:
        out = []
        for i in range(count):
            try:
                n = getter(i).name
                if n:
                    out.append(n)
            except Exception:
                pass
        return out

    joint_names = _all_names(model.njnt, lambda i: model.joint(i))
    actuator_names = _all_names(model.nu, lambda i: model.actuator(i))
    site_names = _all_names(model.nsite, lambda i: model.site(i))

    for n in spec.arm_joint_names + spec.gripper_joint_names:
        if n not in joint_names:
            raise RobotConfigMismatchError("joint", n, joint_names)
    for n in spec.arm_actuator_names + (spec.gripper_actuator_name,):
        if n not in actuator_names:
            raise RobotConfigMismatchError("actuator", n, actuator_names)
    if spec.ee_site_name not in site_names:
        raise RobotConfigMismatchError("site", spec.ee_site_name, site_names)


# -- public API --------------------------------------------------------------


_EE_SITE_INJECTED_NAME = "robosandbox_ee_site"


def load_robot(
    urdf_path: Path, config_path: Path | None
) -> tuple[mujoco.MjSpec, RobotSpec]:
    """Load a URDF/MJCF file + sidecar YAML, return (MjSpec, RobotSpec).

    The returned MjSpec is mutable — callers can add objects, decor, etc.
    before calling spec.compile(). The RobotSpec is final.
    """
    urdf_path = Path(urdf_path)
    if not urdf_path.exists():
        raise RobotConfigError(f"robot file not found: {urdf_path}")
    sidecar_path = resolve_sidecar(urdf_path, config_path)

    with sidecar_path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    parsed = _parse_sidecar(raw)

    try:
        spec = mujoco.MjSpec.from_file(str(urdf_path))
    except (ValueError, RuntimeError) as e:
        raise RobotModelCompileError(urdf_path, e) from e

    # Inject ee_site on the specified body if requested.
    if parsed.ee_site_mode == "inject":
        assert parsed.ee_site_attach_body is not None
        body = _find_body(spec, parsed.ee_site_attach_body)
        body.add_site(
            name=_EE_SITE_INJECTED_NAME,
            pos=list(parsed.ee_site_xyz),
            quat=_quat_wxyz_from_xyzw(parsed.ee_site_quat_xyzw),
        )
        ee_site_name = _EE_SITE_INJECTED_NAME
    else:
        assert parsed.ee_site_name is not None
        ee_site_name = parsed.ee_site_name

    # Place the robot in world by translating its root body.
    root = _find_robot_root_body(spec)
    root.pos = list(parsed.base_xyz)
    root.quat = _quat_wxyz_from_xyzw(parsed.base_quat_xyzw)
    base_body_name = root.name or ""
    if not base_body_name:
        raise RobotConfigError(
            "robot root body has no name; MuJoCo needs a named body for motion-planner lookups"
        )

    robot_spec = RobotSpec(
        arm_joint_names=parsed.arm_joints,
        arm_actuator_names=parsed.arm_actuators,
        gripper_joint_names=parsed.gripper_joints,
        gripper_primary_joint=parsed.gripper_primary_joint,
        gripper_actuator_name=parsed.gripper_actuator,
        ee_site_name=ee_site_name,
        base_body_name=base_body_name,
        home_qpos=parsed.home_qpos,
        gripper_open_qpos=parsed.gripper_open_qpos,
        gripper_closed_qpos=parsed.gripper_closed_qpos,
    )
    return spec, robot_spec


# -- scene object + decor injection ------------------------------------------


def _xyzw_to_wxyz(q: tuple[float, float, float, float]) -> list[float]:
    x, y, z, w = q
    return [w, x, y, z]


def inject_scene_objects(spec: mujoco.MjSpec, scene: Scene) -> None:
    """Add each SceneObject as a free-body child of worldbody."""
    world = spec.worldbody
    for obj in scene.objects:
        if obj.kind == "mesh":
            raise NotImplementedError(
                f"SceneObject(kind='mesh') not supported in URDF path yet (id={obj.id!r})"
            )
        body = world.add_body(
            name=obj.id,
            pos=list(obj.pose.xyz),
            quat=_xyzw_to_wxyz(obj.pose.quat_xyzw),
        )
        body.add_freejoint()
        rgba = list(obj.rgba)
        if obj.kind == "box":
            sx, sy, sz = obj.size
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[sx, sy, sz],
                rgba=rgba,
                friction=[1.5, 0.1, 0.01],
                mass=obj.mass,
            )
        elif obj.kind == "sphere":
            (r,) = obj.size
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[r, 0.0, 0.0],
                rgba=rgba,
                friction=[1.5, 0.1, 0.01],
                mass=obj.mass,
            )
        elif obj.kind == "cylinder":
            r, h = obj.size
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[r, h, 0.0],
                rgba=rgba,
                friction=[1.5, 0.1, 0.01],
                mass=obj.mass,
            )
        else:
            raise ValueError(f"unknown SceneObject.kind: {obj.kind}")


def inject_scene_decor(spec: mujoco.MjSpec, scene: Scene) -> None:
    """Add floor, table, standard cameras, and lights to worldbody.

    Matches the built-in arm's decor so the default camera names ('scene',
    'top') resolve identically on both paths and downstream perception
    stays robot-agnostic.
    """
    world = spec.worldbody

    # Lights
    world.add_light(pos=[0.0, 0.0, 3.0], dir=[0.0, 0.0, -1.0], diffuse=[0.8, 0.8, 0.8])
    world.add_light(pos=[1.5, -1.0, 2.0], dir=[-1.0, 1.0, -1.0], diffuse=[0.4, 0.4, 0.4])

    # Cameras
    world.add_camera(
        name="scene",
        pos=[0.9, -0.9, 0.9],
        xyaxes=[0.707, 0.707, 0.0, -0.4, 0.4, 0.82],
    )
    world.add_camera(name="top", pos=[0.0, 0.0, 1.2], xyaxes=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    # Floor
    world.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[2.0, 2.0, 0.05],
        rgba=[0.9, 0.9, 0.9, 1.0],
    )

    # Table (matches built-in arm: centered at (0.1, 0, 0.02), 0.8m x 0.8m x 0.04m)
    world.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.4, 0.4, 0.02],
        pos=[0.1, 0.0, 0.02],
        rgba=[0.72, 0.55, 0.37, 1.0],
    )


def load_and_compile(scene: Scene) -> tuple[mujoco.MjModel, RobotSpec]:
    """Full URDF path: load robot + sidecar, inject scene + decor, compile.

    This is what scene.build_model(scene) calls when Scene.robot_urdf is set.
    """
    assert scene.robot_urdf is not None
    spec, robot_spec = load_robot(scene.robot_urdf, scene.robot_config)
    inject_scene_objects(spec, scene)
    inject_scene_decor(spec, scene)
    # Apply gravity from scene (built-in path uses the MJCF <option> tag).
    spec.option.gravity = list(scene.gravity)
    try:
        model = spec.compile()
    except (ValueError, RuntimeError) as e:
        raise RobotModelCompileError(Path(scene.robot_urdf), e) from e
    _validate_names_in_model(model, robot_spec)
    return model, robot_spec
