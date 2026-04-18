"""Core data types — the universal currency of RoboSandbox.

Every layer consumes and produces these. Frozen dataclasses keep them
cheap to hash, safe to pass across threads, and impossible to mutate
halfway through a skill.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

XYZ = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # (x, y, z, w)


@dataclass(frozen=True)
class Pose:
    """6-DOF pose in a named frame. Quaternion is (x, y, z, w)."""

    xyz: XYZ
    quat_xyzw: Quat = (0.0, 0.0, 0.0, 1.0)

    def as_array(self) -> np.ndarray:
        return np.array([*self.xyz, *self.quat_xyzw], dtype=np.float64)

    @classmethod
    def from_array(cls, a: np.ndarray) -> Pose:
        a = np.asarray(a, dtype=np.float64)
        if a.shape != (7,):
            raise ValueError(f"Pose.from_array expects shape (7,), got {a.shape}")
        return cls(xyz=tuple(a[:3].tolist()), quat_xyzw=tuple(a[3:].tolist()))


@dataclass(frozen=True)
class SceneObject:
    """One object in the scene, described abstractly.

    Kind is the MJCF geom primitive (box/sphere/cylinder) OR 'mesh' for
    user-supplied OBJ/STL. For v0.1 we ship only the primitives.
    """

    id: str
    kind: str  # 'box' | 'sphere' | 'cylinder' | 'mesh'
    size: tuple[float, ...]  # semantics depend on kind (box: xyz half-extents)
    pose: Pose
    mass: float = 0.1
    rgba: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)
    mesh_path: Path | None = None


@dataclass(frozen=True)
class Scene:
    """A loadable scene description — robot + objects + workspace."""

    robot_urdf: Path | None = None  # None = use built-in simple arm
    objects: tuple[SceneObject, ...] = ()
    workspace_aabb: tuple[XYZ, XYZ] = ((-0.5, -0.5, 0.0), (0.5, 0.5, 0.8))
    table_height: float = 0.0  # z of table surface
    gravity: XYZ = (0.0, 0.0, -9.81)


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(frozen=True)
class Observation:
    """Everything a perception/skill/agent might need from one sim tick."""

    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray | None  # (H, W) float32, meters, or None
    robot_joints: np.ndarray  # (n_dof,)
    ee_pose: Pose
    gripper_width: float
    scene_objects: dict[str, Pose] = field(default_factory=dict)
    timestamp: float = 0.0
    camera_intrinsics: CameraIntrinsics | None = None
    camera_extrinsics: Pose | None = None  # camera pose in world frame


@dataclass(frozen=True)
class DetectedObject:
    label: str
    pixel_xy: tuple[int, int] | None = None
    bbox_2d: tuple[int, int, int, int] | None = None  # x1,y1,x2,y2
    pose_3d: Pose | None = None
    confidence: float = 1.0


@dataclass(frozen=True)
class Grasp:
    pose: Pose  # gripper pose at closed-on-object
    gripper_width: float  # width at which to close
    approach_offset: float = 0.08  # meters above grasp to approach from
    score: float = 1.0


@dataclass(frozen=True)
class JointTrajectory:
    """A sequence of joint waypoints. Time-parameterised via dt."""

    waypoints: np.ndarray  # (T, n_dof)
    dt: float = 1.0 / 200.0  # seconds per step

    def __post_init__(self) -> None:
        wp = np.asarray(self.waypoints, dtype=np.float64)
        if wp.ndim != 2:
            raise ValueError(f"waypoints must be 2D (T, n_dof), got {wp.shape}")
        object.__setattr__(self, "waypoints", wp)

    @property
    def duration(self) -> float:
        return float(len(self.waypoints) * self.dt)


class SkillResult:
    """Return type of every skill. Explicit class (not dataclass) so we can
    construct lightweight, mutable instances during debugging."""

    __slots__ = ("success", "reason", "reason_detail", "artifacts")

    def __init__(
        self,
        success: bool,
        reason: str = "",
        reason_detail: str = "",
        artifacts: dict | None = None,
    ) -> None:
        self.success = success
        self.reason = reason
        self.reason_detail = reason_detail
        self.artifacts = artifacts or {}

    def __repr__(self) -> str:
        tag = "OK" if self.success else f"FAIL:{self.reason}"
        return f"SkillResult({tag})"
