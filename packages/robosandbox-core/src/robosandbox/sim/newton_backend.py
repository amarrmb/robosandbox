"""Newton implementation of the SimBackend protocol.

This is intentionally narrower than the MuJoCo backend today:

- scene loading supports the Franka-style MJCF + primitive box objects
- observations expose robot/object state but not rendered RGB/depth
- the motion-planner stack remains MuJoCo-specific, so planner-driven
  agent runs should stay on MuJoCo for now

The point of this backend is to make Newton a first-class integration
surface for scene loading, policy replay, and interactive viewer demos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robosandbox.types import Observation, Pose, Scene


def _quat_mul(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _quat_conj(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, z, w = q
    return (-x, -y, -z, w)


def _rotate_vec(q: tuple[float, float, float, float], v: tuple[float, float, float]) -> np.ndarray:
    qv = (v[0], v[1], v[2], 0.0)
    out = _quat_mul(_quat_mul(q, qv), _quat_conj(q))
    return np.array(out[:3], dtype=np.float64)


def _body_pose_from_row(row: np.ndarray) -> Pose:
    return Pose(
        xyz=(float(row[0]), float(row[1]), float(row[2])),
        quat_xyzw=(float(row[3]), float(row[4]), float(row[5]), float(row[6])),
    )


@dataclass(frozen=True)
class _RobotConfig:
    arm_joint_names: tuple[str, ...]
    gripper_joint_names: tuple[str, ...]
    home_qpos: tuple[float, ...]
    gripper_open_qpos: float
    gripper_closed_qpos: float
    ee_attach_body: str
    ee_offset_xyz: tuple[float, float, float]


class NewtonBackend:
    """Experimental Newton-backed simulator."""

    def __init__(
        self,
        render_size: tuple[int, int] = (480, 640),
        camera: str = "scene",
        viewer: str = "null",
        port: int = 8080,
        device: str = "cuda:0",
        dt: float = 1.0 / 240.0,
    ):
        self._render_h, self._render_w = render_size
        self._camera = camera
        self._viewer_kind = viewer
        self._port = int(port)
        self._device = device
        self._dt = float(dt)

        self._scene: Scene | None = None
        self._robot: _RobotConfig | None = None
        self._model: Any = None
        self._state_0: Any = None
        self._state_1: Any = None
        self._control: Any = None
        self._contacts: Any = None
        self._solver: Any = None
        self._viewer: Any = None
        self._wp: Any = None
        self._newton: Any = None
        self._joint_target_mode: Any = None
        self._viewer_null_cls: Any = None
        self._viewer_viser_cls: Any = None

        self._arm_joint_names: list[str] = []
        self._arm_joint_q_indices: list[int] = []
        self._gripper_joint_q_indices: list[int] = []
        self._ee_body_idx: int = -1
        self._ee_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._obj_body_indices: dict[str, int] = {}
        self._t: float = 0.0

    def _ensure_runtime(self) -> None:
        if self._newton is not None:
            return
        try:
            import warp as wp
            import newton
            from newton import JointTargetMode
            from newton.viewer import ViewerNull, ViewerViser
        except ImportError as e:
            raise ImportError(
                "Newton backend requires `warp` and `newton` installed in the active environment"
            ) from e

        self._wp = wp
        self._newton = newton
        self._joint_target_mode = JointTargetMode
        self._viewer_null_cls = ViewerNull
        self._viewer_viser_cls = ViewerViser

    def _load_robot_config(self, path: Path) -> _RobotConfig:
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return _RobotConfig(
            arm_joint_names=tuple(str(v) for v in raw["arm"]["joints"]),
            gripper_joint_names=tuple(str(v) for v in raw["gripper"]["joints"]),
            home_qpos=tuple(float(v) for v in raw["arm"]["home_qpos"]),
            gripper_open_qpos=float(raw["gripper"]["open_qpos"]),
            gripper_closed_qpos=float(raw["gripper"]["closed_qpos"]),
            ee_attach_body=str(raw["ee_site"]["inject"]["attach_body"]),
            ee_offset_xyz=tuple(float(v) for v in raw["ee_site"]["inject"]["xyz"]),
        )

    def _create_viewer(self):
        if self._viewer_kind == "null":
            return self._viewer_null_cls(num_frames=1_000_000)
        if self._viewer_kind == "viser":
            return self._viewer_viser_cls(port=self._port)
        raise ValueError(f"unknown Newton viewer {self._viewer_kind!r}")

    def _build_model(self, scene: Scene):
        assert scene.robot_urdf is not None
        assert scene.robot_config is not None
        assert self._robot is not None

        wp = self._wp
        newton = self._newton
        builder = newton.ModelBuilder()
        builder.add_mjcf(
            str(scene.robot_urdf),
            floating=False,
            enable_self_collisions=False,
            parse_mujoco_options=True,
        )

        x0, y0, _ = scene.workspace_aabb[0]
        x1, y1, _ = scene.workspace_aabb[1]
        table_thickness = 0.02
        hx = max((x1 - x0) * 0.5, 0.01)
        hy = max((y1 - y0) * 0.5, 0.01)
        table_center = wp.vec3(
            float((x0 + x1) * 0.5),
            float((y0 + y1) * 0.5),
            float(scene.table_height - table_thickness),
        )
        builder.add_shape_box(
            body=-1,
            hx=hx,
            hy=hy,
            hz=table_thickness,
            xform=wp.transform(table_center, wp.quat_identity()),
        )
        builder.add_ground_plane()

        for obj in scene.objects:
            if obj.kind != "box":
                raise NotImplementedError(
                    f"Newton backend currently supports box objects only, got {obj.kind!r}"
                )
            x, y, z = obj.pose.xyz
            qx, qy, qz, qw = obj.pose.quat_xyzw
            sx, sy, sz = obj.size
            body = builder.add_body(
                xform=wp.transform(wp.vec3(x, y, z), wp.quat(qx, qy, qz, qw)),
                mass=float(obj.mass),
                label=obj.id,
            )
            builder.add_shape_box(body=body, hx=sx, hy=sy, hz=sz)

        target_q = [*self._robot.home_qpos, self._robot.gripper_open_qpos, self._robot.gripper_open_qpos]
        builder.joint_q[: len(target_q)] = target_q
        builder.joint_target_pos[: len(target_q)] = target_q
        builder.joint_target_ke[: len(target_q)] = [4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]
        builder.joint_target_kd[: len(target_q)] = [450, 450, 350, 350, 200, 200, 200, 10, 10]
        for i in range(len(target_q)):
            builder.joint_target_mode[i] = int(self._joint_target_mode.POSITION)
        return builder.finalize()

    def _find_joint_q_index(self, joint_name: str) -> int:
        assert self._model is not None
        labels = list(self._model.joint_label)
        starts = self._model.joint_q_start.numpy()
        for i, label in enumerate(labels):
            if label.endswith(f"/{joint_name}") or label == joint_name:
                return int(starts[i])
        raise KeyError(f"joint {joint_name!r} not found in Newton model")

    def _find_body_index(self, body_name: str) -> int:
        assert self._model is not None
        labels = list(self._model.body_label)
        for i, label in enumerate(labels):
            if label.endswith(f"/{body_name}") or label == body_name:
                return i
        raise KeyError(f"body {body_name!r} not found in Newton model")

    def _log_viewer_state(self) -> None:
        if self._viewer is None or self._state_0 is None:
            return
        self._viewer.begin_frame(self._t)
        self._viewer.log_state(self._state_0)
        self._viewer.end_frame()

    def _write_control_targets(self, target: np.ndarray) -> None:
        assert self._control is not None
        arr = self._wp.array(target, dtype=self._control.joint_target_pos.dtype)
        self._wp.copy(self._control.joint_target_pos, arr)

    def load(self, scene: Scene) -> None:
        self._ensure_runtime()
        self._wp.set_device(self._device)
        if scene.robot_urdf is None or scene.robot_config is None:
            raise NotImplementedError(
                "Newton backend currently requires an explicit robot_urdf and robot_config"
            )
        self._scene = scene
        self._robot = self._load_robot_config(scene.robot_config)
        self._arm_joint_names = list(self._robot.arm_joint_names)
        self._ee_offset_xyz = self._robot.ee_offset_xyz

        self._model = self._build_model(scene)
        self._arm_joint_q_indices = [
            self._find_joint_q_index(name) for name in self._robot.arm_joint_names
        ]
        self._gripper_joint_q_indices = [
            self._find_joint_q_index(name) for name in self._robot.gripper_joint_names
        ]
        self._ee_body_idx = self._find_body_index(self._robot.ee_attach_body)
        self._obj_body_indices = {}
        for obj in scene.objects:
            self._obj_body_indices[obj.id] = self._find_body_index(obj.id)

        self._viewer = self._create_viewer()
        self._viewer.set_model(self._model)
        if hasattr(self._viewer, "set_camera"):
            self._viewer.set_camera(pos=self._wp.vec3(1.1, -1.4, 0.9), pitch=-18.0, yaw=45.0)
        self.reset()

    def reset(self) -> None:
        assert self._model is not None
        self._state_0 = self._model.state()
        self._state_1 = self._model.state()
        self._control = self._model.control()
        self._contacts = self._model.contacts()
        self._solver = self._newton.solvers.SolverMuJoCo(self._model)
        self._t = 0.0
        self._log_viewer_state()

    def close(self) -> None:
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None

    def step(
        self,
        target_joints: np.ndarray | None = None,
        gripper: float | None = None,
    ) -> None:
        assert self._state_0 is not None
        assert self._state_1 is not None
        assert self._control is not None
        assert self._contacts is not None
        assert self._solver is not None
        assert self._robot is not None

        if target_joints is not None:
            arr = np.asarray(target_joints, dtype=np.float64).ravel()
            if arr.shape != (len(self._arm_joint_q_indices),):
                raise ValueError(
                    f"target_joints must have shape ({len(self._arm_joint_q_indices)},), got {arr.shape}"
                )
            target = self._control.joint_target_pos.numpy()
            for q_idx, q in zip(self._arm_joint_q_indices, arr):
                target[q_idx] = float(q)
            self._write_control_targets(target)

        if gripper is not None:
            t = float(np.clip(gripper, 0.0, 1.0))
            finger_q = (
                self._robot.gripper_open_qpos * (1.0 - t)
                + self._robot.gripper_closed_qpos * t
            )
            target = self._control.joint_target_pos.numpy()
            for q_idx in self._gripper_joint_q_indices:
                target[q_idx] = finger_q
            self._write_control_targets(target)

        self._state_0.clear_forces()
        self._model.collide(self._state_0, self._contacts)
        self._solver.step(self._state_0, self._state_1, self._control, self._contacts, self._dt)
        self._state_0, self._state_1 = self._state_1, self._state_0
        self._t += self._dt
        self._log_viewer_state()

    def _ee_pose(self) -> Pose:
        assert self._state_0 is not None
        row = self._state_0.body_q.numpy()[self._ee_body_idx]
        body_pose = _body_pose_from_row(row)
        rotated = _rotate_vec(body_pose.quat_xyzw, self._ee_offset_xyz)
        xyz = np.asarray(body_pose.xyz, dtype=np.float64) + rotated
        return Pose(
            xyz=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
            quat_xyzw=body_pose.quat_xyzw,
        )

    def _body_pose(self, body_idx: int) -> Pose:
        assert self._state_0 is not None
        row = self._state_0.body_q.numpy()[body_idx]
        return _body_pose_from_row(row)

    def observe(self) -> Observation:
        assert self._state_0 is not None
        q = self._state_0.joint_q.numpy()
        arm_joints = np.array([q[i] for i in self._arm_joint_q_indices], dtype=np.float64)
        finger_positions = [float(q[i]) for i in self._gripper_joint_q_indices]
        gripper_width = float(sum(abs(v) for v in finger_positions))
        objects = {oid: self._body_pose(idx) for oid, idx in self._obj_body_indices.items()}
        return Observation(
            rgb=np.zeros((self._render_h, self._render_w, 3), dtype=np.uint8),
            depth=None,
            robot_joints=arm_joints,
            ee_pose=self._ee_pose(),
            gripper_width=gripper_width,
            scene_objects=objects,
            timestamp=self._t,
            camera_intrinsics=None,
            camera_extrinsics=None,
        )

    def get_object_pose(self, object_id: str) -> Pose | None:
        body_idx = self._obj_body_indices.get(object_id)
        if body_idx is None:
            return None
        return self._body_pose(body_idx)

    def set_object_pose(self, object_id: str, pose: Pose) -> None:
        raise NotImplementedError(
            "Newton backend does not yet support teleporting objects in-place"
        )

    @property
    def n_dof(self) -> int:
        return len(self._arm_joint_names)

    @property
    def joint_names(self) -> list[str]:
        return list(self._arm_joint_names)

    @property
    def home_qpos(self) -> np.ndarray:
        assert self._robot is not None
        return np.asarray(self._robot.home_qpos, dtype=np.float64)
