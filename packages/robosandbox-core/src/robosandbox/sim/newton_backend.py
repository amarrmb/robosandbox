"""Newton implementation of the SimBackend protocol.

This is intentionally narrower than the MuJoCo backend today:

- scene loading supports the Franka-style MJCF + primitive box objects
- observations expose robot/object state but not rendered RGB/depth
- the motion-planner stack remains MuJoCo-specific, so planner-driven
  agent runs should stay on MuJoCo for now

The point of this backend is to make Newton a first-class integration
surface for scene loading, policy replay, and interactive viewer demos.

Multi-world (world_count > 1)
------------------------------
Pass ``world_count=N`` to run N identical scenes in parallel on one GPU.
Newton tiles the worlds in a 2-D grid with 2.5 m spacing so they don't
interfere physically.  ``observe_all()`` returns one Observation per world;
``step()`` broadcasts the same joint targets to every world.  This is
exactly what GPU-parallel policy evaluation needs.
"""

from __future__ import annotations

import copy
import math
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


def _grid_offsets(n: int, spacing: float = 2.5) -> list[tuple[float, float, float]]:
    """Lay out N worlds in a square-ish grid with ``spacing`` metres between centres."""
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    offsets: list[tuple[float, float, float]] = []
    for i in range(n):
        r, c = i // cols, i % cols
        x = (c - (cols - 1) * 0.5) * spacing
        y = (r - (rows - 1) * 0.5) * spacing
        offsets.append((x, y, 0.0))
    return offsets


class NewtonBackend:
    """Newton rigid-body sim backend.  Supports ``world_count`` ≥ 1 for
    GPU-parallel policy evaluation."""

    def __init__(
        self,
        render_size: tuple[int, int] = (480, 640),
        camera: str = "scene",
        viewer: str = "null",
        port: int = 8080,
        device: str = "cuda:0",
        dt: float = 1.0 / 240.0,
        world_count: int = 1,
    ):
        self._render_h, self._render_w = render_size
        self._camera = camera
        self._viewer_kind = viewer
        self._port = int(port)
        self._device = device
        self._dt = float(dt)
        self._world_count = max(1, int(world_count))

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

        # Per-world layout (set during load)
        self._bodies_per_world: int = 0
        self._dof_per_world: int = 0
        # Indices *within one world* (0-indexed from world start)
        self._w_arm_q: list[int] = []       # relative joint_q indices for arm
        self._w_gripper_q: list[int] = []   # relative joint_q indices for fingers
        self._w_ee_body: int = -1           # relative body index for EE
        self._w_obj_body: dict[str, int] = {}  # obj_id -> relative body index

        self._arm_joint_names: list[str] = []
        self._ee_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._t: float = 0.0

    # ------------------------------------------------------------------
    # Runtime bootstrap
    # ------------------------------------------------------------------

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

    def _create_viewer(self) -> Any:
        if self._viewer_kind == "null":
            return self._viewer_null_cls(num_frames=1_000_000)
        if self._viewer_kind == "viser":
            return self._viewer_viser_cls(port=self._port)
        raise ValueError(f"unknown Newton viewer {self._viewer_kind!r}")

    # ------------------------------------------------------------------
    # Model building
    # ------------------------------------------------------------------

    def _build_single_builder(self, scene: Scene) -> Any:
        """Build an unfinalized ModelBuilder for one world."""
        assert scene.robot_urdf is not None
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
            hx=hx, hy=hy, hz=table_thickness,
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
            builder.add_body(
                xform=wp.transform(wp.vec3(x, y, z), wp.quat(qx, qy, qz, qw)),
                mass=float(obj.mass),
                label=obj.id,
            )
            builder.add_shape_box(body=builder.body_count - 1, hx=sx, hy=sy, hz=sz)

        target_q = [*self._robot.home_qpos, self._robot.gripper_open_qpos, self._robot.gripper_open_qpos]
        builder.joint_q[: len(target_q)] = target_q
        builder.joint_target_pos[: len(target_q)] = target_q
        builder.joint_target_ke[: len(target_q)] = [4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]
        builder.joint_target_kd[: len(target_q)] = [450, 450, 350, 350, 200, 200, 200, 10, 10]
        for i in range(len(target_q)):
            builder.joint_target_mode[i] = int(self._joint_target_mode.POSITION)
        return builder

    def _build_model(
        self,
        scene: Scene,
        *,
        per_world_scenes: list[Scene] | None = None,
    ) -> Any:
        # `single` defines the per-world topology used for index discovery.
        # When `per_world_scenes` is set, every entry must have matching
        # topology — `load()` already validated that.
        single = self._build_single_builder(scene)

        # Record per-world layout sizes from the unfinalized builder
        self._bodies_per_world = single.body_count
        self._dof_per_world = single.joint_coord_count

        # Discover joint/body indices from a finalized single-world model
        ref_model = copy.deepcopy(single).finalize()
        self._w_arm_q = [self._find_joint_q_index_in(ref_model, n) for n in self._robot.arm_joint_names]
        self._w_gripper_q = [self._find_joint_q_index_in(ref_model, n) for n in self._robot.gripper_joint_names]
        self._w_ee_body = self._find_body_index_in(ref_model, self._robot.ee_attach_body)
        self._w_obj_body = {obj.id: self._find_body_index_in(ref_model, obj.id) for obj in scene.objects}

        if self._world_count == 1:
            # World 0 honours the per-world scene if one was passed.
            if per_world_scenes is not None:
                return self._build_single_builder(per_world_scenes[0]).finalize()
            return single.finalize()

        # Tile N worlds in a grid. Each world gets its own builder when
        # per_world_scenes is set so randomization actually lands in the
        # finalized model (object xform is baked at add_body time).
        wp = self._wp
        newton = self._newton
        multi = newton.ModelBuilder()
        for w, (ox, oy, oz) in enumerate(_grid_offsets(self._world_count)):
            world_scene = per_world_scenes[w] if per_world_scenes is not None else scene
            world_builder = (
                self._build_single_builder(world_scene)
                if per_world_scenes is not None
                else single
            )
            multi.add_world(
                world_builder,
                xform=wp.transform(wp.vec3(ox, oy, oz), wp.quat_identity()),
            )
        return multi.finalize()

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------

    def _find_joint_q_index_in(self, model: Any, joint_name: str) -> int:
        labels = list(model.joint_label)
        starts = model.joint_q_start.numpy()
        for i, label in enumerate(labels):
            if label.endswith(f"/{joint_name}") or label == joint_name:
                return int(starts[i])
        raise KeyError(f"joint {joint_name!r} not found in Newton model")

    def _find_body_index_in(self, model: Any, body_name: str) -> int:
        labels = list(model.body_label)
        for i, label in enumerate(labels):
            if label.endswith(f"/{body_name}") or label == body_name:
                return i
        raise KeyError(f"body {body_name!r} not found in Newton model")

    # absolute index for world w
    def _arm_q_abs(self, w: int) -> list[int]:
        off = w * self._dof_per_world
        return [off + i for i in self._w_arm_q]

    def _gripper_q_abs(self, w: int) -> list[int]:
        off = w * self._dof_per_world
        return [off + i for i in self._w_gripper_q]

    def _ee_body_abs(self, w: int) -> int:
        return w * self._bodies_per_world + self._w_ee_body

    def _obj_body_abs(self, obj_id: str, w: int) -> int:
        return w * self._bodies_per_world + self._w_obj_body[obj_id]

    # ------------------------------------------------------------------
    # Viewer
    # ------------------------------------------------------------------

    def _log_viewer_state(self) -> None:
        if self._viewer is None or self._state_0 is None:
            return
        self._viewer.begin_frame(self._t)
        self._viewer.log_state(self._state_0)
        self._viewer.end_frame()

    # ------------------------------------------------------------------
    # SimBackend protocol
    # ------------------------------------------------------------------

    def load(self, scene: Scene, *, per_world_scenes: list[Scene] | None = None) -> None:
        """Build the multi-world model.

        ``per_world_scenes``: optional list of length ``world_count``. When
        provided, each world is built from the corresponding Scene (so e.g.
        each world can hold a different randomized object pose). The base
        ``scene`` still defines the robot URDF / sidecar / workspace; only
        the per-world ``objects`` differ. Topology must match across the
        list (same number and kinds of objects) — that is the contract
        ``robosandbox.tasks.randomize.jitter_scene`` already satisfies.
        """
        self._ensure_runtime()
        self._wp.set_device(self._device)
        if scene.robot_urdf is None or scene.robot_config is None:
            raise NotImplementedError(
                "Newton backend requires an explicit robot_urdf and robot_config"
            )
        if per_world_scenes is not None:
            if len(per_world_scenes) != self._world_count:
                raise ValueError(
                    f"per_world_scenes has length {len(per_world_scenes)} but "
                    f"world_count={self._world_count}"
                )
            base_kinds = tuple((o.id, o.kind) for o in scene.objects)
            for w, ws in enumerate(per_world_scenes):
                ws_kinds = tuple((o.id, o.kind) for o in ws.objects)
                if ws_kinds != base_kinds:
                    raise ValueError(
                        f"per_world_scenes[{w}] topology differs from base scene "
                        f"({ws_kinds} vs {base_kinds}); only pose/mass/size/rgba "
                        f"may be randomized"
                    )
        self._scene = scene
        self._robot = self._load_robot_config(scene.robot_config)
        self._arm_joint_names = list(self._robot.arm_joint_names)
        self._ee_offset_xyz = self._robot.ee_offset_xyz

        self._model = self._build_model(scene, per_world_scenes=per_world_scenes)
        # Reconcile dof_per_world with the *finalized* model. The unfinalized
        # builder's joint_coord_count under-reports for some Newton free-joint
        # layouts (the cube's quaternion adds an extra coord that isn't
        # counted by `single.joint_coord_count`); the symptom is an OOB into
        # joint_target_pos at world_count >= 16. The total q size after
        # finalize is authoritative.
        try:
            ref_state = self._model.state()
            total_q = int(ref_state.joint_q.shape[0])
            if total_q % self._world_count == 0:
                computed = total_q // self._world_count
                if computed != self._dof_per_world:
                    self._dof_per_world = computed
        except Exception:
            # Don't gate model creation on this defensive check.
            pass
        self._viewer = self._create_viewer()
        self._viewer.set_model(self._model)
        if hasattr(self._viewer, "set_camera"):
            self._viewer.set_camera(
                pos=self._wp.vec3(1.1, -1.4, 0.9), pitch=-18.0, yaw=45.0
            )
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
        assert self._control is not None

        if target_joints is not None:
            arr = np.asarray(target_joints, dtype=np.float64).ravel()
            n_arm = len(self._w_arm_q)
            if arr.shape != (n_arm,):
                raise ValueError(f"target_joints must have shape ({n_arm},), got {arr.shape}")
            target = self._control.joint_target_pos.numpy()
            target_size = target.shape[0]
            # Diagnose stride/index mismatches up front rather than letting
            # numpy raise an opaque OOB N steps later.
            max_local_q = max(self._w_arm_q) if self._w_arm_q else 0
            max_idx = (self._world_count - 1) * self._dof_per_world + max_local_q
            if max_idx >= target_size:
                raise RuntimeError(
                    f"Newton joint_target_pos size mismatch: target.shape[0]={target_size}, "
                    f"world_count={self._world_count}, dof_per_world={self._dof_per_world}, "
                    f"max(_w_arm_q)={max_local_q}, computed max_idx={max_idx}. "
                    f"_w_arm_q={self._w_arm_q}  _w_gripper_q={self._w_gripper_q}"
                )
            for w in range(self._world_count):
                for local_q, q in zip(self._w_arm_q, arr):
                    target[w * self._dof_per_world + local_q] = float(q)
            arr_wp = self._wp.array(target, dtype=self._control.joint_target_pos.dtype)
            self._wp.copy(self._control.joint_target_pos, arr_wp)

        if gripper is not None:
            t = float(np.clip(gripper, 0.0, 1.0))
            finger_q = (
                self._robot.gripper_open_qpos * (1.0 - t)
                + self._robot.gripper_closed_qpos * t
            )
            target = self._control.joint_target_pos.numpy()
            for w in range(self._world_count):
                for local_q in self._w_gripper_q:
                    target[w * self._dof_per_world + local_q] = finger_q
            arr_wp = self._wp.array(target, dtype=self._control.joint_target_pos.dtype)
            self._wp.copy(self._control.joint_target_pos, arr_wp)

        self._state_0.clear_forces()
        self._model.collide(self._state_0, self._contacts)
        self._solver.step(self._state_0, self._state_1, self._control, self._contacts, self._dt)
        self._state_0, self._state_1 = self._state_1, self._state_0
        self._t += self._dt
        self._log_viewer_state()

    def _obs_for_world(self, w: int, q: np.ndarray, body_q: np.ndarray) -> Observation:
        arm_joints = np.array([q[self._dof_per_world * w + i] for i in self._w_arm_q], dtype=np.float64)
        finger_positions = [float(q[self._dof_per_world * w + i]) for i in self._w_gripper_q]
        gripper_width = float(sum(abs(v) for v in finger_positions))
        ee_row = body_q[self._ee_body_abs(w)]
        ee_body_pose = _body_pose_from_row(ee_row)
        rotated = _rotate_vec(ee_body_pose.quat_xyzw, self._ee_offset_xyz)
        ee_xyz = np.asarray(ee_body_pose.xyz, dtype=np.float64) + rotated
        ee_pose = Pose(
            xyz=(float(ee_xyz[0]), float(ee_xyz[1]), float(ee_xyz[2])),
            quat_xyzw=ee_body_pose.quat_xyzw,
        )
        objects = {
            oid: _body_pose_from_row(body_q[self._obj_body_abs(oid, w)])
            for oid in self._w_obj_body
        }
        return Observation(
            rgb=np.zeros((self._render_h, self._render_w, 3), dtype=np.uint8),
            depth=None,
            robot_joints=arm_joints,
            ee_pose=ee_pose,
            gripper_width=gripper_width,
            scene_objects=objects,
            timestamp=self._t,
            camera_intrinsics=None,
            camera_extrinsics=None,
        )

    def step_all(
        self,
        targets: np.ndarray,
        grippers: np.ndarray,
    ) -> None:
        """Per-world joint targets for RL training.

        Args:
            targets:  (N, n_arm) — absolute joint positions per world
            grippers: (N,) — gripper command ∈ [0, 1] per world
        """
        assert self._state_0 is not None
        assert self._control is not None

        targets = np.asarray(targets, dtype=np.float64)
        grippers_arr = np.asarray(grippers, dtype=np.float64)
        N = self._world_count
        n_arm = len(self._w_arm_q)

        if targets.shape != (N, n_arm):
            raise ValueError(f"targets must be ({N}, {n_arm}), got {targets.shape}")
        if grippers_arr.shape != (N,):
            raise ValueError(f"grippers must be ({N},), got {grippers_arr.shape}")

        target = self._control.joint_target_pos.numpy()
        for w in range(N):
            for local_q, q in zip(self._w_arm_q, targets[w]):
                target[w * self._dof_per_world + local_q] = float(q)
            t = float(np.clip(grippers_arr[w], 0.0, 1.0))
            finger_q = (
                self._robot.gripper_open_qpos * (1.0 - t)
                + self._robot.gripper_closed_qpos * t
            )
            for local_q in self._w_gripper_q:
                target[w * self._dof_per_world + local_q] = finger_q

        arr_wp = self._wp.array(target, dtype=self._control.joint_target_pos.dtype)
        self._wp.copy(self._control.joint_target_pos, arr_wp)

        self._state_0.clear_forces()
        self._model.collide(self._state_0, self._contacts)
        self._solver.step(self._state_0, self._state_1, self._control, self._contacts, self._dt)
        self._state_0, self._state_1 = self._state_1, self._state_0
        self._t += self._dt
        self._log_viewer_state()

    def observe(self) -> Observation:
        """World-0 observation (backward-compatible single-world interface)."""
        assert self._state_0 is not None
        q = self._state_0.joint_q.numpy()
        body_q = self._state_0.body_q.numpy()
        return self._obs_for_world(0, q, body_q)

    def observe_all(self) -> list[Observation]:
        """One Observation per parallel world."""
        assert self._state_0 is not None
        q = self._state_0.joint_q.numpy()
        body_q = self._state_0.body_q.numpy()
        return [self._obs_for_world(w, q, body_q) for w in range(self._world_count)]

    def get_object_pose(self, object_id: str) -> Pose | None:
        if object_id not in self._w_obj_body or self._state_0 is None:
            return None
        body_q = self._state_0.body_q.numpy()
        return _body_pose_from_row(body_q[self._obj_body_abs(object_id, 0)])

    def set_object_pose(self, object_id: str, pose: Pose) -> None:
        raise NotImplementedError("Newton backend does not support teleporting objects in-place")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

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

    @property
    def n_worlds(self) -> int:
        return self._world_count
