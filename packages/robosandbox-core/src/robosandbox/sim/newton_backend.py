"""Newton implementation of the SimBackend protocol.

This is intentionally narrower than the MuJoCo backend today:

- scene loading supports the Franka-style MJCF + primitive box objects
- observations expose robot/object state plus optional batched RGB
  (opt-in via ``enable_camera=True``; uses ``newton.sensors.SensorTiledCamera``
  to raytrace one shaded image per world on the GPU)
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


def _rotate_vec_batch(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) by unit quaternion(s) in batch.

    q_xyzw: (N, 4) quaternions in (x, y, z, w) order.
    v:      (3,) single vector or (N, 3) per-quaternion vectors.
    returns (N, 3).

    Uses the identity: v' = v + 2 * q_xyz × (qw * v + q_xyz × v).
    """
    qx, qy, qz, qw = q_xyzw[:, 0], q_xyzw[:, 1], q_xyzw[:, 2], q_xyzw[:, 3]
    if v.ndim == 1:
        v = np.broadcast_to(v[None, :], (q_xyzw.shape[0], 3))
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    c1x = qy * vz - qz * vy
    c1y = qz * vx - qx * vz
    c1z = qx * vy - qy * vx
    tx = c1x + qw * vx
    ty = c1y + qw * vy
    tz = c1z + qw * vz
    c2x = qy * tz - qz * ty
    c2y = qz * tx - qx * tz
    c2z = qx * ty - qy * tx
    return np.stack([vx + 2.0 * c2x, vy + 2.0 * c2y, vz + 2.0 * c2z], axis=1)


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
    base_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


def _look_at_quat(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> Any:
    """OpenGL-style look-at quat (xyzw) used by Newton SensorTiledCamera.

    Camera looks down -Z in its own frame, +Y up. Returns a wp.quatf — caller
    supplies wp via the lazy import on the backend instance.
    """
    import warp as wp  # local: only when camera is enabled

    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-9
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-9
    cam_up = np.cross(right, forward)
    R = np.column_stack([right, cam_up, -forward])
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return wp.quatf(float(qx), float(qy), float(qz), float(qw))


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
        enable_camera: bool = False,
        camera_pos: tuple[float, float, float] = (1.1, -1.4, 0.9),
        camera_look_at: tuple[float, float, float] = (0.4, 0.0, 0.1),
        camera_fov_deg: float = 45.0,
        max_triangle_pairs: int | None = None,
    ):
        self._render_h, self._render_w = render_size
        self._camera = camera
        self._viewer_kind = viewer
        self._port = int(port)
        self._device = device
        self._dt = float(dt)
        self._world_count = max(1, int(world_count))
        # Newton's CollisionPipeline default is 1M triangle pairs, allocated
        # once at model build. At ≥256 worlds with compound-mesh ports the
        # pipeline silently drops contacts above the cap (logged as
        # "Triangle pair buffer overflowed" warnings) — this is what made the
        # 1024-world insertion training metric (64%) diverge from 64-world
        # deployment (98%). Auto-scale with world_count: 8K pairs/world is a
        # safe upper bound for realistic scenes. Caller may override.
        if max_triangle_pairs is None:
            # 16K pairs/world covers compound-mesh ports (4 walls + 4 chamfer
            # plates + base + plug + table + floor) at random positions where
            # broadphase generates more candidate pairs than at fixed pose.
            # Empirically at 256 worlds with random_xy we saw ~2.4M pairs
            # requested; 16K × 256 = 4.1M is comfortably above with headroom.
            max_triangle_pairs = max(2_000_000, self._world_count * 16000)
        self._max_triangle_pairs = int(max_triangle_pairs)
        # When enabled, observe()/observe_all() raytrace one RGB image per
        # world via newton.sensors.SensorTiledCamera. Off by default to keep
        # state-only callers (RL, headless eval) free of GPU render cost.
        self._camera_enabled = bool(enable_camera)
        self._cam_pos = tuple(float(v) for v in camera_pos)
        self._cam_look_at = tuple(float(v) for v in camera_look_at)
        self._cam_fov_deg = float(camera_fov_deg)
        self._sensor: Any = None
        self._cam_rays: Any = None
        self._cam_color_buf: Any = None
        self._cam_xforms: Any = None

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

        # Per-world layout (set during load).
        # _dof_per_world: stride for the FULL state vector (joint_q),
        #   includes the cube freejoint's quaternion (7 coords).
        # _actuators_per_world: stride for the ACTUATOR target vector
        #   (joint_target_pos), sized by joint_qd which uses 6 coords for
        #   freejoints. They differ by 1 per freejoint and conflating them
        #   produces an OOB into joint_target_pos at world_count >= 16.
        self._bodies_per_world: int = 0
        self._dof_per_world: int = 0
        self._actuators_per_world: int = 0
        # Indices *within one world* (0-indexed from world start)
        self._w_arm_q: list[int] = []       # relative joint_q indices for arm
        self._w_gripper_q: list[int] = []   # relative joint_q indices for fingers
        self._w_ee_body: int = -1           # relative body index for EE
        self._w_obj_body: dict[str, int] = {}  # obj_id -> relative body index
        # Static-marker poses per world, in global frame (YAML pose + grid offset).
        # Static objects have no model body; their pose is constant and merged into
        # the observation's scene_objects dict alongside dynamic body poses.
        self._world_static_obj_poses: list[dict[str, Pose]] = []

        self._arm_joint_names: list[str] = []
        self._ee_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._t: float = 0.0
        # Gravity-compensation feedforward. When set, step() reads the
        # current arm + gripper joint values, calls fn(arm_joints,
        # gripper_qpos) → torques, and writes them to control.joint_f
        # before stepping. Cancels gravity sag so the position-control PD
        # only handles small tracking residuals (sub-mm vs 10-20 mm without).
        self._gravity_torque_fn: Any = None

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
        base_raw = raw.get("base_pose") or {}
        base_xyz = tuple(float(v) for v in base_raw.get("xyz", (0.0, 0.0, 0.0)))
        base_quat = tuple(float(v) for v in base_raw.get("quat_xyzw", (0.0, 0.0, 0.0, 1.0)))
        if len(base_xyz) != 3:
            raise ValueError(f"{path}: base_pose.xyz must have 3 components, got {base_xyz}")
        if len(base_quat) != 4:
            raise ValueError(f"{path}: base_pose.quat_xyzw must have 4 components, got {base_quat}")
        return _RobotConfig(
            arm_joint_names=tuple(str(v) for v in raw["arm"]["joints"]),
            gripper_joint_names=tuple(str(v) for v in raw["gripper"]["joints"]),
            home_qpos=tuple(float(v) for v in raw["arm"]["home_qpos"]),
            gripper_open_qpos=float(raw["gripper"]["open_qpos"]),
            gripper_closed_qpos=float(raw["gripper"]["closed_qpos"]),
            ee_attach_body=str(raw["ee_site"]["inject"]["attach_body"]),
            ee_offset_xyz=tuple(float(v) for v in raw["ee_site"]["inject"]["xyz"]),
            base_xyz=base_xyz,
            base_quat_xyzw=base_quat,
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
        # Anchor the robot at base_pose so its forward kinematics agree with
        # MuJoCoBackend (which applies base_pose via scene/robot_loader). Without
        # this, the robot sits at MJCF origin and any IK target expressed in
        # world frame ends up off by base_xyz.
        bx, by, bz = self._robot.base_xyz
        bqx, bqy, bqz, bqw = self._robot.base_quat_xyzw
        builder.add_mjcf(
            str(scene.robot_urdf),
            xform=wp.transform(wp.vec3(bx, by, bz), wp.quat(bqx, bqy, bqz, bqw)),
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

        # Match the elevated friction MuJoCoBackend gives box objects via
        # `robot_loader._inject_objects` (friction=[1.5, 0.1, 0.01]) — Newton's
        # default ShapeConfig is the canonical MuJoCo (1.0, 0.005, 0.0001),
        # which is too slippery to hold a 50g cube against a 1g/cm finger
        # PD residual during lift.  Without this match the gripper closes
        # around the cube in both sims, but Newton's grasp slips at the lift
        # transient while MuJoCo's holds — the dominant cause of the
        # distilled-MLP eval gap (34% MuJoCo vs 0% Newton).
        cube_cfg = copy.copy(builder.default_shape_cfg)
        cube_cfg.mu = 1.5
        cube_cfg.mu_torsional = 0.1
        cube_cfg.mu_rolling = 0.01
        for obj in scene.objects:
            if obj.kind != "box":
                raise NotImplementedError(
                    f"Newton backend currently supports box objects only, got {obj.kind!r}"
                )
            x, y, z = obj.pose.xyz
            qx, qy, qz, qw = obj.pose.quat_xyzw
            sx, sy, sz = obj.size
            sx_f, sy_f, sz_f = float(sx), float(sy), float(sz)
            # Static objects are anchored to the world body (body=-1) — no
            # freejoint, no gravity. Used for visual targets/markers in
            # reach-style tasks where the marker must stay at YAML pose.
            if getattr(obj, "static", False):
                builder.add_shape_box(
                    body=-1,
                    hx=sx_f, hy=sy_f, hz=sz_f,
                    xform=wp.transform(wp.vec3(x, y, z), wp.quat(qx, qy, qz, qw)),
                    cfg=cube_cfg,
                )
                continue
            # Without lock_inertia + an explicit inertia tensor, Newton's
            # finalize() recomputes mass and inertia from shape volume × a
            # default density (~4630 kg/m^3 for boxes), silently ignoring
            # the `mass=` we pass to add_body and clamping inertia by the
            # validate_and_correct_inertia kernel. That puts a 24 mm cube
            # at 0.064 kg / 1.33e-6 kg·m² in Newton vs MuJoCo's 0.05 kg /
            # 4.8e-6 kg·m² — a 28% weight gap that breaks lift parity.
            mass = float(obj.mass)
            ixx = mass * ((2 * sy_f) ** 2 + (2 * sz_f) ** 2) / 12.0
            iyy = mass * ((2 * sx_f) ** 2 + (2 * sz_f) ** 2) / 12.0
            izz = mass * ((2 * sx_f) ** 2 + (2 * sy_f) ** 2) / 12.0
            inertia_mat = wp.mat33(ixx, 0.0, 0.0, 0.0, iyy, 0.0, 0.0, 0.0, izz)
            builder.add_body(
                xform=wp.transform(wp.vec3(x, y, z), wp.quat(qx, qy, qz, qw)),
                mass=mass,
                inertia=inertia_mat,
                lock_inertia=True,
                label=obj.id,
            )
            builder.add_shape_box(
                body=builder.body_count - 1,
                hx=sx_f, hy=sy_f, hz=sz_f,
                cfg=cube_cfg,
            )

        target_q = [*self._robot.home_qpos, self._robot.gripper_open_qpos, self._robot.gripper_open_qpos]
        builder.joint_q[: len(target_q)] = target_q
        builder.joint_target_pos[: len(target_q)] = target_q
        # PD gains taken from MuJoCo's panda.xml. PD alone leaves a
        # gravity-induced steady-state error of ~10-20 mrad on stretched
        # configurations; production-quality tracking requires gravity
        # feedforward via control.joint_f (see set_gravity_compensation).
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

        # Discover joint/body indices from a finalized single-world model.
        # Static objects don't have bodies (anchored to world via add_shape_box(body=-1));
        # they're tracked separately in _world_static_obj_poses for observation injection.
        ref_model = copy.deepcopy(single).finalize()
        self._w_arm_q = [self._find_joint_q_index_in(ref_model, n) for n in self._robot.arm_joint_names]
        self._w_gripper_q = [self._find_joint_q_index_in(ref_model, n) for n in self._robot.gripper_joint_names]
        self._w_ee_body = self._find_body_index_in(ref_model, self._robot.ee_attach_body)
        self._w_obj_body = {
            obj.id: self._find_body_index_in(ref_model, obj.id)
            for obj in scene.objects
            if not getattr(obj, "static", False)
        }

        # Per-world static-object poses, in GLOBAL frame (YAML pose + grid offset)
        # so they're directly comparable to dynamic body_q poses in observation.
        wp = self._wp
        offsets = _grid_offsets(self._world_count)
        self._world_static_obj_poses: list[dict[str, Pose]] = []
        for w in range(self._world_count):
            ox, oy, oz = offsets[w] if self._world_count > 1 else (0.0, 0.0, 0.0)
            world_scene = (
                per_world_scenes[w]
                if per_world_scenes is not None and w < len(per_world_scenes)
                else scene
            )
            poses_w: dict[str, Pose] = {}
            for obj in world_scene.objects:
                if getattr(obj, "static", False):
                    x, y, z = obj.pose.xyz
                    poses_w[obj.id] = Pose(
                        xyz=(float(x + ox), float(y + oy), float(z + oz)),
                        quat_xyzw=tuple(float(v) for v in obj.pose.quat_xyzw),
                    )
            self._world_static_obj_poses.append(poses_w)

        if self._world_count == 1:
            # World 0 honours the per-world scene if one was passed.
            if per_world_scenes is not None:
                return self._build_single_builder(per_world_scenes[0]).finalize()
            return single.finalize()

        # Tile N worlds in a grid. Each world gets its own builder when
        # per_world_scenes is set so randomization actually lands in the
        # finalized model (object xform is baked at add_body time).
        newton = self._newton
        multi = newton.ModelBuilder()
        for w, (ox, oy, oz) in enumerate(offsets):
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
        # Replace Newton's default 1M-triangle CollisionPipeline with one sized
        # to actually fit our compound-mesh scenes at scale. Without this,
        # mujoco_warp drops plug↔chamfer contacts at world_count ≥ 256, which
        # silently degrades insertion success even though the policy is fine.
        # Try the public import path first (newton.sim.collide); fall back to
        # the private one (newton._src.sim.collide) since Newton 1.1 keeps
        # CollisionPipeline behind the underscore namespace.
        _CP = None
        try:
            from newton.sim.collide import CollisionPipeline as _CP  # type: ignore
        except Exception:
            try:
                from newton._src.sim.collide import CollisionPipeline as _CP  # type: ignore
            except Exception as exc:
                print(f"[newton] WARNING: could not import CollisionPipeline: {exc!r}")
        if _CP is not None:
            try:
                self._model._collision_pipeline = _CP(
                    self._model,
                    broad_phase="explicit",
                    max_triangle_pairs=self._max_triangle_pairs,
                )
                print(f"[newton] installed CollisionPipeline with "
                      f"max_triangle_pairs={self._max_triangle_pairs:,}")
            except Exception as exc:
                print(f"[newton] WARNING: CollisionPipeline construction failed "
                      f"(max_triangle_pairs={self._max_triangle_pairs}): {exc!r}")
        # Newton stores actuator targets (joint_target_pos / joint_target_ke /
        # …) on a vector sized to ACTUATED joints only; free joints (the
        # cube) have no actuator slot. So the per-world stride for
        # control vectors is *smaller* than the per-world stride for the
        # full state vector (joint_q) by the freejoint's contribution.
        # Authoritative source: the freshly-built model itself.
        try:
            ref_control = self._model.control()
            ref_state = self._model.state()
            total_targets = int(ref_control.joint_target_pos.shape[0])
            total_q = int(ref_state.joint_q.shape[0])
            if total_targets % self._world_count == 0:
                self._actuators_per_world = total_targets // self._world_count
            else:
                self._actuators_per_world = self._dof_per_world
            if total_q % self._world_count == 0:
                self._dof_per_world = total_q // self._world_count
        except Exception:
            # Don't gate model creation on this defensive recompute — fall
            # back to the unfinalized builder counts.
            self._actuators_per_world = self._dof_per_world
        # Pre-compute flat indices for vectorized observation extraction.
        # Used by observe_all_arrays() to avoid Python per-world loops.
        N = self._world_count
        bpw = self._bodies_per_world
        dpw = self._dof_per_world
        world_strides_q = (np.arange(N) * dpw)[:, None]      # (N, 1)
        world_strides_b = (np.arange(N) * bpw)               # (N,)
        self._arm_q_indices_all = (
            world_strides_q + np.asarray(self._w_arm_q, dtype=np.int64)[None, :]
        )                                                    # (N, n_arm)
        if self._w_gripper_q:
            self._gripper_q_indices_all = (
                world_strides_q + np.asarray(self._w_gripper_q, dtype=np.int64)[None, :]
            )                                                # (N, n_grip)
        else:
            self._gripper_q_indices_all = np.zeros((N, 0), dtype=np.int64)
        self._ee_body_indices_all = world_strides_b + int(self._w_ee_body)  # (N,)
        self._obj_body_indices_all = {
            oid: world_strides_b + int(self._w_obj_body[oid])
            for oid in self._w_obj_body
        }
        # Precompute static (N, 3) and (N, 4) tables.
        self._static_obj_xyz_all: dict[str, np.ndarray] = {}
        self._static_obj_quat_all: dict[str, np.ndarray] = {}
        if self._world_static_obj_poses:
            # Discover the union of static obj ids (assume all worlds have same set)
            static_ids = list(self._world_static_obj_poses[0].keys())
            for oid in static_ids:
                xyz_arr = np.zeros((N, 3), dtype=np.float64)
                quat_arr = np.zeros((N, 4), dtype=np.float64)
                quat_arr[:, 3] = 1.0
                for w in range(N):
                    pose = self._world_static_obj_poses[w].get(oid)
                    if pose is not None:
                        xyz_arr[w] = pose.xyz
                        quat_arr[w] = pose.quat_xyzw
                self._static_obj_xyz_all[oid] = xyz_arr
                self._static_obj_quat_all[oid] = quat_arr

        self._viewer = self._create_viewer()
        self._viewer.set_model(self._model)
        if hasattr(self._viewer, "set_camera"):
            self._viewer.set_camera(
                pos=self._wp.vec3(1.1, -1.4, 0.9), pitch=-18.0, yaw=45.0
            )
        if self._camera_enabled:
            self._setup_tiled_camera()
        self.reset()

    def _setup_tiled_camera(self) -> None:
        """Build SensorTiledCamera + per-world camera transforms (one camera per world).

        Newton's pinhole rays follow the OpenGL convention (camera looks down
        -Z, +Y up). Each world's camera follows the world's grid offset so all
        worlds see their own scene framed identically.
        """
        from newton.sensors import SensorTiledCamera

        wp = self._wp
        self._sensor = SensorTiledCamera(model=self._model)
        self._sensor.utils.create_default_light(enable_shadows=True)
        self._sensor.utils.assign_checkerboard_material_to_all_shapes()

        self._cam_rays = self._sensor.utils.compute_pinhole_camera_rays(
            self._render_w, self._render_h, math.radians(self._cam_fov_deg)
        )
        self._cam_color_buf = self._sensor.utils.create_color_image_output(
            self._render_w, self._render_h, camera_count=1
        )

        eye0 = np.asarray(self._cam_pos, dtype=np.float64)
        target0 = np.asarray(self._cam_look_at, dtype=np.float64)
        up = np.array([0.0, 0.0, 1.0])
        per_world = []
        for ox, oy, oz in _grid_offsets(self._world_count):
            offset = np.array([ox, oy, oz])
            eye_w = eye0 + offset
            q = _look_at_quat(eye_w, target0 + offset, up)
            per_world.append(wp.transformf(wp.vec3f(*eye_w.astype(np.float32)), q))
        # Newton expects shape (camera_count, world_count); one camera per world.
        self._cam_xforms = wp.array([per_world], dtype=wp.transformf)

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
            stride = self._actuators_per_world or self._dof_per_world
            for w in range(self._world_count):
                for local_q, q in zip(self._w_arm_q, arr):
                    target[w * stride + local_q] = float(q)
            arr_wp = self._wp.array(target, dtype=self._control.joint_target_pos.dtype)
            self._wp.copy(self._control.joint_target_pos, arr_wp)

        if gripper is not None:
            t = float(np.clip(gripper, 0.0, 1.0))
            finger_q = (
                self._robot.gripper_open_qpos * (1.0 - t)
                + self._robot.gripper_closed_qpos * t
            )
            target = self._control.joint_target_pos.numpy()
            stride = self._actuators_per_world or self._dof_per_world
            for w in range(self._world_count):
                for local_q in self._w_gripper_q:
                    target[w * stride + local_q] = finger_q
            arr_wp = self._wp.array(target, dtype=self._control.joint_target_pos.dtype)
            self._wp.copy(self._control.joint_target_pos, arr_wp)

        self._apply_gravity_compensation()

        self._state_0.clear_forces()
        self._model.collide(self._state_0, self._contacts)
        self._solver.step(self._state_0, self._state_1, self._control, self._contacts, self._dt)
        self._state_0, self._state_1 = self._state_1, self._state_0
        self._t += self._dt
        self._log_viewer_state()

    def set_gravity_compensation(self, fn: Any) -> None:
        """Register a per-step gravity-compensation feedforward.

        ``fn`` is a callable ``(arm_joints: np.ndarray, gripper_qpos: float)
        → np.ndarray of shape (n_arm + n_gripper,)`` returning the torque
        needed to hold the current configuration against gravity. Called
        each :meth:`step` before the solver advances; result is broadcast
        across all worlds and written into ``control.joint_f``.

        Pair this with :meth:`MuJoCoBackend.compute_gravity_torque` when a
        kinematics-oracle MuJoCo backend is loaded with the same robot:
        PD then only handles tracking residuals (sub-mm error vs 10-20 mm
        without compensation on a stretched-out Franka).

        Pass ``None`` to disable.
        """
        self._gravity_torque_fn = fn

    def _apply_gravity_compensation(self) -> None:
        if self._gravity_torque_fn is None or self._state_0 is None:
            return
        q = self._state_0.joint_q.numpy()
        arm_joints = np.array([q[i] for i in self._w_arm_q], dtype=np.float64)
        gripper_qpos = float(q[self._w_gripper_q[0]]) if self._w_gripper_q else 0.0
        ff = np.asarray(self._gravity_torque_fn(arm_joints, gripper_qpos), dtype=np.float64).ravel()
        n_arm = len(self._w_arm_q)
        n_grip = len(self._w_gripper_q)
        if ff.shape != (n_arm + n_grip,):
            raise ValueError(
                f"gravity_torque_fn returned shape {ff.shape}, "
                f"expected ({n_arm + n_grip},)"
            )
        joint_f = self._control.joint_f.numpy()
        stride = self._actuators_per_world or self._dof_per_world
        for w in range(self._world_count):
            for local_q, t in zip(self._w_arm_q, ff[:n_arm]):
                joint_f[w * stride + local_q] = float(t)
            for local_q, t in zip(self._w_gripper_q, ff[n_arm:]):
                joint_f[w * stride + local_q] = float(t)
        joint_f_wp = self._wp.array(joint_f, dtype=self._control.joint_f.dtype)
        self._wp.copy(self._control.joint_f, joint_f_wp)

    def _render_all_rgb(self) -> np.ndarray | None:
        """Return (W, H, W_pix, 3) uint8 RGB across worlds, or None if disabled."""
        if not self._camera_enabled or self._sensor is None:
            return None
        from newton.sensors import SensorTiledCamera

        self._sensor.update(
            self._state_0,
            self._cam_xforms,
            self._cam_rays,
            color_image=self._cam_color_buf,
            clear_data=SensorTiledCamera.GRAY_CLEAR_DATA,
        )
        # uint32 RGBA per pixel → unpack to uint8 (W, 1, H, W_pix, 4) → drop alpha
        img32 = self._cam_color_buf.numpy()
        rgba = img32.view(np.uint8).reshape(*img32.shape, 4)
        return rgba[:, 0, :, :, :3].copy()  # (W, H, W_pix, 3)

    def _obs_for_world(
        self,
        w: int,
        q: np.ndarray,
        body_q: np.ndarray,
        rgb_all: np.ndarray | None = None,
    ) -> Observation:
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
        # Inject static-marker poses (no body in model — tracked from YAML).
        if self._world_static_obj_poses and w < len(self._world_static_obj_poses):
            objects.update(self._world_static_obj_poses[w])
        rgb = (
            rgb_all[w]
            if rgb_all is not None
            else np.zeros((self._render_h, self._render_w, 3), dtype=np.uint8)
        )
        return Observation(
            rgb=rgb,
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
        # Use the actuator stride (joint_target_pos is sized by joint_qd,
        # which contributes 6 per freejoint vs joint_q's 7 — same fix as
        # commit 3b141b0 applied to the single-world step()).
        stride = self._actuators_per_world or self._dof_per_world
        for w in range(N):
            for local_q, q in zip(self._w_arm_q, targets[w]):
                target[w * stride + local_q] = float(q)
            t = float(np.clip(grippers_arr[w], 0.0, 1.0))
            finger_q = (
                self._robot.gripper_open_qpos * (1.0 - t)
                + self._robot.gripper_closed_qpos * t
            )
            for local_q in self._w_gripper_q:
                target[w * stride + local_q] = finger_q

        arr_wp = self._wp.array(target, dtype=self._control.joint_target_pos.dtype)
        self._wp.copy(self._control.joint_target_pos, arr_wp)

        self._apply_gravity_compensation()

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
        rgb_all = self._render_all_rgb()
        return self._obs_for_world(0, q, body_q, rgb_all)

    def observe_all_arrays(self) -> dict[str, Any]:
        """Vectorized observation: returns batched numpy arrays directly.

        Skips per-world Observation/Pose dataclass construction → ~10-50×
        faster than observe_all() for large N. Used by training rollout loops
        that only need raw arrays for reward + policy input.

        Returns dict:
          - 'joints':         (N, n_arm) arm joint positions
          - 'gripper_width':  (N,)       sum of |finger_q|
          - 'ee_xyz':         (N, 3)     EE position in global frame
          - 'ee_quat':        (N, 4)     EE quaternion (xyzw)
          - 'obj_xyz':        dict[oid, (N, 3)]   object positions (dynamic + static)
          - 'obj_quat':       dict[oid, (N, 4)]   object quaternions
        """
        assert self._state_0 is not None
        q = self._state_0.joint_q.numpy()
        body_q = self._state_0.body_q.numpy()

        joints = q[self._arm_q_indices_all]                          # (N, n_arm)
        if self._gripper_q_indices_all.shape[1] > 0:
            finger_q = q[self._gripper_q_indices_all]                # (N, n_grip)
            gripper_width = np.sum(np.abs(finger_q), axis=1)         # (N,)
        else:
            gripper_width = np.zeros(self._world_count, dtype=np.float64)

        ee_rows = body_q[self._ee_body_indices_all]                  # (N, 7)
        ee_body_xyz = ee_rows[:, :3]
        ee_quat = ee_rows[:, 3:]                                     # (N, 4) xyzw
        ee_offset_v = np.asarray(self._ee_offset_xyz, dtype=np.float64)
        if np.any(ee_offset_v != 0.0):
            ee_xyz = ee_body_xyz + _rotate_vec_batch(ee_quat, ee_offset_v)
        else:
            ee_xyz = ee_body_xyz

        obj_xyz: dict[str, np.ndarray] = {}
        obj_quat: dict[str, np.ndarray] = {}
        for oid, indices in self._obj_body_indices_all.items():
            rows = body_q[indices]
            obj_xyz[oid] = rows[:, :3]
            obj_quat[oid] = rows[:, 3:]
        # Static markers — use precomputed (N, 3) tables.
        for oid, xyz_arr in self._static_obj_xyz_all.items():
            obj_xyz[oid] = xyz_arr
            obj_quat[oid] = self._static_obj_quat_all[oid]

        return {
            "joints": joints,
            "gripper_width": gripper_width,
            "ee_xyz": ee_xyz,
            "ee_quat": ee_quat,
            "obj_xyz": obj_xyz,
            "obj_quat": obj_quat,
        }

    def observe_all(self) -> list[Observation]:
        """One Observation per parallel world."""
        assert self._state_0 is not None
        q = self._state_0.joint_q.numpy()
        body_q = self._state_0.body_q.numpy()
        rgb_all = self._render_all_rgb()
        return [self._obs_for_world(w, q, body_q, rgb_all) for w in range(self._world_count)]

    def get_object_pose(self, object_id: str) -> Pose | None:
        # Static markers have a fixed pose tracked outside the simulation state.
        if self._world_static_obj_poses and object_id in self._world_static_obj_poses[0]:
            return self._world_static_obj_poses[0][object_id]
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
