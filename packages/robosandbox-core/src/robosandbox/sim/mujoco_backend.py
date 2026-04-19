"""MuJoCo implementation of SimBackend.

Compiles a Scene (via scene.build_model) into an MjModel + RobotSpec,
exposes step/observe/object-pose queries, and offers offscreen rendering
for the Observation's rgb/depth channels. The backend is robot-agnostic
— all joint/actuator/site names flow from the RobotSpec produced by the
scene layer, so any arm (built-in, Franka, UR5, …) Just Works once its
MJCF + sidecar are loadable.
"""

from __future__ import annotations

import numpy as np

try:
    import mujoco
except ImportError as e:
    raise ImportError(
        "robosandbox requires mujoco>=3.2 — install via `pip install mujoco`"
    ) from e

from robosandbox.scene import build_model
from robosandbox.scene.robot_spec import RobotSpec
from robosandbox.types import (
    CameraIntrinsics,
    Observation,
    Pose,
    Scene,
)


def _mat_to_quat_xyzw(mat9: np.ndarray) -> tuple[float, float, float, float]:
    """MuJoCo gives 3x3 rotation matrix; convert to (x,y,z,w) quaternion."""
    out = np.empty(4, dtype=np.float64)  # (w, x, y, z)
    mujoco.mju_mat2Quat(out, mat9.ravel())
    return (float(out[1]), float(out[2]), float(out[3]), float(out[0]))


def _quat_wxyz(q_xyzw: tuple[float, float, float, float]) -> np.ndarray:
    """Accept (x,y,z,w), return (w,x,y,z) as numpy array — MuJoCo's convention."""
    x, y, z, w = q_xyzw
    return np.array([w, x, y, z], dtype=np.float64)


class MuJoCoBackend:
    """SimBackend implementation using MuJoCo 3.x."""

    def __init__(self, render_size: tuple[int, int] = (480, 640), camera: str = "scene"):
        self._render_h, self._render_w = render_size
        self._camera = camera
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._renderer: mujoco.Renderer | None = None
        self._depth_renderer: mujoco.Renderer | None = None
        # Show geom groups 0-3 in the default render. Menagerie's Franka
        # marks collision-only geoms as group=3 (MuJoCo's default renderer
        # hides 3/4/5); without this, the bundled collision-only Franka
        # would render as invisible. Group 2 covers our mesh visual geoms,
        # group 3 covers collision. Groups 4/5 stay hidden (debug helpers).
        self._render_opt = mujoco.MjvOption()
        self._render_opt.geomgroup[:] = [1, 1, 1, 1, 0, 0]
        self._scene: Scene | None = None
        self._robot: RobotSpec | None = None
        self._arm_qpos_adr: list[int] = []
        self._gripper_qpos_adr: int = -1
        self._arm_ctrl_adr: list[int] = []
        self._gripper_ctrl_adr: int = -1
        self._ee_site_id: int = -1
        self._obj_body_ids: dict[str, int] = {}
        self._t: float = 0.0

    # ---- lifecycle -------------------------------------------------------
    def load(self, scene: Scene) -> None:
        self._model, self._robot = build_model(scene)
        # MuJoCo's offscreen framebuffer defaults to 640x480; the Renderer
        # refuses sizes beyond it unless bumped here (equivalent to setting
        # <visual><global offwidth offheight/> in the MJCF).
        if self._model.vis.global_.offwidth < self._render_w:
            self._model.vis.global_.offwidth = self._render_w
        if self._model.vis.global_.offheight < self._render_h:
            self._model.vis.global_.offheight = self._render_h
        self._data = mujoco.MjData(self._model)
        self._renderer = mujoco.Renderer(self._model, height=self._render_h, width=self._render_w)
        self._depth_renderer = mujoco.Renderer(
            self._model, height=self._render_h, width=self._render_w
        )
        self._depth_renderer.enable_depth_rendering()
        self._scene = scene

        spec = self._robot
        self._arm_qpos_adr = [self._model.joint(j).qposadr[0] for j in spec.arm_joint_names]
        self._gripper_qpos_adr = self._model.joint(spec.gripper_primary_joint).qposadr[0]
        self._arm_ctrl_adr = [self._model.actuator(a).id for a in spec.arm_actuator_names]
        self._gripper_ctrl_adr = self._model.actuator(spec.gripper_actuator_name).id
        self._ee_site_id = self._model.site(spec.ee_site_name).id
        # Map scene-object id -> body id. For articulated objects (drawer),
        # also register the handle body so perception can locate it by name.
        body_ids: dict[str, int] = {}
        for o in scene.objects:
            body_ids[o.id] = self._model.body(o.id).id
            if o.kind == "drawer":
                handle_name = f"{o.id}_handle"
                body_ids[handle_name] = self._model.body(handle_name).id
        self._obj_body_ids = body_ids
        self.reset()

    def reset(self) -> None:
        assert self._model is not None and self._data is not None and self._robot is not None
        mujoco.mj_resetData(self._model, self._data)
        home = np.asarray(self._robot.home_qpos, dtype=np.float64)
        for adr, q in zip(self._arm_qpos_adr, home):
            self._data.qpos[adr] = float(q)
        self._data.qpos[self._gripper_qpos_adr] = self._robot.gripper_open_qpos
        for adr, q in zip(self._arm_ctrl_adr, home):
            self._data.ctrl[adr] = float(q)
        self._data.ctrl[self._gripper_ctrl_adr] = self._robot.gripper_open_qpos
        mujoco.mj_forward(self._model, self._data)
        self._t = 0.0

    def close(self) -> None:
        # MuJoCo's Renderer.__del__ calls close() a second time at interpreter
        # shutdown when the EGL context is already torn down, raising
        # AttributeError on a half-freed instance. Swallow it — the resources
        # we own are released by the explicit close() here.
        for attr in ("_renderer", "_depth_renderer"):
            r = getattr(self, attr, None)
            if r is not None:
                try:
                    r.close()
                except Exception:
                    pass
                setattr(self, attr, None)

    # ---- stepping --------------------------------------------------------
    def step(
        self,
        target_joints: np.ndarray | None = None,
        gripper: float | None = None,
    ) -> None:
        assert self._model is not None and self._data is not None and self._robot is not None
        n_dof = len(self._robot.arm_joint_names)
        if target_joints is not None:
            arr = np.asarray(target_joints, dtype=np.float64).ravel()
            if arr.shape != (n_dof,):
                raise ValueError(
                    f"target_joints must have shape ({n_dof},), got {arr.shape}"
                )
            for adr, q in zip(self._arm_ctrl_adr, arr):
                self._data.ctrl[adr] = float(q)
        if gripper is not None:
            # Semantic input: 0.0 == open, 1.0 == closed.
            # Lerp form (open*(1-t) + closed*t) — collapses to OLD's
            # `open*(1-t)` when closed==0, preserving bit-exact ctrl values.
            # Physics determinism depends on this; a 1-ulp change compounds
            # over ~1000 steps and can flip grasps from success to failure.
            t = float(np.clip(gripper, 0.0, 1.0))
            ctrl = (
                self._robot.gripper_open_qpos * (1.0 - t)
                + self._robot.gripper_closed_qpos * t
            )
            self._data.ctrl[self._gripper_ctrl_adr] = ctrl
        mujoco.mj_step(self._model, self._data)
        self._t += self._model.opt.timestep

    # ---- observation -----------------------------------------------------
    def observe(self) -> Observation:
        assert self._model is not None and self._data is not None and self._renderer is not None
        assert self._depth_renderer is not None and self._robot is not None

        self._renderer.update_scene(self._data, camera=self._camera, scene_option=self._render_opt)
        rgb = self._renderer.render().copy()
        self._depth_renderer.update_scene(self._data, camera=self._camera, scene_option=self._render_opt)
        depth = self._depth_renderer.render().copy()

        arm_joints = np.array(
            [self._data.qpos[adr] for adr in self._arm_qpos_adr], dtype=np.float64
        )
        gripper_qpos = float(self._data.qpos[self._gripper_qpos_adr])
        # Width = 2 * |qpos|. Works for every parallel gripper we ship where
        # closed_qpos == 0 (built-in: open=-0.035→w=0.07; Franka:
        # open=0.04→w=0.08). Matches OLD backend byte-for-byte so physics
        # determinism is preserved across the refactor.
        gripper_width = 2.0 * abs(gripper_qpos)
        ee_pose = self._ee_pose()
        objects = {
            oid: self._body_pose(bid) for oid, bid in self._obj_body_ids.items()
        }
        intrinsics = self._camera_intrinsics()
        extrinsics = self._camera_extrinsics()
        return Observation(
            rgb=rgb,
            depth=depth,
            robot_joints=arm_joints,
            ee_pose=ee_pose,
            gripper_width=gripper_width,
            scene_objects=objects,
            timestamp=self._t,
            camera_intrinsics=intrinsics,
            camera_extrinsics=extrinsics,
        )

    def _ee_pose(self) -> Pose:
        assert self._data is not None
        xyz = tuple(float(v) for v in self._data.site_xpos[self._ee_site_id])
        mat = np.asarray(self._data.site_xmat[self._ee_site_id]).reshape(3, 3)
        quat = _mat_to_quat_xyzw(mat)
        return Pose(xyz=xyz, quat_xyzw=quat)

    def _body_pose(self, body_id: int) -> Pose:
        assert self._data is not None and self._model is not None
        xyz = tuple(float(v) for v in self._data.xpos[body_id])
        mat = np.asarray(self._data.xmat[body_id]).reshape(3, 3)
        quat = _mat_to_quat_xyzw(mat)
        return Pose(xyz=xyz, quat_xyzw=quat)

    def _camera_intrinsics(self) -> CameraIntrinsics:
        assert self._model is not None
        cam_id = self._model.camera(self._camera).id
        fovy_deg = float(self._model.cam_fovy[cam_id])
        h, w = self._render_h, self._render_w
        fovy = np.deg2rad(fovy_deg)
        fy = 0.5 * h / np.tan(0.5 * fovy)
        fx = fy  # square pixels
        return CameraIntrinsics(fx=fx, fy=fy, cx=w / 2, cy=h / 2, width=w, height=h)

    def _camera_extrinsics(self) -> Pose:
        assert self._data is not None and self._model is not None
        cam_id = self._model.camera(self._camera).id
        xyz = tuple(float(v) for v in self._data.cam_xpos[cam_id])
        mat = np.asarray(self._data.cam_xmat[cam_id]).reshape(3, 3)
        # MuJoCo camera looks down -Z of its own frame; we report the camera
        # body pose directly — downstream math must know this convention.
        quat = _mat_to_quat_xyzw(mat)
        return Pose(xyz=xyz, quat_xyzw=quat)

    # ---- scene queries ---------------------------------------------------
    def get_object_pose(self, object_id: str) -> Pose | None:
        if object_id not in self._obj_body_ids:
            return None
        return self._body_pose(self._obj_body_ids[object_id])

    def set_object_pose(self, object_id: str, pose: Pose) -> None:
        """Teleport a free-body object. Only works if the object has a
        freejoint, which is the default for SceneObjects spawned here.
        """
        assert self._model is not None and self._data is not None
        body = self._model.body(object_id)
        jnt_adr = body.jntadr[0]
        if jnt_adr < 0:
            raise ValueError(f"object '{object_id}' has no freejoint")
        qpos_adr = int(self._model.jnt_qposadr[jnt_adr])
        x, y, z = pose.xyz
        self._data.qpos[qpos_adr : qpos_adr + 3] = (x, y, z)
        self._data.qpos[qpos_adr + 3 : qpos_adr + 7] = _quat_wxyz(pose.quat_xyzw)
        mujoco.mj_forward(self._model, self._data)

    @property
    def n_dof(self) -> int:
        assert self._robot is not None
        return len(self._robot.arm_joint_names)

    @property
    def joint_names(self) -> list[str]:
        assert self._robot is not None
        return list(self._robot.arm_joint_names)

    # ---- internal accessors (for MotionPlanner + Skills) ----------------
    @property
    def model(self) -> mujoco.MjModel:
        assert self._model is not None
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        assert self._data is not None
        return self._data

    @property
    def ee_site_id(self) -> int:
        return self._ee_site_id

    @property
    def arm_qpos_adr(self) -> list[int]:
        return list(self._arm_qpos_adr)

    @property
    def base_body_name(self) -> str:
        assert self._robot is not None
        return self._robot.base_body_name
