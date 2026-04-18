"""``RealRobotBackend`` — stub that satisfies the ``SimBackend`` protocol
for real hardware.

RoboSandbox's skills, motion planners, grasp planners, and the Agent
loop all consume the ``SimBackend`` Protocol (see ``protocols.py``).
They don't care whether they're driving MuJoCo or a real robot — as long
as the backend exposes the same shape:

    load(scene)        -> load/prepare the workspace
    reset()            -> return to home pose
    step(target_joints, gripper)  -> send one control tick
    observe()          -> Observation (rgb, joints, ee_pose, objects)
    get_object_pose(id), set_object_pose(id, pose)
    n_dof, joint_names
    close()

Swapping from sim to real is a constructor change:

    # sim:
    sim = MuJoCoBackend(render_size=(240, 320))
    sim.load(scene)

    # real:
    sim = RealRobotBackend.from_yaml("my_so101.yaml")
    sim.load(scene)   # initializes cameras, zero-pose, safety limits

    # everything else — skills, motion, grasp, agent — unchanged.

This stub class raises ``NotImplementedError`` with actionable messages
from every method. Subclass it and fill in the hardware driver for your
target platform (SO-101, Franka ROS2, UR5 RTDE, LeRobot LePort, etc.).

A production hardware backend typically wires:

- **load**: spin up cameras, zero the arm, home the gripper, verify
  workspace boundaries against ``scene.workspace_aabb``.
- **step**: stream `target_joints` to the robot's position controller
  at roughly the same rate MuJoCo ticks (200 Hz). Clamp against joint
  limits and max-velocity safety.
- **observe**: read current joints from the robot's state interface,
  capture one frame from the overhead camera, optionally fuse with an
  external pose tracker for scene_objects.
- **reset**: send the stored home_qpos to the robot's trajectory
  controller and block until settled.
- **get_object_pose**: query whatever your pose estimator is (AprilTag,
  OptiTrack, learned keypoint detector). If you don't have one, return
  ``None`` and rely on VLM perception for localization.

See ``examples/real_robot_swap.py`` (when present) for a concrete
template and the RoboSandbox docs for integration recipes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from robosandbox.types import Observation, Pose, Scene

if TYPE_CHECKING:  # pragma: no cover
    pass


@dataclass
class RealRobotBackendConfig:
    """Configuration for a hardware bridge.

    The fields here are the minimum any hardware driver needs; add more
    in subclasses (IP, USB device path, camera index, home_qpos, etc.).
    """

    n_dof: int
    joint_names: tuple[str, ...]
    control_hz: float = 200.0
    home_qpos: tuple[float, ...] = ()
    gripper_open: float = 0.04
    gripper_closed: float = 0.0
    safety_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] = field(
        default=((-0.5, -0.5, 0.0), (0.5, 0.5, 0.8))
    )

    def __post_init__(self) -> None:
        if len(self.joint_names) != self.n_dof:
            raise ValueError(
                f"joint_names length {len(self.joint_names)} != n_dof {self.n_dof}"
            )
        if self.home_qpos and len(self.home_qpos) != self.n_dof:
            raise ValueError(
                f"home_qpos length {len(self.home_qpos)} != n_dof {self.n_dof}"
            )


_NYI_MSG = (
    "RealRobotBackend is a stub — subclass it and implement {method}. "
    "See the module docstring for hardware-integration guidance."
)


class RealRobotBackend:
    """Stub backend that satisfies ``SimBackend`` at the Protocol level
    but raises ``NotImplementedError`` from every method.

    Subclasses plug in a hardware driver:

    .. code-block:: python

        class MySO101Backend(RealRobotBackend):
            def __init__(self, serial_port: str, **cfg):
                super().__init__(RealRobotBackendConfig(
                    n_dof=6, joint_names=("j1",...,"j6"), ...
                ))
                self._arm = SO101Arm(serial_port)

            def load(self, scene: Scene) -> None:
                self._arm.connect()
                self._arm.home()

            def step(self, target_joints=None, gripper=None) -> None:
                if target_joints is not None:
                    self._arm.write_joint_positions(target_joints)
                if gripper is not None:
                    self._arm.write_gripper(gripper)

            # ...etc for observe/get_object_pose/close.

    The existing motion planner, skills, and agent loop will then work
    unchanged against ``MySO101Backend``.
    """

    def __init__(self, config: RealRobotBackendConfig) -> None:
        self._config = config
        self._loaded_scene: Scene | None = None

    # -- SimBackend protocol -----------------------------------------------

    def load(self, scene: Scene) -> None:
        self._loaded_scene = scene
        raise NotImplementedError(_NYI_MSG.format(method="load(scene)"))

    def reset(self) -> None:
        raise NotImplementedError(_NYI_MSG.format(method="reset()"))

    def step(
        self,
        target_joints: np.ndarray | None = None,
        gripper: float | None = None,
    ) -> None:
        raise NotImplementedError(_NYI_MSG.format(method="step(target_joints, gripper)"))

    def observe(self) -> Observation:
        raise NotImplementedError(_NYI_MSG.format(method="observe()"))

    def get_object_pose(self, object_id: str) -> Pose | None:
        raise NotImplementedError(
            _NYI_MSG.format(method="get_object_pose(object_id)")
        )

    def set_object_pose(self, object_id: str, pose: Pose) -> None:
        raise NotImplementedError(
            _NYI_MSG.format(method="set_object_pose(object_id, pose)")
        )

    @property
    def n_dof(self) -> int:
        return self._config.n_dof

    @property
    def joint_names(self) -> list[str]:
        return list(self._config.joint_names)

    def close(self) -> None:
        # Default close is a no-op so subclasses don't have to override
        # when they hold no external resources.
        return None

    # -- helpers -----------------------------------------------------------

    @property
    def config(self) -> RealRobotBackendConfig:
        return self._config

    @property
    def scene(self) -> Scene | None:
        return self._loaded_scene
