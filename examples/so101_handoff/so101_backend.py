"""Concrete SO-101 backend *skeleton* — fill in the marked sections.

This is the starting point for wiring a real SO-101 arm (or any
LeRobot-bus-compatible robot) into RoboSandbox. It extends
``RealRobotBackend`` and implements every method *as a stub that
tracks commanded joint state* — enough to run observation+step skills
(``Home``, a custom ``Wave``, teleop) against it unchanged.

To take this to real hardware, replace the clearly-marked TODO blocks
in each method with calls to your driver:

    self._arm = feetech.FeetechController(serial_port="/dev/ttyUSB0")
    self._camera = cv2.VideoCapture(0)

Before you touch hardware, see ``tutorials/sim-to-real-handoff.md``
for the "first real run" safety checklist.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from robosandbox.backends.real import RealRobotBackend, RealRobotBackendConfig
from robosandbox.types import Observation, Pose, Scene

# SO-101 ships in a 5-arm-joint + 1-gripper configuration. Matches
# mujoco_menagerie's `trs_so_arm100` asset used in the BYO-robot
# tutorial — sim-trained policies for that asset should transfer to
# this backend without remapping joint order.
_SO101_JOINT_NAMES = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll")
_SO101_HOME_QPOS = (0.0, -1.4, 1.4, 0.0, 0.0)
_SO101_GRIPPER_OPEN = 1.5     # Jaw rad — wide
_SO101_GRIPPER_CLOSED = 0.0   # Jaw rad — pinched


class SO101Backend(RealRobotBackend):
    """Skeleton real-hardware backend for the SO-101 arm.

    Commanded joint state is tracked in software so observation+step
    skills can be smoke-tested before any serial cable is plugged in.
    The ``_TODO`` blocks mark where a real driver call replaces the
    stub.
    """

    def __init__(self, *, serial_port: str | None = None) -> None:
        super().__init__(RealRobotBackendConfig(
            n_dof=len(_SO101_JOINT_NAMES),
            joint_names=_SO101_JOINT_NAMES,
            home_qpos=_SO101_HOME_QPOS,
            gripper_open=_SO101_GRIPPER_OPEN,
            gripper_closed=_SO101_GRIPPER_CLOSED,
        ))
        self._serial_port = serial_port
        self._joints = np.array(_SO101_HOME_QPOS, dtype=np.float64)
        self._gripper_qpos = float(_SO101_GRIPPER_OPEN)
        self._t = 0.0
        self._arm: Any = None      # your driver handle
        self._camera: Any = None   # your camera handle

    # -- SimBackend protocol -----------------------------------------------

    def load(self, scene: Scene) -> None:
        self._loaded_scene = scene
        # _TODO(real): connect to motor bus + camera, run calibration.
        # Example for a Feetech-based LeRobot bus:
        #
        #   from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
        #   self._arm = FeetechMotorsBus(port=self._serial_port, motors={...})
        #   self._arm.connect()
        #   self._arm.enable_torque()
        #
        #   import cv2
        #   self._camera = cv2.VideoCapture(0)
        #
        # Also validate scene.workspace_aabb against your physical workspace
        # bounds here and raise early if the sim task would command the
        # arm outside safe reach.
        pass

    def reset(self) -> None:
        self._joints = np.array(self._config.home_qpos, dtype=np.float64)
        self._gripper_qpos = float(self._config.gripper_open)
        self._t = 0.0
        # _TODO(real): send a blocking trajectory command to home.
        #   self._arm.write_joint_positions(self._joints, blocking=True)

    def step(
        self,
        target_joints: np.ndarray | None = None,
        gripper: float | None = None,
    ) -> None:
        if target_joints is not None:
            arr = np.asarray(target_joints, dtype=np.float64).ravel()
            if arr.shape != (self._config.n_dof,):
                raise ValueError(
                    f"target_joints shape {arr.shape} != ({self._config.n_dof},)"
                )
            # _TODO(real): clamp against joint limits, verify velocity
            # since last tick, then stream to the motor bus:
            #   self._arm.write_joint_positions(arr)
            self._joints = arr.copy()
        if gripper is not None:
            t = float(np.clip(gripper, 0.0, 1.0))
            self._gripper_qpos = (
                self._config.gripper_open * (1.0 - t)
                + self._config.gripper_closed * t
            )
            # _TODO(real): self._arm.write_gripper(self._gripper_qpos)
        self._t += 1.0 / self._config.control_hz

    def observe(self) -> Observation:
        # _TODO(real): swap echoed state for self._arm.read_joint_positions()
        # and grab an actual frame from self._camera.read().
        rgb = np.zeros((240, 320, 3), dtype=np.uint8)

        # `gripper_width = 2 * |qpos|` matches the convention RoboSandbox
        # uses elsewhere so downstream skills (and export-lerobot) see a
        # consistent ordering: larger == more open.
        gripper_width = 2.0 * abs(self._gripper_qpos)

        return Observation(
            rgb=rgb,
            depth=None,
            robot_joints=self._joints.copy(),
            # Using home-pose ee as a placeholder. A real backend either
            # runs FK against a URDF, or reads a tool-tracker pose.
            ee_pose=Pose(xyz=(0.0, -0.25, 0.15)),
            gripper_width=gripper_width,
            scene_objects={},  # _TODO(real): pose-estimator / AprilTag poses
            timestamp=self._t,
        )

    def get_object_pose(self, object_id: str) -> Pose | None:
        # _TODO(real): query your pose estimator by object_id.
        # If you rely on VLM perception for localization, returning None
        # here is fine — the VLM grounds the object directly in the RGB
        # frame.
        return None

    def set_object_pose(self, object_id: str, pose: Pose) -> None:
        # You cannot teleport objects on real hardware. This method is
        # a no-op on the real path; the sandbox only calls it for sim
        # scene initialisation.
        return None

    def close(self) -> None:
        # _TODO(real): disable torque, release camera, close serial.
        #   if self._arm is not None: self._arm.disable_torque()
        #   if self._camera is not None: self._camera.release()
        return None
