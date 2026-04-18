"""Swap from sim to real-robot backend without changing skill code.

Illustrates the ``SimBackend`` protocol as the seam between "simulated"
and "real" execution. Subclass ``RealRobotBackend`` with your hardware
driver; everything else — skills, motion, grasp, agent loop — works
unchanged.

Run the stub (no hardware needed — just shows the protocol holds):
    uv run python examples/real_robot_swap.py
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robosandbox.backends.real import RealRobotBackend, RealRobotBackendConfig
from robosandbox.protocols import SimBackend
from robosandbox.types import Observation, Pose, Scene


class FakeSO101Backend(RealRobotBackend):
    """Minimal example: a FAKE hardware bridge that prints what it would do.

    Replace the print/assignment logic with real driver calls:
      - `self._arm.write_joint_positions(target_joints)` on a LeRobot bus
      - `self._arm.read_joints()` for state
      - `self._camera.read()` for the RGB frame
    """

    def __init__(self) -> None:
        super().__init__(RealRobotBackendConfig(
            n_dof=6,
            joint_names=("j1", "j2", "j3", "j4", "j5", "j6"),
            home_qpos=(0.0, 0.2, -0.8, 0.0, 0.5, 0.0),
        ))
        self._t = 0.0
        self._joints = np.array(self._config.home_qpos, dtype=np.float64)

    def load(self, scene: Scene) -> None:
        super().load(scene) if False else None  # stash scene on self._loaded_scene
        self._loaded_scene = scene
        print(f"[fake_so101] load scene with {len(scene.objects)} objects")

    def reset(self) -> None:
        self._joints = np.array(self._config.home_qpos, dtype=np.float64)
        self._t = 0.0
        print("[fake_so101] reset to home")

    def step(self, target_joints=None, gripper=None) -> None:
        if target_joints is not None:
            # Real driver would send a position command to each servo here.
            self._joints = np.asarray(target_joints, dtype=np.float64).copy()
        if gripper is not None:
            # Real driver would write the gripper value.
            pass
        self._t += 1.0 / self._config.control_hz

    def observe(self) -> Observation:
        return Observation(
            rgb=np.zeros((240, 320, 3), dtype=np.uint8),  # camera would fill this
            depth=None,
            robot_joints=self._joints.copy(),
            ee_pose=Pose(xyz=(0.3, 0.0, 0.2)),            # FK or external tracker
            gripper_width=0.08,
            scene_objects={},
            timestamp=self._t,
        )

    def get_object_pose(self, object_id: str) -> Pose | None:
        # Would query your pose estimator. Returning None is OK when you
        # rely on VLM perception for localization instead.
        return None

    def set_object_pose(self, object_id: str, pose: Pose) -> None:
        # Real hardware can't teleport objects — this is a no-op on the
        # real backend. The sandbox doesn't call it during normal runs.
        pass


def main() -> None:
    backend: SimBackend = FakeSO101Backend()
    backend.load(Scene())
    backend.reset()
    for _ in range(5):
        backend.step(target_joints=np.zeros(6), gripper=0.0)
    obs = backend.observe()
    print(f"[example] sim-or-real run ok.  n_dof={backend.n_dof}  t={obs.timestamp:.3f}")
    backend.close()


if __name__ == "__main__":
    main()
