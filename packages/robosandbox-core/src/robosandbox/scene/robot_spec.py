"""Everything a SimBackend needs to drive a specific robot.

Produced by scene/robot_loader.py (URDF/MJCF path) or scene/mjcf_builder.py
(built-in arm). Names refer to elements in the compiled MjModel; a backend
looks them up to cache qpos/ctrl addresses and site ids.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RobotSpec:
    arm_joint_names: tuple[str, ...]
    arm_actuator_names: tuple[str, ...]
    gripper_joint_names: tuple[str, ...]
    gripper_primary_joint: str
    gripper_actuator_name: str
    ee_site_name: str
    base_body_name: str  # Robot root body; motion planner needs its world xpos
    home_qpos: tuple[float, ...]
    gripper_open_qpos: float
    gripper_closed_qpos: float

    def __post_init__(self) -> None:
        if len(self.home_qpos) != len(self.arm_joint_names):
            raise ValueError(
                f"RobotSpec.home_qpos length {len(self.home_qpos)} != "
                f"arm_joint_names length {len(self.arm_joint_names)}"
            )
        if len(self.arm_actuator_names) != len(self.arm_joint_names):
            raise ValueError(
                f"RobotSpec.arm_actuator_names length {len(self.arm_actuator_names)} != "
                f"arm_joint_names length {len(self.arm_joint_names)}"
            )
        if self.gripper_primary_joint not in self.gripper_joint_names:
            raise ValueError(
                f"gripper_primary_joint {self.gripper_primary_joint!r} not in "
                f"gripper_joint_names {self.gripper_joint_names}"
            )
