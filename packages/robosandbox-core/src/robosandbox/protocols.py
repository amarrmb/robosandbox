"""Plugin-facing Protocols. Implementations live in core or in sibling
packages (robosandbox-anygrasp, robosandbox-curobo, ...) and register
themselves via Python entry points declared in pyproject.toml.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from robosandbox.types import (
    DetectedObject,
    Grasp,
    JointTrajectory,
    Observation,
    Pose,
    Scene,
)


@runtime_checkable
class SimBackend(Protocol):
    """A physics sim that loads a scene and steps through time."""

    def load(self, scene: Scene) -> None: ...
    def reset(self) -> None: ...
    def step(self, target_joints: np.ndarray | None = None, gripper: float | None = None) -> None: ...
    def observe(self) -> Observation: ...
    def get_object_pose(self, object_id: str) -> Pose | None: ...
    def set_object_pose(self, object_id: str, pose: Pose) -> None: ...
    @property
    def n_dof(self) -> int: ...
    @property
    def joint_names(self) -> list[str]: ...
    def close(self) -> None: ...


@runtime_checkable
class Perception(Protocol):
    """Text query + observation → located objects."""

    def locate(self, query: str, obs: Observation) -> list[DetectedObject]: ...


@runtime_checkable
class GraspPlanner(Protocol):
    """Observation + target object → candidate grasps."""

    def plan(self, obs: Observation, target: DetectedObject) -> list[Grasp]: ...


@runtime_checkable
class MotionPlanner(Protocol):
    """Start joints + target pose → joint trajectory."""

    def plan(
        self,
        sim: SimBackend,
        start_joints: np.ndarray,
        target_pose: Pose,
        constraints: dict[str, Any] | None = None,
    ) -> JointTrajectory: ...


@runtime_checkable
class RecordSink(Protocol):
    def start_episode(self, task: str, metadata: dict) -> str: ...
    def write_frame(self, obs: Observation, action: dict | None = None) -> None: ...
    def end_episode(self, success: bool, result: dict) -> None: ...


@runtime_checkable
class VLMClient(Protocol):
    def chat(self, messages: list, tools: list | None = None, **kwargs) -> dict: ...


@runtime_checkable
class Skill(Protocol):
    """The action vocabulary the agent exposes to the VLM."""

    name: str
    description: str
    parameters_schema: dict

    def __call__(self, ctx: Any, **kwargs) -> Any: ...
