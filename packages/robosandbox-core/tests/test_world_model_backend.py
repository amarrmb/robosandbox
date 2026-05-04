"""Contract tests for WorldModelBackend.

Verifies the SimBackend Protocol surface is satisfied and that the
slot wiring (bootstrap -> predict_next -> observe) actually routes
data through. Uses IdentityPredictor as the reference predictor; a
real predictor swap follows the same Protocol.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

sys.path.insert(
    0,
    str(Path(__file__).parent.parent / "src"),
)

try:
    import mujoco  # noqa: F401
except ImportError:
    pytest.skip("mujoco not installed", allow_module_level=True)

from robosandbox.protocols import SimBackend
from robosandbox.sim.factory import create_sim_backend, list_sim_backends
from robosandbox.sim.world_model_backend import (
    IdentityPredictor,
    WorldModelBackend,
    WorldModelPredictor,
)
from robosandbox.tasks.loader import load_builtin_task
from robosandbox.types import Observation, Pose


def _bootstrap_kwargs() -> dict:
    # MuJoCoBackend takes render_size; keep it small to make the test
    # cheap.
    return {"render_size": (64, 96)}


def test_world_model_listed_in_factory() -> None:
    assert "world_model" in list_sim_backends()


def test_identity_predictor_satisfies_protocol() -> None:
    assert isinstance(IdentityPredictor(), WorldModelPredictor)


def test_world_model_backend_satisfies_sim_backend_protocol() -> None:
    backend = WorldModelBackend(bootstrap_kwargs=_bootstrap_kwargs())
    try:
        assert isinstance(backend, SimBackend)
    finally:
        # close() is idempotent before load() — bootstrap.close handles it.
        try:
            backend.close()
        except Exception:
            pass


def test_factory_creates_world_model_backend() -> None:
    backend = create_sim_backend(
        "world_model", bootstrap_kwargs=_bootstrap_kwargs()
    )
    try:
        assert isinstance(backend, WorldModelBackend)
    finally:
        try:
            backend.close()
        except Exception:
            pass


def test_load_observe_step_roundtrip_with_identity_predictor() -> None:
    """End-to-end: load a real task, step a few times, verify state stays
    constant under IdentityPredictor (proves the slot is wired without
    secretly falling through to the bootstrap's physics)."""
    task = load_builtin_task("pick_cube_franka")
    backend = WorldModelBackend(bootstrap_kwargs=_bootstrap_kwargs())
    try:
        backend.load(task.scene)
        obs0 = backend.observe()
        assert obs0.robot_joints.shape == (backend.n_dof,)
        assert backend.n_dof == 7  # Franka
        assert isinstance(obs0.ee_pose, Pose)

        # Drive a non-trivial action: bias every joint by +0.1 rad.
        target = obs0.robot_joints + 0.1
        for _ in range(5):
            backend.step(target_joints=target, gripper=0.0)

        obs5 = backend.observe()

        # Identity predictor → joints did not change despite 5 steps of
        # commanded motion. This is the failure mode that makes it 0% on
        # every task; it's also the proof that the predictor is in the
        # data path, not the bootstrap's physics.
        np.testing.assert_array_equal(obs5.robot_joints, obs0.robot_joints)

        # But timestamp DID advance — the backend kept its own clock.
        assert obs5.timestamp > obs0.timestamp
    finally:
        backend.close()


def test_get_object_pose_reads_from_predicted_observation() -> None:
    """Object queries route through the cached predicted observation,
    not through the bootstrap. Confirms world-model divergence from
    classical sim shows up in the API."""
    task = load_builtin_task("pick_cube_franka")
    backend = WorldModelBackend(bootstrap_kwargs=_bootstrap_kwargs())
    try:
        backend.load(task.scene)
        # IdentityPredictor preserves scene_objects, so the query
        # returns the bootstrap's initial cube pose. The point of the
        # test is the routing — for a real model, this is where you'd
        # see the prediction diverge from physics.
        obs0 = backend.observe()
        cube_xyz_obs = obs0.scene_objects["red_cube"].xyz
        cube_xyz_query = backend.get_object_pose("red_cube").xyz
        assert cube_xyz_obs == cube_xyz_query
    finally:
        backend.close()


def test_custom_predictor_observation_is_what_observe_returns() -> None:
    """Drop in a custom predictor that returns a known observation;
    verify observe() round-trips it. This is the integration shape a
    real model wrapper (V-JEPA-2, Cosmos, DreamerV3) would follow."""

    class FixedJointsPredictor:
        """Returns the input observation but with robot_joints set to a
        constant — a real model would instead run inference here."""

        def __init__(self, target: np.ndarray) -> None:
            self._target = target

        def predict_next(
            self,
            obs: Observation,
            target_joints: np.ndarray | None,
            gripper: float | None,
        ) -> Observation:
            return Observation(
                rgb=obs.rgb,
                depth=obs.depth,
                robot_joints=self._target.copy(),
                ee_pose=obs.ee_pose,
                gripper_width=obs.gripper_width,
                scene_objects=obs.scene_objects,
                timestamp=obs.timestamp,
                camera_intrinsics=obs.camera_intrinsics,
                camera_extrinsics=obs.camera_extrinsics,
            )

        def reset(self) -> None:
            pass

    target = np.array([0.1, 0.2, 0.3, -1.4, 0.0, 1.5, -0.7])
    backend = WorldModelBackend(
        predictor=FixedJointsPredictor(target),
        bootstrap_kwargs=_bootstrap_kwargs(),
    )
    try:
        task = load_builtin_task("pick_cube_franka")
        backend.load(task.scene)
        backend.step(target_joints=target, gripper=0.0)
        obs = backend.observe()
        np.testing.assert_array_equal(obs.robot_joints, target)
    finally:
        backend.close()
