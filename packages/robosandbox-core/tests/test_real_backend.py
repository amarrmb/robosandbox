"""Tests for ``RealRobotBackend`` stub — shape + NotImplementedError surface.

These tests encode the contract that anyone implementing a hardware
backend must satisfy. A subclass that neglects one of the core methods
still surfaces a clear NotImplementedError; the NoImpl "I'm a stub"
message tells downstream users to subclass.
"""

from __future__ import annotations

import numpy as np
import pytest

from robosandbox.backends.real import RealRobotBackend, RealRobotBackendConfig
from robosandbox.protocols import SimBackend
from robosandbox.types import Pose, Scene


def _basic_cfg(**overrides) -> RealRobotBackendConfig:
    defaults = dict(
        n_dof=6,
        joint_names=("j1", "j2", "j3", "j4", "j5", "j6"),
        home_qpos=(0.0,) * 6,
    )
    defaults.update(overrides)
    return RealRobotBackendConfig(**defaults)


def test_satisfies_simbackend_protocol() -> None:
    b = RealRobotBackend(_basic_cfg())
    # runtime_checkable Protocol — instance-check should pass.
    assert isinstance(b, SimBackend)


def test_n_dof_and_joint_names() -> None:
    b = RealRobotBackend(_basic_cfg())
    assert b.n_dof == 6
    assert b.joint_names == ["j1", "j2", "j3", "j4", "j5", "j6"]


def test_close_is_noop_by_default() -> None:
    b = RealRobotBackend(_basic_cfg())
    b.close()  # should not raise


@pytest.mark.parametrize(
    "call",
    [
        lambda b: b.load(Scene()),
        lambda b: b.reset(),
        lambda b: b.step(target_joints=np.zeros(6), gripper=0.0),
        lambda b: b.observe(),
        lambda b: b.get_object_pose("x"),
        lambda b: b.set_object_pose("x", Pose(xyz=(0, 0, 0))),
    ],
)
def test_methods_raise_actionable_not_implemented(call) -> None:
    b = RealRobotBackend(_basic_cfg())
    with pytest.raises(NotImplementedError) as ei:
        call(b)
    msg = str(ei.value).lower()
    assert "stub" in msg and "subclass" in msg


def test_config_validation_joint_names_length() -> None:
    with pytest.raises(ValueError, match="joint_names length"):
        RealRobotBackendConfig(
            n_dof=7,
            joint_names=("a", "b", "c"),
        )


def test_config_validation_home_qpos_length() -> None:
    with pytest.raises(ValueError, match="home_qpos length"):
        RealRobotBackendConfig(
            n_dof=6,
            joint_names=("a", "b", "c", "d", "e", "f"),
            home_qpos=(0.0, 0.0, 0.0),
        )


def test_subclass_override_works() -> None:
    """A minimal subclass that overrides the stub can be driven normally."""
    calls: list[str] = []

    class FakeBackend(RealRobotBackend):
        def load(self, scene):
            calls.append(f"load({type(scene).__name__})")

        def reset(self):
            calls.append("reset")

        def step(self, target_joints=None, gripper=None):
            calls.append(f"step(j={target_joints is not None},g={gripper})")

        def observe(self):
            # Stub test — real subclasses return a proper Observation.
            return None  # type: ignore[return-value]

        def get_object_pose(self, object_id):
            return None

        def set_object_pose(self, object_id, pose):
            pass

    b = FakeBackend(_basic_cfg())
    b.load(Scene())
    b.reset()
    b.step(target_joints=np.zeros(6), gripper=1.0)
    assert calls == ["load(Scene)", "reset", "step(j=True,g=1.0)"]
