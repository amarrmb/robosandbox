"""Tests for ``LeRobotPolicyAdapter`` — batch construction + action shaping.

No real LeRobot install required; we mock the policy with a callable
that verifies batch shape and returns a hand-crafted action.
"""

from __future__ import annotations

import numpy as np
import pytest
from robosandbox.policy import LeRobotPolicyAdapter, Policy
from robosandbox.types import Observation, Pose


def _obs(
    rgb_hw: tuple[int, int] = (32, 40),
    n_dof: int = 7,
    gripper_width: float = 0.05,
) -> Observation:
    h, w = rgb_hw
    return Observation(
        rgb=np.zeros((h, w, 3), dtype=np.uint8),
        depth=None,
        robot_joints=np.arange(n_dof, dtype=np.float64),
        ee_pose=Pose(xyz=(0.3, 0.0, 0.2)),
        gripper_width=float(gripper_width),
        scene_objects={},
        timestamp=0.0,
    )


class _FakeLeRobotPolicy:
    """Records the batch it was given; returns a fixed tensor-shaped action."""

    def __init__(self, action_out: np.ndarray) -> None:
        self._action_out = np.asarray(action_out, dtype=np.float32)
        self.last_batch: dict | None = None

    def select_action(self, batch):
        self.last_batch = batch
        # Return the LeRobot-typical (1, N) numpy tensor; adapter should
        # strip the batch dim.
        return self._action_out[np.newaxis, :]


def test_adapter_builds_lerobot_batch_shape() -> None:
    policy = _FakeLeRobotPolicy(np.zeros(8, dtype=np.float32))
    adapter = LeRobotPolicyAdapter(policy, camera_name="scene")
    obs = _obs(rgb_hw=(32, 40), n_dof=7)

    _ = adapter.act(obs)

    batch = policy.last_batch
    assert batch is not None
    # Image key uses the configured camera name.
    assert "observation.images.scene" in batch
    # Image is (1, C, H, W) float32 in [0, 1].
    img = batch["observation.images.scene"]
    assert img.shape == (1, 3, 32, 40)
    assert img.dtype == np.float32
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0
    # State is (1, n_dof + 1).
    state = batch["observation.state"]
    assert state.shape == (1, 8)


def test_adapter_is_a_policy() -> None:
    policy = _FakeLeRobotPolicy(np.zeros(8, dtype=np.float32))
    adapter = LeRobotPolicyAdapter(policy)
    assert isinstance(adapter, Policy)


def test_action_gripper_clamped_to_unit_interval() -> None:
    # Policy returns a gripper value outside [0, 1]; adapter must clamp.
    raw = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.7], dtype=np.float32)
    policy = _FakeLeRobotPolicy(raw)
    adapter = LeRobotPolicyAdapter(policy)
    a = adapter.act(_obs(n_dof=7))
    assert a[-1] == pytest.approx(1.0)
    # Joint values unchanged.
    assert np.allclose(a[:-1], raw[:-1])


def test_action_dim_validation() -> None:
    policy = _FakeLeRobotPolicy(np.zeros(8, dtype=np.float32))
    # Adapter expects 7 (wrong shape) — act() should raise.
    adapter = LeRobotPolicyAdapter(policy, action_dim=7)
    with pytest.raises(ValueError, match="action dim 8"):
        adapter.act(_obs(n_dof=7))


def test_adapter_handles_torch_style_tensor_output() -> None:
    """Adapter must handle torch-tensor-shaped returns (with .detach/.cpu/.numpy)
    without actually importing torch.
    """

    class _FakeTorchTensor:
        def __init__(self, arr: np.ndarray) -> None:
            self._arr = arr
            self.detached = False
            self.cpued = False
            self.numpyied = False

        def detach(self):
            self.detached = True
            return self

        def cpu(self):
            self.cpued = True
            return self

        def numpy(self):
            self.numpyied = True
            return self._arr

    class _TensorReturningPolicy:
        def __init__(self, arr: np.ndarray) -> None:
            self._t = _FakeTorchTensor(arr)

        def select_action(self, batch):
            return self._t

    arr = np.arange(8, dtype=np.float32)[np.newaxis, :]
    policy = _TensorReturningPolicy(arr)
    adapter = LeRobotPolicyAdapter(policy)
    out = adapter.act(_obs(n_dof=7))
    assert out.shape == (8,)
    assert policy._t.detached and policy._t.cpued and policy._t.numpyied


def test_image_resize_passes_through_when_size_matches() -> None:
    policy = _FakeLeRobotPolicy(np.zeros(8, dtype=np.float32))
    adapter = LeRobotPolicyAdapter(policy, image_size=(32, 40))
    adapter.act(_obs(rgb_hw=(32, 40)))
    img = policy.last_batch["observation.images.scene"]
    assert img.shape == (1, 3, 32, 40)


def test_image_resize_downsamples_to_target() -> None:
    policy = _FakeLeRobotPolicy(np.zeros(8, dtype=np.float32))
    adapter = LeRobotPolicyAdapter(policy, image_size=(16, 20))
    adapter.act(_obs(rgb_hw=(240, 320)))
    img = policy.last_batch["observation.images.scene"]
    assert img.shape == (1, 3, 16, 20)


def test_torch_module_policy_receives_tensors() -> None:
    """Regression for codex finding: the adapter must feed torch.Tensors
    when the wrapped policy is a torch.nn.Module, but keep numpy for
    mock policies even when torch happens to be installed.
    """
    torch = pytest.importorskip("torch")

    # Torch module that records whatever it received.
    class _TorchPolicyModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Register a real parameter so `parameters()` yields torch params.
            self._p = torch.nn.Parameter(torch.zeros(1))
            self.last_batch: dict | None = None

        def select_action(self, batch):  # pragma: no cover - exercised below
            self.last_batch = batch
            return torch.zeros(1, 8, dtype=torch.float32)

    policy = _TorchPolicyModule()
    adapter = LeRobotPolicyAdapter(policy, camera_name="scene")
    adapter.act(_obs(n_dof=7))
    assert policy.last_batch is not None
    img = policy.last_batch["observation.images.scene"]
    state = policy.last_batch["observation.state"]
    assert isinstance(img, torch.Tensor), f"expected torch.Tensor, got {type(img)!r}"
    assert isinstance(state, torch.Tensor), f"expected torch.Tensor, got {type(state)!r}"


def test_mock_policy_still_receives_numpy_when_torch_is_importable() -> None:
    """The numpy contract for non-torch mocks must not depend on whether
    torch is present in the environment."""
    pytest.importorskip("torch")  # ensure torch IS importable
    policy = _FakeLeRobotPolicy(np.zeros(8, dtype=np.float32))
    adapter = LeRobotPolicyAdapter(policy, camera_name="scene")
    adapter.act(_obs(n_dof=7))
    img = policy.last_batch["observation.images.scene"]
    assert isinstance(img, np.ndarray), f"mock policy should get numpy, got {type(img)!r}"
    assert img.dtype == np.float32
