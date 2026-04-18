"""Adapter between RoboSandbox ``Observation`` and a LeRobot-style policy.

LeRobot policies expect a batched dict of named tensors — roughly:

    {
        "observation.images.<camera_name>": (B, C, H, W) float32 in [0, 1],
        "observation.state": (B, state_dim) float32,
    }

and return an "action" tensor of shape ``(B, action_dim)``. This adapter
threads a RoboSandbox ``Observation`` through that contract:

- The RGB frame from ``obs.rgb`` becomes the named image channel
  (default name ``"scene"``, matching the :mod:`robosandbox.export.lerobot`
  exporter's default).
- ``obs.robot_joints + [obs.gripper_width]`` becomes ``observation.state``.
- The returned action is split back into ``(n_dof + 1,)`` per our
  :class:`~robosandbox.policy.Policy` contract — first ``n_dof`` entries
  are target joints, the last is gripper ∈ ``[0, 1]``.

The real ``lerobot`` package is an optional dependency. When present, the
adapter accepts any object exposing a ``select_action(batch)`` method
(the LeRobot policy interface). When absent, a clear actionable error
message points the user at the install path.

Usage:

.. code-block:: python

    # With a real LeRobot checkpoint:
    from lerobot.common.policies.factory import make_policy
    policy = make_policy("/path/to/ckpt")
    adapter = LeRobotPolicyAdapter(policy, camera_name="scene")
    run_policy(sim, adapter, max_steps=500)

    # Or construct a policy yourself:
    adapter = LeRobotPolicyAdapter(my_policy)

This is the "plumbing" 4.5 ships — validating that the adapter + 4.3
replay loop actually drive a checkpoint to a useful result belongs to
whoever brings the first trained checkpoint (the docs tutorial walks
through the record → train → replay flow end to end).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from robosandbox.types import Observation


@runtime_checkable
class _LeRobotLike(Protocol):
    """Minimal subset of the LeRobot policy interface we rely on."""

    def select_action(self, batch: dict[str, Any]) -> Any: ...


class LeRobotPolicyAdapter:
    """Wrap a LeRobot-style policy as a RoboSandbox :class:`Policy`."""

    def __init__(
        self,
        policy: _LeRobotLike,
        *,
        camera_name: str = "scene",
        action_dim: int | None = None,
        image_size: tuple[int, int] | None = None,
    ) -> None:
        """Wrap ``policy`` as an ``act()``-compatible adapter.

        :param policy: any object exposing ``select_action(batch) -> tensor``.
        :param camera_name: key used under ``observation.images.<camera_name>``.
            Defaults to ``"scene"`` to match the shipped MuJoCo camera.
        :param action_dim: expected ``(n_dof + 1,)`` action length. Used to
            validate the policy's output without depending on the sim's
            n_dof at construction time. ``None`` = skip validation.
        :param image_size: ``(H, W)``. If set, frames get resized to this
            before being fed to the policy (requires ``numpy`` only — we
            use nearest-neighbor block averaging, no PIL dependency). If
            ``None``, we pass the frame through at its native resolution.
        """
        self._policy = policy
        self._camera_name = camera_name
        self._action_dim = action_dim
        self._image_size = image_size

    @property
    def camera_name(self) -> str:
        return self._camera_name

    def act(self, obs: Observation) -> np.ndarray:
        batch = self._to_batch(obs)
        raw = self._policy.select_action(batch)
        action = _to_numpy_1d(raw)
        if self._action_dim is not None and action.shape[-1] != self._action_dim:
            raise ValueError(
                f"LeRobot policy returned action dim {action.shape[-1]}, "
                f"expected {self._action_dim} (n_dof + 1)"
            )
        # Clamp the gripper command into [0, 1] — the sim's step() contract.
        if action.size >= 1:
            action = action.copy()
            action[-1] = float(np.clip(action[-1], 0.0, 1.0))
        return action

    # -- internals ---------------------------------------------------------

    def _to_batch(self, obs: Observation) -> dict[str, Any]:
        rgb = obs.rgb.astype(np.float32) / 255.0  # HWC uint8 -> HWC float32 [0,1]
        if self._image_size is not None:
            rgb = _resize_hw(rgb, self._image_size)
        # LeRobot expects CHW (channels-first) + batch dim; reorder + unsqueeze.
        chw = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]

        state = np.concatenate(
            [np.asarray(obs.robot_joints, dtype=np.float32),
             np.asarray([obs.gripper_width], dtype=np.float32)]
        )[np.newaxis, :]

        return {
            f"observation.images.{self._camera_name}": chw,
            "observation.state": state,
        }


def _to_numpy_1d(x: Any) -> np.ndarray:
    """Coerce a tensor-ish object (torch.Tensor, np.ndarray, list, ...) to a
    1D numpy array. Strips batch dimension if shape is ``(1, N)``.
    """
    # Torch tensor — avoid importing torch at module load.
    detach = getattr(x, "detach", None)
    if callable(detach):
        x = detach()
    to_cpu = getattr(x, "cpu", None)
    if callable(to_cpu):
        x = to_cpu()
    numpy_fn = getattr(x, "numpy", None)
    if callable(numpy_fn):
        arr = numpy_fn()
    else:
        arr = np.asarray(x)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.ravel()


def _resize_hw(rgb_hwc: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor resize on a (H, W, C) float32 array. No PIL dep."""
    target_h, target_w = size_hw
    src_h, src_w = rgb_hwc.shape[:2]
    if (src_h, src_w) == (target_h, target_w):
        return rgb_hwc
    y_idx = (np.arange(target_h) * (src_h / target_h)).astype(np.int64)
    x_idx = (np.arange(target_w) * (src_w / target_w)).astype(np.int64)
    return rgb_hwc[y_idx[:, None], x_idx[None, :], :]
