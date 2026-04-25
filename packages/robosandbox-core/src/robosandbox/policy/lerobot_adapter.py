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
        image_keys: list[str] | None = None,
        action_dim: int | None = None,
        image_size: tuple[int, int] | None = None,
        preprocessor: Any = None,
        postprocessor: Any = None,
    ) -> None:
        """Wrap ``policy`` as an ``act()``-compatible adapter.

        :param policy: any object exposing ``select_action(batch) -> tensor``.
        :param camera_name: convenience for the common single-camera case;
            translates to ``image_keys=["observation.images.<camera_name>"]``
            when ``image_keys`` is not given. Ignored otherwise.
        :param image_keys: explicit list of batch keys under which the
            sim's RGB frame should be exposed. Use this when the wrapped
            policy expects something other than ``observation.images.scene``
            — e.g. legacy ``"observation.image"`` (pusht-era checkpoints)
            or multiple wrist/base cameras (the same frame is duplicated
            into each key). When ``None``, falls back to ``camera_name``.
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
        self._image_keys = (
            list(image_keys)
            if image_keys is not None
            else [f"observation.images.{camera_name}"]
        )
        self._action_dim = action_dim
        self._image_size = image_size
        # See _load_lerobot_processors in policy/__init__.py for why both
        # are required (lerobot 0.4 moved normalization out of the model).
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        # Lazily-allocated scratch buffer for the (1, n_dof + 1) state vector.
        # Re-used across act() calls so the per-frame hot path doesn't
        # allocate three intermediate arrays per step. Sized on first use
        # once we see the actual robot_joints length.
        self._state_buf: np.ndarray | None = None
        # Whether to convert batch values to torch.Tensor in _to_batch().
        # Keyed on the *wrapped policy's* type (is it a torch.nn.Module?),
        # not on ambient torch availability — so non-torch mock policies
        # keep receiving numpy arrays even when torch is installed.
        self._use_torch_batch = _policy_wants_torch(policy)

    @property
    def camera_name(self) -> str:
        return self._camera_name

    def reset(self) -> None:
        """Reset wrapped policy state between eval trials.

        ACT (and most chunked policies) maintain a per-episode action queue
        that's populated by ``predict_action_chunk`` every ``n_action_steps``
        calls. Without an explicit reset between trials, trial N starts with
        whatever residual state trial N-1 left behind — most often a
        partially-consumed chunk that gets popped on trial N's first
        ``select_action`` call instead of a fresh chunk conditioned on
        trial N's actual initial observation. The eval CLI calls
        ``policy.reset()`` between trials; without this method that call
        was silently a no-op (``getattr(adapter, 'reset', None)`` returned
        ``None``), making per-trial outcomes depend on trial order.
        """
        inner_reset = getattr(self._policy, "reset", None)
        if callable(inner_reset):
            inner_reset()

    def act(self, obs: Observation) -> np.ndarray:
        batch = self._to_batch(obs)
        if self._preprocessor is not None:
            # The preprocessor pipeline expects observation keys at the
            # transition's top level (NOT nested under an `observation`
            # sub-dict): {observation.state: ..., observation.images.X: ...}.
            # Steps: rename + add-batch-dim (idempotent on already-batched
            # tensors) + device transfer + per-feature normalization (state
            # via dataset stats, images via ImageNet stats).
            transition = self._preprocessor(dict(batch))
            batch = {k: transition[k] for k in batch if k in transition}
        raw = self._policy.select_action(batch)
        if self._postprocessor is not None:
            # Un-normalize the predicted action back into raw joint /
            # gripper-command units. Postprocessor expects an `action` key
            # in the transition.
            out = self._postprocessor({"action": raw})
            raw = out["action"]
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
        # Image path: avoid a full HxWx3 float32 copy when rgb is already
        # float32 (rare, but common after augmentation/resizing by callers).
        if obs.rgb.dtype == np.float32:
            rgb = obs.rgb if obs.rgb.max() <= 1.0 else obs.rgb / 255.0
        else:
            rgb = obs.rgb.astype(np.float32) / 255.0  # HWC uint8 -> HWC float32 [0,1]
        if self._image_size is not None:
            rgb = _resize_hw(rgb, self._image_size)
        chw = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]

        # State path: fill a pre-allocated (1, n_dof + 1) buffer in place so
        # the hot loop doesn't churn three intermediate allocs per frame.
        joints = np.asarray(obs.robot_joints)
        state_dim = joints.shape[0] + 1
        if self._state_buf is None or self._state_buf.shape != (1, state_dim):
            self._state_buf = np.empty((1, state_dim), dtype=np.float32)
        self._state_buf[0, :-1] = joints
        self._state_buf[0, -1] = obs.gripper_width

        # LeRobot's PreTrainedPolicy forward passes call nn.Linear on the
        # state tensor directly, so the batch values must be torch.Tensors
        # (not numpy). Only convert when the wrapped policy is actually a
        # torch module — non-torch mocks keep receiving numpy so their
        # tests/examples don't break when torch happens to be in the env.
        if self._use_torch_batch:
            import torch  # deferred import: only reached for torch policies
            img_val: Any = torch.from_numpy(chw)
            state_val: Any = torch.from_numpy(self._state_buf)
        else:
            img_val = chw
            state_val = self._state_buf

        batch: dict[str, Any] = {"observation.state": state_val}
        for key in self._image_keys:
            batch[key] = img_val
        return batch


def _policy_wants_torch(policy: Any) -> bool:
    """Decide whether to hand ``policy.select_action`` torch.Tensors.

    True when the wrapped object is (or smells like) a torch.nn.Module.
    False otherwise — preserves the numpy batch contract for test mocks,
    docs-demo policies, and any custom callable that doesn't need tensors.
    """
    if type(policy).__name__ in ("module", "Module"):  # unlikely, but cheap
        return True
    # Duck-type: real torch.nn.Module exposes parameters() → an iterator of
    # torch.nn.Parameter and a `forward` method. Avoid importing torch here.
    params = getattr(policy, "parameters", None)
    if callable(params):
        try:
            first = next(iter(params()))
        except (StopIteration, TypeError, ValueError):
            return False
        return type(first).__module__.startswith("torch")
    return False


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
