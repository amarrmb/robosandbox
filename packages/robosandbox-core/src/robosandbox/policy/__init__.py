"""Policy-in-the-loop replay.

An alternative to the planner-driven :class:`~robosandbox.agent.agent.Agent`
for the "deploy" half of the sim-first loop: a trained (or replayed)
policy observes the sim and commands joint targets directly.

Public surface:

- :class:`Policy` — runtime-checkable Protocol. Implement ``act(obs) -> (n_dof+1,)``
  where the first ``n_dof`` entries are target joint positions and the
  last is gripper command in ``[0, 1]`` (0 == open, 1 == closed).
- :func:`run_policy` — observe → act → step loop with an optional
  success criterion and a ``(obs, action)`` step callback.
- :class:`ReplayTrajectoryPolicy` — reference impl that reads a JSONL
  of ``{joints, gripper}`` rows (or a ``events.jsonl`` from the
  :class:`~robosandbox.recorder.local.LocalRecorder`) and replays it
  open-loop.
- :func:`load_policy` — stub loader. Recognises a directory containing
  ``policy.json`` + a trajectory file for :class:`ReplayTrajectoryPolicy`.
  Any other path raises :class:`ImportError` with instructions for the
  user to wire their own LeRobot/ACT checkpoint loader here.

Real policies plug in via :func:`load_policy` (extend the dispatch in this
module) or by direct construction — any object with a compatible ``act``
method satisfies the :class:`Policy` protocol.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from robosandbox.policy.lerobot_adapter import LeRobotPolicyAdapter
from robosandbox.types import Observation

__all__ = [
    "LeRobotPolicyAdapter",
    "NeuralPolicy",
    "Policy",
    "ReplayTrajectoryPolicy",
    "load_policy",
    "run_eval_parallel",
    "run_policy",
]


# ---- Policy protocol --------------------------------------------------


@runtime_checkable
class Policy(Protocol):
    """A closed-loop joint-space policy.

    Implementers must return a flat ``(n_dof + 1,)`` array per call:
    ``[j1, ..., jN, gripper]`` where ``gripper ∈ [0, 1]`` (0 == open,
    1 == closed), matching :meth:`MuJoCoBackend.step`'s semantics.
    """

    def act(self, obs: Observation) -> np.ndarray: ...


# ---- ReplayTrajectoryPolicy -------------------------------------------


class ReplayTrajectoryPolicy:
    """Open-loop replay of a pre-recorded joint-space trajectory.

    Accepts either a simple JSONL (``{joints: [...], gripper: float}``) or
    the ``events.jsonl`` emitted by
    :class:`~robosandbox.recorder.local.LocalRecorder` (``robot_joints`` +
    ``gripper_width``). Gripper width is normalised to the ``[0, 1]``
    command space using the max-open width (``0.07`` m).

    Ignores ``obs`` — useful for sanity-checking a recording or for
    demonstrating the :class:`Policy` protocol without a trained model.
    """

    # MuJoCoBackend opens to ~0.07m (2 * 0.035). Treat that as fully open.
    _MAX_WIDTH_M = 0.07

    def __init__(
        self,
        actions: np.ndarray,
        *,
        action_lookahead: int = 1,
    ) -> None:
        if actions.ndim != 2 or actions.shape[0] == 0:
            raise ValueError(
                f"actions must be 2D (T, n_dof+1), got shape {actions.shape}"
            )
        if action_lookahead < 1:
            raise ValueError(f"action_lookahead must be >= 1, got {action_lookahead}")
        self._actions = actions
        self._lookahead = int(action_lookahead)
        self._i = 0

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        action_lookahead: int = 1,
    ) -> ReplayTrajectoryPolicy:
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"Trajectory {path} is empty")

        actions = []
        for r in rows:
            if "joints" in r:
                joints = np.asarray(r["joints"], dtype=np.float64).ravel()
            elif "robot_joints" in r:
                joints = np.asarray(r["robot_joints"], dtype=np.float64).ravel()
            else:
                raise ValueError(
                    f"Row missing 'joints' or 'robot_joints': {r!r}"
                )
            if "gripper" in r:
                gripper = float(r["gripper"])
            elif "gripper_width" in r:
                # width (m) → closed fraction ∈ [0, 1]
                w = float(r["gripper_width"])
                gripper = float(np.clip(1.0 - (w / cls._MAX_WIDTH_M), 0.0, 1.0))
            else:
                gripper = 0.0
            actions.append(np.concatenate([joints, [gripper]]))

        arr = np.stack(actions, axis=0)
        return cls(arr, action_lookahead=action_lookahead)

    def act(self, obs: Observation) -> np.ndarray:
        i = min(self._i, self._actions.shape[0] - 1)
        action = self._actions[i].copy()
        self._i += self._lookahead
        return action

    def reset(self) -> None:
        self._i = 0

    def __len__(self) -> int:
        return int(self._actions.shape[0])


# ---- run_policy --------------------------------------------------------


def run_policy(
    sim: Any,
    policy: Policy,
    max_steps: int = 1000,
    *,
    success: Any = None,  # SuccessCriterion | None — imported lazily to dodge cycles
    on_step: Callable[[Observation, np.ndarray], None] | None = None,
) -> dict:
    """Drive ``sim`` with ``policy`` for up to ``max_steps`` ticks.

    Returns a dict: ``{success, steps, final_obs, initial_obs}``. ``success``
    is the boolean outcome of ``success`` evaluated against
    ``(initial_obs, final_obs)``, or ``None`` when no criterion is supplied.
    """
    n_dof = getattr(sim, "n_dof", 6)
    initial_obs = sim.observe()

    last_obs = initial_obs
    steps_done = 0
    for step_i in range(max_steps):
        obs = sim.observe()
        action = np.asarray(policy.act(obs), dtype=np.float64).ravel()
        if action.shape != (n_dof + 1,):
            raise ValueError(
                f"policy.act must return shape ({n_dof + 1},), got {action.shape}"
            )
        target_joints = action[:n_dof]
        gripper = float(action[n_dof])
        sim.step(target_joints=target_joints, gripper=gripper)
        last_obs = obs
        steps_done = step_i + 1
        if on_step is not None:
            on_step(obs, action)

    final_obs = sim.observe()

    success_ok: bool | None
    if success is None:
        success_ok = None
    else:
        # Lazy import to avoid tasks→policy cycles if any.
        from robosandbox.tasks.runner import _eval_criterion

        success_ok, _detail = _eval_criterion(success, initial_obs, final_obs)

    return {
        "success": success_ok,
        "steps": steps_done,
        "initial_obs": initial_obs,
        "final_obs": final_obs,
        "last_obs_before_final": last_obs,
    }


# ---- run_eval_parallel -------------------------------------------------


def run_eval_parallel(
    sim: Any,
    policy: Policy,
    max_steps: int = 500,
    *,
    success: Any = None,
    settle_steps: int = 100,
) -> dict:
    """GPU-parallel policy evaluation across all worlds in ``sim``.

    Calls ``sim.observe_all()`` once per step, runs ``policy.act`` on
    world-0's observation (broadcast to all worlds), then checks
    ``success`` per world.  Returns aggregated stats.

    ``sim`` must expose ``observe_all() -> list[Observation]`` and
    ``n_worlds: int`` — i.e., a :class:`~robosandbox.sim.newton_backend.NewtonBackend`
    with ``world_count > 1`` (also works with ``world_count=1``).
    """
    import time

    n_worlds: int = getattr(sim, "n_worlds", 1)
    n_dof: int = getattr(sim, "n_dof", 6)

    observe_all = getattr(sim, "observe_all", None)
    if observe_all is None:
        raise TypeError("sim must expose observe_all() for parallel eval")

    # Settle physics before recording initial state
    for _ in range(settle_steps):
        sim.step()

    initial_obs_all: list[Any] = observe_all()

    done = [False] * n_worlds
    success_per_world = [False] * n_worlds
    steps_done = 0

    t0 = time.time()
    for _ in range(max_steps):
        obs_all: list[Any] = observe_all()
        # Drive policy on world-0 observation; broadcast to all worlds
        obs_0 = obs_all[0]
        action = np.asarray(policy.act(obs_0), dtype=np.float64).ravel()
        if action.shape != (n_dof + 1,):
            raise ValueError(
                f"policy.act must return shape ({n_dof + 1},), got {action.shape}"
            )
        target_joints = action[:n_dof]
        gripper = float(action[n_dof])
        sim.step(target_joints=target_joints, gripper=gripper)
        steps_done += 1

        # Evaluate success criterion per world
        if success is not None:
            from robosandbox.tasks.runner import _eval_criterion

            for w in range(n_worlds):
                if done[w]:
                    continue
                ok, _ = _eval_criterion(success, initial_obs_all[w], obs_all[w])
                if ok:
                    success_per_world[w] = True
                    done[w] = True

            if all(done):
                break

    wall = time.time() - t0
    n_success = sum(success_per_world)
    total_env_steps = steps_done * n_worlds
    throughput = total_env_steps / wall if wall > 0 else 0.0

    return {
        "n_worlds": n_worlds,
        "successes": n_success,
        "rate": n_success / n_worlds if n_worlds > 0 else 0.0,
        "steps": steps_done,
        "wall": wall,
        "throughput": throughput,
        "success_per_world": success_per_world,
    }


# ---- load_policy stub --------------------------------------------------


_BRING_YOUR_OWN_CHECKPOINT_HINT = (
    "No policy loader matched {path!r}. RoboSandbox ships a reference "
    "ReplayTrajectoryPolicy but does not bundle a LeRobot/ACT/Diffusion-Policy "
    "checkpoint loader. To run a trained checkpoint, either: (1) drop a "
    "policy.json + events.jsonl under a directory and point --policy at it, "
    "or (2) extend robosandbox.policy.load_policy to dispatch on your "
    "checkpoint format (LeRobot, torchscript, onnx, ...)."
)


def load_policy(path: str | Path) -> Policy:
    """Load a policy from a checkpoint-or-episode directory.

    Currently handles one case: a directory containing ``policy.json``
    of the form ``{"kind": "replay_trajectory", "trajectory": "<file>"}``.

    Raises :class:`ImportError` with an explicit bring-your-own-checkpoint
    message otherwise — this is the extension seam for real policies.
    """
    p = Path(path)

    if p.is_dir():
        cfg_file = p / "policy.json"
        if cfg_file.exists():
            cfg = json.loads(cfg_file.read_text())
            kind = cfg.get("kind", "replay_trajectory")
            if kind == "replay_trajectory":
                traj_rel = cfg.get("trajectory", "events.jsonl")
                traj_path = p / traj_rel
                if not traj_path.exists():
                    raise FileNotFoundError(
                        f"policy.json references trajectory {traj_rel!r} "
                        f"but {traj_path} does not exist"
                    )
                lookahead = int(cfg.get("action_lookahead", 1))
                return ReplayTrajectoryPolicy.from_jsonl(
                    traj_path, action_lookahead=lookahead
                )
            if kind == "ppo_neural":
                from robosandbox.rl.ppo import NeuralPolicy
                return NeuralPolicy.load(p)
            raise ImportError(
                _BRING_YOUR_OWN_CHECKPOINT_HINT.format(path=str(p))
                + f" (policy.json kind={kind!r} not recognised)"
            )
        # Directory w/o policy.json but with events.jsonl — auto-wrap.
        events = p / "events.jsonl"
        if events.exists():
            return ReplayTrajectoryPolicy.from_jsonl(events)

    raise ImportError(_BRING_YOUR_OWN_CHECKPOINT_HINT.format(path=str(p)))
