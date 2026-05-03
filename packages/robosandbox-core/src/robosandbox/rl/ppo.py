"""PPO training loop for robosandbox Newton multi-world backend.

Requires PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu121

Usage:
    from robosandbox.rl.ppo import train_ppo, NeuralPolicy
    policy = train_ppo(sim, task, total_steps=5_000_000, device="cuda:0")
    policy.save("runs/rl/pick_cube")
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from robosandbox.rl.obs_encoder import ObsEncoder
from robosandbox.rl.reward import compute_shaped_reward
from robosandbox.tasks.loader import Task
from robosandbox.types import Observation


# ---- Actor-Critic --------------------------------------------------------


def _layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    """Two-head MLP: shared trunk → policy mean + log-std, value."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_d = obs_dim
        for h in hidden:
            layers += [_layer_init(nn.Linear(in_d, h)), nn.Tanh()]
            in_d = h
        self.trunk = nn.Sequential(*layers)
        self.actor_mean = _layer_init(nn.Linear(in_d, act_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))
        self.critic = _layer_init(nn.Linear(in_d, 1), std=1.0)
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self._hidden = hidden

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(self.trunk(x))

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        mean = self.actor_mean(h)
        logstd = self.actor_logstd.expand_as(mean)
        dist = torch.distributions.Normal(mean, logstd.exp())
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), self.critic(h)

    def get_action_mean(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor_mean(self.trunk(x))

    def extra_state(self) -> dict:
        return {"obs_dim": self._obs_dim, "act_dim": self._act_dim, "hidden": list(self._hidden)}

    @classmethod
    def from_checkpoint(cls, path: Path) -> ActorCritic:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        meta = ckpt["meta"]
        ac = cls(meta["obs_dim"], meta["act_dim"], tuple(meta["hidden"]))
        ac.load_state_dict(ckpt["state_dict"])
        return ac

    def save_checkpoint(self, path: Path) -> None:
        torch.save(
            {"state_dict": self.state_dict(), "meta": self.extra_state()},
            path,
        )


# ---- Neural policy (inference wrapper) -----------------------------------


class NeuralPolicy:
    """Wraps a trained ActorCritic as a robosandbox Policy (deterministic mean).

    Supports two action spaces:
      - "joint":  policy outputs (joint_deltas, gripper). Default.
      - "ee_xyz": policy outputs (dx, dy, dz, gripper); converted to joint
                  deltas via DLS pseudoinverse using a CPU MuJoCo Jacobian.
                  Requires ``scene`` at load time.
    """

    def __init__(
        self,
        actor_critic: ActorCritic,
        encoder: ObsEncoder,
        delta_scale: float = 0.05,
        action_space: str = "joint",
        ee_delta_scale: float = 0.02,
        ee_body: str = "hand",
        ee_kine: Any = None,
        policy_id: str | None = None,
        parent_policy_id: str | None = None,
        lineage_op: str | None = None,
    ) -> None:
        self._ac = actor_critic.eval()
        self._enc = encoder
        self._delta_scale = delta_scale
        self._n_dof = encoder.n_dof
        self._action_space = action_space
        self._ee_delta_scale = ee_delta_scale
        self._ee_body = ee_body
        self._ee_kine = ee_kine
        self._policy_id = policy_id
        self._parent_policy_id = parent_policy_id
        self._lineage_op = lineage_op
        # ee_kine is required to call act() in ee_xyz mode but not to save();
        # validation deferred to act() so save-only flows (e.g. logging
        # lineage at training-end) don't need to materialize the IK helper.

    def act(self, obs: Observation) -> np.ndarray:
        vec = self._enc.normalize(self._enc.encode(obs))
        x = torch.from_numpy(vec).unsqueeze(0)
        with torch.no_grad():
            mean = self._ac.get_action_mean(x).squeeze(0)
        if self._action_space == "ee_xyz":
            if self._ee_kine is None:
                raise ValueError(
                    "NeuralPolicy with action_space='ee_xyz' requires ee_kine "
                    "(an instance of FrankaEEKine) — pass scene to NeuralPolicy.load()."
                )
            current_q = np.asarray(obs.robot_joints, dtype=np.float64)
            delta_ee = mean[:3].numpy().astype(np.float64) * self._ee_delta_scale  # (3,)
            delta_q = self._ee_kine.delta_q_from_delta_ee(
                current_q[None, :], delta_ee[None, :]
            )[0]                                                                   # (n_dof,)
            target_q = current_q + delta_q
            gripper = float(torch.sigmoid(mean[3]).item())
        elif self._action_space == "ee_xyz_rpy":
            if self._ee_kine is None:
                raise ValueError(
                    "NeuralPolicy with action_space='ee_xyz_rpy' requires ee_kine — "
                    "pass scene to NeuralPolicy.load()."
                )
            current_q = np.asarray(obs.robot_joints, dtype=np.float64)
            # mean[:3] = trans delta, mean[3:6] = angular delta. We scale both
            # by ee_delta_scale; rotation in radians, translation in meters.
            delta_ee = np.zeros(6, dtype=np.float64)
            delta_ee[:3] = mean[:3].numpy().astype(np.float64) * self._ee_delta_scale
            delta_ee[3:] = mean[3:6].numpy().astype(np.float64) * self._ee_delta_scale
            delta_q = self._ee_kine.delta_q_from_delta_ee_6d(
                current_q[None, :], delta_ee[None, :]
            )[0]
            target_q = current_q + delta_q
            gripper = float(torch.sigmoid(mean[6]).item())
        else:
            deltas = mean[: self._n_dof].numpy() * self._delta_scale
            target_q = np.asarray(obs.robot_joints, dtype=np.float64) + deltas
            gripper = float(torch.sigmoid(mean[self._n_dof]).item())
        return np.concatenate([target_q, [gripper]])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._ac.save_checkpoint(path / "actor_critic.pt")
        self._enc.save(path / "obs_config.json")
        pid = self._policy_id or path.resolve().name
        (path / "policy.json").write_text(
            json.dumps(
                {
                    "kind": "ppo_neural",
                    "model": "actor_critic.pt",
                    "obs_config": "obs_config.json",
                    "delta_scale": self._delta_scale,
                    "action_space": self._action_space,
                    "ee_delta_scale": self._ee_delta_scale,
                    "ee_body": self._ee_body,
                    "policy_id": pid,
                    "parent_policy_id": self._parent_policy_id,
                    "lineage_op": self._lineage_op,
                },
                indent=2,
            )
        )
        # Best-effort eval_log emission — do not crash training on log failure.
        # append_policy is idempotent on policy_id so checkpoint_interval saves
        # don't multiply rows.
        try:
            from datetime import datetime, timezone

            from robosandbox.eval_log import EvalLogStore, PolicyRow
            EvalLogStore("runs/eval_log").append_policy(PolicyRow(
                policy_id=pid,
                kind="ppo_neural",
                parent_policy_id=self._parent_policy_id,
                lineage_op=self._lineage_op,
                training_demo_set_id=None,
                training_steps=None,
                training_dist_summary=None,
                trained_at=datetime.now(timezone.utc).isoformat(),
                checkpoint_path=str(path.resolve()),
                metadata={},
            ))
        except Exception:
            pass

    @classmethod
    def load(cls, path: str | Path, scene: Any = None, arm_joint_names: list[str] | None = None) -> NeuralPolicy:
        path = Path(path)
        cfg = json.loads((path / "policy.json").read_text())
        encoder = ObsEncoder.load(path / cfg["obs_config"])
        ac = ActorCritic.from_checkpoint(path / cfg["model"])
        action_space = str(cfg.get("action_space", "joint"))
        ee_delta_scale = float(cfg.get("ee_delta_scale", 0.02))
        ee_body = str(cfg.get("ee_body", "hand"))
        ee_kine = None
        if action_space in ("ee_xyz", "ee_xyz_rpy"):
            if scene is None:
                raise ValueError(
                    f"Loading a {action_space} NeuralPolicy requires the task scene; "
                    "pass scene= to NeuralPolicy.load()."
                )
            from robosandbox.rl.ee_ik import FrankaEEKine
            # Default arm joint names for Franka; caller can override.
            arm_names = arm_joint_names or [
                "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7",
            ]
            ee_kine = FrankaEEKine(scene=scene, ee_body=ee_body, arm_joint_names=arm_names)
        return cls(
            ac, encoder,
            delta_scale=float(cfg.get("delta_scale", 0.05)),
            action_space=action_space,
            ee_delta_scale=ee_delta_scale,
            ee_body=ee_body,
            ee_kine=ee_kine,
            policy_id=cfg.get("policy_id"),
            parent_policy_id=cfg.get("parent_policy_id"),
            lineage_op=cfg.get("lineage_op"),
        )


# ---- PPO training --------------------------------------------------------


def _compute_gae(
    rewards: np.ndarray,     # (T, N)
    values: np.ndarray,      # (T, N)
    last_values: np.ndarray, # (N,)
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    T, N = rewards.shape
    advantages = np.zeros_like(rewards)
    last_adv = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        next_val = last_values if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val - values[t]
        last_adv = delta + gamma * lam * last_adv
        advantages[t] = last_adv
    return advantages, advantages + values


def train_ppo(
    sim: Any,
    task: Task,
    *,
    total_steps: int = 5_000_000,
    n_steps: int = 256,
    n_epochs: int = 4,
    batch_size: int = 4096,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    delta_scale: float = 0.05,
    settle_steps: int = 50,
    device: str = "cuda:0",
    save_path: str | Path | None = None,
    log_interval: int = 5,
    checkpoint_interval: int = 20,
    warm_start: str | Path | None = None,
    freeze_encoder_stats: bool = False,
    action_repeat: int = 1,
    init_logstd: float = 0.0,
    hidden: tuple[int, ...] = (256, 256),
    action_space: str = "joint",          # "joint" | "ee_xyz"
    ee_delta_scale: float = 0.02,          # max EE displacement per policy step (m)
    ee_body: str = "hand",
    ee_ik_damping: float = 0.05,
    policy_id: str | None = None,
    parent_policy_id: str | None = None,
    lineage_op: str | None = None,
) -> NeuralPolicy:
    """Train a PPO policy against a Newton multi-world sim.

    ``sim`` must expose ``n_worlds``, ``n_dof``, ``observe_all()``,
    ``step_all(targets, grippers)``, and ``reset()``.
    """
    from robosandbox.tasks.runner import _eval_criterion

    if action_repeat < 1:
        raise ValueError(f"action_repeat must be >= 1, got {action_repeat}")

    N: int = sim.n_worlds
    n_dof: int = sim.n_dof

    object_ids = [obj.id for obj in task.scene.objects]
    encoder = ObsEncoder(object_ids, n_dof=n_dof)

    # Cache the success-target object id for the batched reward fast-path.
    from robosandbox.tasks.runner import criterion_target_object
    from robosandbox.rl.reward import (
        compute_shaped_reward_batch,
        compute_inserted_reward_batch,
    )
    success_target_oid = criterion_target_object(task.success)
    success_kind = task.success.data.get("kind") if task.success is not None else None
    use_batched_reward = (
        task.success is not None
        and success_kind in ("ee_near", "inserted")
        and success_target_oid is not None
    )
    use_insertion_reward = success_kind == "inserted"

    # EE-space action setup: actor outputs (dx, dy, dz, gripper) instead of
    # (joint_deltas, gripper). DLS pseudoinverse converts to joint deltas at
    # each inner step. Action space matches the goal space → policy doesn't
    # have to learn 7-DOF coordination implicitly.
    use_ee_action = action_space in ("ee_xyz", "ee_xyz_rpy")
    use_6dof_action = action_space == "ee_xyz_rpy"
    ee_kine: Any = None
    if use_ee_action:
        from robosandbox.rl.ee_ik import FrankaEEKine
        arm_names = list(getattr(sim, "_arm_joint_names", []) or [])
        if not arm_names:
            arm_names = list(getattr(sim, "joint_names", []))
        if not arm_names:
            raise ValueError(f"{action_space} action space needs sim.joint_names or sim._arm_joint_names")
        ee_kine = FrankaEEKine(
            scene=task.scene, ee_body=ee_body,
            arm_joint_names=arm_names, damping=ee_ik_damping,
        )
        print(f"[train] action_space: {action_space}  (ee_body={ee_body!r}, "
              f"delta_scale={ee_delta_scale}m, damping={ee_ik_damping})")
    else:
        print(f"[train] action_space: joint  (delta_scale={delta_scale}rad)")
    obs_dim = encoder.obs_dim
    if use_6dof_action:
        act_dim = 6 + 1  # (dx,dy,dz,droll,dpitch,dyaw) + gripper
    elif use_ee_action:
        act_dim = 3 + 1  # (dx,dy,dz) + gripper
    else:
        act_dim = n_dof + 1  # joint deltas + gripper

    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    if str(device_t) != device and "cuda" in device:
        print(f"[train] WARNING: {device} unavailable, falling back to {device_t}")

    if warm_start is not None:
        # Replace the random-init actor with a behaviorally-cloned one.
        # Also adopt the matching ObsEncoder so PPO's normalization stats
        # start from where distillation left off — random-init Welford
        # stats would otherwise re-shape the input distribution every
        # rollout and wipe the prior in the first few iterations.
        ws = Path(warm_start)
        ac_ckpt = ws / "actor_critic.pt"
        enc_ckpt = ws / "obs_config.json"
        if not ac_ckpt.exists() or not enc_ckpt.exists():
            raise FileNotFoundError(
                f"warm_start={ws} missing actor_critic.pt or obs_config.json"
            )
        ac = ActorCritic.from_checkpoint(ac_ckpt).to(device_t)
        if ac._obs_dim != obs_dim or ac._act_dim != act_dim:
            raise ValueError(
                f"warm_start actor shape (obs_dim={ac._obs_dim}, act_dim={ac._act_dim}) "
                f"does not match task (obs_dim={obs_dim}, act_dim={act_dim})"
            )
        loaded_enc = ObsEncoder.load(enc_ckpt)
        if loaded_enc.obs_dim != obs_dim:
            raise ValueError(
                f"warm_start encoder obs_dim={loaded_enc.obs_dim} != {obs_dim}"
            )
        encoder = loaded_enc
        print(f"[train] warm-start:    {ws} (loaded actor + encoder)")
    else:
        ac = ActorCritic(obs_dim, act_dim, hidden=hidden).to(device_t)
    if init_logstd != 0.0:
        with torch.no_grad():
            ac.actor_logstd.fill_(init_logstd)
    optimizer = torch.optim.Adam(ac.parameters(), lr=lr, eps=1e-5)

    # Rollout buffers — preallocated for the full trajectory
    obs_buf = np.zeros((n_steps, N, obs_dim), dtype=np.float32)
    act_buf = np.zeros((n_steps, N, act_dim), dtype=np.float32)
    logp_buf = np.zeros((n_steps, N), dtype=np.float32)
    val_buf = np.zeros((n_steps, N), dtype=np.float32)
    rew_buf = np.zeros((n_steps, N), dtype=np.float32)

    n_params = sum(p.numel() for p in ac.parameters())
    print(f"[train] obs_dim:       {obs_dim}")
    print(f"[train] act_dim:       {act_dim}")
    print(f"[train] world_count:   {N}")
    print(f"[train] n_steps:       {n_steps}")
    print(f"[train] total_steps:   {total_steps:,}")
    print(f"[train] parameters:    {n_params:,}")
    print(f"[train] device:        {device_t}")
    print(f"[train] action_repeat: {action_repeat}")
    print(f"[train] init_logstd:   {float(ac.actor_logstd[0,0].item()):.3f} "
          f"(std={float(ac.actor_logstd[0,0].exp().item()):.3f})")

    total_env_steps = 0
    iteration = 0
    t0 = time.time()

    # Per-iter diagnostics: track which worlds ever hit threshold during the
    # rollout (success "in flight") vs which are still inside at the last step.
    # Big delta between the two means "policy reaches but can't stay" — calls
    # for smoothness penalty / smaller logstd / hold-still reward.
    ever_within = np.zeros(N, dtype=bool)
    iter_dist_sum = 0.0
    iter_dist_n = 0
    # Insertion-specific: track which worlds aligned within axis_tol_deg AND
    # achieved min_depth past the port plane during rollout.
    ever_aligned = np.zeros(N, dtype=bool)
    ever_inserted = np.zeros(N, dtype=bool)
    iter_align_deg_sum = 0.0
    iter_depth_mm_sum = 0.0
    iter_align_n = 0
    # Per-world consecutive-inserted-step streak, persists across PPO iters.
    # Resets to 0 when a world drops out of inserted state. Used by the
    # hold-still bonus in the inserted reward.
    inserted_streak = np.zeros(N, dtype=np.int64)

    # Fast-path detection: backends that expose observe_all_arrays() return
    # raw numpy batches and skip the per-world Observation/Pose dataclass
    # construction — ~3-5x throughput at large world counts.
    use_array_api = hasattr(sim, "observe_all_arrays")
    if use_array_api:
        print(f"[train] obs api:       observe_all_arrays (vectorized fast path)")

    while total_env_steps < total_steps:
        # Reset rollout diagnostics
        ever_within = np.zeros(N, dtype=bool)
        iter_dist_sum = 0.0
        iter_dist_n = 0
        ever_aligned = np.zeros(N, dtype=bool)
        ever_inserted = np.zeros(N, dtype=bool)
        iter_align_deg_sum = 0.0
        iter_depth_mm_sum = 0.0
        iter_align_n = 0
        # Reset hold-still streak at iter boundaries — sim.reset() returns
        # the world to home pose, which is not inserted.
        inserted_streak = np.zeros(N, dtype=np.int64)

        # ---- Rollout collection ------------------------------------------
        sim.reset()
        for _ in range(settle_steps):
            sim.step()

        if use_array_api:
            obs_arrays = sim.observe_all_arrays()
            initial_obs_all = None  # not used in array path
        else:
            obs_all = sim.observe_all()
            initial_obs_all = obs_all

        for t in range(n_steps):
            if use_array_api:
                raw_vecs = encoder.encode_arrays(obs_arrays)
            else:
                raw_vecs = encoder.encode_batch(obs_all)
            norm_vecs = encoder.normalize_batch(raw_vecs)      # (N, obs_dim) normalized

            obs_buf[t] = norm_vecs

            obs_t = torch.from_numpy(norm_vecs).to(device_t)
            with torch.no_grad():
                actions, logps, _, values = ac.get_action_and_value(obs_t)

            act_np = actions.cpu().numpy()                     # (N, act_dim)
            act_buf[t] = act_np
            logp_buf[t] = logps.cpu().numpy()
            val_buf[t] = values.squeeze(-1).cpu().numpy()

            # Convert to sim targets.
            if use_array_api:
                current_qs = obs_arrays["joints"]              # (N, n_dof) already
            else:
                current_qs = np.array(
                    [o.robot_joints for o in obs_all], dtype=np.float64
                )                                              # (N, n_dof)
            if use_6dof_action:
                # action[:3] = trans delta, action[3:6] = angular delta.
                delta_ee = np.zeros((act_np.shape[0], 6), dtype=np.float64)
                delta_ee[:, :3] = act_np[:, :3].astype(np.float64) * ee_delta_scale
                delta_ee[:, 3:] = act_np[:, 3:6].astype(np.float64) * ee_delta_scale
                deltas_q = ee_kine.delta_q_from_delta_ee_6d(
                    current_qs, delta_ee
                )
                targets = current_qs + deltas_q
                gripper_logits = act_np[:, 6]
            elif use_ee_action:
                # action[:3] = ee delta in (dx, dy, dz). Convert via DLS pseudoinverse.
                delta_ee = act_np[:, :3].astype(np.float64) * ee_delta_scale  # (N, 3)
                deltas_q = ee_kine.delta_q_from_delta_ee(
                    current_qs, delta_ee
                )                                              # (N, n_arm)
                targets = current_qs + deltas_q
                gripper_logits = act_np[:, 3]
            else:
                deltas = act_np[:, :n_dof].astype(np.float64) * delta_scale
                targets = current_qs + deltas                  # (N, n_dof)
                gripper_logits = act_np[:, n_dof]
            grippers = 1.0 / (1.0 + np.exp(-np.clip(gripper_logits, -20, 20)))

            for _ in range(action_repeat):
                sim.step_all(targets, grippers)

            if use_array_api:
                obs_arrays = sim.observe_all_arrays()
            else:
                obs_all = sim.observe_all()

            # Shaped reward — vectorized fast path for ee_near + inserted.
            if use_batched_reward and use_array_api:
                ee_xyz_batch = obs_arrays["ee_xyz"]
                tgt_xyz_batch = obs_arrays["obj_xyz"][success_target_oid]
                if use_insertion_reward:
                    ee_quat_batch = obs_arrays["ee_quat"]
                    port_quat_batch = obs_arrays.get("obj_quat", {}).get(success_target_oid)
                    rew_buf[t] = compute_inserted_reward_batch(
                        task.success, ee_xyz_batch, ee_quat_batch, tgt_xyz_batch,
                        port_quat_batch=port_quat_batch,
                        inserted_streak=inserted_streak,
                        last_action=act_np,
                    )
                    # Update streak for next step's hold-still bonus. Compute
                    # success per-world from the same predicate the reward used
                    # (depth + axis + yaw_ok). Cheap inline reproduction —
                    # only needs depth here since align/yaw were already
                    # computed by the reward call's diagnostics block below.
                    # Insertion diagnostics: tip-to-port distance, alignment, depth.
                    check = task.success.data
                    port_axis_v = np.asarray(
                        check.get("port_axis", [0.0, 0.0, 1.0]), dtype=np.float64
                    )
                    port_axis_v /= max(float(np.linalg.norm(port_axis_v)), 1e-9)
                    plane_z_v = float(check.get("port_plane_z", 0.0))
                    offset_v = np.asarray(
                        check.get("plug_tip_offset_ee", [0.0, 0.0, 0.05]),
                        dtype=np.float64,
                    )
                    min_depth_v = float(check.get("min_depth_m", 0.005))
                    axis_tol_deg_v = float(check.get("axis_tol_deg", 5.0))
                    qx = ee_quat_batch[:, 0]; qy = ee_quat_batch[:, 1]
                    qz = ee_quat_batch[:, 2]; qw = ee_quat_batch[:, 3]
                    x2_, y2_, z2_ = qx + qx, qy + qy, qz + qz
                    wx_, wy_, wz_ = qw * x2_, qw * y2_, qw * z2_
                    xx_, xy_, xz_ = qx * x2_, qx * y2_, qx * z2_
                    yy_, yz_, zz_ = qy * y2_, qy * z2_, qz * z2_
                    R_ = np.empty((ee_quat_batch.shape[0], 3, 3), dtype=np.float64)
                    R_[:, 0, 0] = 1.0 - (yy_ + zz_); R_[:, 0, 1] = xy_ - wz_;        R_[:, 0, 2] = xz_ + wy_
                    R_[:, 1, 0] = xy_ + wz_;        R_[:, 1, 1] = 1.0 - (xx_ + zz_); R_[:, 1, 2] = yz_ - wx_
                    R_[:, 2, 0] = xz_ - wy_;        R_[:, 2, 1] = yz_ + wx_;         R_[:, 2, 2] = 1.0 - (xx_ + yy_)
                    tip_b = ee_xyz_batch + np.einsum("nij,j->ni", R_, offset_v)
                    plug_axis_b = np.einsum("nij,j->ni", R_, np.array([0.0, 0.0, 1.0]))
                    _dist = np.linalg.norm(tip_b - tgt_xyz_batch, axis=1)
                    iter_dist_sum += float(_dist.mean())
                    iter_dist_n += 1
                    cos_a = np.clip(
                        np.einsum("ni,i->n", plug_axis_b, -port_axis_v), -1.0, 1.0
                    )
                    align_deg_b = np.degrees(np.arccos(cos_a))
                    depth_b = plane_z_v - np.einsum("ni,i->n", tip_b, port_axis_v)
                    iter_align_deg_sum += float(align_deg_b.mean())
                    iter_depth_mm_sum += float(depth_b.mean()) * 1000.0
                    iter_align_n += 1
                    ever_aligned |= align_deg_b <= axis_tol_deg_v
                    ever_inserted |= depth_b >= min_depth_v
                    # Update streak: increment for worlds in inserted state,
                    # reset to 0 for worlds out of it. Used by next step's
                    # hold-still bonus.
                    full_inserted = (
                        (depth_b >= min_depth_v) & (align_deg_b <= axis_tol_deg_v)
                    )
                    inserted_streak = np.where(full_inserted, inserted_streak + 1, 0)
                else:
                    rew_buf[t] = compute_shaped_reward_batch(
                        task.success, ee_xyz_batch, tgt_xyz_batch
                    )
                    # Diagnostics: track distance and threshold-crossings
                    _dist = np.linalg.norm(ee_xyz_batch - tgt_xyz_batch, axis=1)
                    iter_dist_sum += float(_dist.mean())
                    iter_dist_n += 1
                    _thresh = float(task.success.data.get("threshold_m", 0.05))
                    ever_within |= (_dist <= _thresh)
            elif use_batched_reward:
                ee_xyz_batch = np.array(
                    [o.ee_pose.xyz for o in obs_all], dtype=np.float64
                )
                tgt_xyz_batch = np.array(
                    [o.scene_objects[success_target_oid].xyz for o in obs_all],
                    dtype=np.float64,
                )
                if use_insertion_reward:
                    ee_quat_batch = np.array(
                        [o.ee_pose.quat_xyzw for o in obs_all], dtype=np.float64
                    )
                    port_quat_batch = np.array(
                        [o.scene_objects[success_target_oid].quat_xyzw for o in obs_all],
                        dtype=np.float64,
                    )
                    rew_buf[t] = compute_inserted_reward_batch(
                        task.success, ee_xyz_batch, ee_quat_batch, tgt_xyz_batch,
                        port_quat_batch=port_quat_batch,
                    )
                else:
                    rew_buf[t] = compute_shaped_reward_batch(
                        task.success, ee_xyz_batch, tgt_xyz_batch
                    )
            else:
                for w in range(N):
                    rew_buf[t, w] = compute_shaped_reward(
                        task.success, initial_obs_all[w], obs_all[w]
                    )

            if not freeze_encoder_stats:
                encoder.update_stats_batch(raw_vecs)

        # Bootstrap value
        if use_array_api:
            last_raw = encoder.encode_arrays(obs_arrays)
        else:
            last_raw = encoder.encode_batch(obs_all)
        last_norm = encoder.normalize_batch(last_raw)
        with torch.no_grad():
            last_vals = (
                ac.get_value(torch.from_numpy(last_norm).to(device_t))
                .squeeze(-1)
                .cpu()
                .numpy()
            )

        advantages, returns = _compute_gae(rew_buf, val_buf, last_vals, gamma, gae_lambda)

        # ---- PPO update --------------------------------------------------
        n_samples = n_steps * N
        b_obs = torch.from_numpy(obs_buf.reshape(n_samples, obs_dim)).to(device_t)
        b_act = torch.from_numpy(act_buf.reshape(n_samples, act_dim)).to(device_t)
        b_logp = torch.from_numpy(logp_buf.reshape(n_samples)).to(device_t)
        b_adv = torch.from_numpy(advantages.reshape(n_samples).astype(np.float32)).to(device_t)
        b_ret = torch.from_numpy(returns.reshape(n_samples).astype(np.float32)).to(device_t)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        for _ in range(n_epochs):
            perm = np.random.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                mb = torch.from_numpy(perm[start : start + batch_size]).to(device_t)
                _, new_logp, entropy, new_val = ac.get_action_and_value(b_obs[mb], b_act[mb])
                ratio = torch.exp(new_logp - b_logp[mb])
                mb_adv = b_adv[mb]
                pg_loss = -torch.min(
                    ratio * mb_adv,
                    torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv,
                ).mean()
                v_loss = 0.5 * ((new_val.squeeze(-1) - b_ret[mb]) ** 2).mean()
                loss = pg_loss + 0.5 * v_loss - entropy_coef * entropy.mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), max_grad_norm)
                optimizer.step()

        total_env_steps += n_steps * N * action_repeat
        iteration += 1

        if iteration % log_interval == 0:
            mean_rew = float(rew_buf.mean())
            if use_array_api and use_batched_reward and use_insertion_reward:
                # End-of-rollout success: full inserted criterion (depth + align)
                check = task.success.data
                port_axis_v = np.asarray(
                    check.get("port_axis", [0.0, 0.0, 1.0]), dtype=np.float64
                )
                port_axis_v /= max(float(np.linalg.norm(port_axis_v)), 1e-9)
                plane_z_v = float(check.get("port_plane_z", 0.0))
                offset_v = np.asarray(
                    check.get("plug_tip_offset_ee", [0.0, 0.0, 0.05]),
                    dtype=np.float64,
                )
                min_depth_v = float(check.get("min_depth_m", 0.005))
                axis_tol_deg_v = float(check.get("axis_tol_deg", 5.0))
                ee_xyz_b = obs_arrays["ee_xyz"]
                ee_quat_b = obs_arrays["ee_quat"]
                qx = ee_quat_b[:, 0]; qy = ee_quat_b[:, 1]
                qz = ee_quat_b[:, 2]; qw = ee_quat_b[:, 3]
                x2_, y2_, z2_ = qx + qx, qy + qy, qz + qz
                wx_, wy_, wz_ = qw * x2_, qw * y2_, qw * z2_
                xx_, xy_, xz_ = qx * x2_, qx * y2_, qx * z2_
                yy_, yz_, zz_ = qy * y2_, qy * z2_, qz * z2_
                R_ = np.empty((ee_quat_b.shape[0], 3, 3), dtype=np.float64)
                R_[:, 0, 0] = 1.0 - (yy_ + zz_); R_[:, 0, 1] = xy_ - wz_;        R_[:, 0, 2] = xz_ + wy_
                R_[:, 1, 0] = xy_ + wz_;        R_[:, 1, 1] = 1.0 - (xx_ + zz_); R_[:, 1, 2] = yz_ - wx_
                R_[:, 2, 0] = xz_ - wy_;        R_[:, 2, 1] = yz_ + wx_;         R_[:, 2, 2] = 1.0 - (xx_ + yy_)
                tip_b = ee_xyz_b + np.einsum("nij,j->ni", R_, offset_v)
                plug_axis_b = np.einsum("nij,j->ni", R_, np.array([0.0, 0.0, 1.0]))
                cos_a = np.clip(
                    np.einsum("ni,i->n", plug_axis_b, -port_axis_v), -1.0, 1.0
                )
                align_deg_b = np.degrees(np.arccos(cos_a))
                tgt_xyz_b = obs_arrays["obj_xyz"][success_target_oid]
                depth_b = plane_z_v - np.einsum("ni,i->n", tip_b, port_axis_v)
                end_inserted = (depth_b >= min_depth_v) & (align_deg_b <= axis_tol_deg_v)
                n_success = int(end_inserted.sum())
            elif use_array_api and use_batched_reward:
                # ee_near: vectorized success check
                threshold_m = float(task.success.data.get("threshold_m", 0.05))
                ee_xyz_b = obs_arrays["ee_xyz"]
                tgt_xyz_b = obs_arrays["obj_xyz"][success_target_oid]
                dist_b = np.linalg.norm(ee_xyz_b - tgt_xyz_b, axis=1)
                n_success = int(np.sum(dist_b <= threshold_m))
            else:
                n_success = sum(
                    1
                    for w in range(N)
                    if _eval_criterion(task.success, initial_obs_all[w], obs_all[w])[0]
                )
            rate = n_success / N * 100.0
            fps = total_env_steps / (time.time() - t0)
            mean_dist_m = (iter_dist_sum / iter_dist_n) if iter_dist_n else float("nan")
            if use_insertion_reward and iter_align_n > 0:
                mean_align_deg = iter_align_deg_sum / iter_align_n
                mean_depth_mm = iter_depth_mm_sum / iter_align_n
                ever_aligned_pct = 100.0 * float(ever_aligned.mean())
                ever_inserted_pct = 100.0 * float(ever_inserted.mean())
                print(
                    f"iter {iteration:>5} | steps {total_env_steps:>10,} | "
                    f"rew {mean_rew:.3f} | tipDist {mean_dist_m*100:5.1f}cm | "
                    f"align {mean_align_deg:5.1f}° | depth {mean_depth_mm:+6.1f}mm | "
                    f"ever_align {ever_aligned_pct:5.1f}% | ever_ins {ever_inserted_pct:5.1f}% | "
                    f"end_ins {rate:5.1f}% | fps {fps:,.0f}"
                )
            else:
                ever_pct = 100.0 * float(ever_within.mean()) if ever_within.size else 0.0
                print(
                    f"iter {iteration:>5} | steps {total_env_steps:>10,} | "
                    f"rew {mean_rew:.3f} | dist {mean_dist_m*100:5.1f}cm | "
                    f"ever<thr {ever_pct:5.1f}% | end<thr {rate:5.1f}% | fps {fps:,.0f}"
                )

        if save_path is not None and iteration % checkpoint_interval == 0:
            p = NeuralPolicy(
                ac, encoder, delta_scale,
                action_space=action_space,
                ee_delta_scale=ee_delta_scale,
                ee_body=ee_body,
                ee_kine=ee_kine,
                policy_id=policy_id,
                parent_policy_id=parent_policy_id,
                lineage_op=lineage_op,
            )
            p.save(save_path)

    policy = NeuralPolicy(
        ac, encoder, delta_scale,
        action_space=action_space,
        ee_delta_scale=ee_delta_scale,
        ee_body=ee_body,
        ee_kine=ee_kine,
        policy_id=policy_id,
        parent_policy_id=parent_policy_id,
        lineage_op=lineage_op,
    )
    if save_path is not None:
        policy.save(save_path)
        print(f"[train] checkpoint → {save_path}/")

    return policy
