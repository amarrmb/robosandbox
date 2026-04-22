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
    """Wraps a trained ActorCritic as a robosandbox Policy (deterministic mean)."""

    def __init__(
        self,
        actor_critic: ActorCritic,
        encoder: ObsEncoder,
        delta_scale: float = 0.05,
    ) -> None:
        self._ac = actor_critic.eval()
        self._enc = encoder
        self._delta_scale = delta_scale
        self._n_dof = encoder.n_dof

    def act(self, obs: Observation) -> np.ndarray:
        vec = self._enc.normalize(self._enc.encode(obs))
        x = torch.from_numpy(vec).unsqueeze(0)
        with torch.no_grad():
            mean = self._ac.get_action_mean(x).squeeze(0)
        deltas = mean[: self._n_dof].numpy() * self._delta_scale
        target_q = np.asarray(obs.robot_joints, dtype=np.float64) + deltas
        gripper = float(torch.sigmoid(mean[self._n_dof]).item())
        return np.concatenate([target_q, [gripper]])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._ac.save_checkpoint(path / "actor_critic.pt")
        self._enc.save(path / "obs_config.json")
        (path / "policy.json").write_text(
            json.dumps(
                {
                    "kind": "ppo_neural",
                    "model": "actor_critic.pt",
                    "obs_config": "obs_config.json",
                    "delta_scale": self._delta_scale,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> NeuralPolicy:
        path = Path(path)
        cfg = json.loads((path / "policy.json").read_text())
        encoder = ObsEncoder.load(path / cfg["obs_config"])
        ac = ActorCritic.from_checkpoint(path / cfg["model"])
        return cls(ac, encoder, float(cfg.get("delta_scale", 0.05)))


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
) -> NeuralPolicy:
    """Train a PPO policy against a Newton multi-world sim.

    ``sim`` must expose ``n_worlds``, ``n_dof``, ``observe_all()``,
    ``step_all(targets, grippers)``, and ``reset()``.
    """
    from robosandbox.tasks.runner import _eval_criterion

    N: int = sim.n_worlds
    n_dof: int = sim.n_dof

    object_ids = [obj.id for obj in task.scene.objects]
    encoder = ObsEncoder(object_ids, n_dof=n_dof)
    obs_dim = encoder.obs_dim
    act_dim = n_dof + 1  # joint deltas + gripper logit

    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    if str(device_t) != device and "cuda" in device:
        print(f"[train] WARNING: {device} unavailable, falling back to {device_t}")

    ac = ActorCritic(obs_dim, act_dim).to(device_t)
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

    total_env_steps = 0
    iteration = 0
    t0 = time.time()

    while total_env_steps < total_steps:
        # ---- Rollout collection ------------------------------------------
        sim.reset()
        for _ in range(settle_steps):
            sim.step()

        initial_obs_all = sim.observe_all()
        obs_all = initial_obs_all

        for t in range(n_steps):
            raw_vecs = encoder.encode_batch(obs_all)           # (N, obs_dim)
            norm_vecs = encoder.normalize_batch(raw_vecs)      # (N, obs_dim) normalized

            obs_buf[t] = norm_vecs

            obs_t = torch.from_numpy(norm_vecs).to(device_t)
            with torch.no_grad():
                actions, logps, _, values = ac.get_action_and_value(obs_t)

            act_np = actions.cpu().numpy()                     # (N, act_dim)
            act_buf[t] = act_np
            logp_buf[t] = logps.cpu().numpy()
            val_buf[t] = values.squeeze(-1).cpu().numpy()

            # Convert to sim targets: delta for joints, sigmoid for gripper
            current_qs = np.stack(
                [np.asarray(o.robot_joints, dtype=np.float64) for o in obs_all]
            )                                                  # (N, n_dof)
            deltas = act_np[:, :n_dof].astype(np.float64) * delta_scale
            targets = current_qs + deltas                      # (N, n_dof)
            gripper_logits = act_np[:, n_dof]
            grippers = 1.0 / (1.0 + np.exp(-np.clip(gripper_logits, -20, 20)))

            sim.step_all(targets, grippers)
            obs_all = sim.observe_all()

            # Shaped reward per world
            for w in range(N):
                rew_buf[t, w] = compute_shaped_reward(
                    task.success, initial_obs_all[w], obs_all[w]
                )

            encoder.update_stats_batch(raw_vecs)

        # Bootstrap value
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

        total_env_steps += n_steps * N
        iteration += 1

        if iteration % log_interval == 0:
            mean_rew = float(rew_buf.mean())
            n_success = sum(
                1
                for w in range(N)
                if _eval_criterion(task.success, initial_obs_all[w], obs_all[w])[0]
            )
            rate = n_success / N * 100.0
            fps = total_env_steps / (time.time() - t0)
            print(
                f"iter {iteration:>5} | steps {total_env_steps:>10,} | "
                f"rew {mean_rew:.3f} | success {rate:.1f}% | fps {fps:,.0f}"
            )

        if save_path is not None and iteration % checkpoint_interval == 0:
            p = NeuralPolicy(ac, encoder, delta_scale)
            p.save(save_path)

    policy = NeuralPolicy(ac, encoder, delta_scale)
    if save_path is not None:
        policy.save(save_path)
        print(f"[train] checkpoint → {save_path}/")

    return policy
