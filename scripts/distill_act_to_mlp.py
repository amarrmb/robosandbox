"""Distill an ACT (image+state, action chunks) into a state-only MLP that
matches PPO's ActorCritic so it can serve as the warm-start for RL.

Why this exists
---------------
- ACT is image+state with action-chunk outputs; PPO's actor is a tiny state-only
  MLP that emits a per-step joint *delta* + gripper logit.
- Their architectures and inputs don't overlap, so weight transfer is impossible.
- Distillation gives PPO a behavioural prior shaped like ACT's policy, expressed
  in PPO's own action parameterization.

What it does
------------
1. Roll out ACT in MuJoCo across N randomized scenes.
2. At each step record (encoded_obs, joint_delta, gripper_logit) where:
     joint_delta   = (act_target_joints - current_joints) / delta_scale
     gripper_logit = logit(act_gripper)
3. Train an ActorCritic (same hidden dims as PPO defaults) by MSE on the
   actor_mean head only — actor_logstd and the critic stay at their init.
4. Save in ActorCritic.from_checkpoint format + the matching ObsEncoder
   (with stats already populated) so PPO can load both directly.

Output layout
-------------
    <out>/actor_critic.pt           (ActorCritic.save_checkpoint)
    <out>/obs_config.json           (ObsEncoder.save)
    <out>/distill_meta.json         (provenance: act ckpt, n_episodes, etc.)

PPO loads this via --warm-start <out>.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "robosandbox-core" / "src"))


def _logit(p: float, eps: float = 1e-4) -> float:
    """sigmoid^-1, clamped away from {0,1} so we don't return ±inf."""
    p = float(np.clip(p, eps, 1.0 - eps))
    return float(np.log(p / (1.0 - p)))


def collect_act_rollouts(
    act_policy_dir: Path,
    task_name: str,
    n_episodes: int,
    max_steps: int,
    settle_steps: int,
    action_repeat: int,
    delta_scale: float,
    seed_start: int,
) -> tuple[list, list, list]:
    """Returns (obs_list, delta_targets, gripper_logits) collected across episodes.

    obs_list:        list of Observation
    delta_targets:   list of (n_dof,) float arrays — what actor_mean[:n_dof] should predict
    gripper_logits:  list of floats — what actor_mean[n_dof] should predict
    """
    from robosandbox.policy import load_policy
    from robosandbox.sim.mujoco_backend import MuJoCoBackend
    from robosandbox.tasks.loader import load_builtin_task
    from robosandbox.tasks.randomize import jitter_scene

    task = load_builtin_task(task_name)
    if not task.randomize:
        raise RuntimeError(f"task {task_name!r} must have a randomize spec")

    obs_records: list = []
    delta_records: list = []
    grip_records: list = []

    for ep_idx in range(n_episodes):
        seed = seed_start + ep_idx
        scene = jitter_scene(task.scene, task.randomize, seed=seed)
        sim = MuJoCoBackend(render_size=(240, 320), camera="scene")
        sim.load(scene)
        # Reload ACT per episode: chunked policies (ACT, diffusion) keep an
        # internal action queue + RNG state that must reset between trials.
        policy = load_policy(act_policy_dir)
        try:
            for _ in range(settle_steps):
                sim.step()
            held_action: np.ndarray | None = None
            n_dof = sim.n_dof
            for step_i in range(max_steps):
                obs = sim.observe()
                if step_i % action_repeat == 0:
                    act = np.asarray(policy.act(obs), dtype=np.float64).ravel()
                    held_action = act
                else:
                    act = held_action
                target_q = act[:n_dof]
                gripper = float(np.clip(act[n_dof], 0.0, 1.0))
                # Distillation target: what would PPO's actor_mean need to
                # output so NeuralPolicy.act() reproduces this command?
                #   target_q = current_q + delta * delta_scale
                #   gripper  = sigmoid(logit)
                delta = (target_q - np.asarray(obs.robot_joints, dtype=np.float64)) / delta_scale
                obs_records.append(obs)
                delta_records.append(delta.astype(np.float32))
                grip_records.append(_logit(gripper))
                sim.step(target_joints=target_q, gripper=gripper)
        finally:
            sim.close()
        if (ep_idx + 1) % 5 == 0 or ep_idx == n_episodes - 1:
            print(
                f"[distill] rolled out {ep_idx + 1}/{n_episodes} episodes "
                f"({len(obs_records):,} samples)"
            )
    return obs_records, delta_records, grip_records


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--act-policy", required=True, type=Path,
                   help="Path to a LeRobot ACT checkpoint dir")
    p.add_argument("--task", default="pick_cube_franka_random")
    p.add_argument("--n-episodes", type=int, default=50,
                   help="Number of randomized rollouts to collect")
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--settle-steps", type=int, default=100)
    p.add_argument("--action-repeat", type=int, default=6,
                   help="Match ACT's training cadence (200Hz sim / 30fps data = 6)")
    p.add_argument("--delta-scale", type=float, default=0.05,
                   help="Must match PPO's --delta-scale")
    p.add_argument("--seed-start", type=int, default=1)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    p.add_argument("--out", required=True, type=Path,
                   help="Output dir for the distilled actor_critic + encoder")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[distill] act_policy:  {args.act_policy}")
    print(f"[distill] task:        {args.task}")
    print(f"[distill] n_episodes:  {args.n_episodes}")
    print(f"[distill] action_rep:  {args.action_repeat}")
    print(f"[distill] delta_scale: {args.delta_scale}")
    print(f"[distill] hidden:      {args.hidden}")
    print(f"[distill] epochs:      {args.epochs}")
    print(f"[distill] out:         {args.out}")

    t0 = time.time()
    obs_records, delta_records, grip_records = collect_act_rollouts(
        act_policy_dir=args.act_policy,
        task_name=args.task,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        settle_steps=args.settle_steps,
        action_repeat=args.action_repeat,
        delta_scale=args.delta_scale,
        seed_start=args.seed_start,
    )
    rollout_wall = time.time() - t0
    print(f"[distill] collected {len(obs_records):,} samples in {rollout_wall:.0f}s")

    # ---- Build encoder + populate normalization stats ----------------------
    from robosandbox.rl.obs_encoder import ObsEncoder
    from robosandbox.rl.ppo import ActorCritic
    from robosandbox.tasks.loader import load_builtin_task
    import torch

    task = load_builtin_task(args.task)
    object_ids = [o.id for o in task.scene.objects]
    n_dof = int(np.asarray(obs_records[0].robot_joints).shape[0])
    encoder = ObsEncoder(object_ids, n_dof=n_dof)
    raw = np.array([encoder.encode(o) for o in obs_records], dtype=np.float64)
    encoder.update_stats_batch(raw)
    norm = encoder.normalize_batch(raw)

    # ---- Build target tensor ----------------------------------------------
    deltas = np.stack(delta_records, axis=0).astype(np.float32)        # (T, n_dof)
    grippers = np.array(grip_records, dtype=np.float32).reshape(-1, 1)  # (T, 1)
    targets = np.concatenate([deltas, grippers], axis=1)               # (T, n_dof+1)

    # ---- Train MLP via supervised MSE on actor_mean head -------------------
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if str(device) != args.device and "cuda" in args.device:
        print(f"[distill] WARNING: {args.device} unavailable, using {device}")

    obs_dim = encoder.obs_dim
    act_dim = n_dof + 1
    ac = ActorCritic(obs_dim, act_dim, hidden=tuple(args.hidden)).to(device)
    optimizer = torch.optim.Adam(ac.parameters(), lr=args.lr)

    X = torch.from_numpy(norm).float().to(device)
    Y = torch.from_numpy(targets).float().to(device)
    n = X.shape[0]

    print(f"[distill] training MLP: obs_dim={obs_dim} act_dim={act_dim} samples={n:,}")
    losses: list[float] = []
    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, args.batch_size):
            idx = perm[i : i + args.batch_size]
            xb, yb = X[idx], Y[idx]
            pred = ac.get_action_mean(xb)
            loss = torch.nn.functional.mse_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        avg = epoch_loss / max(n_batches, 1)
        losses.append(avg)
        if (epoch + 1) % max(args.epochs // 20, 1) == 0:
            print(f"[distill] epoch {epoch + 1:3d}/{args.epochs}  loss={avg:.6f}")

    # ---- Save in PPO-compatible layout ------------------------------------
    ac.cpu().save_checkpoint(args.out / "actor_critic.pt")
    encoder.save(args.out / "obs_config.json")
    # Policy wrapper so `robo-sandbox eval --policy <out>` works directly
    # via load_policy → NeuralPolicy.load. Mirrors NeuralPolicy.save.
    (args.out / "policy.json").write_text(json.dumps(
        {
            "kind": "ppo_neural",
            "model": "actor_critic.pt",
            "obs_config": "obs_config.json",
            "delta_scale": args.delta_scale,
        },
        indent=2,
    ))
    (args.out / "distill_meta.json").write_text(json.dumps(
        {
            "kind": "distilled_act",
            "act_policy": str(args.act_policy),
            "task": args.task,
            "n_episodes": args.n_episodes,
            "samples": int(n),
            "epochs": args.epochs,
            "final_loss": losses[-1] if losses else None,
            "delta_scale": args.delta_scale,
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "hidden": list(args.hidden),
            "rollout_wall_s": rollout_wall,
        },
        indent=2,
    ))

    print()
    print(f"[distill] DONE — wrote {args.out}/actor_critic.pt + obs_config.json")
    print(f"[distill] final loss: {losses[-1]:.6f}")
    print()
    print(f"[distill] Next: warm-start PPO in Newton:")
    print(
        f"  robo-sandbox train --task {args.task} \\\n"
        f"    --warm-start {args.out} \\\n"
        f"    --sim-backend newton --world-count 256 \\\n"
        f"    --total-steps 500000 \\\n"
        f"    --output outputs/act_ppo_finetuned"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
