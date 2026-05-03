"""Record 1024 (or N) parallel Newton worlds running a trained reach policy.

Each world has a randomized target marker. Each world is driven by its
own observation (per-world actions via sim.step_all).

Renders to a grid video with green borders around worlds that have
reached the target during the rollout. Used for Beat 2 of the
build-in-public campaign.

Usage:
    /home/amar/newton/.venv/bin/python scripts/record_reach_rollout.py \\
        --policy outputs/reach_ppo_1024_v2 \\
        --world-count 256 \\
        --max-steps 200 \\
        --out outputs/beat2/reach_256w_trained.mp4
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

sys.path.insert(
    0, str(Path(__file__).parent.parent / "packages" / "robosandbox-core" / "src")
)

import numpy as np

from robosandbox.policy import load_policy
from robosandbox.sim.newton_backend import NewtonBackend
from robosandbox.tasks.loader import load_builtin_task
from robosandbox.tasks.randomize import jitter_scene
from robosandbox.tasks.runner import _eval_criterion


def grid_shape(n: int) -> tuple[int, int]:
    presets = {
        24: (6, 4), 32: (8, 4), 64: (8, 8), 128: (16, 8),
        144: (12, 12), 256: (16, 16), 400: (20, 20), 1024: (32, 32),
    }
    if n in presets:
        return presets[n]
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    return cols, rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="reach_target_franka")
    p.add_argument("--policy", required=True)
    p.add_argument("--world-count", type=int, default=256)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--settle-steps", type=int, default=50)
    p.add_argument("--frame-size", type=int, nargs=2, default=[120, 160],
                   metavar=("H", "W"))
    p.add_argument("--video-size", type=int, nargs=2, default=[1080, 1920],
                   metavar=("H", "W"))
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Explicit per-world seed list (overrides --seed). "
                        "Length must match --world-count. Use to hand-pick "
                        "well-separated port positions for zoom hero shots.")
    p.add_argument("--clearance-m", type=float, default=None,
                   help="Rebuild port assembly at this clearance (insert task)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    cols, rows = grid_shape(args.world_count)
    pw_h, pw_w = args.frame_size
    grid_h, grid_w = pw_h * rows, pw_w * cols
    out_h, out_w = args.video_size
    pad_top = (out_h - grid_h) // 2
    pad_left = (out_w - grid_w) // 2
    if pad_top < 0 or pad_left < 0:
        raise SystemExit(
            f"grid {grid_h}x{grid_w} doesn't fit in {out_h}x{out_w}; "
            f"reduce --world-count or --frame-size or increase --video-size"
        )

    print(f"[rec] task={args.task} world_count={args.world_count} grid={cols}x{rows}")
    print(f"[rec] per-world {pw_h}x{pw_w} → grid {grid_h}x{grid_w} → output {out_h}x{out_w}")

    task = load_builtin_task(args.task)
    if args.clearance_m is not None:
        from robosandbox.tasks.insertion_geometry import rescale_port_clearance
        task.scene = rescale_port_clearance(task.scene, args.clearance_m)
        print(f"[rec] rebuilt port at clearance={args.clearance_m*1000:.2f} mm")
    # Some tasks (e.g. insert_connector_franka in Phase 1) have no randomize
    # block — keep all worlds at the canonical scene pose.
    if task.randomize and (
        float(task.randomize.get("xy_jitter", 0.0)) > 0.0
        or float(task.randomize.get("yaw_jitter", 0.0)) > 0.0
        or float(task.randomize.get("group_xy_jitter", 0.0)) > 0.0
        or float(task.randomize.get("group_yaw_jitter", 0.0)) > 0.0
    ):
        if args.seeds is not None:
            if len(args.seeds) != args.world_count:
                raise SystemExit(
                    f"--seeds has {len(args.seeds)} values but --world-count is "
                    f"{args.world_count}; lengths must match"
                )
            seeds = list(args.seeds)
        else:
            seeds = [args.seed + w + 1 for w in range(args.world_count)]
        print(f"[rec] per-world seeds: {seeds}")
        per_world_scenes = [
            jitter_scene(task.scene, task.randomize, seed=s)
            for s in seeds
        ]
    else:
        per_world_scenes = None

    sim = NewtonBackend(
        world_count=args.world_count,
        enable_camera=True,
        render_size=(pw_h, pw_w),
    )
    if per_world_scenes is not None:
        sim.load(task.scene, per_world_scenes=per_world_scenes)
    else:
        sim.load(task.scene)
    for _ in range(args.settle_steps):
        sim.step()

    # If checkpoint is an ee_xyz NeuralPolicy, build the IK helper using the
    # task scene + robot's arm joint names. Falls back to load_policy for
    # joint-space / replay-trajectory policies.
    import json as _json
    policy_dir = Path(args.policy)
    cfg_path = policy_dir / "policy.json"
    if cfg_path.exists():
        cfg = _json.loads(cfg_path.read_text())
        if cfg.get("kind") == "ppo_neural" and cfg.get("action_space") in ("ee_xyz", "ee_xyz_rpy"):
            from robosandbox.rl.ppo import NeuralPolicy
            arm_names = list(getattr(sim, "_arm_joint_names", []) or sim.joint_names)
            policy = NeuralPolicy.load(policy_dir, scene=task.scene, arm_joint_names=arm_names)
            print(f"[rec] loaded EE-space NeuralPolicy (action_space={cfg.get('action_space')})")
        else:
            policy = load_policy(policy_dir, scene=task.scene)
    else:
        policy = load_policy(policy_dir, scene=task.scene)
    n_dof = sim.n_dof

    # Capture initial obs for per-world success eval (ee_near uses final-only,
    # but we keep the API uniform)
    initial_obs_all = sim.observe_all()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}", "-pix_fmt", "rgb24", "-r", str(args.fps),
        "-i", "-",
        "-vcodec", "libx264", "-profile:v", "main", "-pix_fmt", "yuv420p",
        "-preset", "medium", "-crf", "20",
        "-movflags", "+faststart",
        "-r", str(args.fps),
        str(args.out),
    ]
    print(f"[rec] ffmpeg → {args.out}")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    frame = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    success_step: dict[int, int] = {}

    try:
        for step_i in range(args.max_steps):
            obs_list = sim.observe_all()
            actions = np.stack(
                [np.asarray(policy.act(o), dtype=np.float64).ravel()
                 for o in obs_list],
                axis=0,
            )
            sim.step_all(actions[:, :n_dof], actions[:, n_dof])

            # Render frame
            for w, obs in enumerate(obs_list):
                r = w // cols
                c = w % cols
                y0, y1 = pad_top + r * pw_h, pad_top + (r + 1) * pw_h
                x0, x1 = pad_left + c * pw_w, pad_left + (c + 1) * pw_w
                frame[y0:y1, x0:x1] = obs.rgb

                # Success check per world (ee_near criterion)
                if w not in success_step:
                    ok, _ = _eval_criterion(task.success, initial_obs_all[w], obs)
                    if ok:
                        success_step[w] = step_i

                # Once a world succeeds, draw a green border on every subsequent frame
                if w in success_step:
                    frame[y0:y0+2, x0:x1] = [40, 220, 40]
                    frame[y1-2:y1, x0:x1] = [40, 220, 40]
                    frame[y0:y1, x0:x0+2] = [40, 220, 40]
                    frame[y0:y1, x1-2:x1] = [40, 220, 40]

            proc.stdin.write(frame.tobytes())

            if (step_i + 1) % 50 == 0:
                print(f"[rec] step {step_i + 1}/{args.max_steps}  "
                      f"successes so far: {len(success_step)}/{args.world_count}")
    finally:
        proc.stdin.close()
        proc.wait()
        sim.close()

    print(f"[rec] DONE — {len(success_step)}/{args.world_count} worlds reached the target "
          f"({100.0*len(success_step)/args.world_count:.1f}%)")
    print(f"[rec] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
