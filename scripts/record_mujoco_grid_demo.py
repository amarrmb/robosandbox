"""Render an N-world grid video from sequential MuJoCo single-world rollouts.

MuJoCoBackend is single-world by design (per CLAUDE.md). This script runs
N independent rollouts sequentially with per-rollout cube-pose randomization,
captures per-step RGB for each, then stitches them into a grid video so the
visual matches what record_newton_parallel_demo.py produces — but sourced
from MuJoCo where the policy actually picks the cube.

Why bother: Newton's mujoco_warp solver allows ~3.4 mm contact penetration
which kills the grasp at scale. For "policy works" visual proof we need
MuJoCo. This recorder is for that demo only.
"""
from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import numpy as np

from robosandbox.policy import load_policy
from robosandbox.sim.mujoco_backend import MuJoCoBackend
from robosandbox.tasks.loader import load_builtin_task
from robosandbox.tasks.randomize import jitter_scene


def grid_shape(n: int) -> tuple[int, int]:
    presets = {1: (1, 1), 2: (2, 1), 4: (2, 2), 6: (3, 2), 8: (4, 2),
               9: (3, 3), 12: (4, 3), 16: (4, 4), 24: (6, 4), 32: (8, 4)}
    if n in presets:
        return presets[n]
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    return cols, rows


def run_one_episode(
    sim: MuJoCoBackend, scene, policy, n_dof: int, max_steps: int,
    action_repeat: int, settle_steps: int, success_check,
) -> tuple[np.ndarray, bool, int]:
    """Run one rollout. Return (frames (T,H,W,3) uint8, success, success_step)."""
    sim.load(scene)
    for _ in range(settle_steps):
        sim.step()
    initial_obs = sim.observe()
    frames: list[np.ndarray] = []
    success = False
    success_step = -1
    held: np.ndarray | None = None
    for step_i in range(max_steps):
        obs = sim.observe()
        frames.append(obs.rgb.copy())
        if step_i % action_repeat == 0:
            held = np.asarray(policy.act(obs), dtype=np.float64).ravel()
        sim.step(target_joints=held[:n_dof], gripper=float(held[n_dof]))
        if not success and success_check is not None:
            from robosandbox.tasks.runner import _eval_criterion
            ok, _ = _eval_criterion(success_check, initial_obs, sim.observe())
            if ok:
                success = True
                success_step = step_i
    return np.stack(frames, axis=0), success, success_step


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="pick_cube_franka_random")
    p.add_argument("--policy", required=True, help="Path to policy dir")
    p.add_argument("--world-count", type=int, default=24)
    p.add_argument("--max-steps", type=int, default=480)
    p.add_argument("--settle-steps", type=int, default=50)
    p.add_argument("--action-repeat", type=int, default=6)
    p.add_argument("--frame-size", type=int, nargs=2, default=[240, 320],
                   metavar=("H", "W"))
    p.add_argument("--video-size", type=int, nargs=2, default=[1080, 1920],
                   metavar=("H", "W"))
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    cols, rows = grid_shape(args.world_count)
    pw_h, pw_w = args.frame_size
    out_h, out_w = args.video_size
    grid_h, grid_w = pw_h * rows, pw_w * cols
    pad_top = (out_h - grid_h) // 2
    pad_left = (out_w - grid_w) // 2

    task = load_builtin_task(args.task)
    if not task.randomize:
        raise SystemExit(f"task {args.task} has no randomize spec")

    policy = load_policy(Path(args.policy))
    sim = MuJoCoBackend(render_size=(pw_h, pw_w))
    n_dof = 7  # Franka

    print(f"[mj-rec] task={args.task} world_count={args.world_count} grid={cols}x{rows}")
    print(f"[mj-rec] running {args.world_count} sequential MuJoCo episodes...")

    all_frames: list[np.ndarray] = []
    all_success: list[bool] = []
    all_success_step: list[int] = []
    for w in range(args.world_count):
        scene = jitter_scene(task.scene, task.randomize, seed=args.seed + w + 1)
        frames, success, success_step = run_one_episode(
            sim, scene, policy, n_dof, args.max_steps,
            args.action_repeat, args.settle_steps, task.success,
        )
        all_frames.append(frames)
        all_success.append(success)
        all_success_step.append(success_step)
        print(f"[mj-rec]   ep {w+1}/{args.world_count}: "
              f"{'SUCCESS' if success else 'fail'} "
              f"({'step '+str(success_step) if success else f'after {args.max_steps} steps'})")

    n_success = sum(all_success)
    print(f"[mj-rec] {n_success}/{args.world_count} successes "
          f"({100.0*n_success/args.world_count:.1f}%)")

    # Build grid video
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
    print(f"[mj-rec] ffmpeg → {args.out}")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    frame = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    T = args.max_steps
    try:
        for t in range(T):
            for w in range(args.world_count):
                r = w // cols
                c = w % cols
                frame[
                    pad_top + r * pw_h : pad_top + (r + 1) * pw_h,
                    pad_left + c * pw_w : pad_left + (c + 1) * pw_w,
                ] = all_frames[w][t]
                # Green border once success has been reached (latched)
                if all_success[w] and t >= all_success_step[w]:
                    y0 = pad_top + r * pw_h
                    y1 = pad_top + (r + 1) * pw_h
                    x0 = pad_left + c * pw_w
                    x1 = pad_left + (c + 1) * pw_w
                    frame[y0:y0+4, x0:x1] = [40, 220, 40]
                    frame[y1-4:y1, x0:x1] = [40, 220, 40]
                    frame[y0:y1, x0:x0+4] = [40, 220, 40]
                    frame[y0:y1, x1-4:x1] = [40, 220, 40]
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
        proc.wait()
        sim.close()

    print(f"[mj-rec] DONE — wrote {args.out} ({n_success}/{args.world_count} green-bordered)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
