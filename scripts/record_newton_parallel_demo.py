"""Record a tiled-grid video of N Newton worlds executing in parallel.

For each frame, calls sim.observe_all() (RGB raytraced via SensorTiledCamera),
arranges the per-world frames into a grid, writes to ffmpeg as raw RGB. Uses
a replay policy so all worlds run the same scripted joint trajectory but
each world's cube starts at a different randomized pose, so picks and
misses are visible per tile.

Output is Twitter-friendly: 1920x1080 H.264 yuv420p +faststart.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "robosandbox-core" / "src"))

from robosandbox.policy import load_policy
from robosandbox.sim.newton_backend import NewtonBackend
from robosandbox.tasks.loader import load_builtin_task
from robosandbox.tasks.randomize import jitter_scene


def grid_shape(n: int) -> tuple[int, int]:
    """Pick (cols, rows) for n worlds — prefer wider-than-tall (16:9-ish)."""
    presets = {
        4: (2, 2),
        6: (3, 2),
        8: (4, 2),
        9: (3, 3),
        12: (4, 3),
        16: (4, 4),
        18: (6, 3),
        24: (6, 4),
        32: (8, 4),
    }
    if n in presets:
        return presets[n]
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    return cols, rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="pick_cube_franka_random")
    p.add_argument("--policy", required=True, help="Path to replay policy dir (events.jsonl)")
    p.add_argument("--world-count", type=int, default=24)
    p.add_argument("--max-steps", type=int, default=720)
    p.add_argument("--settle-steps", type=int, default=100)
    p.add_argument("--action-repeat", type=int, default=8,
                   help="Hold each policy action for N sim steps. 8 matches "
                        "30 fps recordings replayed in Newton's 240 Hz sim.")
    p.add_argument("--frame-size", type=int, nargs=2, default=[240, 320],
                   metavar=("H", "W"), help="Per-world frame H W")
    p.add_argument("--video-size", type=int, nargs=2, default=[1080, 1920],
                   metavar=("H", "W"), help="Output video H W (Twitter 1080p default)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=1, help="Per-world randomization seed base")
    p.add_argument("--no-randomize", action="store_true",
                   help="Run all worlds with identical scenes (matches the demo). "
                        "Useful when the replay policy was recorded against a "
                        "single fixed cube pose — every world will succeed.")
    p.add_argument("--out", type=Path, default=Path("/tmp/newton_parallel_demo.mp4"))
    args = p.parse_args()

    cols, rows = grid_shape(args.world_count)
    pw_h, pw_w = args.frame_size
    grid_h, grid_w = pw_h * rows, pw_w * cols
    out_h, out_w = args.video_size
    print(f"[rec] task={args.task} world_count={args.world_count} grid={cols}x{rows}")
    print(f"[rec] per-world {pw_h}x{pw_w} → grid {grid_h}x{grid_w} → output {out_h}x{out_w} @ {args.fps}fps")

    task = load_builtin_task(args.task)
    if not task.randomize:
        print(f"[rec] WARNING: task {args.task} has no randomize spec — all worlds identical")
    per_world_scenes = (
        [jitter_scene(task.scene, task.randomize, seed=args.seed + w + 1)
         for w in range(args.world_count)]
        if task.randomize and not args.no_randomize else None
    )
    if args.no_randomize:
        print("[rec] --no-randomize: all worlds share the demo's scene")

    sim = NewtonBackend(
        world_count=args.world_count,
        enable_camera=True,
        render_size=(pw_h, pw_w),
    )
    sim.load(task.scene, per_world_scenes=per_world_scenes)

    for _ in range(args.settle_steps):
        sim.step()

    policy = load_policy(Path(args.policy))
    n_dof = sim.n_dof

    # Pre-compute padding to center grid in the output frame.
    pad_top = (out_h - grid_h) // 2
    pad_bot = out_h - grid_h - pad_top
    pad_left = (out_w - grid_w) // 2
    pad_right = out_w - grid_w - pad_left
    if pad_top < 0 or pad_left < 0:
        raise SystemExit(
            f"grid {grid_h}x{grid_w} doesn't fit in {out_h}x{out_w}; "
            f"reduce --world-count or --frame-size"
        )

    # ffmpeg writer: raw rgb24 in, h264 yuv420p out, +faststart for Twitter.
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
    print(f"[rec] ffmpeg: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    frame = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    success_step: dict[int, int] = {}

    try:
        for step_i in range(args.max_steps):
            obs_list = sim.observe_all()
            # Arrange per-world frames into grid
            for w, obs in enumerate(obs_list):
                r = w // cols
                c = w % cols
                frame[
                    pad_top + r * pw_h : pad_top + (r + 1) * pw_h,
                    pad_left + c * pw_w : pad_left + (c + 1) * pw_w,
                ] = obs.rgb
                # Optional: tint border green if cube has been lifted
                if w not in success_step:
                    cube = obs.scene_objects.get("red_cube")
                    if cube is not None and cube.xyz[2] > task.scene.table_height + 0.05:
                        success_step[w] = step_i
                if w in success_step:
                    # 4-pixel green frame
                    y0 = pad_top + r * pw_h
                    y1 = pad_top + (r + 1) * pw_h
                    x0 = pad_left + c * pw_w
                    x1 = pad_left + (c + 1) * pw_w
                    frame[y0:y0+4, x0:x1] = [40, 220, 40]
                    frame[y1-4:y1, x0:x1] = [40, 220, 40]
                    frame[y0:y1, x0:x0+4] = [40, 220, 40]
                    frame[y0:y1, x1-4:x1] = [40, 220, 40]
            proc.stdin.write(frame.tobytes())

            # Step the sim. Replay a new policy action every action_repeat
            # steps; hold the previous action otherwise. Each world gets its
            # own action driven by its own observation — that's the whole
            # point of "parallel eval at scale". Broadcasting world-0's
            # action would defeat the randomization.
            if step_i % args.action_repeat == 0:
                actions = np.stack(
                    [np.asarray(policy.act(o), dtype=np.float64).ravel()
                     for o in obs_list],
                    axis=0,
                )  # (W, n_dof + 1)
            sim.step_all(actions[:, :n_dof], actions[:, n_dof])

            if (step_i + 1) % 50 == 0:
                print(f"[rec] step {step_i + 1}/{args.max_steps}  successes so far: {len(success_step)}/{args.world_count}")
    finally:
        proc.stdin.close()
        proc.wait()
        sim.close()

    print(f"[rec] DONE — {len(success_step)}/{args.world_count} worlds picked the cube")
    print(f"[rec] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
