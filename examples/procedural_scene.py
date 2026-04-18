"""Build a procedurally randomized tabletop scene via ScenePresets.

``tabletop_clutter(n_objects, seed)`` returns a Scene with the bundled
Franka + N YCB distractors at non-overlapping poses. Each seed is a
distinct, reproducible layout; give the runner a range of seeds to
generate a randomized benchmark.

Run:
    uv run python examples/procedural_scene.py
    uv run python examples/procedural_scene.py --n 7 --seed 3 --out examples/scene.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio

from robosandbox.scene.presets import tabletop_clutter
from robosandbox.sim.mujoco_backend import MuJoCoBackend


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=5, help="Number of objects (default 5)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="If set, save one frame here")
    args = ap.parse_args()

    scene = tabletop_clutter(n_objects=args.n, seed=args.seed)
    print(f"tabletop_clutter(n={args.n}, seed={args.seed}):")
    for obj in scene.objects:
        x, y, z = obj.pose.xyz
        print(f"  {obj.id:<20}  at ({x:.3f}, {y:.3f}, {z:.3f})")

    sim = MuJoCoBackend(render_size=(480, 640), camera="scene")
    sim.load(scene)
    try:
        for _ in range(140):
            sim.step()  # let them settle
        obs = sim.observe()
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            iio.imwrite(args.out, obs.rgb)
            print(f"\nwrote {args.out}")
        print("\nsettled poses:")
        for obj in scene.objects:
            settled = obs.scene_objects[obj.id].xyz
            print(f"  {obj.id:<20}  at ({settled[0]:.3f}, {settled[1]:.3f}, {settled[2]:.3f})")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
