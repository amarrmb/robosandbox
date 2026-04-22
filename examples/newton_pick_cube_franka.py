"""Run the bundled pick_cube_franka task through RoboSandbox's Newton backend."""

from __future__ import annotations

import argparse
import time

from robosandbox.sim import create_sim_backend
from robosandbox.tasks.loader import load_builtin_task


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", choices=("viser", "null"), default="viser")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hold", action="store_true")
    args = parser.parse_args()

    task = load_builtin_task("pick_cube_franka")
    sim = create_sim_backend(
        "newton",
        render_size=(480, 640),
        viewer=args.viewer,
        port=args.port,
        device=args.device,
    )
    sim.load(task.scene)
    try:
        for _ in range(args.steps):
            sim.step()
        obs = sim.observe()
        cube = obs.scene_objects["red_cube"]
        print("scene=pick_cube_franka")
        print(f"cube_final_xyz=({cube.xyz[0]:.4f}, {cube.xyz[1]:.4f}, {cube.xyz[2]:.4f})")
        print(f"viewer={args.viewer}")
        print(f"steps={args.steps}")
        if args.viewer == "viser":
            print(f"viewer_url=http://127.0.0.1:{args.port}")
            if args.hold:
                try:
                    while True:
                        time.sleep(1.0)
                except KeyboardInterrupt:
                    pass
    finally:
        sim.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
