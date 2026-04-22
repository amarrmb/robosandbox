"""Newton cloth showcase for RoboSandbox demos — standalone runner.

The ClothFrankaShowcase class lives in robosandbox.sim.cloth_showcase so
both this script and the newton_cloth backend can share the same simulation.
"""

from __future__ import annotations

import argparse

import warp as wp

import newton.examples
from newton.viewer import ViewerNull, ViewerViser

from robosandbox.sim.cloth_showcase import ClothFrankaShowcase


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--viewer", choices=("viser", "null"), default="viser")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--world-count", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=200000)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.quiet:
        wp.config.quiet = True
    if args.device:
        wp.set_device(args.device)

    if args.viewer == "viser":
        viewer = ViewerViser(port=args.port)
    else:
        viewer = ViewerNull(num_frames=args.num_frames)

    example = ClothFrankaShowcase(viewer=viewer, world_count=args.world_count)
    newton.examples.run(example, argparse.Namespace(test=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
