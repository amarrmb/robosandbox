#!/usr/bin/env python3
"""Download Franka Panda visual meshes from mujoco_menagerie.

Thin wrapper over ``robosandbox.assets.franka_visuals.cli`` so the
script is discoverable from the repo root. The CLI shortcut
``robo-sandbox download-franka-visuals`` does the same thing.

Usage:
    uv run python scripts/download_franka_visuals.py [--cache-dir PATH] [--force]
"""

from __future__ import annotations

import argparse


def main() -> int:
    from robosandbox.assets.franka_visuals import cli

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    return cli(cache_dir=args.cache_dir, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
