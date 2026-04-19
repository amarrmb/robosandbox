"""Migrate pre-0.5 LeRobot checkpoint configs to the current schema.

Public SO-100 ACT checkpoints on the Hub were mostly trained with
``lerobot < 0.5`` and have a config.json shaped like::

    {
      "input_shapes":             {"observation.state": [7], "observation.images.laptop": [3,480,640], ...},
      "input_normalization_modes":{"observation.state": "mean_std", ...},
      "output_shapes":            {"action": [7]},
      "output_normalization_modes": {"action": "mean_std"},
      ...ACT hparams...
    }

``lerobot >= 0.5`` expects::

    {
      "type": "act",
      "normalization_mapping": {"VISUAL": "MEAN_STD", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
      "input_features":  {name: {"type": "VISUAL"|"STATE", "shape": [...]}},
      "output_features": {name: {"type": "ACTION",         "shape": [...]}},
      ...ACT hparams (unchanged)...
    }

This utility rewrites the file in place. It's a mechanical migration —
if your checkpoint uses a non-mean_std normalization or has other
custom fields, hand-edit the result.

Usage (on a snapshot downloaded via huggingface_hub)::

    uv run python examples/so_arm100/migrate_lerobot_config.py /path/to/ckpt/config.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_DROP_KEYS = (
    "input_shapes",
    "output_shapes",
    "input_normalization_modes",
    "output_normalization_modes",
)


def migrate_config(old: dict) -> dict:
    """Return a new config dict in lerobot-0.5 schema."""
    new = {k: v for k, v in old.items() if k not in _DROP_KEYS and k != "type"}
    new["type"] = "act"
    # Coarse default — lerobot 0.5 expects a normalization_mapping per
    # feature-type. Override manually if your checkpoint doesn't use
    # mean/std on every input.
    new["normalization_mapping"] = {
        "VISUAL": "MEAN_STD",
        "STATE": "MEAN_STD",
        "ACTION": "MEAN_STD",
    }
    new["input_features"] = {
        name: {
            "type": "VISUAL" if "image" in name.lower() else "STATE",
            "shape": shape,
        }
        for name, shape in old.get("input_shapes", {}).items()
    }
    new["output_features"] = {
        name: {"type": "ACTION", "shape": shape}
        for name, shape in old.get("output_shapes", {}).items()
    }
    return new


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config_json", type=Path, help="config.json to rewrite in place")
    ap.add_argument("--backup", action="store_true", help="Write <name>.old as a backup first")
    args = ap.parse_args()

    path: Path = args.config_json
    old = json.loads(path.read_text())
    if "input_features" in old and "type" in old:
        print(f"{path}: already in current schema, nothing to do.", file=sys.stderr)
        return 0
    if "input_shapes" not in old:
        print(f"{path}: does not look like a legacy config (no `input_shapes`).", file=sys.stderr)
        return 2

    if args.backup:
        backup = path.with_suffix(".json.old")
        backup.write_text(json.dumps(old, indent=2))
        print(f"wrote backup: {backup}")

    new = migrate_config(old)
    path.write_text(json.dumps(new, indent=2))
    print(f"migrated: {path}")
    print(f"  input_features:  {list(new['input_features'])}")
    print(f"  output_features: {list(new['output_features'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
