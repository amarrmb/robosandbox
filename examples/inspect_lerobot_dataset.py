"""Inspect a LeRobot v3 dataset RoboSandbox exported.

Loads the parquet frame table + metadata JSON files and prints a
human-readable summary: episode/frame counts, state/action dims,
sample rows. Useful as a smoke check that the export is well-formed.

Usage:
    uv run python examples/inspect_lerobot_dataset.py datasets/pick_demo
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path, help="Root of exported LeRobot dataset")
    p.add_argument("--sample-rows", type=int, default=2, help="Rows to preview")
    args = p.parse_args()

    root: Path = args.dataset
    meta = root / "meta"
    info = json.loads((meta / "info.json").read_text())
    tasks = [json.loads(line) for line in (meta / "tasks.jsonl").read_text().splitlines() if line]
    episodes = [json.loads(line) for line in (meta / "episodes.jsonl").read_text().splitlines() if line]

    print(f"Dataset:        {root}")
    print(f"LeRobot:        {info.get('codebase_version')}")
    print(f"Episodes:       {info['total_episodes']}")
    print(f"Total frames:   {info['total_frames']}")
    print(f"fps:            {info['fps']}")
    print()
    print(f"State dim:      {info['features']['observation.state']['shape'][0]}")
    print(f"Action dim:     {info['features']['action']['shape'][0]}")
    print(f"State names:    {info['features']['observation.state']['names']}")
    print(f"Video key:      observation.images.scene  (h264, {info['fps']} fps)")
    print()
    print(f"Tasks ({len(tasks)}):")
    for t in tasks:
        print(f"  [{t['task_index']}] {t['task']!r}")
    print()
    print(f"Episodes ({len(episodes)}):")
    for e in episodes:
        print(f"  episode_{e['episode_index']:06d}  length={e['length']}  tasks={e['tasks']}")

    # Parquet preview. pyarrow is a direct import so the error message is
    # actionable if the user skipped the `[lerobot]` extra.
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print(
            "\n(install the `[lerobot]` extra with "
            "`uv pip install -e 'packages/robosandbox-core[lerobot]'` "
            "or install `pyarrow` directly to preview frame rows)",
            file=sys.stderr,
        )
        return 0

    parquet = root / "data" / "chunk-000" / "episode_000000.parquet"
    table = pq.read_table(parquet)
    print(f"\nFrame table: {parquet}")
    print(f"  rows:    {table.num_rows}")
    print(f"  columns: {table.column_names}")
    print(f"\nFirst {args.sample_rows} rows:")
    sub = table.slice(0, args.sample_rows).to_pylist()
    for row in sub:
        print("  -")
        for k, v in row.items():
            if isinstance(v, list) and len(v) > 4:
                short = [round(x, 4) if isinstance(x, float) else x for x in v[:4]]
                print(f"    {k:<22s} {short} … ({len(v)} elems)")
            elif isinstance(v, float):
                print(f"    {k:<22s} {v:.6f}")
            else:
                print(f"    {k:<22s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
