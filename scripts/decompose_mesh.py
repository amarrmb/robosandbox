#!/usr/bin/env python3
"""Offline tool: convex-decompose a mesh with CoACD and emit bundled assets.

Intended for authoring ``assets/objects/<pack>/<obj>/`` directories in the
repo. Run it once per bundled object; commit the resulting OBJs + sidecar
YAML. Runtime never calls this script — runtime just loads the cached hulls.

Usage:
    python scripts/decompose_mesh.py \\
        --input /tmp/ycb_mug/025_mug/google_16k/nontextured.stl \\
        --out-dir packages/robosandbox-core/src/robosandbox/assets/objects/ycb/025_mug \\
        --name mug \\
        --mass 0.15 \\
        --center-bottom

Flags:
    --input PATH        Source mesh (OBJ/STL/PLY, anything trimesh reads)
    --out-dir PATH      Destination directory (created if missing)
    --name NAME         Basename for output files (``<name>_visual.obj`` etc.)
    --mass KG           Default mass written into the sidecar
    --rgba R G B A      Default colour (default 0.7 0.7 0.7 1.0)
    --threshold FLOAT   CoACD concavity threshold (default 0.05 — lower = more hulls)
    --center-bottom     Translate so centroid-xy is at (0,0) and min-z is at 0
    --scale FLOAT       Apply a uniform scale before export (default 1.0)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh
import yaml

try:
    import coacd  # type: ignore
except ImportError:
    print(
        "coacd is not installed. Run:  uv sync --extra meshes  (or pip install coacd)",
        file=sys.stderr,
    )
    sys.exit(2)


def _center_bottom(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Translate so xy-centroid is at origin, min-z at 0."""
    centroid_xy = mesh.centroid.copy()
    centroid_xy[2] = 0.0
    mesh = mesh.copy()
    mesh.apply_translation(-centroid_xy)
    mesh.apply_translation([0.0, 0.0, -mesh.bounds[0][2]])
    return mesh


def _apply_scale(mesh: trimesh.Trimesh, s: float) -> trimesh.Trimesh:
    if s == 1.0:
        return mesh
    mesh = mesh.copy()
    mesh.apply_scale(s)
    return mesh


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--mass", type=float, default=0.1)
    ap.add_argument("--rgba", nargs=4, type=float, default=[0.7, 0.7, 0.7, 1.0])
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--center-bottom", action="store_true")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument(
        "--friction",
        nargs=3,
        type=float,
        default=[1.5, 0.1, 0.01],
        help="sliding torsional rolling",
    )
    args = ap.parse_args()

    mesh = trimesh.load(str(args.input), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        print(f"Input did not load as a single mesh: {type(mesh).__name__}", file=sys.stderr)
        return 2

    if args.center_bottom:
        mesh = _center_bottom(mesh)
    mesh = _apply_scale(mesh, args.scale)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Visual: export the (optionally centered/scaled) mesh as OBJ.
    visual_path = args.out_dir / f"{args.name}_visual.obj"
    mesh.export(str(visual_path), file_type="obj")
    print(f"visual: {visual_path}  ({len(mesh.vertices)} v / {len(mesh.faces)} f)")

    # Collision: CoACD decomposition. We feed a watertight-ish patched mesh.
    # CoACD's API accepts (vertices, faces) as numpy arrays.
    co_mesh = coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    parts = coacd.run_coacd(co_mesh, threshold=args.threshold)
    print(f"coacd: {len(parts)} hull(s) at threshold={args.threshold}")

    hull_files: list[str] = []
    for i, (verts, faces) in enumerate(parts):
        hull = trimesh.Trimesh(vertices=np.asarray(verts), faces=np.asarray(faces), process=False)
        # CoACD output is already convex; MuJoCo's compiler accepts it as-is.
        hull_path = args.out_dir / f"{args.name}_hull_{i}.obj"
        hull.export(str(hull_path), file_type="obj")
        hull_files.append(hull_path.name)

    sidecar_path = args.out_dir / f"{args.name}.robosandbox.yaml"
    sidecar_path.write_text(
        yaml.safe_dump(
            {
                "visual_mesh": visual_path.name,
                "collision_meshes": hull_files,
                "scale": 1.0,
                "mass": args.mass,
                "friction": list(args.friction),
                "rgba": list(args.rgba),
            },
            sort_keys=False,
        )
    )
    print(f"sidecar: {sidecar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
