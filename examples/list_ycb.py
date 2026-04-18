"""List every bundled YCB object + its properties.

Run:
    uv run python examples/list_ycb.py
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from robosandbox.scene.mesh_conversion import load_bundled_mesh
from robosandbox.tasks.loader import _ycb_short_name, list_builtin_ycb_objects


def main() -> None:
    ids = list_builtin_ycb_objects()
    print(f"{len(ids)} bundled YCB objects:\n")
    print(f"  {'id':<25} {'mass':>8}   hulls   bytes(visual+hulls)")
    print(f"  {'-'*25} {'----':>8}   -----   --------------------")

    for ycb_id in ids:
        short = _ycb_short_name(ycb_id)
        sidecar = Path(
            str(
                files("robosandbox").joinpath(
                    "assets", "objects", "ycb", ycb_id, f"{short}.robosandbox.yaml"
                )
            )
        )
        asset = load_bundled_mesh(sidecar, obj_id=short)
        total_bytes = sum(
            p.stat().st_size for p in asset.visual_files + asset.collision_files
        )
        print(
            f"  {ycb_id:<25} {asset.mass:>6.3f}kg   "
            f"{len(asset.collision_files):>5}   {total_bytes/1024:>8.1f} KB"
        )

    print(
        "\nUse in a task YAML with:\n"
        '    objects:\n'
        '      - id: <your_id>\n'
        '        kind: mesh\n'
        '        mesh: "@ycb:<ycb_id>"\n'
        '        pose: {xyz: [0.4, 0.0, 0.05]}\n'
    )


if __name__ == "__main__":
    main()
