"""Spawn three YCB objects on the Franka and render one frame.

Shows the mesh-import pipeline end to end: @ycb: resolver in task-shaped
Scene construction -> MjSpec mesh injection -> MuJoCo render.

Run:
    uv run python examples/spawn_ycb_scene.py --out examples/out.png
"""

from __future__ import annotations

import argparse
from importlib.resources import files
from pathlib import Path

import imageio.v3 as iio

from robosandbox.scene.mjcf_builder import build_model
from robosandbox.sim.mujoco_backend import MuJoCoBackend
from robosandbox.tasks.loader import _ycb_short_name
from robosandbox.types import Pose, Scene, SceneObject


def _ycb_sidecar(ycb_id: str) -> Path:
    short = _ycb_short_name(ycb_id)
    return Path(
        str(
            files("robosandbox").joinpath(
                "assets", "objects", "ycb", ycb_id, f"{short}.robosandbox.yaml"
            )
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("examples/out.png"))
    args = ap.parse_args()

    urdf = Path(str(files("robosandbox").joinpath("assets/robots/franka_panda/panda.xml")))
    cfg = Path(
        str(files("robosandbox").joinpath("assets/robots/franka_panda/panda.robosandbox.yaml"))
    )

    scene = Scene(
        robot_urdf=urdf,
        robot_config=cfg,
        objects=(
            SceneObject(
                id="apple",
                kind="mesh",
                size=(0.0,),
                pose=Pose(xyz=(0.35, -0.1, 0.06)),
                mass=0.0,  # use sidecar default
                mesh_sidecar=_ycb_sidecar("013_apple"),
            ),
            SceneObject(
                id="soup_can",
                kind="mesh",
                size=(0.0,),
                pose=Pose(xyz=(0.45, 0.0, 0.065)),
                mass=0.0,
                mesh_sidecar=_ycb_sidecar("005_tomato_soup_can"),
            ),
            SceneObject(
                id="banana",
                kind="mesh",
                size=(0.0,),
                pose=Pose(xyz=(0.35, 0.1, 0.04)),
                mass=0.0,
                mesh_sidecar=_ycb_sidecar("011_banana"),
            ),
        ),
    )

    # Confirm the scene compiles before paying for a sim.
    model, _ = build_model(scene)
    print(f"compiled: {model.nbody} bodies, {model.nmesh} meshes, {model.ngeom} geoms")

    sim = MuJoCoBackend(render_size=(480, 640), camera="scene")
    sim.load(scene)
    try:
        for _ in range(120):  # let objects settle
            sim.step()
        obs = sim.observe()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(args.out, obs.rgb)
        print(f"wrote {args.out}")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
