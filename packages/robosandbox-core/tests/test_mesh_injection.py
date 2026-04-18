"""Integration tests for scene.mesh_injection — mesh objects compile.

Uses trimesh to generate small valid OBJs on the fly (a single triangle
is not enough for MuJoCo's mesh compiler). The URDF path (Franka) is our
baseline — once the mesh branch there works, step 3 unifies the built-in
arm through the same entry point.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import mujoco
import numpy as np
import pytest
import trimesh
import yaml

from robosandbox.scene.mesh_conversion import (
    MeshConfigError,
    load_bundled_mesh,
    resolve_mesh_asset,
)
from robosandbox.scene.mesh_injection import inject_mesh_object
from robosandbox.scene.robot_loader import load_and_compile
from robosandbox.types import Pose, Scene, SceneObject


def _write_box_obj(path: Path, extents: tuple[float, float, float]) -> Path:
    """Export a small axis-aligned box as OBJ for MuJoCo to compile."""
    mesh = trimesh.creation.box(extents=list(extents))
    mesh.export(str(path), file_type="obj")
    return path


def _make_bundled_mug_fixture(dir_: Path) -> Path:
    """Write a self-contained per-object sidecar + hulls under ``dir_``.

    Mesh geometry is a tiny box decomposed into two halves so we test the
    multi-collision-geom path. Dimensions are ~mug-sized for realism.
    """
    visual = _write_box_obj(dir_ / "widget_visual.obj", (0.06, 0.06, 0.08))
    h0 = _write_box_obj(dir_ / "widget_hull_0.obj", (0.06, 0.06, 0.04))
    h1 = _write_box_obj(dir_ / "widget_hull_1.obj", (0.06, 0.06, 0.04))
    sidecar = dir_ / "widget.robosandbox.yaml"
    sidecar.write_text(
        yaml.safe_dump({
            "visual_mesh": visual.name,
            "collision_meshes": [h0.name, h1.name],
            "scale": 1.0,
            "mass": 0.15,
            "friction": [1.5, 0.1, 0.01],
            "rgba": [0.85, 0.85, 0.85, 1.0],
        })
    )
    return sidecar


@pytest.fixture
def franka_urdf() -> Path:
    return Path(str(files("robosandbox").joinpath("assets/robots/franka_panda/panda.xml")))


@pytest.fixture
def franka_config() -> Path:
    return Path(
        str(files("robosandbox").joinpath("assets/robots/franka_panda/panda.robosandbox.yaml"))
    )


def test_franka_plus_mesh_object_compiles(
    tmp_path: Path, franka_urdf: Path, franka_config: Path
) -> None:
    """URDF path + one bundled mesh object compiles and the free-body exists."""
    sidecar = _make_bundled_mug_fixture(tmp_path)

    scene = Scene(
        robot_urdf=franka_urdf,
        robot_config=franka_config,
        objects=(
            SceneObject(
                id="widget",
                kind="mesh",
                size=(0.0,),  # unused for mesh
                pose=Pose(xyz=(0.45, 0.0, 0.1)),
                mass=0.0,  # defer to sidecar
                mesh_sidecar=sidecar,
            ),
        ),
    )

    model, _ = load_and_compile(scene)

    # The free-body body should exist under worldbody with the right name.
    body_id = model.body("widget").id
    assert body_id > 0, "widget body should be created"
    assert model.body("widget").jntnum >= 1, "free-body should have at least one joint"

    # Should have at least two collision mesh geoms (one per hull).
    widget_geom_ids = [
        i for i in range(model.ngeom) if model.geom(i).bodyid == body_id
    ]
    mesh_geom_ids = [i for i in widget_geom_ids if model.geom(i).type == mujoco.mjtGeom.mjGEOM_MESH]
    assert len(mesh_geom_ids) >= 2, f"expected >= 2 mesh geoms, got {len(mesh_geom_ids)}"

    # Body mass is summed from geom mass at compile time; visuals contribute 0.
    body_mass = float(model.body("widget").mass[0])
    assert body_mass == pytest.approx(0.15, rel=1e-4), (
        f"expected body mass 0.15 kg, got {body_mass}"
    )


def test_mesh_object_mass_override_from_task_yaml(
    tmp_path: Path, franka_urdf: Path, franka_config: Path
) -> None:
    """SceneObject.mass > 0 overrides the sidecar default."""
    sidecar = _make_bundled_mug_fixture(tmp_path)
    scene = Scene(
        robot_urdf=franka_urdf,
        robot_config=franka_config,
        objects=(
            SceneObject(
                id="widget",
                kind="mesh",
                size=(0.0,),
                pose=Pose(xyz=(0.45, 0.0, 0.1)),
                mass=0.03,  # override
                mesh_sidecar=sidecar,
            ),
        ),
    )
    model, _ = load_and_compile(scene)
    body_mass = float(model.body("widget").mass[0])
    assert body_mass == pytest.approx(0.03, rel=1e-4)


def test_resolve_mesh_asset_rejects_both_set(tmp_path: Path) -> None:
    sidecar = _make_bundled_mug_fixture(tmp_path)
    obj = SceneObject(
        id="widget",
        kind="mesh",
        size=(0.0,),
        pose=Pose(xyz=(0, 0, 0)),
        mesh_sidecar=sidecar,
        mesh_path=tmp_path / "irrelevant.obj",
    )
    with pytest.raises(MeshConfigError) as exc:
        resolve_mesh_asset(obj)
    assert "exactly one" in str(exc.value)


def test_resolve_mesh_asset_rejects_neither_set(tmp_path: Path) -> None:
    obj = SceneObject(
        id="widget",
        kind="mesh",
        size=(0.0,),
        pose=Pose(xyz=(0, 0, 0)),
    )
    with pytest.raises(MeshConfigError) as exc:
        resolve_mesh_asset(obj)
    assert "exactly one" in str(exc.value)


def test_resolve_mesh_asset_rejects_non_mesh_kind(tmp_path: Path) -> None:
    obj = SceneObject(
        id="b",
        kind="box",
        size=(0.01, 0.01, 0.01),
        pose=Pose(xyz=(0, 0, 0)),
    )
    with pytest.raises(MeshConfigError):
        resolve_mesh_asset(obj)
