"""Unit tests for scene.mesh_conversion — sidecar schema + validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robosandbox.scene.mesh_conversion import (
    MeshAsset,
    MeshConfigNotFoundError,
    MeshConfigValidationError,
    load_bundled_mesh,
    load_byo_mesh,
)


# Minimal valid OBJ (single triangle). Enough to satisfy "file exists" —
# mesh_conversion doesn't parse geometry, only threads paths to MuJoCo.
_TRI_OBJ = """\
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
"""


def _write_tri(path: Path) -> Path:
    path.write_text(_TRI_OBJ)
    return path


def _make_sidecar(
    dir_: Path,
    overrides: dict | None = None,
    hulls: int = 2,
    include_files: bool = True,
) -> Path:
    sidecar = dir_ / "widget.robosandbox.yaml"
    data = {
        "visual_mesh": "widget_visual.obj",
        "collision_meshes": [f"widget_hull_{i}.obj" for i in range(hulls)],
        "scale": 1.0,
        "mass": 0.2,
        "friction": [1.5, 0.1, 0.01],
        "rgba": [0.8, 0.2, 0.2, 1.0],
    }
    if overrides:
        data.update(overrides)
    sidecar.write_text(yaml.safe_dump(data))
    if include_files:
        _write_tri(dir_ / "widget_visual.obj")
        for i in range(hulls):
            _write_tri(dir_ / f"widget_hull_{i}.obj")
    return sidecar


def test_load_bundled_parses_all_fields(tmp_path: Path) -> None:
    sidecar = _make_sidecar(tmp_path, hulls=3)
    asset = load_bundled_mesh(sidecar, obj_id="widget")

    assert isinstance(asset, MeshAsset)
    assert asset.obj_id == "widget"
    assert len(asset.visual_files) == 1
    assert asset.visual_files[0] == tmp_path / "widget_visual.obj"
    assert len(asset.collision_files) == 3
    assert all(p.is_absolute() for p in asset.collision_files)
    assert asset.scale == (1.0, 1.0, 1.0)
    assert asset.mass == pytest.approx(0.2)
    assert asset.friction == (1.5, 0.1, 0.01)
    assert asset.rgba == (0.8, 0.2, 0.2, 1.0)


def test_load_bundled_accepts_per_axis_scale(tmp_path: Path) -> None:
    sidecar = _make_sidecar(tmp_path, overrides={"scale": [0.5, 1.0, 2.0]})
    asset = load_bundled_mesh(sidecar, obj_id="widget")
    assert asset.scale == (0.5, 1.0, 2.0)


def test_load_bundled_missing_sidecar(tmp_path: Path) -> None:
    with pytest.raises(MeshConfigNotFoundError):
        load_bundled_mesh(tmp_path / "does_not_exist.yaml", obj_id="x")


def test_load_bundled_missing_visual(tmp_path: Path) -> None:
    sidecar = _make_sidecar(tmp_path)
    (tmp_path / "widget_visual.obj").unlink()
    with pytest.raises(MeshConfigNotFoundError):
        load_bundled_mesh(sidecar, obj_id="widget")


def test_load_bundled_missing_hull(tmp_path: Path) -> None:
    sidecar = _make_sidecar(tmp_path, hulls=2)
    (tmp_path / "widget_hull_1.obj").unlink()
    with pytest.raises(MeshConfigNotFoundError):
        load_bundled_mesh(sidecar, obj_id="widget")


def test_load_bundled_empty_collision_list_raises(tmp_path: Path) -> None:
    sidecar = _make_sidecar(tmp_path, overrides={"collision_meshes": []})
    with pytest.raises(MeshConfigValidationError) as exc:
        load_bundled_mesh(sidecar, obj_id="widget")
    assert "collision_meshes" in str(exc.value)


def test_load_bundled_missing_required_field(tmp_path: Path) -> None:
    sidecar = tmp_path / "bad.robosandbox.yaml"
    sidecar.write_text(yaml.safe_dump({"visual_mesh": "a.obj"}))  # no collision_meshes
    with pytest.raises(MeshConfigValidationError) as exc:
        load_bundled_mesh(sidecar, obj_id="x")
    assert "collision_meshes" in str(exc.value)


def test_load_bundled_bad_scale_type(tmp_path: Path) -> None:
    sidecar = _make_sidecar(tmp_path, overrides={"scale": "big"})
    with pytest.raises(MeshConfigValidationError) as exc:
        load_bundled_mesh(sidecar, obj_id="widget")
    assert "scale" in str(exc.value)


def test_load_bundled_bad_friction_length(tmp_path: Path) -> None:
    sidecar = _make_sidecar(tmp_path, overrides={"friction": [1.5, 0.1]})
    with pytest.raises(MeshConfigValidationError) as exc:
        load_bundled_mesh(sidecar, obj_id="widget")
    assert "friction" in str(exc.value)


def test_load_bundled_uses_defaults_for_optional_fields(tmp_path: Path) -> None:
    sidecar = tmp_path / "minimal.yaml"
    _write_tri(tmp_path / "v.obj")
    _write_tri(tmp_path / "h0.obj")
    sidecar.write_text(yaml.safe_dump({
        "visual_mesh": "v.obj",
        "collision_meshes": ["h0.obj"],
    }))
    asset = load_bundled_mesh(sidecar, obj_id="minimal")
    assert asset.scale == (1.0, 1.0, 1.0)
    assert asset.mass == pytest.approx(0.1)
    assert asset.friction == (1.5, 0.1, 0.01)
    assert asset.rgba == (0.7, 0.7, 0.7, 1.0)


def test_byo_not_yet_implemented(tmp_path: Path) -> None:
    obj = _write_tri(tmp_path / "thing.obj")
    with pytest.raises(NotImplementedError):
        load_byo_mesh(obj, obj_id="thing", collision_mode="coacd")


def test_mesh_asset_requires_nonempty_collision() -> None:
    with pytest.raises(MeshConfigValidationError):
        MeshAsset(
            obj_id="x",
            visual_files=(Path("/abs/v.obj"),),
            collision_files=(),
            scale=(1.0, 1.0, 1.0),
            mass=0.1,
            friction=(1.5, 0.1, 0.01),
            rgba=(0.7, 0.7, 0.7, 1.0),
        )


def test_mesh_asset_rejects_relative_paths() -> None:
    with pytest.raises(MeshConfigValidationError):
        MeshAsset(
            obj_id="x",
            visual_files=(Path("v.obj"),),  # relative, should fail
            collision_files=(Path("/abs/h.obj"),),
            scale=(1.0, 1.0, 1.0),
            mass=0.1,
            friction=(1.5, 0.1, 0.01),
            rgba=(0.7, 0.7, 0.7, 1.0),
        )
