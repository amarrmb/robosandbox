"""Unit tests for scene.mesh_conversion — sidecar schema + validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robosandbox.scene.mesh_conversion import (
    MeshAsset,
    MeshConfigError,
    MeshConfigNotFoundError,
    MeshConfigValidationError,
    _byo_cache_key,
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


def test_byo_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(MeshConfigNotFoundError):
        load_byo_mesh(
            tmp_path / "does_not_exist.obj",
            obj_id="x",
            collision_mode="hull",
            cache_dir=tmp_path / "cache",
        )


def test_byo_invalid_mode_raises(tmp_path: Path) -> None:
    obj = _write_tri(tmp_path / "thing.obj")
    with pytest.raises(MeshConfigValidationError):
        load_byo_mesh(
            obj,
            obj_id="thing",
            collision_mode="wrong",
            cache_dir=tmp_path / "cache",
        )


def test_byo_hull_passthrough(tmp_path: Path) -> None:
    """Hull mode: copies the mesh into the cache and returns a MeshAsset.

    Uses trimesh to create a valid convex mesh so MuJoCo would accept it.
    """
    trimesh = pytest.importorskip("trimesh")
    mesh_path = tmp_path / "cube.obj"
    trimesh.creation.box(extents=[0.02, 0.02, 0.02]).export(str(mesh_path), file_type="obj")

    cache = tmp_path / "cache"
    asset = load_byo_mesh(
        mesh_path,
        obj_id="cube",
        collision_mode="hull",
        cache_dir=cache,
        mass=0.05,
        rgba=(0.2, 0.3, 0.4, 1.0),
    )

    assert asset.obj_id == "cube"
    assert asset.mass == pytest.approx(0.05)
    assert asset.rgba == (0.2, 0.3, 0.4, 1.0)
    assert len(asset.collision_files) == 1
    assert asset.collision_files[0].exists()
    # Cache subdir should have the key directory and a manifest.
    subdirs = list(cache.iterdir())
    assert len(subdirs) == 1
    assert (subdirs[0] / "manifest.yaml").exists()


def test_byo_hull_cache_hit(tmp_path: Path) -> None:
    """Second call with same mesh+mode should read the cache, not recompute.

    We proxy 'cache hit' by overwriting the hull file after the first call
    and checking the second call reads that overwritten content (not the
    original).
    """
    trimesh = pytest.importorskip("trimesh")
    mesh_path = tmp_path / "cube.obj"
    trimesh.creation.box(extents=[0.02, 0.02, 0.02]).export(str(mesh_path), file_type="obj")
    cache = tmp_path / "cache"

    asset1 = load_byo_mesh(
        mesh_path,
        obj_id="cube",
        collision_mode="hull",
        cache_dir=cache,
    )
    sentinel = "# CACHE HIT MARKER\n"
    asset1.collision_files[0].write_text(sentinel, encoding="utf-8")

    asset2 = load_byo_mesh(
        mesh_path,
        obj_id="cube",
        collision_mode="hull",
        cache_dir=cache,
    )
    assert asset2.collision_files[0] == asset1.collision_files[0]
    assert asset2.collision_files[0].read_text(encoding="utf-8") == sentinel


def test_byo_coacd_real_mesh(tmp_path: Path) -> None:
    """CoACD mode: decompose a procedural mug into multiple convex hulls.

    Skipped if coacd isn't installed (should always be installed when the
    [meshes] extra is active, but we keep the skip for non-extra installs).
    """
    pytest.importorskip("coacd")
    trimesh = pytest.importorskip("trimesh")

    # A torus is non-convex; CoACD should split it into 2+ hulls.
    mesh_path = tmp_path / "torus.obj"
    trimesh.creation.torus(major_radius=0.03, minor_radius=0.01).export(
        str(mesh_path), file_type="obj"
    )
    cache = tmp_path / "cache"
    asset = load_byo_mesh(
        mesh_path,
        obj_id="torus",
        collision_mode="coacd",
        cache_dir=cache,
    )
    assert len(asset.collision_files) >= 2, (
        f"CoACD should decompose a torus into >=2 hulls, got {len(asset.collision_files)}"
    )
    for p in asset.collision_files:
        assert p.exists()


def test_byo_cache_key_depends_on_mode(tmp_path: Path) -> None:
    trimesh = pytest.importorskip("trimesh")
    mesh_path = tmp_path / "cube.obj"
    trimesh.creation.box(extents=[0.02, 0.02, 0.02]).export(str(mesh_path), file_type="obj")
    assert _byo_cache_key(mesh_path, "hull") != _byo_cache_key(mesh_path, "coacd")


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
