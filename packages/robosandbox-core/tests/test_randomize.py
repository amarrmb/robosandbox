"""Tests for tasks.randomize — per-seed scene jittering."""

from __future__ import annotations

import math

import pytest
from robosandbox.tasks.randomize import (
    _quat_xyzw_from_yaw,
    _yaw_from_quat_xyzw,
    jitter_scene,
)
from robosandbox.types import Pose, Scene, SceneObject


def _obj(id_: str = "c", xyz=(0.3, 0.0, 0.05)) -> SceneObject:
    return SceneObject(id=id_, kind="box", size=(0.01, 0.01, 0.01), pose=Pose(xyz=xyz))


def test_seed_zero_is_identity() -> None:
    scene = Scene(objects=(_obj(),))
    out = jitter_scene(scene, {"xy_jitter": 0.05, "yaw_jitter": 0.5}, seed=0)
    assert out is scene  # identity short-circuit


def test_no_spec_is_identity() -> None:
    scene = Scene(objects=(_obj(),))
    assert jitter_scene(scene, None, seed=7) is scene
    assert jitter_scene(scene, {}, seed=7) is scene


def test_zero_amplitude_is_identity() -> None:
    scene = Scene(objects=(_obj(),))
    out = jitter_scene(scene, {"xy_jitter": 0.0, "yaw_jitter": 0.0}, seed=7)
    assert out is scene


def test_jitter_deterministic_per_seed() -> None:
    scene = Scene(objects=(_obj(),))
    a = jitter_scene(scene, {"xy_jitter": 0.05}, seed=3)
    b = jitter_scene(scene, {"xy_jitter": 0.05}, seed=3)
    assert a.objects[0].pose.xyz == b.objects[0].pose.xyz


def test_jitter_different_seeds_diverge() -> None:
    scene = Scene(objects=(_obj(),))
    a = jitter_scene(scene, {"xy_jitter": 0.05}, seed=1)
    b = jitter_scene(scene, {"xy_jitter": 0.05}, seed=2)
    assert a.objects[0].pose.xyz != b.objects[0].pose.xyz


def test_xy_jitter_stays_within_bounds() -> None:
    base = _obj(xyz=(0.3, 0.0, 0.05))
    scene = Scene(objects=(base,))
    for seed in range(1, 100):
        out = jitter_scene(scene, {"xy_jitter": 0.04}, seed=seed)
        x, y, z = out.objects[0].pose.xyz
        assert abs(x - 0.3) <= 0.04 + 1e-12
        assert abs(y - 0.0) <= 0.04 + 1e-12
        assert z == 0.05  # z unchanged


def test_yaw_jitter_within_bounds() -> None:
    scene = Scene(objects=(_obj(),))
    for seed in range(1, 50):
        out = jitter_scene(scene, {"yaw_jitter": 0.3}, seed=seed)
        yaw = _yaw_from_quat_xyzw(out.objects[0].pose.quat_xyzw)
        assert -0.3 - 1e-9 <= yaw <= 0.3 + 1e-9


def test_jitter_applies_to_all_objects_independently() -> None:
    scene = Scene(objects=(_obj("a", (0.3, 0.0, 0.05)), _obj("b", (0.4, 0.1, 0.05))))
    out = jitter_scene(scene, {"xy_jitter": 0.02}, seed=5)
    assert out.objects[0].pose.xyz != scene.objects[0].pose.xyz
    assert out.objects[1].pose.xyz != scene.objects[1].pose.xyz
    # They should have different perturbations (not the same delta applied).
    da = (
        out.objects[0].pose.xyz[0] - scene.objects[0].pose.xyz[0],
        out.objects[0].pose.xyz[1] - scene.objects[0].pose.xyz[1],
    )
    db = (
        out.objects[1].pose.xyz[0] - scene.objects[1].pose.xyz[0],
        out.objects[1].pose.xyz[1] - scene.objects[1].pose.xyz[1],
    )
    assert da != db


def test_quat_roundtrip() -> None:
    for yaw in [-1.5, -0.3, 0.0, 0.3, 1.5, 3.0]:
        back = _yaw_from_quat_xyzw(_quat_xyzw_from_yaw(yaw))
        # Quat roundtrip may wrap near ±π; normalize modulo 2π.
        diff = (back - yaw + math.pi) % (2 * math.pi) - math.pi
        assert abs(diff) < 1e-9


# ---------------------------------------------------------------------------
# rgba / size / mass jitter (v0.2 knobs)
# ---------------------------------------------------------------------------


def _cube(id_: str = "c", rgba=(0.5, 0.5, 0.5, 1.0), size=(0.01, 0.01, 0.01), mass=0.05):
    return SceneObject(
        id=id_,
        kind="box",
        size=size,
        pose=Pose(xyz=(0.3, 0.0, 0.05)),
        mass=mass,
        rgba=rgba,
    )


def test_rgba_jitter_perturbs_rgb_not_alpha() -> None:
    base = _cube(rgba=(0.5, 0.5, 0.5, 1.0))
    scene = Scene(objects=(base,))
    out = jitter_scene(scene, {"rgba_jitter": 0.1}, seed=1)
    r, g, b, a = out.objects[0].rgba
    assert r != 0.5 and g != 0.5 and b != 0.5
    assert a == 1.0  # alpha untouched


def test_rgba_jitter_clamped_to_unit_interval() -> None:
    # Start at the boundary so any non-zero delta would push out of [0,1].
    base = _cube(rgba=(0.0, 1.0, 0.0, 1.0))
    scene = Scene(objects=(base,))
    for seed in range(1, 50):
        out = jitter_scene(scene, {"rgba_jitter": 0.5}, seed=seed)
        r, g, b, a = out.objects[0].rgba
        assert 0.0 <= r <= 1.0
        assert 0.0 <= g <= 1.0
        assert 0.0 <= b <= 1.0
        assert a == 1.0


def test_size_jitter_perturbs_every_component() -> None:
    base = _cube(size=(0.01, 0.02, 0.03))
    scene = Scene(objects=(base,))
    out = jitter_scene(scene, {"size_jitter": 0.15}, seed=1)
    ns = out.objects[0].size
    assert len(ns) == 3
    for s_new, s_old in zip(ns, base.size):
        assert s_new != s_old


def test_size_jitter_clamped_to_valid_range() -> None:
    base = _cube(size=(0.01, 0.02, 0.03))
    scene = Scene(objects=(base,))
    # Wildly large amplitude would drive the size outside sane bounds.
    for seed in range(1, 50):
        out = jitter_scene(scene, {"size_jitter": 10.0}, seed=seed)
        for s_new, s_old in zip(out.objects[0].size, base.size):
            assert 0.3 * s_old - 1e-12 <= s_new <= 3.0 * s_old + 1e-12


def test_mass_jitter_perturbs_and_clamps() -> None:
    base = _cube(mass=0.05)
    scene = Scene(objects=(base,))
    out = jitter_scene(scene, {"mass_jitter": 0.3}, seed=1)
    assert out.objects[0].mass != 0.05
    # Clamp: huge negative relative deltas still floor at 1e-4.
    tiny = _cube(mass=1e-5)
    out_tiny = jitter_scene(Scene(objects=(tiny,)), {"mass_jitter": 0.9}, seed=3)
    assert out_tiny.objects[0].mass >= 1e-4


def test_seed_zero_identity_across_all_knobs() -> None:
    base = _cube()
    scene = Scene(objects=(base,))
    out = jitter_scene(
        scene,
        {
            "xy_jitter": 0.05,
            "yaw_jitter": 0.5,
            "rgba_jitter": 0.3,
            "size_jitter": 0.2,
            "mass_jitter": 0.4,
        },
        seed=0,
    )
    assert out is scene


def test_mesh_object_size_jitter_is_no_op() -> None:
    mesh = SceneObject(
        id="m",
        kind="mesh",
        size=(0.0,),
        pose=Pose(xyz=(0.3, 0.0, 0.05)),
        mass=0.2,
        rgba=(0.4, 0.5, 0.6, 1.0),
    )
    scene = Scene(objects=(mesh,))
    out = jitter_scene(
        scene,
        {"size_jitter": 0.5, "rgba_jitter": 0.2, "mass_jitter": 0.3},
        seed=1,
    )
    o = out.objects[0]
    # Size unchanged (mesh skip).
    assert o.size == (0.0,)
    # rgba and mass still jittered.
    assert o.rgba[:3] != (0.4, 0.5, 0.6)
    assert o.rgba[3] == 1.0
    assert o.mass != 0.2


def test_drawer_object_all_non_pose_jitters_are_no_op() -> None:
    drawer = SceneObject(
        id="d",
        kind="drawer",
        size=(0.15, 0.12, 0.05),
        pose=Pose(xyz=(0.42, 0.0, 0.08)),
        mass=0.1,
        rgba=(0.55, 0.35, 0.2, 1.0),
        drawer_max_open=0.12,
    )
    scene = Scene(objects=(drawer,))
    out = jitter_scene(
        scene,
        {
            "xy_jitter": 0.02,
            "yaw_jitter": 0.2,
            "rgba_jitter": 0.3,
            "size_jitter": 0.3,
            "mass_jitter": 0.4,
        },
        seed=1,
    )
    o = out.objects[0]
    # rgba / size / mass untouched.
    assert o.rgba == drawer.rgba
    assert o.size == drawer.size
    assert o.mass == drawer.mass
    # Pose still jittered.
    assert o.pose.xyz != drawer.pose.xyz


def test_rgba_size_mass_deterministic_per_seed() -> None:
    scene = Scene(objects=(_cube(),))
    spec = {"rgba_jitter": 0.2, "size_jitter": 0.1, "mass_jitter": 0.3}
    a = jitter_scene(scene, spec, seed=5)
    b = jitter_scene(scene, spec, seed=5)
    assert a.objects[0].rgba == b.objects[0].rgba
    assert a.objects[0].size == b.objects[0].size
    assert a.objects[0].mass == b.objects[0].mass


def test_loader_rejects_negative_jitter(tmp_path) -> None:
    import yaml as _yaml
    from robosandbox.tasks.loader import load_task

    task_yaml = {
        "name": "bad",
        "prompt": "x",
        "scene": {"objects": [{"id": "a", "kind": "box"}]},
        "success": {"kind": "lifted", "object": "a", "min_mm": 1},
        "randomize": {"rgba_jitter": -0.1},
    }
    p = tmp_path / "bad.yaml"
    p.write_text(_yaml.safe_dump(task_yaml))
    with pytest.raises(ValueError, match="rgba_jitter"):
        load_task(p)


def test_loader_rejects_non_numeric_jitter(tmp_path) -> None:
    import yaml as _yaml
    from robosandbox.tasks.loader import load_task

    task_yaml = {
        "name": "bad",
        "prompt": "x",
        "scene": {"objects": [{"id": "a", "kind": "box"}]},
        "success": {"kind": "lifted", "object": "a", "min_mm": 1},
        "randomize": {"size_jitter": "lots"},
    }
    p = tmp_path / "bad.yaml"
    p.write_text(_yaml.safe_dump(task_yaml))
    with pytest.raises(ValueError, match="size_jitter"):
        load_task(p)
