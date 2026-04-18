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