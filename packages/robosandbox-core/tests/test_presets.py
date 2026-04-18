"""Tests for scene.presets — procedural scene generators."""

from __future__ import annotations

import pytest

from robosandbox.scene.mjcf_builder import build_model
from robosandbox.scene.presets import tabletop_clutter
from robosandbox.types import Scene


def test_tabletop_clutter_deterministic_per_seed() -> None:
    a = tabletop_clutter(n_objects=4, seed=42)
    b = tabletop_clutter(n_objects=4, seed=42)
    assert [o.id for o in a.objects] == [o.id for o in b.objects]
    assert [o.pose.xyz for o in a.objects] == [o.pose.xyz for o in b.objects]


def test_tabletop_clutter_different_seeds_differ() -> None:
    a = tabletop_clutter(n_objects=5, seed=0)
    b = tabletop_clutter(n_objects=5, seed=1)
    # Either the picks or the poses should differ.
    assert (
        [o.id for o in a.objects] != [o.id for o in b.objects]
        or [o.pose.xyz for o in a.objects] != [o.pose.xyz for o in b.objects]
    )


def test_tabletop_clutter_compiles() -> None:
    scene = tabletop_clutter(n_objects=3, seed=0)
    model, robot = build_model(scene)
    # Each object should exist as a body in the compiled model.
    for obj in scene.objects:
        assert model.body(obj.id).id > 0


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_tabletop_clutter_ten_seeds_all_compile(seed: int) -> None:
    """Smoke-check 10 seeds — catches a sampler that wedges on edge cases."""
    scene = tabletop_clutter(n_objects=5, seed=seed)
    model, _ = build_model(scene)
    for obj in scene.objects:
        assert model.body(obj.id).id > 0


def test_tabletop_clutter_respects_min_spacing() -> None:
    scene = tabletop_clutter(n_objects=5, seed=0, min_spacing=0.08)
    poses = [o.pose.xyz for o in scene.objects]
    for i, (xi, yi, _) in enumerate(poses):
        for xj, yj, _ in poses[i + 1 :]:
            d2 = (xi - xj) ** 2 + (yi - yj) ** 2
            assert d2 >= 0.08**2 - 1e-9, (
                f"objects too close: distance^2 = {d2}, min_spacing^2 = {0.08**2}"
            )


def test_tabletop_clutter_zero_objects() -> None:
    scene = tabletop_clutter(n_objects=0, seed=0)
    assert isinstance(scene, Scene)
    assert len(scene.objects) == 0


def test_tabletop_clutter_too_many_objects_raises() -> None:
    # Only 10 YCB objects bundled; asking for 11 should fail loudly.
    with pytest.raises(ValueError, match="needs >="):
        tabletop_clutter(n_objects=11, seed=0)
