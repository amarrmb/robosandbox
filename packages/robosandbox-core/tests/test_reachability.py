"""Tests for scene reachability pre-flight check."""
from __future__ import annotations

import os

import pytest

from robosandbox.scene.reachability import check_scene_reachability
from robosandbox.tasks.loader import load_builtin_task
from robosandbox.types import Pose, Scene, SceneObject


# MuJoCo rendering requires an OpenGL backend. EGL works headlessly in CI.
os.environ.setdefault("MUJOCO_GL", "egl")


def test_bundled_pick_cube_franka_is_reachable():
    """Sanity: the flagship Franka pick task's objects are all reachable."""
    task = load_builtin_task("pick_cube_franka")
    assert check_scene_reachability(task.scene) == []


def test_bundled_pick_ycb_mug_is_reachable():
    """Mesh-import task's mug should be reachable (no regressions)."""
    task = load_builtin_task("pick_ycb_mug")
    assert check_scene_reachability(task.scene) == []


def test_far_object_is_flagged_unreachable():
    """A cube placed 1.2 m from the Franka (reach ~0.85 m) must be flagged."""
    franka_task = load_builtin_task("pick_cube_franka")
    # Replace the cube with one at an impossibly far x.
    obj = franka_task.scene.objects[0]
    far_obj = SceneObject(
        id=obj.id, kind=obj.kind, size=obj.size,
        pose=Pose(xyz=(1.2, 0.0, 0.06), quat_xyzw=obj.pose.quat_xyzw),
        rgba=obj.rgba, mass=obj.mass,
    )
    scene = Scene(
        robot_urdf=franka_task.scene.robot_urdf,
        robot_config=franka_task.scene.robot_config,
        objects=(far_obj,),
    )
    warnings = check_scene_reachability(scene)
    assert len(warnings) == 1
    w = warnings[0]
    assert w.id == obj.id
    assert w.first_failed_phase in {"approach", "grasp", "lift"}


def test_mixed_scene_flags_only_the_bad_one():
    """A scene with one good and one bad object returns only the bad one."""
    franka_task = load_builtin_task("pick_cube_franka")
    good = franka_task.scene.objects[0]
    bad = SceneObject(
        id="far_cube", kind="box", size=good.size,
        pose=Pose(xyz=(1.2, 0.0, 0.06)),
        rgba=good.rgba, mass=good.mass,
    )
    scene = Scene(
        robot_urdf=franka_task.scene.robot_urdf,
        robot_config=franka_task.scene.robot_config,
        objects=(good, bad),
    )
    warnings = check_scene_reachability(scene)
    assert [w.id for w in warnings] == ["far_cube"]


def test_drawer_objects_are_skipped():
    """Drawers use OpenDrawer/CloseDrawer, not Pick — reachability should skip them."""
    task = load_builtin_task("open_drawer")
    warnings = check_scene_reachability(task.scene)
    # The drawer itself must not be flagged. Any non-drawer objects in the
    # scene should still be checked.
    for w in warnings:
        obj = next((o for o in task.scene.objects if o.id == w.id), None)
        assert obj is None or obj.kind != "drawer", (
            f"drawer {w.id!r} should be skipped but was flagged: {w.detail}"
        )
