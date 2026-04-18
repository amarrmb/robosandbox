"""Unit tests for scene.robot_loader — sidecar schema + MjSpec injection."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robosandbox.scene.robot_loader import (
    RobotConfigMismatchError,
    RobotConfigNotFoundError,
    RobotConfigValidationError,
    load_and_compile,
    load_robot,
    resolve_sidecar,
)
from robosandbox.types import Scene, SceneObject, Pose


@pytest.fixture
def franka_urdf() -> Path:
    from importlib.resources import files

    return Path(str(files("robosandbox").joinpath("assets/robots/franka_panda/panda.xml")))


@pytest.fixture
def franka_config() -> Path:
    from importlib.resources import files

    return Path(
        str(files("robosandbox").joinpath("assets/robots/franka_panda/panda.robosandbox.yaml"))
    )


def test_load_bundled_franka_returns_robot_spec(franka_urdf: Path, franka_config: Path) -> None:
    _, spec = load_robot(franka_urdf, franka_config)
    assert spec.arm_joint_names == ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7")
    assert spec.arm_actuator_names == tuple(f"actuator{i}" for i in range(1, 8))
    assert spec.gripper_primary_joint == "finger_joint1"
    assert spec.gripper_actuator_name == "actuator8"
    assert spec.ee_site_name == "robosandbox_ee_site"
    assert spec.base_body_name == "link0"
    assert len(spec.home_qpos) == 7
    assert spec.gripper_open_qpos == pytest.approx(0.04)
    assert spec.gripper_closed_qpos == pytest.approx(0.0)


def test_sidecar_explicit_path_overrides_sibling(
    franka_urdf: Path, franka_config: Path, tmp_path: Path
) -> None:
    # Place an explicit config in a different location
    alt = tmp_path / "my_config.yaml"
    alt.write_text(franka_config.read_text())
    result = resolve_sidecar(franka_urdf, alt)
    assert result == alt


def test_sidecar_sibling_fallback_resolves(franka_urdf: Path, franka_config: Path) -> None:
    # With config_path=None, loader must find the sibling .robosandbox.yaml
    result = resolve_sidecar(franka_urdf, None)
    assert result == franka_config


def test_sidecar_missing_raises_not_found(tmp_path: Path) -> None:
    fake_urdf = tmp_path / "nonexistent.xml"
    fake_urdf.write_text("<mujoco/>")
    with pytest.raises(RobotConfigNotFoundError) as exc_info:
        resolve_sidecar(fake_urdf, None)
    assert "nonexistent.robosandbox.yaml" in str(exc_info.value)


def test_sidecar_explicit_missing_raises_not_found(tmp_path: Path) -> None:
    fake_urdf = tmp_path / "fake.xml"
    fake_urdf.write_text("<mujoco/>")
    missing_config = tmp_path / "does_not_exist.yaml"
    with pytest.raises(RobotConfigNotFoundError):
        resolve_sidecar(fake_urdf, missing_config)


def _write_sidecar(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data))


def _minimal_valid_sidecar() -> dict:
    # Shape only — no compile happens in validation tests.
    return {
        "arm": {
            "joints": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"],
            "actuators": ["actuator1", "actuator2", "actuator3", "actuator4",
                          "actuator5", "actuator6", "actuator7"],
            "home_qpos": [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853],
        },
        "gripper": {
            "joints": ["finger_joint1", "finger_joint2"],
            "primary_joint": "finger_joint1",
            "actuator": "actuator8",
            "open_qpos": 0.04,
            "closed_qpos": 0.0,
        },
        "ee_site": {
            "inject": {"attach_body": "hand", "xyz": [0.0, 0.0, 0.1034]},
        },
        "base_pose": {"xyz": [-0.3, 0.0, 0.0]},
    }


def test_sidecar_missing_required_field_raises(
    franka_urdf: Path, tmp_path: Path
) -> None:
    bad = _minimal_valid_sidecar()
    del bad["arm"]["joints"]
    sidecar = tmp_path / "bad.yaml"
    _write_sidecar(sidecar, bad)
    with pytest.raises(RobotConfigValidationError) as exc:
        load_robot(franka_urdf, sidecar)
    assert "arm.joints" in str(exc.value)


def test_sidecar_ee_site_both_modes_raises(
    franka_urdf: Path, tmp_path: Path
) -> None:
    bad = _minimal_valid_sidecar()
    bad["ee_site"]["name"] = "some_site"
    # now has both "name" and "inject"
    sidecar = tmp_path / "bad.yaml"
    _write_sidecar(sidecar, bad)
    with pytest.raises(RobotConfigValidationError) as exc:
        load_robot(franka_urdf, sidecar)
    assert "ee_site" in str(exc.value)
    assert "both" in str(exc.value).lower()


def test_sidecar_ee_site_neither_mode_raises(
    franka_urdf: Path, tmp_path: Path
) -> None:
    bad = _minimal_valid_sidecar()
    bad["ee_site"] = {}
    sidecar = tmp_path / "bad.yaml"
    _write_sidecar(sidecar, bad)
    with pytest.raises(RobotConfigValidationError) as exc:
        load_robot(franka_urdf, sidecar)
    assert "neither" in str(exc.value).lower()


def test_sidecar_home_qpos_length_mismatch_raises(
    franka_urdf: Path, tmp_path: Path
) -> None:
    bad = _minimal_valid_sidecar()
    bad["arm"]["home_qpos"] = [0.0, 0.0, 0.0]  # only 3, arm has 7 joints
    sidecar = tmp_path / "bad.yaml"
    _write_sidecar(sidecar, bad)
    with pytest.raises(RobotConfigValidationError) as exc:
        load_robot(franka_urdf, sidecar)
    assert "home_qpos" in str(exc.value)


def test_sidecar_unknown_joint_name_raises_at_compile(
    franka_urdf: Path, tmp_path: Path
) -> None:
    bad = _minimal_valid_sidecar()
    bad["arm"]["joints"][0] = "not_a_real_joint"
    sidecar = tmp_path / "bad.yaml"
    _write_sidecar(sidecar, bad)
    scene = Scene(robot_urdf=franka_urdf, robot_config=sidecar)
    with pytest.raises(RobotConfigMismatchError) as exc:
        load_and_compile(scene)
    assert exc.value.kind == "joint"
    assert exc.value.name == "not_a_real_joint"
    assert "joint1" in exc.value.available  # real joints are still listed


def test_ee_site_inject_lands_at_correct_world_position(
    franka_urdf: Path, franka_config: Path
) -> None:
    """After compile + settle, the injected ee_site should sit 0.1034 m
    along the hand body's +z axis, relative to hand's world pose."""
    import numpy as np

    from robosandbox.sim.mujoco_backend import MuJoCoBackend

    scene = Scene(robot_urdf=franka_urdf, robot_config=franka_config)
    sim = MuJoCoBackend(render_size=(240, 320))
    sim.load(scene)
    for _ in range(80):
        sim.step()
    obs = sim.observe()
    ee_world = np.asarray(obs.ee_pose.xyz)
    # Grab hand body world pose via the sim model
    hand_xyz = np.asarray(sim.data.body("hand").xpos)
    hand_mat = np.asarray(sim.data.body("hand").xmat).reshape(3, 3)
    # Expected site world pos = hand_xyz + R_hand @ [0, 0, 0.1034]
    expected = hand_xyz + hand_mat @ np.array([0.0, 0.0, 0.1034])
    sim.close()
    assert np.allclose(ee_world, expected, atol=1e-6), (ee_world, expected)


def test_full_franka_scene_compiles_with_cube(
    franka_urdf: Path, franka_config: Path
) -> None:
    """End-to-end: build_model on a Franka scene with a cube produces a
    working MjModel with 7 arm joints + cube free joint + injected ee_site."""
    scene = Scene(
        robot_urdf=franka_urdf,
        robot_config=franka_config,
        objects=(
            SceneObject(
                id="red_cube",
                kind="box",
                size=(0.012, 0.012, 0.012),
                pose=Pose(xyz=(0.4, 0.0, 0.06)),
                mass=0.05,
                rgba=(0.85, 0.2, 0.2, 1.0),
            ),
        ),
    )
    model, spec = load_and_compile(scene)
    assert spec.base_body_name == "link0"
    # njnt = 7 arm + 2 finger + 1 free (cube) = 10
    assert model.njnt == 10
    assert model.nu == 8  # 7 arm actuators + 1 gripper
    assert model.nsite == 1  # robosandbox_ee_site
    # Cube body should be reachable by name
    assert model.body("red_cube").id > 0
