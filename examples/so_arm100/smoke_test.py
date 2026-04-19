"""Smoke-test the SO-ARM100 URDF import.

Exercises the full BYO-robot flow documented in
``docs/site/docs/guides/bring-your-own-robot.md``:

    1. Load the Menagerie MJCF + sidecar YAML into a Scene.
    2. Reachability pre-flight.
    3. Step the sim, verify joint names / DoF / gripper convention.

Run from the repo root::

    uv run python examples/so_arm100/smoke_test.py
"""
from __future__ import annotations

from pathlib import Path

from robosandbox.scene.reachability import check_scene_reachability, format_warnings
from robosandbox.sim.mujoco_backend import MuJoCoBackend
from robosandbox.types import Pose, Scene, SceneObject


HERE = Path(__file__).parent
ROBOT_XML = HERE / "so_arm100.xml"
ROBOT_YAML = HERE / "so_arm100.robosandbox.yaml"


def main() -> int:
    scene = Scene(
        robot_urdf=ROBOT_XML,
        robot_config=ROBOT_YAML,
        objects=(
            SceneObject(
                id="red_cube", kind="box",
                size=(0.012, 0.012, 0.012),
                # SO-ARM100's forward axis is the world -Y direction at home;
                # put the cube there so it sits inside the reach envelope.
                pose=Pose(xyz=(0.0, -0.25, 0.06)),
                rgba=(0.85, 0.2, 0.2, 1.0),
                mass=0.05,
            ),
        ),
    )

    print("== reachability pre-flight ==")
    print(format_warnings(check_scene_reachability(scene)))
    print()

    print("== sim load + settle ==")
    sim = MuJoCoBackend(render_size=(360, 480), camera="scene")
    sim.load(scene)
    for _ in range(100):
        sim.step()
    obs = sim.observe()
    print(f"n_dof:       {sim.n_dof}")
    print(f"joint_names: {sim.joint_names}")
    print(f"ee_pose:     ({obs.ee_pose.xyz[0]:+.3f}, {obs.ee_pose.xyz[1]:+.3f}, {obs.ee_pose.xyz[2]:+.3f})")
    print(f"gripper_w:   {obs.gripper_width:.4f}  (open_qpos → large; closed_qpos → ~0)")
    cube = obs.scene_objects["red_cube"].xyz
    print(f"red_cube:    ({cube[0]:+.3f}, {cube[1]:+.3f}, {cube[2]:+.3f})")
    print()

    print("== gripper sanity — command closed, then back to open ==")
    for _ in range(150):
        sim.step(gripper=1.0)  # closed (0 rad, jaws clamped)
    closed_w = sim.observe().gripper_width
    for _ in range(150):
        sim.step(gripper=0.0)  # open (1.5 rad, jaws spread)
    open_w = sim.observe().gripper_width
    print(f"closed:      {closed_w:.4f}")
    print(f"open:        {open_w:.4f}   (should be noticeably larger than closed)")
    sim.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
