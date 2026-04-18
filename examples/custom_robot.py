"""Load any URDF/MJCF robot via Scene(robot_urdf=...) + sidecar YAML.

RoboSandbox ships the Franka Panda pre-configured; this example shows
how the same pattern works for an arbitrary robot — drop a URDF on
disk, write a small sidecar next to it, done.

Run:
    uv run python examples/custom_robot.py
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from robosandbox.scene.robot_loader import load_robot
from robosandbox.types import Scene


def main() -> None:
    # These paths point at the bundled Franka; swap to your own URDF + sidecar
    # (e.g. UR5, SO-101) and the rest of the stack works unchanged.
    urdf = Path(str(files("robosandbox").joinpath("assets/robots/franka_panda/panda.xml")))
    cfg = Path(
        str(files("robosandbox").joinpath("assets/robots/franka_panda/panda.robosandbox.yaml"))
    )

    _, robot_spec = load_robot(urdf, cfg)
    print(f"Loaded robot from {urdf.name}")
    print(f"  arm joints:       {robot_spec.arm_joint_names}")
    print(f"  arm actuators:    {robot_spec.arm_actuator_names}")
    print(f"  gripper primary:  {robot_spec.gripper_primary_joint}")
    print(f"  end-effector site: {robot_spec.ee_site_name}")
    print(f"  home qpos:        {robot_spec.home_qpos}")

    # Using the robot in a Scene is the same as any other scene, just with
    # robot_urdf / robot_config set.
    scene = Scene(robot_urdf=urdf, robot_config=cfg)
    print(f"\nReady: Scene(robot_urdf={urdf.name}, robot_config={cfg.name})")
    print(f"Objects: {len(scene.objects)} (add SceneObject(...) tuples to populate)")

    # Sidecar schema reference:
    ref = Path(
        str(files("robosandbox").joinpath("assets/robots/franka_panda/panda.robosandbox.yaml"))
    )
    print(f"\nSidecar schema example: {ref}")


if __name__ == "__main__":
    main()
