"""Build a self-contained MJCF from a Scene description.

v0.1 uses a built-in 6-DOF arm (no external meshes) so that the demo
runs with a single `pip install` and nothing else. Supplying a
`robot_urdf` is reserved for a future slice — the slot exists in the
Scene dataclass but is currently unused.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

from robosandbox.types import Scene, SceneObject


_ARM_XML = """\
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <light pos="1.5 -1 2" dir="-1 1 -1" diffuse="0.4 0.4 0.4"/>

    <!-- Fixed wrist-mounted camera looking at the workspace -->
    <camera name="scene" pos="0.9 -0.9 0.9" xyaxes="0.707 0.707 0 -0.4 0.4 0.82"/>
    <camera name="top" pos="0 0 1.2" xyaxes="1 0 0 0 1 0"/>

    <!-- Floor -->
    <geom name="floor" type="plane" size="2 2 0.05" rgba="0.9 0.9 0.9 1"/>

    <!-- Table -->
    <geom name="table" type="box" size="0.4 0.4 0.02" pos="0.1 0 0.02" rgba="0.72 0.55 0.37 1"/>

    <!-- Arm -->
    <body name="base" pos="-0.32 0 0.04">
      <geom type="cylinder" size="0.045 0.04" rgba="0.15 0.15 0.15 1"/>
      <body name="L1" pos="0 0 0.04">
        <joint name="j1" type="hinge" axis="0 0 1" range="-3.0 3.0"/>
        <geom type="cylinder" size="0.04 0.035" rgba="0.25 0.55 0.9 1"/>
        <body name="L2" pos="0 0 0.035">
          <joint name="j2" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.18" size="0.028" rgba="0.25 0.55 0.9 1"/>
          <body name="L3" pos="0 0 0.18">
            <joint name="j3" type="hinge" axis="0 1 0" range="-2.0 2.0"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.16" size="0.023" rgba="0.25 0.55 0.9 1"/>
            <body name="L4" pos="0 0 0.16">
              <joint name="j4" type="hinge" axis="0 1 0" range="-2.0 2.0"/>
              <geom type="capsule" fromto="0 0 0 0 0 0.08" size="0.02" rgba="0.25 0.55 0.9 1"/>
              <body name="L5" pos="0 0 0.08">
                <joint name="j5" type="hinge" axis="0 0 1" range="-3.0 3.0"/>
                <geom type="cylinder" size="0.02 0.02" rgba="0.25 0.55 0.9 1"/>
                <body name="L6" pos="0 0 0.025">
                  <joint name="j6" type="hinge" axis="0 1 0" range="-1.8 1.8"/>
                  <geom type="capsule" fromto="0 0 0 0 0 0.05" size="0.018" rgba="0.25 0.55 0.9 1"/>
                  <!-- Gripper.
                       ee_site sits at the mid-finger height: call that
                       the "grasp frame". Fingers are 4cm tall and the
                       fingertip lands 2cm below ee_site in world (when
                       the wrist is flipped 180°). Target ee_site at the
                       object grasp point; fingers wrap around it. -->
                  <body name="ee" pos="0 0 0.03">
                    <geom name="palm" type="box" size="0.030 0.045 0.010" rgba="0.1 0.1 0.1 1"/>
                    <site name="ee_site" pos="0 0 0.032" size="0.004" rgba="1 0 0 1"/>
                    <body name="left_finger" pos="-0.012 0 0.012">
                      <joint name="left_finger_joint" type="slide" axis="1 0 0" range="-0.035 0.0"/>
                      <geom type="box" size="0.006 0.008 0.02" pos="0.006 0 0.02" rgba="0.15 0.15 0.15 1" friction="2.0 0.1 0.01"/>
                    </body>
                    <body name="right_finger" pos="0.012 0 0.012">
                      <joint name="right_finger_joint" type="slide" axis="1 0 0" range="0.0 0.035"/>
                      <geom type="box" size="0.006 0.008 0.02" pos="-0.006 0 0.02" rgba="0.15 0.15 0.15 1" friction="2.0 0.1 0.01"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

__OBJECTS__

  </worldbody>

  <equality>
    <!-- Finger mimic: right_finger = -left_finger -->
    <joint joint1="right_finger_joint" joint2="left_finger_joint" polycoef="0 -1 0 0 0"/>
  </equality>

  <actuator>
    <position name="a1" joint="j1" kp="80" kv="5" ctrlrange="-3.0 3.0"/>
    <position name="a2" joint="j2" kp="120" kv="8" ctrlrange="-1.5 1.5"/>
    <position name="a3" joint="j3" kp="100" kv="6" ctrlrange="-2.0 2.0"/>
    <position name="a4" joint="j4" kp="80" kv="4" ctrlrange="-2.0 2.0"/>
    <position name="a5" joint="j5" kp="40" kv="2" ctrlrange="-3.0 3.0"/>
    <position name="a6" joint="j6" kp="30" kv="2" ctrlrange="-1.8 1.8"/>
    <position name="a_gripper" joint="left_finger_joint" kp="300" kv="15" ctrlrange="-0.035 0.0" forcerange="-20 20"/>
  </actuator>
"""


def _object_xml(obj: SceneObject) -> str:
    """Render one SceneObject as a free-body MJCF <body>."""
    x, y, z = obj.pose.xyz
    qx, qy, qz, qw = obj.pose.quat_xyzw
    r, g, b, a = obj.rgba
    rgba = f"{r} {g} {b} {a}"
    quat = f"{qw} {qx} {qy} {qz}"  # MuJoCo uses (w, x, y, z) ordering

    if obj.kind == "box":
        sx, sy, sz = obj.size
        geom = f'<geom type="box" size="{sx} {sy} {sz}" rgba="{rgba}" friction="1.5 0.1 0.01"/>'
    elif obj.kind == "sphere":
        (r_,) = obj.size
        geom = f'<geom type="sphere" size="{r_}" rgba="{rgba}" friction="1.5 0.1 0.01"/>'
    elif obj.kind == "cylinder":
        r_, h = obj.size
        geom = f'<geom type="cylinder" size="{r_} {h}" rgba="{rgba}" friction="1.5 0.1 0.01"/>'
    elif obj.kind == "mesh":
        # v0.1 placeholder — mesh import lands in v0.2
        raise NotImplementedError("mesh objects deferred to v0.2")
    else:
        raise ValueError(f"unknown SceneObject.kind: {obj.kind}")

    body_name = escape(obj.id)
    return (
        f'    <body name="{body_name}" pos="{x} {y} {z}" quat="{quat}">\n'
        f'      <freejoint/>\n'
        f'      {geom}\n'
        f'    </body>'
    )


def build_mjcf(scene: Scene) -> str:
    """Return a complete MJCF XML string for the given scene."""
    objects_xml = "\n".join(_object_xml(o) for o in scene.objects)
    gx, gy, gz = scene.gravity
    arm = _ARM_XML.replace("__OBJECTS__", objects_xml)
    return f"""<mujoco model="robosandbox">
  <compiler angle="radian" coordinate="local" autolimits="true"/>
  <option integrator="implicitfast" gravity="{gx} {gy} {gz}" timestep="0.005"/>

  <visual>
    <headlight active="0"/>
    <rgba haze="0.15 0.2 0.25 1"/>
  </visual>

  <default>
    <joint damping="0.5" armature="0.02"/>
    <geom condim="4"/>
  </default>

{arm}
</mujoco>
"""
