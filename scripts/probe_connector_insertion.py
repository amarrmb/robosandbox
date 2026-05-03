"""Probe: chamfered connector-insertion physics.

Phase 0 of the connector-insertion plan. Tests whether the solver can keep
a rigid plug from penetrating chamfer/wall geometry when commanded to insert.
This is the precondition for any RL training on this task.

Approach:
  * 6-DoF 'carrier' articulation (3 slide + 3 hinge joints) drives a plug.
  * Port = base + 4 walls + 4 chamfer plates, all static box geoms.
  * Three clearance variants: 1.5mm (USB-A), 4mm (USB-C class), 8mm (barrel-jack).
  * Run a scripted approach + descent + insert; measure penetration, settled
    depth, and oscillation per clearance.
  * Run in classical MuJoCo first (laptop sanity); Newton port follows once
    classical works.

Usage:
    python3 scripts/probe_connector_insertion.py --clearance-mm 4.0 --backend mujoco
    python3 scripts/probe_connector_insertion.py --backend newton  # runs all 3 clearances
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Plug half-extents (m). 4cm × 1cm × 0.5cm thick plug — generic connector-shaped.
PLUG_HX = 0.020
PLUG_HY = 0.005
PLUG_HZ = 0.0025

# Port geometry
WALL_T = 0.005      # 5mm wall thickness
WALL_H = 0.025      # 2.5cm tall walls (= half-extent in z)
CHAMFER_LEN = 0.006  # 6mm chamfer plates
PORT_BASE_Z = 0.0   # port base sits on z=0

# Carrier home pose: 20cm above port top
CARRIER_HOME_Z = 0.20


def build_mjcf(clearance_mm: float, with_chamfer: bool = True) -> str:
    """Construct a probe MJCF with a 6-DoF carrier + plug + chamfered port."""
    clearance = clearance_mm / 1000.0  # mm to m

    # Hole inner half-extents = plug half + clearance/2 per side
    hole_hx = PLUG_HX + clearance / 2.0
    hole_hy = PLUG_HY + clearance / 2.0

    outer_hx = hole_hx + WALL_T
    outer_hy = hole_hy + WALL_T

    port_top_z = PORT_BASE_Z + 2 * WALL_H

    # Chamfer plate geometry: 45° angled boxes above each wall, lead-in 6mm
    # Each chamfer plate is wall_long × thin × CHAMFER_LEN, rotated 45° outward.
    # Geom is a thin slab; we tilt it by 45° around the appropriate axis so its
    # inner edge lies on the wall top inner corner and its outer edge extends
    # outward + upward at 45°.
    chamfer_thick = 0.0015  # 1.5mm thin
    cosA = np.cos(np.pi / 4)
    sinA = np.sin(np.pi / 4)
    chamf_half_len = CHAMFER_LEN / 2.0

    # Center of each chamfer plate: at wall top inner corner + (CHAMFER_LEN/2) outward+up at 45°
    # +x chamfer plate
    cxp_center_x = hole_hx + chamf_half_len * cosA
    cxp_center_z = port_top_z + chamf_half_len * sinA
    # -x chamfer plate (mirror)
    cxn_center_x = -cxp_center_x

    # +y chamfer plate
    cyp_center_y = hole_hy + chamf_half_len * cosA
    cyp_center_z = port_top_z + chamf_half_len * sinA
    cyn_center_y = -cyp_center_y

    # Quaternions for 45° rotations (MuJoCo: w x y z order)
    # x-side chamfers: rotate around y-axis by -45° (so inner edge stays anchored, outer goes up+out)
    # +x side: outer at +x, inner at -x. Tilt outer up = rotate around y by -45°
    #   q = (cos(-22.5°), 0, sin(-22.5°), 0) = (cos(22.5°), 0, -sin(22.5°), 0)
    qcos = np.cos(np.pi / 8)
    qsin = np.sin(np.pi / 8)
    # x+: rotate -45° around y
    q_xp = f"{qcos} 0 {-qsin} 0"
    # x-: rotate +45° around y
    q_xn = f"{qcos} 0 {qsin} 0"
    # y+: rotate +45° around x (so outer goes +y up)
    q_yp = f"{qcos} {qsin} 0 0"
    # y-: rotate -45° around x
    q_yn = f"{qcos} {-qsin} 0 0"

    chamfer_xml = ""
    if with_chamfer:
        chamfer_xml = f"""
        <!-- Chamfer plates: 45° lead-in above each wall -->
        <geom name="chamfer_xp" type="box"
              pos="{cxp_center_x} 0 {cxp_center_z}"
              size="{chamf_half_len} {outer_hy} {chamfer_thick}"
              quat="{q_xp}" rgba="0.65 0.65 0.65 1"/>
        <geom name="chamfer_xn" type="box"
              pos="{cxn_center_x} 0 {cxp_center_z}"
              size="{chamf_half_len} {outer_hy} {chamfer_thick}"
              quat="{q_xn}" rgba="0.65 0.65 0.65 1"/>
        <geom name="chamfer_yp" type="box"
              pos="0 {cyp_center_y} {cyp_center_z}"
              size="{outer_hx} {chamf_half_len} {chamfer_thick}"
              quat="{q_yp}" rgba="0.65 0.65 0.65 1"/>
        <geom name="chamfer_yn" type="box"
              pos="0 {cyn_center_y} {cyp_center_z}"
              size="{outer_hx} {chamf_half_len} {chamfer_thick}"
              quat="{q_yn}" rgba="0.65 0.65 0.65 1"/>
"""

    return textwrap.dedent(f"""\
    <mujoco model="probe_insertion">
      <option integrator="implicitfast" gravity="0 0 -9.81" timestep="0.002"/>
      <default>
        <joint damping="2.0" armature="0.01"/>
        <geom condim="4" friction="1.0 0.05 0.001"/>
      </default>
      <worldbody>
        <light pos="0.5 0.5 1" dir="-0.3 -0.3 -1" diffuse="0.8 0.8 0.8"/>
        <camera name="scene" pos="0.25 -0.25 0.25" xyaxes="0.707 0.707 0 -0.4 0.4 0.82" fovy="60"/>

        <!-- Floor for visual reference -->
        <geom name="floor" type="plane" size="1 1 0.01" pos="0 0 -0.05" rgba="0.9 0.9 0.9 1"/>

        <!-- Carrier root: a tiny static body anchored at CARRIER_HOME_Z. Necessary
             because Newton's add_mjcf treats the very first joint of the world's
             child as a root joint (different DoF accounting), causing a
             joint_target_pos/joint_count mismatch downstream. Anchoring the chain
             to a static parent makes all 6 carrier joints uniform. -->
        <body name="carrier_root" pos="0 0 {CARRIER_HOME_Z}">
          <inertial pos="0 0 0" mass="0.0001" diaginertia="1e-8 1e-8 1e-8"/>
          <!-- 6-DoF carrier driving the plug. Each intermediate body needs
               a tiny inertia or MuJoCo refuses to compile (massless moving body). -->
          <body name="carrier_x">
          <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>
          <joint name="cx" type="slide" axis="1 0 0" range="-0.3 0.3"/>
          <body name="carrier_y">
            <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>
            <joint name="cy" type="slide" axis="0 1 0" range="-0.3 0.3"/>
            <body name="carrier_z">
              <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>
              <joint name="cz" type="slide" axis="0 0 1" range="-0.25 0.05"/>
              <body name="carrier_yaw">
                <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>
                <joint name="cyaw" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
                <body name="carrier_pitch">
                  <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>
                  <joint name="cpitch" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
                  <body name="carrier_roll">
                    <joint name="croll" type="hinge" axis="1 0 0" range="-1.5 1.5"/>
                    <geom name="plug" type="box"
                          size="{PLUG_HX} {PLUG_HY} {PLUG_HZ}"
                          rgba="0.2 0.6 1.0 1" mass="0.05"/>
                    <site name="plug_tip" pos="0 0 {-PLUG_HZ}" size="0.001" rgba="1 0 0 1"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
          </body>
        </body>

        <!-- Port base -->
        <geom name="port_base" type="box"
              pos="0 0 {PORT_BASE_Z + WALL_T/2}"
              size="{outer_hx} {outer_hy} {WALL_T/2}"
              rgba="0.4 0.4 0.4 1"/>

        <!-- Port walls (4 sides, hollow center) -->
        <geom name="port_wall_xp" type="box"
              pos="{hole_hx + WALL_T/2} 0 {PORT_BASE_Z + WALL_H}"
              size="{WALL_T/2} {outer_hy} {WALL_H}"
              rgba="0.5 0.5 0.5 1"/>
        <geom name="port_wall_xn" type="box"
              pos="{-(hole_hx + WALL_T/2)} 0 {PORT_BASE_Z + WALL_H}"
              size="{WALL_T/2} {outer_hy} {WALL_H}"
              rgba="0.5 0.5 0.5 1"/>
        <geom name="port_wall_yp" type="box"
              pos="0 {hole_hy + WALL_T/2} {PORT_BASE_Z + WALL_H}"
              size="{hole_hx} {WALL_T/2} {WALL_H}"
              rgba="0.5 0.5 0.5 1"/>
        <geom name="port_wall_yn" type="box"
              pos="0 {-(hole_hy + WALL_T/2)} {PORT_BASE_Z + WALL_H}"
              size="{hole_hx} {WALL_T/2} {WALL_H}"
              rgba="0.5 0.5 0.5 1"/>
        {chamfer_xml}
      </worldbody>

      <actuator>
        <position name="ax"     joint="cx"     kp="500"  kv="50"  ctrlrange="-0.2 0.2"/>
        <position name="ay"     joint="cy"     kp="500"  kv="50"  ctrlrange="-0.2 0.2"/>
        <position name="az"     joint="cz"     kp="1000" kv="100" ctrlrange="-0.25 0.05"/>
        <position name="ayaw"   joint="cyaw"   kp="100"  kv="10"  ctrlrange="-3.14 3.14"/>
        <position name="apitch" joint="cpitch" kp="100"  kv="10"  ctrlrange="-1.5 1.5"/>
        <position name="aroll"  joint="croll"  kp="100"  kv="10"  ctrlrange="-1.5 1.5"/>
      </actuator>
    </mujoco>
    """)


@dataclass
class InsertionResult:
    clearance_mm: float
    backend: str
    settled_z_mm: float       # plug-tip z relative to port top, after final settle
    commanded_z_mm: float     # commanded z target relative to port top
    max_penetration_mm: float # max negative tip z below the wall geometry the solver allowed
    oscillation_mm: float     # peak-to-peak z range over last 100 steps
    success: bool             # tip ≥ 5mm past port plane AND oscillation < 1mm
    raw_trace: np.ndarray | None = None  # (T, 3) tip xyz over time, optional


def run_classical_mujoco(clearance_mm: float, with_chamfer: bool = True,
                          n_settle: int = 200, n_descend: int = 600,
                          n_final_settle: int = 200) -> InsertionResult:
    """Run probe in classical MuJoCo. Returns measurement summary."""
    import mujoco

    xml = build_mjcf(clearance_mm, with_chamfer=with_chamfer)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    plug_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "plug")
    plug_tip_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "plug_tip")

    # Port top z in world frame = PORT_BASE_Z + 2*WALL_H
    port_top_z_world = PORT_BASE_Z + 2 * WALL_H

    # Phase 1 — settle at home (carrier_z = 0 means plug is at CARRIER_HOME_Z)
    data.ctrl[:] = 0.0
    for _ in range(n_settle):
        mujoco.mj_step(model, data)

    # Phase 2 — descend: command carrier_z to drive plug-tip to (port_top_z - 8mm)
    # carrier base is at world z = CARRIER_HOME_Z; plug center sits at carrier_z position
    # plug tip = plug center - PLUG_HZ
    # We want plug tip at port_top_z_world - 0.008 (i.e. 8mm into the port)
    # carrier z (relative joint coord) such that:
    #   CARRIER_HOME_Z + carrier_z - PLUG_HZ = port_top_z_world - 0.008
    target_tip_world_z = port_top_z_world - 0.008  # 8mm past port plane
    target_carrier_z = target_tip_world_z + PLUG_HZ - CARRIER_HOME_Z
    target_carrier_z = max(target_carrier_z, -0.20)  # respect joint range
    commanded_descent_mm = (port_top_z_world - target_tip_world_z) * 1000.0
    # Actually we want commanded_z_mm = how far past port plane we asked for, NEGATIVE if past
    commanded_z_past_plane_mm = (target_tip_world_z - port_top_z_world) * 1000.0  # negative = inserted
    # Linear ramp the z target to avoid step impulse
    trace = []
    for k in range(n_descend):
        alpha = (k + 1) / n_descend
        data.ctrl[2] = alpha * target_carrier_z  # az is index 2
        mujoco.mj_step(model, data)
        tip_pos = data.site_xpos[plug_tip_site].copy()
        trace.append(tip_pos)

    # Phase 3 — final settle at target, hold
    final_zs = []
    for _ in range(n_final_settle):
        data.ctrl[2] = target_carrier_z
        mujoco.mj_step(model, data)
        tip_pos = data.site_xpos[plug_tip_site].copy()
        trace.append(tip_pos)
        final_zs.append(tip_pos[2])

    final_zs = np.array(final_zs)
    settled_z_mm = (final_zs[-50:].mean() - port_top_z_world) * 1000.0  # negative = inserted
    osc_mm = (final_zs[-100:].max() - final_zs[-100:].min()) * 1000.0

    # Penetration tracking: scan trace for any tip below port_base_z (= 0)
    # If tip ever went below z=0 (the bottom of the receptacle), that's penetration through base.
    # If wall_xp etc walls are reached but tip y or x is outside hole, that's wall penetration.
    trace_arr = np.array(trace)
    # Wall penetration: tip x outside ±hole_hx but tip z below port top → it sliced through wall
    hole_hx_local = PLUG_HX + (clearance_mm / 1000.0) / 2.0
    hole_hy_local = PLUG_HY + (clearance_mm / 1000.0) / 2.0
    below_top_mask = trace_arr[:, 2] < port_top_z_world
    outside_hole_mask = ((np.abs(trace_arr[:, 0]) > hole_hx_local + 0.001) |
                         (np.abs(trace_arr[:, 1]) > hole_hy_local + 0.001))
    wall_penetration_mask = below_top_mask & outside_hole_mask
    if wall_penetration_mask.any():
        # Maximum depth tip went into wall region
        max_pen_depth_mm = (port_top_z_world - trace_arr[wall_penetration_mask, 2].min()) * 1000.0
    else:
        max_pen_depth_mm = 0.0

    # Success: tip went ≥ 5mm past port plane AND oscillation < 1mm
    success = (settled_z_mm <= -5.0) and (osc_mm < 1.0) and (max_pen_depth_mm < 0.5)

    return InsertionResult(
        clearance_mm=clearance_mm,
        backend="mujoco",
        settled_z_mm=settled_z_mm,
        commanded_z_mm=commanded_z_past_plane_mm,
        max_penetration_mm=max_pen_depth_mm,
        oscillation_mm=osc_mm,
        success=success,
        raw_trace=trace_arr,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clearance-mm", type=float, default=None,
                        help="Single clearance to test (mm). If unset, sweeps 1.5/4/8.")
    parser.add_argument("--backend", choices=["mujoco", "newton", "both"], default="mujoco")
    parser.add_argument("--no-chamfer", action="store_true",
                        help="Disable chamfer plates (test bare wall edge)")
    parser.add_argument("--save-mjcf", type=str, default=None,
                        help="Path to save the generated MJCF for visual inspection")
    args = parser.parse_args()

    if args.save_mjcf:
        clear = args.clearance_mm if args.clearance_mm else 4.0
        Path(args.save_mjcf).write_text(build_mjcf(clear, with_chamfer=not args.no_chamfer))
        print(f"[probe] wrote MJCF to {args.save_mjcf} (clearance={clear}mm)")
        return 0

    clearances = [args.clearance_mm] if args.clearance_mm else [8.0, 4.0, 1.5]

    results = []
    for c in clearances:
        if args.backend in ("mujoco", "both"):
            r = run_classical_mujoco(c, with_chamfer=not args.no_chamfer)
            results.append(r)
            print(f"[mujoco] clearance={c:>5.2f}mm  "
                  f"commanded={r.commanded_z_mm:+.1f}mm  "
                  f"settled={r.settled_z_mm:+6.2f}mm  "
                  f"osc={r.oscillation_mm:5.2f}mm  "
                  f"penetration={r.max_penetration_mm:5.2f}mm  "
                  f"success={'YES' if r.success else 'NO '}")
        if args.backend in ("newton", "both"):
            print(f"[newton] clearance={c}mm  -- newton path not yet wired (needs DGX)")

    print()
    print("Summary (classical MuJoCo):")
    for r in results:
        verdict = "OK" if r.success else "FAIL"
        print(f"  {r.clearance_mm:>5.2f}mm clearance: {verdict} "
              f"(settled {r.settled_z_mm:+.2f}mm, osc {r.oscillation_mm:.2f}mm, "
              f"penetration {r.max_penetration_mm:.2f}mm)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
