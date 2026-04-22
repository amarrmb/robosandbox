"""Newton cloth showcase for RoboSandbox demos.

This keeps Newton's upstream Franka + cloth scene geometry intact for the 1x
demo, then replicates that same world physically for the 10x demo so the
viewer shows coherent cloth, tables, and robot poses.
"""

from __future__ import annotations

import argparse
import copy

import numpy as np

try:
    from pxr import Usd
except ImportError as e:
    raise ImportError("This showcase requires USD support (`pxr`) in the active environment") from e

import warp as wp

import newton
import newton.examples
import newton.usd
import newton.utils
from newton import ModelBuilder, eval_fk
from newton.solvers import SolverFeatherstone, SolverVBD
from newton.viewer import ViewerNull, ViewerViser


@wp.kernel
def scale_positions(src: wp.array[wp.vec3], scale: float, dst: wp.array[wp.vec3]):
    i = wp.tid()
    dst[i] = src[i] * scale


@wp.kernel
def scale_body_transforms(src: wp.array[wp.transform], scale: float, dst: wp.array[wp.transform]):
    i = wp.tid()
    p = wp.transform_get_translation(src[i])
    q = wp.transform_get_rotation(src[i])
    dst[i] = wp.transform(p * scale, q)


@wp.kernel
def compute_ee_delta(
    body_q: wp.array[wp.transform],
    offset: wp.transform,
    body_id: int,
    bodies_per_world: int,
    targets: wp.array[wp.transform],
    ee_delta: wp.array[wp.spatial_vector],
):
    world_id = wp.tid()
    tf = body_q[bodies_per_world * world_id + body_id] * offset
    pos = wp.transform_get_translation(tf)
    pos_des = wp.transform_get_translation(targets[world_id])
    pos_diff = pos_des - pos
    rot = wp.transform_get_rotation(tf)
    rot_des = wp.transform_get_rotation(targets[world_id])
    ang_diff = rot_des * wp.quat_inverse(rot)
    ee_delta[world_id] = wp.spatial_vector(pos_diff[0], pos_diff[1], pos_diff[2], ang_diff[0], ang_diff[1], ang_diff[2])


class ClothFrankaShowcase:
    def __init__(self, viewer, args: argparse.Namespace):
        self.viewer = viewer
        self.world_count = int(args.world_count)
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.iterations = 5

        self.viz_scale = 0.01

        self.cloth_particle_radius = 0.8
        self.cloth_body_contact_margin = 0.8
        self.particle_self_contact_radius = 0.2
        self.particle_self_contact_margin = 0.2
        self.soft_contact_ke = 1e4
        self.soft_contact_kd = 1e-2
        self.robot_contact_ke = 5e4
        self.robot_contact_kd = 1e-3
        self.robot_contact_mu = 1.5
        self.self_contact_friction = 0.25
        self.tri_ke = 1e4
        self.tri_ka = 1e4
        self.tri_kd = 1.5e-6
        self.bending_ke = 5
        self.bending_kd = 1e-2
        self.table_hx_cm = 40.0
        self.table_hy_cm = 40.0
        self.table_hz_cm = 10.0
        self.table_pos_cm = np.array([0.0, -50.0, 10.0], dtype=np.float32)

        world_builder = self._build_world()
        single_world_builder = copy.deepcopy(world_builder)
        self.bodies_per_world = world_builder.body_count
        self.dof_q_per_world = world_builder.joint_coord_count
        self.dof_qd_per_world = world_builder.joint_dof_count

        if self.world_count > 1:
            scene = ModelBuilder(gravity=-981.0)
            self.world_offsets_cm = self._demo_world_offsets(self.world_count)
            for world_offset in self.world_offsets_cm:
                scene.add_world(
                    world_builder,
                    xform=wp.transform(wp.vec3(*world_offset), wp.quat_identity()),
                )
            scene.color()
            self.model = scene.finalize(requires_grad=False)
        else:
            self.world_offsets_cm = [(0.0, 0.0, 0.0)]
            self.model = world_builder.finalize(requires_grad=False)

        self.table_viz_scale = (
            self.table_hx_cm * self.viz_scale,
            self.table_hy_cm * self.viz_scale,
            self.table_hz_cm * self.viz_scale,
        )
        self.table_viz_color = wp.array([wp.vec3(0.82, 0.72, 0.42)] * self.world_count, dtype=wp.vec3)
        self.table_viz_xform = wp.array(
            [
                wp.transform(
                    (
                        float(offset[0] + self.table_pos_cm[0]) * self.viz_scale,
                        float(offset[1] + self.table_pos_cm[1]) * self.viz_scale,
                        float(offset[2] + self.table_pos_cm[2]) * self.viz_scale,
                    ),
                    wp.quat_identity(),
                )
                for offset in self.world_offsets_cm
            ],
            dtype=wp.transform,
        )

        self.model.soft_contact_ke = self.soft_contact_ke
        self.model.soft_contact_kd = self.soft_contact_kd
        self.model.soft_contact_mu = self.self_contact_friction

        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu = self.model.shape_material_mu.numpy()
        shape_ke[...] = self.robot_contact_ke
        shape_kd[...] = self.robot_contact_kd
        shape_mu[...] = self.robot_contact_mu
        self.model.shape_material_ke = wp.array(
            shape_ke, dtype=self.model.shape_material_ke.dtype, device=self.model.shape_material_ke.device
        )
        self.model.shape_material_kd = wp.array(
            shape_kd, dtype=self.model.shape_material_kd.dtype, device=self.model.shape_material_kd.device
        )
        self.model.shape_material_mu = wp.array(
            shape_mu, dtype=self.model.shape_material_mu.dtype, device=self.model.shape_material_mu.device
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.viz_state = self.model.state()
        self.control = self.model.control()
        self.target_joint_qd = wp.zeros(
            self.model.joint_dof_count,
            dtype=self.state_0.joint_qd.dtype,
            device=self.state_0.joint_qd.device,
        )

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_margin=self.cloth_body_contact_margin,
        )
        self.contacts = self.collision_pipeline.contacts()
        self.robot_solver = SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)
        self.cloth_solver = SolverVBD(
            self.model,
            iterations=self.iterations,
            integrate_with_external_rigid_solver=True,
            particle_self_contact_radius=self.particle_self_contact_radius,
            particle_self_contact_margin=self.particle_self_contact_margin,
            particle_topological_contact_filter_threshold=1,
            particle_rest_shape_contact_exclusion_radius=0.5,
            particle_enable_self_contact=True,
            particle_vertex_contact_buffer_size=16,
            particle_edge_contact_buffer_size=20,
            particle_collision_detection_interval=-1,
            rigid_contact_k_start=self.soft_contact_ke,
        )

        self.viewer.set_model(self.model)
        if self.world_count > 1 and hasattr(self.viewer, "set_world_offsets"):
            self.viewer.set_world_offsets((0.0, 0.0, 0.0))
        if hasattr(self.viewer, "set_camera"):
            if self.world_count == 1:
                self.viewer.set_camera(wp.vec3(-0.6, 0.6, 1.24), -42.0, -58.0)
            else:
                self.viewer.set_camera(wp.vec3(-0.1, 1.75, 2.15), -31.0, -88.0)

        self.sim_shape_transform = self.model.shape_transform
        self.sim_shape_scale = self.model.shape_scale
        xform_np = self.model.shape_transform.numpy().copy()
        xform_np[:, :3] *= self.viz_scale
        self.viz_shape_transform = wp.array(xform_np, dtype=wp.transform, device=self.model.device)
        scale_np = self.model.shape_scale.numpy().copy()
        scale_np *= self.viz_scale
        self.viz_shape_scale = wp.array(scale_np, dtype=wp.vec3, device=self.model.device)

        if hasattr(self.viewer, "_shape_instances"):
            for shapes in self.viewer._shape_instances.values():
                xi = shapes.xforms.numpy()
                xi[:, :3] *= self.viz_scale
                shapes.xforms = wp.array(xi, dtype=wp.transform, device=shapes.device)
                sc = shapes.scales.numpy()
                sc *= self.viz_scale
                shapes.scales = wp.array(sc, dtype=wp.vec3, device=shapes.device)

        self.gravity_zero = wp.zeros(1, dtype=wp.vec3)
        self.gravity_cm = wp.array(wp.vec3(0.0, 0.0, -981.0), dtype=wp.vec3)

        self.model_single = single_world_builder.finalize(requires_grad=False)
        self._setup_control()
        eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

    def _build_world(self) -> ModelBuilder:
        scene = ModelBuilder(gravity=-981.0)
        self._create_robot(scene)

        scene.add_shape_box(
            -1,
            wp.transform(wp.vec3(*self.table_pos_cm), wp.quat_identity()),
            hx=self.table_hx_cm,
            hy=self.table_hy_cm,
            hz=self.table_hz_cm,
        )

        usd_stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
        usd_prim = usd_stage.GetPrimAtPath("/root/shirt")
        shirt_mesh = newton.usd.get_mesh(usd_prim)
        vertices = [wp.vec3(v) for v in shirt_mesh.vertices]
        scene.add_cloth_mesh(
            vertices=vertices,
            indices=shirt_mesh.indices,
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi),
            pos=wp.vec3(5.0, 87.0, 30.0),
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            scale=1.0,
            tri_ke=self.tri_ke,
            tri_ka=self.tri_ka,
            tri_kd=self.tri_kd,
            edge_ke=self.bending_ke,
            edge_kd=self.bending_kd,
            particle_radius=self.cloth_particle_radius,
        )
        scene.color()
        scene.add_ground_plane()
        return scene

    def _demo_world_offsets(self, world_count: int) -> list[tuple[float, float, float]]:
        if world_count == 10:
            xs = [-320.0, -160.0, 0.0, 160.0, 320.0]
            ys = [-120.0, 120.0]
            return [(x, y, 0.0) for y in ys for x in xs]

        side = int(np.ceil(np.sqrt(world_count)))
        spacing_x = 160.0
        spacing_y = 240.0
        offsets: list[tuple[float, float, float]] = []
        for i in range(world_count):
            row = i // side
            col = i % side
            x = (col - (side - 1) * 0.5) * spacing_x
            y = (row - (side - 1) * 0.5) * spacing_y
            offsets.append((x, y, 0.0))
        return offsets

    def _create_robot(self, builder: ModelBuilder) -> None:
        asset_path = newton.utils.download_asset("franka_emika_panda")
        builder.add_urdf(
            str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
            xform=wp.transform((-50.0, -50.0, 0.0), wp.quat_identity()),
            floating=False,
            scale=100.0,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        builder.joint_q[:6] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307]

        open_val = 0.8
        close_val = 0.1
        self.robot_key_poses = np.array(
            [
                [4.0, 31.0, -60.0, 40.0, 0.8536, -0.3536, 0.3536, -0.1464, open_val],
                [2.0, 31.0, -60.0, 20.0, 0.8536, -0.3536, 0.3536, -0.1464, open_val],
                [2.0, 31.0, -60.0, 20.0, 0.8536, -0.3536, 0.3536, -0.1464, close_val],
                [2.0, 26.0, -60.0, 26.0, 0.8536, -0.3536, 0.3536, -0.1464, close_val],
                [2.0, 12.0, -60.0, 31.0, 0.8536, -0.3536, 0.3536, -0.1464, close_val],
                [3.0, -6.0, -60.0, 31.0, 0.8536, -0.3536, 0.3536, -0.1464, close_val],
                [1.0, -6.0, -60.0, 31.0, 0.8536, -0.3536, 0.3536, -0.1464, open_val],
                [2.0, 15.0, -33.0, 31.0, 0.8536, -0.3536, 0.3536, -0.1464, open_val],
                [3.0, 15.0, -33.0, 21.0, 0.8536, -0.3536, 0.3536, -0.1464, open_val],
                [3.0, 15.0, -33.0, 21.0, 0.8536, -0.3536, 0.3536, -0.1464, close_val],
                [2.0, 15.0, -33.0, 28.0, 0.8536, -0.3536, 0.3536, -0.1464, close_val],
                [3.0, -2.0, -33.0, 28.0, 0.8536, -0.3536, 0.3536, -0.1464, close_val],
                [1.0, -2.0, -33.0, 28.0, 0.8536, -0.3536, 0.3536, -0.1464, open_val],
                [2.0, -28.0, -60.0, 28.0, 0.9239, -0.3827, 0.0, 0.0, open_val],
                [2.0, -28.0, -60.0, 20.0, 0.9239, -0.3827, 0.0, 0.0, open_val],
                [2.0, -28.0, -60.0, 20.0, 0.9239, -0.3827, 0.0, 0.0, close_val],
                [2.0, -18.0, -60.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, close_val],
                [3.0, 5.0, -60.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, close_val],
                [1.0, 5.0, -60.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, open_val],
                [3.0, -18.0, -30.0, 20.5, 0.9239, -0.3827, 0.0, 0.0, open_val],
                [3.0, -18.0, -30.0, 20.5, 0.9239, -0.3827, 0.0, 0.0, close_val],
                [2.0, -3.0, -30.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, close_val],
                [3.0, -3.0, -30.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, close_val],
                [2.0, -3.0, -30.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, open_val],
                [2.0, 0.0, -20.0, 30.0, 0.9239, -0.3827, 0.0, 0.0, open_val],
                [2.0, 0.0, -20.0, 19.5, 0.9239, -0.3827, 0.0, 0.0, open_val],
                [2.0, 0.0, -20.0, 19.5, 0.9239, -0.3827, 0.0, 0.0, close_val],
                [2.0, 0.0, -20.0, 35.0, 0.9239, -0.3827, 0.0, 0.0, close_val],
                [1.0, 0.0, -30.0, 35.0, 0.9239, -0.3827, 0.0, 0.0, close_val],
                [1.5, 0.0, -30.0, 35.0, 0.9239, -0.3827, 0.0, 0.0, close_val],
                [1.5, 0.0, -40.0, 35.0, 0.9239, -0.3827, 0.0, 0.0, close_val],
                [1.5, 0.0, -40.0, 35.0, 0.9239, -0.3827, 0.0, 0.0, open_val],
                [2.0, -28.0, -60.0, 28.0, 0.9239, -0.3827, 0.0, 0.0, open_val],
            ],
            dtype=np.float32,
        )
        self.targets = self.robot_key_poses[:, 1:]
        self.transition_duration = self.robot_key_poses[:, 0]
        self.robot_key_poses_time = np.cumsum(self.robot_key_poses[:, 0])
        self.endeffector_id = builder.body_count - 3
        self.endeffector_offset = wp.transform((0.0, 0.0, 22.0), wp.quat_identity())
        self.initial_pose = np.array(builder.joint_q[: builder.joint_coord_count], dtype=np.float32)

    def _setup_control(self) -> None:
        out_dim = 6
        self.jac_state = self.model_single.state(requires_grad=True)
        self.body_out = wp.empty(out_dim, dtype=float, requires_grad=True)
        self.Jacobian_one_hots = [
            wp.array([1.0 if j == i else 0.0 for j in range(out_dim)], dtype=float)
            for i in range(out_dim)
        ]
        self.J_flat = wp.empty(out_dim * self.dof_qd_per_world, dtype=float)
        self.ee_delta = wp.empty(self.world_count, dtype=wp.spatial_vector)
        self.ee_targets = wp.empty(self.world_count, dtype=wp.transform)

        @wp.kernel
        def compute_body_out(
            body_q: wp.array[wp.transform],
            body_qd: wp.array[wp.spatial_vector],
            body_com: wp.array[wp.vec3],
            body_out: wp.array[float],
        ):
            ee_id = wp.static(self.endeffector_id)
            ee_offset = wp.static(wp.vec3(0.0, 0.0, 22.0))
            X_wb = body_q[ee_id]
            r_world = wp.transform_vector(X_wb, ee_offset - body_com[ee_id])
            qd = body_qd[ee_id]
            omega = wp.spatial_bottom(qd)
            v_com = wp.spatial_top(qd)
            v_tip = v_com + wp.cross(omega, r_world)
            body_out[0] = v_tip[0]
            body_out[1] = v_tip[1]
            body_out[2] = v_tip[2]
            body_out[3] = omega[0]
            body_out[4] = omega[1]
            body_out[5] = omega[2]

        self.compute_body_out_kernel = compute_body_out

    def _compute_body_jacobian(self, joint_q_np: np.ndarray, joint_qd_np: np.ndarray) -> np.ndarray:
        joint_q = wp.array(joint_q_np, dtype=wp.float32, requires_grad=True)
        joint_qd = wp.array(joint_qd_np, dtype=wp.float32, requires_grad=True)
        tape = wp.Tape()
        with tape:
            eval_fk(self.model_single, joint_q, joint_qd, self.jac_state)
            wp.launch(
                self.compute_body_out_kernel,
                dim=1,
                inputs=[self.jac_state.body_q, self.jac_state.body_qd, self.model_single.body_com],
                outputs=[self.body_out],
            )
        for i in range(6):
            tape.backward(grads={self.body_out: self.Jacobian_one_hots[i]})
            wp.copy(self.J_flat[i * self.dof_qd_per_world : (i + 1) * self.dof_qd_per_world], joint_qd.grad)
            tape.zero()
        return self.J_flat.numpy().reshape(6, self.dof_qd_per_world)

    def _target_transforms(self, target_row: np.ndarray) -> wp.array:
        targets = [wp.transform(*target_row[:7]) for _ in range(self.world_count)]
        self.ee_targets.assign(targets)
        return self.ee_targets

    def _solve_damped_least_squares(self, J: np.ndarray, delta_target: np.ndarray) -> np.ndarray:
        J = np.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)
        delta_target = np.nan_to_num(delta_target, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)
        damping = 1e-3
        lhs = J.T @ J + damping * np.eye(J.shape[1], dtype=np.float64)
        rhs = J.T @ delta_target
        try:
            return np.linalg.solve(lhs, rhs).astype(np.float32, copy=False)
        except np.linalg.LinAlgError:
            return np.zeros(J.shape[1], dtype=np.float32)

    def generate_control_joint_qd(self) -> None:
        if self.sim_time >= self.robot_key_poses_time[-1]:
            self.target_joint_qd.zero_()
            return

        current_interval = int(np.searchsorted(self.robot_key_poses_time, self.sim_time))
        target = self.targets[current_interval]

        wp.launch(
            compute_ee_delta,
            dim=self.world_count,
            inputs=[
                self.state_0.body_q,
                self.endeffector_offset,
                self.endeffector_id,
                self.bodies_per_world,
                self._target_transforms(target),
            ],
            outputs=[self.ee_delta],
        )

        q = self.state_0.joint_q.numpy()[: self.dof_q_per_world].copy()
        qd = self.state_0.joint_qd.numpy()[: self.dof_qd_per_world].copy()
        J = self._compute_body_jacobian(q, qd)
        delta_target = np.array(self.ee_delta.numpy()[0], dtype=np.float32)

        q_des = q.copy()
        q_des[1:] = self.initial_pose[1:]
        delta_q_null = q_des - q
        delta_q_task = self._solve_damped_least_squares(J, delta_target)
        delta_q_null_proj = self._solve_damped_least_squares(J, J @ delta_q_null)
        N_delta_q_null = delta_q_null - delta_q_null_proj
        delta_q = delta_q_task + N_delta_q_null
        delta_q[-2] = target[-1] * 4.0 - q[-2]
        delta_q[-1] = target[-1] * 4.0 - q[-1]

        self.target_joint_qd.assign(np.tile(delta_q.astype(np.float32), self.world_count))

    def step(self) -> None:
        self.generate_control_joint_qd()
        self.cloth_solver.rebuild_bvh(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            self.viewer.apply_forces(self.state_0)

            particle_count = self.model.particle_count
            self.model.particle_count = 0
            self.model.gravity.assign(self.gravity_zero)
            self.model.shape_contact_pair_count = 0
            self.state_0.joint_qd.assign(self.target_joint_qd)
            self.robot_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0.particle_f.zero_()
            self.model.particle_count = particle_count
            self.model.gravity.assign(self.gravity_cm)

            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.cloth_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def render(self) -> None:
        wp.launch(
            scale_positions,
            dim=self.model.particle_count,
            inputs=[self.state_0.particle_q, self.viz_scale],
            outputs=[self.viz_state.particle_q],
        )
        if self.model.body_count > 0:
            wp.launch(
                scale_body_transforms,
                dim=self.model.body_count,
                inputs=[self.state_0.body_q, self.viz_scale],
                outputs=[self.viz_state.body_q],
            )

        self.model.shape_transform = self.viz_shape_transform
        self.model.shape_scale = self.viz_shape_scale

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.viz_state)
        self.viewer.log_shapes(
            "/table",
            newton.GeoType.BOX,
            self.table_viz_scale,
            self.table_viz_xform,
            self.table_viz_color,
        )
        self.viewer.end_frame()

        self.model.shape_transform = self.sim_shape_transform
        self.model.shape_scale = self.sim_shape_scale


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--viewer", choices=("viser", "null"), default="viser")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--world-count", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=200000)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.quiet:
        wp.config.quiet = True
    if args.device:
        wp.set_device(args.device)

    if args.viewer == "viser":
        viewer = ViewerViser(port=args.port)
    else:
        viewer = ViewerNull(num_frames=args.num_frames)

    example = ClothFrankaShowcase(viewer, args)
    newton.examples.run(example, argparse.Namespace(test=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
