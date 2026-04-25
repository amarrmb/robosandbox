"""Probe SensorTiledCamera against our newton_backend scene.

Loads pick_cube_franka via NewtonBackend, then sets up a SensorTiledCamera
externally (the backend itself doesn't render RGB yet), renders one frame,
and saves PNGs per world. If the cube + arm are visible, the camera-pose
convention is right and we can port this into newton_backend.observe_all().

Usage:
    /home/amar/newton/.venv/bin/python3 scripts/probe_newton_rgb.py [--world-count 4]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "robosandbox-core" / "src"))

from robosandbox.sim.newton_backend import NewtonBackend
from robosandbox.tasks.loader import load_builtin_task


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--world-count", type=int, default=4)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/newton_rgb_probe"))
    p.add_argument("--cam-pos", type=float, nargs=3, default=[1.1, -1.4, 0.9],
                   help="Camera position in world-0 frame (matches existing viewer default)")
    p.add_argument("--look-at", type=float, nargs=3, default=[0.4, 0.0, 0.1])
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    import warp as wp
    from newton.sensors import SensorTiledCamera

    task = load_builtin_task("pick_cube_franka")
    sim = NewtonBackend(world_count=args.world_count)
    sim.load(task.scene)

    # Settle physics so the cube isn't mid-fall.
    for _ in range(60):
        sim.step()

    # --- camera setup (manual; backend doesn't expose this yet) ---
    sensor = SensorTiledCamera(model=sim._model)  # noqa: SLF001
    sensor.utils.create_default_light(enable_shadows=True)
    sensor.utils.assign_checkerboard_material_to_all_shapes()

    fov_deg = 45.0
    rays = sensor.utils.compute_pinhole_camera_rays(
        args.width, args.height, math.radians(fov_deg)
    )
    color = sensor.utils.create_color_image_output(args.width, args.height, camera_count=1)

    # Build a pose-only camera transform per world. Newton's pinhole rays are
    # +Z forward in camera space; we need a quat that aligns +Z with the
    # look direction. Use a "look_at" transform via look-rotation.
    def look_at_quat(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> wp.quatf:
        # Newton/OpenGL convention: camera looks down -Z, +Y is up, +X is right.
        # Build world-from-camera rotation: cam_x=right, cam_y=up, cam_z=back.
        forward = target - eye
        forward /= np.linalg.norm(forward) + 1e-9
        right = np.cross(forward, up)
        right /= np.linalg.norm(right) + 1e-9
        cam_up = np.cross(right, forward)
        back = -forward
        # Columns of R map camera-axis vectors into world.
        R = np.column_stack([right, cam_up, back])
        # quat from rotation matrix (xyzw)
        m = R
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            s = math.sqrt(tr + 1.0) * 2
            qw = 0.25 * s
            qx = (m[2, 1] - m[1, 2]) / s
            qy = (m[0, 2] - m[2, 0]) / s
            qz = (m[1, 0] - m[0, 1]) / s
        elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
        return wp.quatf(qx, qy, qz, qw)

    eye0 = np.asarray(args.cam_pos, dtype=np.float64)
    target = np.asarray(args.look_at, dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0])

    # Per-world: camera follows the world by the same grid offset
    from robosandbox.sim.newton_backend import _grid_offsets  # type: ignore
    offsets = _grid_offsets(args.world_count)
    # Newton expects shape (camera_count, world_count) — outer list = cameras,
    # inner = one transform per world.
    per_world = []
    for ox, oy, oz in offsets:
        eye_w = eye0 + np.array([ox, oy, oz])
        q = look_at_quat(eye_w, target + np.array([ox, oy, oz]), up)
        per_world.append(wp.transformf(wp.vec3f(*eye_w.astype(np.float32)), q))
    cam_xforms = wp.array([per_world], dtype=wp.transformf)

    # Render
    sensor.update(
        sim._state_0,
        cam_xforms,
        rays,
        color_image=color,
        clear_data=SensorTiledCamera.GRAY_CLEAR_DATA,
    )

    # color is wp.array4d[wp.uint32] shape (W, 1, H, W, ?) — actually (worlds, cams, H, W) uint32 RGBA
    img = color.numpy()  # shape (worlds, cams, H, W) uint32
    print(f"[probe] color image shape={img.shape}, dtype={img.dtype}")
    # uint32 packed as 0xAABBGGRR (little-endian) — extract RGB bytes
    arr = img.view(np.uint8).reshape(*img.shape, 4)  # (worlds, cams, H, W, 4)
    rgb = arr[..., :3]  # drop alpha
    print(f"[probe] rgb shape after view: {rgb.shape}")

    # Save per-world PNG
    try:
        from PIL import Image
        for w in range(args.world_count):
            path = args.out_dir / f"world_{w:02d}.png"
            Image.fromarray(rgb[w, 0]).save(path)
            mean = rgb[w, 0].mean()
            print(f"[probe] world {w}: saved {path} (mean intensity={mean:.1f})")
    except ImportError:
        np.save(args.out_dir / "rgb_all.npy", rgb)
        print(f"[probe] PIL not available — saved {args.out_dir / 'rgb_all.npy'}")

    sim.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
