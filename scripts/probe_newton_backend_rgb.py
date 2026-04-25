"""End-to-end probe: NewtonBackend(enable_camera=True).observe_all() per world."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "robosandbox-core" / "src"))

from robosandbox.sim.newton_backend import NewtonBackend
from robosandbox.tasks.loader import load_builtin_task


def main() -> int:
    task = load_builtin_task("pick_cube_franka")
    sim = NewtonBackend(world_count=4, enable_camera=True, render_size=(240, 320))
    sim.load(task.scene)
    for _ in range(60):
        sim.step()
    obs_list = sim.observe_all()
    print(f"[probe] worlds={len(obs_list)}, rgb shape={obs_list[0].rgb.shape}, dtype={obs_list[0].rgb.dtype}")
    out = Path("/tmp/newton_backend_rgb")
    out.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        for w, obs in enumerate(obs_list):
            p = out / f"world_{w:02d}.png"
            Image.fromarray(obs.rgb).save(p)
            print(f"[probe] world {w}: mean={obs.rgb.mean():.1f}  saved {p}")
    except ImportError:
        import numpy as np
        np.save(out / "all.npy", [obs.rgb for obs in obs_list])
    sim.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
