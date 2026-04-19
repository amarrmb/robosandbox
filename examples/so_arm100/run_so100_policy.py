"""Drive the SO-ARM100 sim with a public LeRobot ACT checkpoint.

Stages this tutorial exercises end-to-end:

  1. Non-bundled URDF import (the SO-ARM100 from mujoco_menagerie).
  2. Public LeRobot checkpoint download + schema migration.
  3. LeRobotPolicyAdapter + a small shim that reconciles the policy's
     state/action dimensionality with this sim's (6 vs 7 here).
  4. Closed-loop rollout in sim with recording.

Checkpoint used: ``satvikahuja/act_so100_test`` (community ACT, trained
pre-lerobot-0.5, expects 7-dim state / two cameras / 7-dim action).
Menagerie's ``trs_so_arm100`` is 5 arm joints + 1 gripper = 6 dims, so
the shim pads the state to 7 and truncates the action to 6. This is a
brittle, pragmatic dogfood; cross-embodiment transfer is not the claim.

Run from the repo root::

    uv run python examples/so_arm100/run_so100_policy.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from robosandbox.policy.lerobot_adapter import LeRobotPolicyAdapter
from robosandbox.recorder.local import LocalRecorder
from robosandbox.sim.mujoco_backend import MuJoCoBackend
from robosandbox.types import Observation, Pose, Scene, SceneObject

HERE = Path(__file__).parent
ROBOT_XML = HERE / "so_arm100.xml"
ROBOT_YAML = HERE / "so_arm100.robosandbox.yaml"

_CHECKPOINT = "satvikahuja/act_so100_test"
_CKPT_DIR = Path("/tmp/so100_ckpt")


def prepare_checkpoint() -> Path:
    """Download (cached) + migrate config if needed. Returns local path."""
    from huggingface_hub import snapshot_download

    from examples.so_arm100.migrate_lerobot_config import migrate_config

    local = Path(snapshot_download(_CHECKPOINT, local_dir=str(_CKPT_DIR)))
    cfg_path = local / "config.json"
    cfg = json.loads(cfg_path.read_text())
    if "input_features" not in cfg:
        print(f"  migrating config.json → lerobot-0.5 schema")
        cfg_path.write_text(json.dumps(migrate_config(cfg), indent=2))
    return local


class DimShimAdapter(LeRobotPolicyAdapter):
    """Adapter that pads sim state → policy state, and truncates action.

    The SO-100 ACT checkpoints on HF expect a 7-dim state/action vector
    (6 joints + gripper). Menagerie's SO-ARM100 exposes 5 joints + 1
    gripper = 6 dims. This shim:

    - pads the state to 7 by appending a zero where the extra joint
      would be (position 5 — wrist-roll vs something else, depending
      on the checkpoint's joint order convention)
    - truncates the 7-dim output action back to 6 before the sim
      consumes it

    This is a **brittle demo shim** — it will not produce successful
    rollouts, but it lets us prove the integration works end-to-end.
    """

    def __init__(self, policy: Any, camera_names: tuple[str, ...], **kwargs: Any) -> None:
        super().__init__(policy, camera_name=camera_names[0], **kwargs)
        self._extra_camera_names = camera_names[1:]

    def _to_batch(self, obs: Observation) -> dict[str, Any]:
        batch = super()._to_batch(obs)
        # Duplicate the scene frame across any additional expected cameras.
        primary_key = f"observation.images.{self._camera_name}"
        for extra in self._extra_camera_names:
            batch[f"observation.images.{extra}"] = batch[primary_key]
        # Pad state 6 → 7 (append a zero).
        import torch
        state = batch["observation.state"]
        pad = torch.zeros(state.shape[0], 1, dtype=state.dtype, device=state.device)
        batch["observation.state"] = torch.cat([state, pad], dim=1)
        return batch

    def act(self, obs: Observation) -> np.ndarray:
        action_7 = super().act(obs)
        # Truncate to 6-dim: keep the first 5 arm joints, drop the padded
        # 6th, keep the gripper (which was the 7th element of the
        # policy's output).
        action_6 = np.concatenate([action_7[:5], action_7[6:7]])
        return action_6


def main() -> int:
    print("[1/4] prepare checkpoint")
    ckpt = prepare_checkpoint()

    print("[2/4] load policy")
    from lerobot.policies.act.modeling_act import ACTPolicy

    policy = ACTPolicy.from_pretrained(str(ckpt))
    policy.eval()

    # The trained SO-100 policy expected two cameras (laptop + phone).
    # Our sim has one 'scene' camera — we feed the same frame as both.
    # The checkpoint's visual contract was 480x640.
    cam_keys = tuple(policy.config.input_features.keys())
    cameras = [k.split(".")[-1] for k in cam_keys if "image" in k]
    # action_dim=7 matches the policy's output contract; the shim
    # truncates to 6 before the sim consumes the action.
    adapter = DimShimAdapter(
        policy,
        camera_names=tuple(cameras),
        image_size=(480, 640),
        action_dim=7,
    )
    print(f"  policy expects cameras: {cameras}")
    print(f"  policy state dim:       {policy.config.input_features['observation.state'].shape[0]}")
    print(f"  sim state dim:          6 (shim pads to 7 and truncates action to 6)")

    print("[3/4] load sim")
    scene = Scene(
        robot_urdf=ROBOT_XML,
        robot_config=ROBOT_YAML,
        objects=(
            SceneObject(
                id="red_cube", kind="box", size=(0.012,)*3,
                pose=Pose(xyz=(0.0, -0.25, 0.06)), rgba=(0.85, 0.2, 0.2, 1.0),
                mass=0.05,
            ),
        ),
    )
    sim = MuJoCoBackend(render_size=(480, 640), camera="scene")
    sim.load(scene)
    for _ in range(100):
        sim.step()

    print("[4/4] rollout via robosandbox.policy.run_policy")
    from robosandbox.policy import run_policy

    recorder = LocalRecorder(Path("runs"))
    recorder.start_episode(
        task="pick up the red cube (so100 + public ACT checkpoint)",
        metadata={"checkpoint": _CHECKPOINT, "embodiment": "trs_so_arm100", "note": "dim-shim demo"},
    )
    # run_policy handles the observe → act → step loop; wiring the
    # recorder here hooks frame capture onto the same callback the
    # viewer uses for live streaming. No monkey-patching of sim.step.
    def _frame_hook(obs: Observation, action: np.ndarray) -> None:
        recorder.write_frame(obs)

    t0 = time.time()
    out = run_policy(sim, adapter, max_steps=80, on_step=_frame_hook)
    wall = time.time() - t0
    obs_final = out["final_obs"]
    cube_dz_mm = (obs_final.scene_objects["red_cube"].xyz[2] - 0.06) * 1000
    recorder.end_episode(
        success=False,  # cross-embodiment shim — not a real success claim
        result={"wall": wall, "cube_dz_mm": cube_dz_mm, "steps": out["steps"]},
    )
    sim.close()
    print(f"  rollout wall: {wall:.1f}s  (run_policy steps={out['steps']})")
    print(f"  cube vertical delta: {cube_dz_mm:+.1f}mm  (not expected to succeed)")
    print()
    print("DEMO 2 VERDICT: non-bundled SO-ARM100 + public ACT checkpoint ran end-to-end")
    print("through LeRobotPolicyAdapter. Cross-embodiment action quality is a separate question.")
    return 0


if __name__ == "__main__":
    # Allow `python examples/so_arm100/run_so100_policy.py` from repo root
    # to find the sibling migrate module.
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    raise SystemExit(main())
