# Tutorial — LeRobot Policy Replay (pre-trained checkpoint)

The second tutorial in the [LeRobot workflow](./policy-replay.md)
track. Demo 1 proved the data path. This one proves the **policy
seam** — load a public ACT checkpoint off Hugging Face and drive it
through RoboSandbox's `run_policy` loop, on a robot that isn't
bundled with the core.

![so100 policy rollout](../assets/demos/so100_policy_run.gif){ loading=lazy }

The non-bundled SO-ARM100 (from
[mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie))
acting under
[`satvikahuja/act_so100_test`](https://huggingface.co/satvikahuja/act_so100_test),
a public ACT checkpoint, with RoboSandbox's `LeRobotPolicyAdapter`
as the seam.

## What this tutorial proves

- **RoboSandbox can run a non-bundled embodiment** via the documented
  [Bring Your Own Robot](../guides/bring-your-own-robot.md) path.
- **`LeRobotPolicyAdapter` correctly marshals observations** into
  LeRobot's batch contract (torch tensors, named image keys, 
  `(1, state_dim)` float32 state).
- **Any public LeRobot ACT checkpoint loads** after a one-line config
  schema migration for pre-0.5 checkpoints.
- **The full stack runs end to end** — sim observation → policy
  forward pass → action → `sim.step` — with recording via the same
  `LocalRecorder` the export tutorial uses.

**What it does NOT claim:** that the policy actually succeeds at the
task. The checkpoint's training distribution, state/action dimension,
and camera layout rarely match a given sim 1:1. Cross-embodiment
transfer is a separate research problem; this tutorial is about
**plumbing fidelity**, not policy quality.

## The four stages

![so100 policy terminal](../assets/demos/so100_policy.gif){ loading=lazy }

The companion script
[`examples/so_arm100/run_so100_policy.py`](https://github.com/amarrmb/robosandbox/blob/main/examples/so_arm100/run_so100_policy.py)
runs all four in one go:

```bash
uv run python examples/so_arm100/run_so100_policy.py
```

### 1. Import the non-bundled robot

The SO-ARM100 MJCF + 18 STL meshes live under
[`examples/so_arm100/`](https://github.com/amarrmb/robosandbox/tree/main/examples/so_arm100/)
(Apache-2.0, copied from Menagerie). The hand-authored
`so_arm100.robosandbox.yaml` sidecar (see the
[BYO-robot guide](../guides/bring-your-own-robot.md) for the schema)
declares:

- 5 arm joints: `Rotation`, `Pitch`, `Elbow`, `Wrist_Pitch`, `Wrist_Roll`
- 1 gripper joint: `Jaw`
- `open_qpos=1.5` / `closed_qpos=0.0` — **verified empirically** by
  measuring the pad gap, not guessed
- `home_qpos` placing the gripper over a reach-forward workspace
- Injected `ee_site` at the Fixed_Jaw body with a -10 cm local-Y offset

Automated coverage in
[`tests/test_so_arm100_import.py`](https://github.com/amarrmb/robosandbox/blob/main/packages/robosandbox-core/tests/test_so_arm100_import.py)
locks the DoF count, joint order, gripper open/closed ordering, and
reachability from home.

### 2. Download and migrate a public checkpoint

```python
from huggingface_hub import snapshot_download
local = Path(snapshot_download("satvikahuja/act_so100_test", ...))
```

**Every public SO-100 ACT checkpoint on the Hub was trained with
`lerobot < 0.5`.** The older config schema uses `input_shapes` +
`input_normalization_modes` rather than the current
`input_features`/`output_features` + `normalization_mapping`. Loading
the old config crashes with a `DecodingError`.

Fix:
[`examples/so_arm100/migrate_lerobot_config.py`](https://github.com/amarrmb/robosandbox/blob/main/examples/so_arm100/migrate_lerobot_config.py)
rewrites `config.json` in-place. Run it once per checkpoint:

```bash
uv run python examples/so_arm100/migrate_lerobot_config.py /path/to/ckpt/config.json
```

The rollout script above does this automatically on first download.

### 3. Wrap with `LeRobotPolicyAdapter` (+ compatibility shims as needed)

```python
from lerobot.policies.act.modeling_act import ACTPolicy
from robosandbox.policy.lerobot_adapter import LeRobotPolicyAdapter

policy = ACTPolicy.from_pretrained(str(local))
policy.eval()

adapter = LeRobotPolicyAdapter(
    policy,
    camera_name="laptop",                 # policy's primary image key
    image_size=(480, 640),                 # policy's expected HxW
    action_dim=7,                          # policy's output dim
)
```

`LeRobotPolicyAdapter` auto-detects that the wrapped policy is a
`torch.nn.Module` and hands `select_action` torch tensors; mock
policies keep receiving numpy (see
[`tests/test_lerobot_adapter.py`](https://github.com/amarrmb/robosandbox/blob/main/packages/robosandbox-core/tests/test_lerobot_adapter.py)
for the regression locks).

**Two real mismatches this checkpoint has vs Menagerie's SO-ARM100**,
each typical and worth documenting honestly:

| Dimension | Checkpoint expects | Our sim provides | Reconciliation |
|---|---|---|---|
| Cameras | two — `laptop` + `phone` | one — `scene` | Duplicate the scene frame across both keys |
| State dim | 7 (6 joints + gripper) | 6 (5 joints + gripper) | Zero-pad state 6 → 7; truncate action 7 → 6 |

The rollout script implements both with a `DimShimAdapter` subclass
of `LeRobotPolicyAdapter`. Look at its
[source](https://github.com/amarrmb/robosandbox/blob/main/examples/so_arm100/run_so100_policy.py)
— it's 20 lines and shows the general pattern for reconciling any
LeRobot checkpoint with a non-matching sim embodiment. **A working
shim like this proves the seam holds; it does not make the policy
succeed at the task.**

### 4. Run in sim with recording

```python
for _ in range(80):
    obs = sim.observe()
    action = adapter.act(obs)
    sim.step(target_joints=action[:-1], gripper=float(action[-1]))
```

80 steps at sim `dt=0.005` s = 400 ms of simulated time. The rollout
writes `runs/<id>/video.mp4 + events.jsonl + result.json` via the
same `LocalRecorder` the [export tutorial](./lerobot-export.md) uses
— closing the loop with Demo 1's data path.

## What success looks like at each stage

| Stage | Signal |
|---|---|
| Import | `MuJoCoBackend.load(scene)` returns; `sim.n_dof == 5`; reachability pre-flight empty |
| Migrate | `config.json` now has `type`, `input_features`, `output_features` keys |
| Wrap | `ACTPolicy.from_pretrained(local)` succeeds; `policy.config.input_features` matches what the sim will provide |
| Run | Terminal prints `rollout wall: ~3s` with no traceback; `runs/<id>/video.mp4` exists |

## What success does NOT look like

- The cube being lifted. This is a cross-embodiment shim on a
  checkpoint trained on real SO-100 hardware with cameras the sim
  doesn't have. Expect the arm to move toward plausible configurations
  but not complete the task.
- A drop-in policy replacement. Treat this as scaffolding for the
  case where you _have_ a checkpoint trained on the specific embodiment
  and cameras you're simulating.

## When will actual task success land?

Three things need to line up for a meaningful policy run in sim:

1. **Embodiment match.** Checkpoint's joint count, joint order, and
   gripper convention must match the sim's URDF. Menagerie's
   `trs_so_arm100` is 5-DoF; most public SO-100 ACT checkpoints were
   trained on 6-DoF SO-101 variants. Bringing in the SO-101 URDF
   collapses that gap.
2. **Camera match.** The checkpoint's image keys (`laptop` / `phone`
   here) need real camera views, not duplicated scene frames. Adding
   a second MuJoCo camera at the right extrinsics is a scene-level
   change.
3. **Normalization match.** The checkpoint's
   `normalization_mapping` ships the mean/std statistics from its
   training distribution. Sim observations that fall well outside
   that distribution produce garbage actions even when plumbing is
   perfect. This is rarely a hard blocker but is often a subtle one.

None of these are in scope for "prove the seam holds." All three are
in scope for a future *Sim-to-Real Handoff* (Demo 3) tutorial that
pairs a sim-trained policy with a matching hardware backend.

## Where this tutorial fits

The [policy-replay umbrella](./policy-replay.md) frames the larger
"record → train → deploy" loop:

1. **[LeRobot Export](./lerobot-export.md)** — proves the data path.
2. **LeRobot Policy Replay with a pre-trained checkpoint** (you are
   here) — proves the policy seam under cross-embodiment mismatch.
3. **Sim-to-Real Handoff** — coming. The deployment recipe and
   backend template for taking a sim-validated policy to real
   hardware.

## Requirements

```bash
uv pip install -e 'packages/robosandbox-core[lerobot]'
uv pip install lerobot        # brings torch, torchvision, lerobot's policy code
```

The first line matches the [export tutorial](./lerobot-export.md) and
pulls `pyarrow`. The second is only needed for *this* tutorial — the
export doesn't need the `lerobot` package itself.

Footprint: ~2 GB for torch + torchvision + lerobot dependencies.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `ParsingError: Expected a dict with a 'type' key` | Pre-lerobot-0.5 checkpoint; run `migrate_lerobot_config.py` |
| `DecodingError: The fields input_normalization_modes, input_shapes ... are not valid` | Same — the full migration adds `input_features`/`output_features`, not just `type` |
| `TypeError: linear(): ... must be Tensor, not numpy.ndarray` | Older RoboSandbox without the torch-gating fix; pull latest or pin `robosandbox >= <version-with-fix>` |
| `ValueError: LeRobot policy returned action dim N, expected M` | Set `action_dim=N` on the adapter to match the checkpoint's output, then handle the sim mismatch in a shim |
| Arm moves but doesn't grasp | Expected — cross-embodiment policy action quality is not the claim |

## Credits

- SO-ARM100 URDF + meshes: [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)
  (`trs_so_arm100`, Apache 2.0).
- Public ACT checkpoint: [`satvikahuja/act_so100_test`](https://huggingface.co/satvikahuja/act_so100_test).
- LeRobot policies: [huggingface/lerobot](https://github.com/huggingface/lerobot).
