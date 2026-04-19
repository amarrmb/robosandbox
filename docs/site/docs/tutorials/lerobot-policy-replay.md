# Tutorial — LeRobot Policy Replay (pre-trained checkpoint)

The second tutorial in the [LeRobot workflow](./policy-replay.md)
track. Demo 1 proved the data path. This one runs **one specific
public ACT checkpoint** through RoboSandbox's
`robosandbox.policy.run_policy` loop on a robot that isn't bundled
with the core, so you can see what the policy integration actually looks
like in code.

!!! warning "What this tutorial is — and isn't"
    **This is an advanced integration demo under cross-embodiment
    mismatch.** It proves RoboSandbox's policy runtime plumbing
    (`LeRobotPolicyAdapter` + `run_policy` + the `LocalRecorder`
    loop-back) holds against a real public checkpoint whose
    observation/action contract doesn't match our sim exactly.

    **It is NOT ready-made policy replay support** for arbitrary
    checkpoints. The reusable product path is
    `LeRobotPolicyAdapter(policy) → run_policy(sim, adapter)` **when
    the checkpoint's embodiment matches your sim's** (same joint
    count, matching camera keys, compatible normalization). Anything
    else — including the `DimShimAdapter` in this tutorial's example
    script — is user-level glue, not a stable API. When your
    checkpoint and sim line up, skip the shim and drop the vanilla
    adapter into `run_policy` directly.

![so100 policy rollout](../assets/demos/so100_policy_run.gif){ loading=lazy }

The non-bundled SO-ARM100 (from
[mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie))
acting under
[`satvikahuja/act_so100_test`](https://huggingface.co/satvikahuja/act_so100_test),
a public pre-lerobot-0.5 ACT checkpoint, wrapped with
`LeRobotPolicyAdapter` and driven via the standard `run_policy`
entry point.

## What this tutorial proves

- **RoboSandbox runs a non-bundled embodiment** via the documented
  [Bring Your Own Robot](../guides/bring-your-own-robot.md) path
  (SO-ARM100 under `examples/so_arm100/`).
- **`LeRobotPolicyAdapter` marshals observations** into LeRobot's
  batch contract (torch tensors, named image keys, `(1, state_dim)`
  float32 state; auto-detected for `torch.nn.Module` policies).
- **`robosandbox.policy.run_policy` drives the rollout end to end** —
  `observe → act → step` — with recording via the same
  `LocalRecorder` the export tutorial uses.

**What it does NOT claim:**

- Task success. The checkpoint's training distribution, state/action
  dimension, and camera layout rarely match a given sim 1:1.
  Cross-embodiment transfer is a separate research problem.
- That arbitrary LeRobot checkpoints load automatically. Only the one
  checkpoint above is known-working end to end in this tutorial. The
  schema-migration utility below handles the common pre-lerobot-0.5
  config shape; variants that use non-standard normalization,
  multi-horizon action chunks, or bespoke observation keys will need
  hand edits.
- That `DimShimAdapter` (below) is a stable API. It exists specifically
  to paper over the 7-vs-6 dim mismatch between this checkpoint and
  Menagerie's SO-ARM100 — treat it as an **escape hatch for cross-
  embodiment experimentation**, not as the template for a production
  replay path. When the checkpoint's embodiment matches your sim's,
  the vanilla `LeRobotPolicyAdapter` is the API you build on.

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

**If your sim's embodiment matches the checkpoint's, you stop here.**
The plain `LeRobotPolicyAdapter` slots into `run_policy` directly,
which is the production replay path:

```python
from robosandbox.policy import run_policy

out = run_policy(sim, adapter, max_steps=80)
print(out["success"], out["steps"])
```

In this tutorial the checkpoint and sim **don't** match — and handling
that gap is its own section.

#### Advanced: cross-embodiment escape hatch (experimental)

`satvikahuja/act_so100_test` was trained on a robot whose state/action
vector is 7-dim (6 arm joints + gripper) with two cameras (`laptop`
and `phone`). Menagerie's SO-ARM100 exposes 5 arm joints + a gripper
(6-dim) with one scene camera. The dimensions don't line up:

| Dimension | Checkpoint expects | Our sim provides |
|---|---|---|
| Cameras | two — `laptop` + `phone` | one — `scene` |
| State dim | 7 | 6 |

To prove the `run_policy` path works *at all* under this mismatch,
the example script uses a small `DimShimAdapter` that duplicates the
scene frame across both camera keys, zero-pads state 6 → 7, and
truncates the 7-dim action back to 6 before `run_policy` consumes it.
**This is an experimental workaround, not an API contract.** Read
the
[full ~20-line source](https://github.com/amarrmb/robosandbox/blob/main/examples/so_arm100/run_so100_policy.py)
and treat it as illustrative of the kind of adapter a user might
need; do not import `DimShimAdapter` into production code.

When a checkpoint trained for your exact embodiment lands (same DoF,
same camera keys, same normalization statistics), skip the shim and
drop the vanilla `LeRobotPolicyAdapter` into `run_policy` directly.

### 4. Run via `run_policy` with recording

```python
from robosandbox.policy import run_policy
from robosandbox.recorder.local import LocalRecorder

recorder = LocalRecorder(Path("runs"))
recorder.start_episode(task="so100 rollout", metadata={})

def _frame_hook(obs, action):
    recorder.write_frame(obs)

out = run_policy(sim, adapter, max_steps=80, on_step=_frame_hook)
recorder.end_episode(success=False, result={"steps": out["steps"]})
```

80 steps at sim `dt=0.005` s ≈ 400 ms of simulated time. The rollout
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

None of these are in scope for "prove the integration holds." All three are
in scope for a future *Sim-to-Real Handoff* (Demo 3) tutorial that
pairs a sim-trained policy with a matching hardware backend.

## Where this tutorial fits

The [policy-replay umbrella](./policy-replay.md) frames the larger
"record → train → deploy" loop:

1. **[LeRobot Export](./lerobot-export.md)** — proves the data path.
2. **LeRobot Policy Replay with a pre-trained checkpoint** (you are
   here) — proves the policy integration under cross-embodiment mismatch.
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
