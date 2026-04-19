# Tutorial: policy replay

From a recorded episode → LeRobot v3 parquet → run it back in sim.

!!! tip "Looking for the layered LeRobot workflow?"
    This page is the original umbrella tutorial for record → replay
    in sim. For the three-step workflow that covers exporting to
    LeRobot datasets, driving a pre-trained checkpoint in sim, and
    handing a sim-validated skill off to real hardware, start here:

    1. **[LeRobot Export](./lerobot-export.md)** — record an episode
       and export it into a LeRobot v3 dataset; inspect the parquet
       + metadata with standard tooling.
    2. **[LeRobot Policy Replay](./lerobot-policy-replay.md)** — run
       a public ACT checkpoint through `LeRobotPolicyAdapter` +
       `run_policy` on a non-bundled robot.
    3. **[Sim-to-Real Handoff](./sim-to-real-handoff.md)** — what
       carries over to real hardware, what doesn't, and a concrete
       SO-101 backend skeleton.

    The page below stays useful for the `ReplayTrajectoryPolicy`
    path: given an events.jsonl from `LocalRecorder`, open-loop
    replay the joint trajectory with no training and no checkpoint.

This closes the "record → train → deploy" half of the loop. v0.1
ships the plumbing and a `ReplayTrajectoryPolicy` reference; wiring a
real LeRobot/ACT/Diffusion checkpoint is the
[LeRobot Policy Replay](./lerobot-policy-replay.md) tutorial.

## 1. Record an episode

```bash
uv run python examples/record_demo.py --out-dir runs
```

Produces:

```
runs/20260418-094533-1a2b3c4d/
├── episode.json
├── events.jsonl
├── result.json
└── video.mp4
```

See [recording & export](../concepts/recording-and-export.md) for the
`events.jsonl` schema.

## 2. (Optional) Export to LeRobot v3

```bash
uv pip install -e 'packages/robosandbox-core[lerobot]'
robo-sandbox export-lerobot \
  runs/20260418-094533-1a2b3c4d \
  /tmp/my_dataset
```

Produces a LeRobot v3 dataset at `/tmp/my_dataset/` with
`data/chunk-000/episode_000000.parquet` + `meta/` + `videos/`. Pass
this to any LeRobot-compatible training loop.

## 3. Replay the trajectory

The fastest path — no training needed — is the bundled
`ReplayTrajectoryPolicy`. It treats `events.jsonl` as an open-loop
action trace and drives the sim through it tick-by-tick.

Directly via CLI:

```bash
robo-sandbox run --policy runs/20260418-094533-1a2b3c4d \
                 --task pick_cube_franka \
                 --max-steps 1000
```

What happens:

1. `load_policy(path)` inspects the directory. An `events.jsonl`
   present → wraps in `ReplayTrajectoryPolicy`.
2. The task's scene is loaded into `MuJoCoBackend` and settled under
   gravity.
3. `run_policy(sim, policy, max_steps, success=task.success)` loops
   observe → act → step.
4. The task's success criterion runs against the final observation
   and is printed at the end.

Example output:

```
[run --policy] task:        pick_cube_franka
[run --policy] policy:      runs/20260418-094533-1a2b3c4d
[run --policy] verdict:     success
[run --policy] steps:       1000
[run --policy] final_reason: policy_completed_1000_steps
[run --policy] wall:        18.3s
```

## 4. Wire your own policy

Anything with `act(obs: Observation) -> np.ndarray` of shape
`(n_dof + 1,)` (joints + gripper in `[0, 1]`) satisfies the `Policy`
protocol:

```python
from robosandbox.policy import Policy, run_policy

class MyAwesomePolicy:
    def __init__(self, checkpoint: str):
        self._model = load_my_model(checkpoint)

    def act(self, obs):
        joints, gripper = self._model.infer(obs.rgb, obs.robot_joints)
        return np.concatenate([joints, [gripper]])

result = run_policy(sim, MyAwesomePolicy("ckpt.pt"),
                    max_steps=1000, success=task.success)
# {"success": True, "steps": 1000, "initial_obs": ..., "final_obs": ...}
```

To route a checkpoint directory through the CLI, extend
`robosandbox.policy.load_policy` to dispatch on your checkpoint
format (LeRobot, torchscript, onnx, whatever):

```python
# in your own package
from robosandbox.policy import load_policy as _core_load_policy

def load_policy(path):
    p = Path(path)
    if (p / "config.json").exists():
        return MyAwesomePolicy(p)
    return _core_load_policy(p)   # fall through to replay
```

## `policy.json` alternative

For a directory that doesn't auto-match, drop a `policy.json`:

```json
{
  "kind": "replay_trajectory",
  "trajectory": "events.jsonl",
  "action_lookahead": 1
}
```

`action_lookahead > 1` skips that many rows per `act()` — useful to
replay a 200 Hz recording at 100 Hz.

## Action semantics

`Policy.act(obs)` returns a flat `(n_dof + 1,)` vector:

- first `n_dof` entries — target joint positions
- last entry — gripper in `[0, 1]` (0 = open, 1 = closed)

This matches `MuJoCoBackend.step(target_joints=..., gripper=...)`.
Values outside range are clamped by the sim, not the policy.

## Tips

- **`verdict: unknown`** in the CLI means the task didn't declare a
  success criterion. That's fine for free-form exploratory runs.
- **Policy runs forever** — ReplayTrajectoryPolicy repeats its last
  action after the trajectory ends. Use `--max-steps` to cap.
- **Sim lag** — `run_policy` does one sim step per `act` call. At
  200 Hz sim timestep, 1000 steps = 5 sim seconds.

## See also

- [Recording & export](../concepts/recording-and-export.md) —
  `LocalRecorder` layout + `events.jsonl` schema.
- [Real-robot bridge](../concepts/real-robot.md) — same `Policy`
  protocol runs against `RealRobotBackend`.
- [CLI: `robo-sandbox run --policy`](../reference/cli.md#run-policy).
