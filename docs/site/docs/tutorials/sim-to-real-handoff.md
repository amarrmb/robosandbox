# Tutorial — Sim-to-Real Handoff

This page is about the handoff from sim code to a real backend. It does
not ship a hardware driver; it shows the contract that a hardware driver
has to satisfy and the pieces that already sit on top of that contract.

Everything here runs against a software skeleton that tracks commanded
state in memory. The useful part is the shape of the interface: what a
real backend needs to implement, what should carry over unchanged, and
what still depends on MuJoCo.

![so101 handoff terminal](../assets/demos/so101_handoff.gif){ loading=lazy }

The screenshot above shows the `Home` skill — the same one the sim uses
— driving a
`RealRobotBackend` subclass from an arbitrary start pose to its
declared home, with zero joint residual. No branches for
"sim vs real" in the skill; the `SimBackend` Protocol is the interface.

## What carries over

| Layer | Status on `RealRobotBackend` |
|---|---|
| `Skill` Protocol (`name`, `description`, `parameters_schema`, `__call__`) | **certified** by the contract tests below |
| `Home` skill end-to-end | **certified** by `test_home_skill_runs_against_real_backend` |
| Other observation+step skills (`Wave`, teleop primitives) | **expected to work** — same Protocol surface, not individually tested |
| `LocalRecorder` + `export-lerobot` | **expected to work** — reads `Observation`, which the fake backend produces |
| `LeRobotPolicyAdapter` + `run_policy` | **expected to work when observation/action dims and camera keys match the policy contract** — the runtime loop only calls `observe` / `step`, but the adapter still needs the backend's image keys, state dimension, and normalization to line up with what the checkpoint was trained for |
| `Agent` ReAct loop | **expected to work** — only composes skill calls + observations |

Here, "certified" means a test in
[`test_real_backend_contract.py`](https://github.com/amarrmb/robosandbox/blob/main/packages/robosandbox-core/tests/test_real_backend_contract.py)
exercises the path end-to-end against a `RealRobotBackend` subclass.
"Expected to work" means the layer only consumes the same `SimBackend`
Protocol surface the certified paths do — verify in your own backend
before relying on it in a production loop.

## What you still have to implement

| Method | What your driver does |
|---|---|
| `load(scene)` | connect to motor bus + camera, run calibration, validate scene workspace bounds |
| `reset()` | send a blocking trajectory to the configured home pose |
| `step(target_joints, gripper)` | stream one position command at the configured control rate; clamp against joint + velocity limits |
| `observe()` | read current joint positions; grab one RGB frame; optionally populate `scene_objects` from a pose estimator |
| `get_object_pose(id)` | query your pose estimator (AprilTag, OptiTrack, learned keypoints) — return `None` if you rely on VLM perception instead |
| `set_object_pose(id, pose)` | no-op on real hardware; the sandbox only calls it for sim scene init |
| `close()` | disable torque, release camera, close serial |

## What does not carry over

**Motion-planning skills (`Pick`, `PlaceOn`, `Push`) depend on
MuJoCo's kinematic model** via `sim.model` / `sim.data` for their IK
solver. A real-hardware backend won't have those. Options:

1. **Ship the same URDF into your real backend** and expose
   `model`/`data` compatible with MuJoCo's Python API (possible with
   `mujoco.MjSpec`, but now the sim IS the real backend).
2. **Plan in sim, execute on real:** run `Pick` against a MuJoCo sim
   to get a `JointTrajectory`, then stream the waypoints to your
   real backend's `step()`. This is the mainstream pattern.
3. **Replace the motion planner** with one that speaks directly to
   your robot's native kinematics (e.g. curobo on real URDF).

`Home` and any observation+step policy can run on either backend
without special branching.

## Implementing your SO-101 backend

The skeleton at
[`examples/so101_handoff/so101_backend.py`](https://github.com/amarrmb/robosandbox/blob/main/examples/so101_handoff/so101_backend.py)
is a working starting point with every method stubbed as "echo
commanded state back through `observe`." Search for `_TODO(real)` to
find the places where real driver code should go.

Key fields in `RealRobotBackendConfig`:

```python
RealRobotBackendConfig(
    n_dof=5,                                  # length of target_joints
    joint_names=("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"),
    home_qpos=(0.0, -1.4, 1.4, 0.0, 0.0),
    gripper_open=1.5,                         # jaw radians — widest
    gripper_closed=0.0,                       # jaw radians — pinched
    control_hz=200.0,                         # target tick rate for step()
    safety_bounds=((-0.5, -0.5, 0.0), (0.5, 0.5, 0.8)),
)
```

Three things are worth preserving when you wire in a real driver:

- **Skills read `home_qpos` via `sim.home_qpos`** (a public
  property on both `MuJoCoBackend` and `RealRobotBackend`). Removes
  the per-skill hardcoded home vectors that used to break on any
  DoF other than 6.
- **`LeRobotPolicyAdapter` constructs the same batch shape** it
  does in sim — `(1, C, H, W)` float32 images, `(1, state_dim)`
  float32 state, correct named keys. Policy rollouts whose
  observation+action dims match your embodiment can be driven with
  the vanilla adapter via `run_policy`.
- **Regression tests in `packages/robosandbox-core/tests/test_real_backend_contract.py`**
  guard the contract: `n_dof`/`joint_names`/`home_qpos` stay
  consistent, `step` mutates observed state, gripper ordering is
  sane, `Home` runs to within 1 mm-equivalent joint norm.

The intended workflow is simple: copy the skeleton, rename the class,
replace the `_TODO(real)` blocks with calls into your motor bus and
camera, and run the same regression tests against your subclass before
you power the arm.

## First real run: safety checklist

Before enabling torque on the arm for the first time with your new
backend:

1. **Joint limits in config match hardware.** Read a servo at each
   limit with the arm powered off (gravity only); confirm the
   ``joint_names`` + ``RealRobotBackendConfig`` ranges match what
   the motors physically allow.
2. **`reset()` runs without torque.** Drive the arm to your declared
   home manually; run `backend.reset()` with torque off; verify
   `observe().robot_joints` matches `backend.home_qpos` ± a few
   counts. Catches sign-flip and zero-offset bugs.
3. **Slow first motion.** First step with torque on sends a
   trajectory to home at ≤ 10% of max velocity. Stand near the
   E-stop.
4. **Workspace bounds sanity.** `safety_bounds` encloses the physical
   workspace with room to spare. Force a test command that should
   clamp; verify it does.
5. **Gripper open/closed ordering.** Run the
   `test_gripper_open_width_exceeds_closed` style check against your
   backend (see
   [`tests/test_real_backend_contract.py`](https://github.com/amarrmb/robosandbox/blob/main/packages/robosandbox-core/tests/test_real_backend_contract.py)):
   send `gripper=1.0`, read width; send `gripper=0.0`, read width;
   the second must be larger. Catches a swapped
   `gripper_open`/`gripper_closed`.
6. **`Home` skill dry run.** With torque on but the arm in open air,
   run `Home` from its current pose. Zero joint residual is a pass.
   Any residual = your driver's position controller isn't tracking
   the commanded trajectory.

Only after those six checks pass should you enable faster motions,
teleop, or policy rollouts.

## Running it

The zero-hardware skeleton demo is:

```bash
uv run python examples/so101_handoff/run_home_skill.py
```

Expected output:

```
before home: [ 0.5 -0.5  0.5  0.3 -0.3]
after home:  [ 0.  -1.4  1.4  0.   0. ]
result:      success=True reason='homed'
home error:  0.00mm-equivalent joint norm
```

Substitute a real driver by replacing the `_TODO(real)` blocks in
`examples/so101_handoff/so101_backend.py`, then run the contract
tests:

```bash
uv run pytest packages/robosandbox-core/tests/test_real_backend_contract.py
```

If all five pass, the observation+step contract holds for your backend.
That is enough for `Home`, teleop primitives, custom open-loop skills,
and `LeRobotPolicyAdapter`-wrapped policies via `run_policy`. It is not
enough to certify motion-planning skills like `Pick`, `PlaceOn`, or
`Push`; those still depend on MuJoCo kinematics.

## Where this fits

1. **[LeRobot Export](./lerobot-export.md)** — proves the data path.
2. **[LeRobot Policy Replay](./lerobot-policy-replay.md)** —
   proves the policy integration.
3. **Sim-to-Real Handoff** (you are here) — the deployment side of the
   story: what has to happen between a sim-validated skill or policy
   and real hardware.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `ValueError: backend reports n_dof=N but home_qpos has length M` | `RealRobotBackendConfig` out of sync — `joint_names` and `home_qpos` must both match `n_dof` |
| `home_dim_mismatch` reason from `Home` | Your backend's `observe().robot_joints.shape[0] != len(home_qpos)` — usually a driver returning padded state |
| Gripper closes when you expect open | `gripper_open`/`gripper_closed` swapped in config; the open-vs-closed width test catches this |
| First `step()` jerks the arm | `control_hz` too low for the distance the controller needs to cover in one tick; prefer smaller joint deltas + higher tick rate |
| `Pick` skill fails with `AttributeError: 'MyBackend' object has no attribute 'model'` | Expected — motion planners need MuJoCo kinematics. See "what does NOT carry over" above |
