# How it works in 3 minutes

Text goes in, an arm picks something up. Four clean layers, each
swappable, each ~200 lines of readable Python.

![franka picking](../assets/demos/franka_pick.gif){ loading=lazy }

## The four layers

```
"pick up the red cube"
       │
       ▼
┌──────────────┐
│   Planner    │  text + image → list[SkillCall]
└──────┬───────┘
       │  [SkillCall("pick", {"object": "red_cube"})]
       ▼
┌──────────────┐
│    Skill     │  orchestrates perception → grasp → motion → sim.step
└──────┬───────┘
       │
       ├─► Perception    "red_cube" → 3D point
       ├─► Grasp         3D point → Grasp(pose, width)
       └─► Motion/IK     joint trajectory → sim.step loop
                              │
                              ▼
                         ┌─────────┐
                         │ MuJoCo  │   physics advances one dt=5ms tick
                         └─────────┘
```

| Layer | Input | Output | Where |
|---|---|---|---|
| **Planner** | text + image + prior failures | `list[SkillCall]` | `agent/planner.py` |
| **Skill** | `SkillCall.arguments` | `SkillResult(success, reason, artifacts)` | `skills/*.py` |
| **Motion / IK** | start joints + target pose | `JointTrajectory` | `motion/ik.py` |
| **`sim.step`** | one joint-target vector | advances physics `dt` | `sim/mujoco_backend.py` |

**The arm only moves inside `sim.step`.** Everything above is choosing
what to do, points in space, or lists of joint targets. No magic.

## Watch the phases

Run the benchmark — the agent prints one line per phase:

![phase logs](../assets/demos/phases_log.gif){ loading=lazy }

```
PLAN:    task='pick up the red cube' replan=0
EXECUTE: pick({'object': 'red_cube'})
TASK               SEED  RESULT   SECS  REPLANS DETAIL
---------------------------------------------------------
pick_cube_franka   0     OK        1.1        0 dz_mm=166.905
```

One `PLAN` → one `EXECUTE` → success. On failure the agent appends the
failure to `prior_attempts` and re-enters `PLAN` (up to `max_replans=3`
times) with the failed steps fed back to the planner as "don't repeat
these."

## Read one pick

`skills/pick.py` is ~100 lines. End to end:

```python
# 1. Perception: text → 3D point
detected = ctx.perception.locate("red_cube", obs)
target = max(detected, key=lambda d: d.confidence)

# 2. Grasp: 3D point → gripper pose
grasps = ctx.grasp.plan(obs, target)
grasp = grasps[0]
approach = pose_offset_z(grasp.pose, 0.08)           # 8 cm above

# 3. Motion: pose → joint trajectory
traj = ctx.motion.plan(sim, start=obs.robot_joints,
                       target_pose=approach,
                       constraints={"orientation": "z_down"})

# 4. Execute: feed waypoints to sim.step
execute_trajectory(ctx, traj, gripper=0.0)

# Descend Cartesian-linear (not joint-space — avoids swinging the gripper sideways)
traj = plan_linear_cartesian(sim, start_joints, grasp.pose,
                             n_waypoints=60, orientation="z_down")
execute_trajectory(ctx, traj, gripper=0.0)

# Close, lift Cartesian-linear, verify the object rose 50 mm.
set_gripper(ctx, closed=1.0, hold_steps=60)
# ...lift + verify...
return SkillResult(success=True, reason="picked",
                   artifacts={"lifted_m": 0.167})
```

Each of the four methods above is a `@runtime_checkable Protocol` —
swap the implementation, the rest of the stack doesn't care.

## Why Damped Least Squares IK

MuJoCo gives you forward kinematics and the Jacobian for free.
Turning that into an inverse is:

```
dq = Jᵀ(JJᵀ + λ²·I)⁻¹ · err       # apply per iteration
qpos ← qpos + α·dq, clipped to joint limits
```

That's the whole iteration (`motion/ik.py`). Add singularity-escape
retries from a few seed poses, add Cartesian-space linear interpolation
for straight-line motions, and you have enough motion for pick/place/push
on a flat table. v0.2 swaps in curobo (GPU, collision-aware) by the
same Protocol.

## The ReAct loop

`agent/agent.py`:

```
IDLE → PLAN → EXECUTE (one skill) → ok? → next skill
                                │
                                └─ fail? → append to prior_attempts
                                           → REPLAN (≤ max_replans)
                                           → next PLAN with failures fed back
```

The planner doesn't know whether it's a regex (`StubPlanner`) or a
VLM (`VLMPlanner`) on the other side of the call. The agent doesn't
know what kind of skill it's invoking. Each layer sees only its
neighbors.

## Plug in your own piece

Every swappable component is a Protocol in `protocols.py`:

```python
SimBackend, Perception, GraspPlanner, MotionPlanner, RecordSink,
VLMClient, Skill, Planner
```

A plugin that implements one of these Protocols drops into the same
call sites the built-in implementations use — e.g., swap the bundled
`AnalyticTopDown` grasp planner for your own by passing a different
`GraspPlanner` into `AgentContext`. `SimBackend` and `MotionPlanner`
are the largest surfaces; the others are a method or two each. Plugin
packages register via `robosandbox.*` entry points.

The sim-to-real tutorial documents the concrete caveat for
`SimBackend` replacements: observation+step skills carry over, but
anything that reads MuJoCo's kinematic model (the current motion
planner does) needs either a kinematics-carrying real backend or a
different `MotionPlanner` implementation.

## What's next

- [Bring your own robot](./bring-your-own-robot.md) — swap the URDF.
- [Bring your own object](./bring-your-own-object.md) — drop in a YCB mesh or any OBJ.
- [Add a skill](./add-a-skill.md) — extend the action vocabulary.
