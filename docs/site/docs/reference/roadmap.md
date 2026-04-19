# Roadmap

Mission-aligned: a playground where someone can build and evaluate
manipulation agents. The four pillars are object diversity, task
diversity, interaction, and loop closure (record → train → deploy).

This page is the shipped / in-flight snapshot. For the concrete test
and benchmark status, run `robo-sandbox-bench` locally or check the
[CI badge](https://github.com/amarrmb/robosandbox/actions).

## Shipped

**Core**

- MuJoCo backend + built-in 6-DOF arm.
- 9 skills: `pick`, `place_on`, `push`, `home`, `pour`, `tap`,
  `open_drawer`, `close_drawer`, `stack`.
- Stub planner + OpenAI-compatible VLM planner.

**Robot + object diversity**

- URDF import — `Scene(robot_urdf=...)` + sidecar YAML; bundled Franka Panda.
- Mesh import — `SceneObject(kind="mesh")` with CoACD decomposition + hull cache.
- 10 bundled YCB items reachable via `@ycb:<id>` shorthand.
- Procedural scenes — `scene.presets.tabletop_clutter(n, seed)`.
- Drawer primitive — `SceneObject(kind="drawer")`, first articulated primitive.

**Benchmark + evaluation**

- Declarative success criteria (`lifted`, `moved_above`, `displaced`, `all`, `any`).
- `randomize:` YAML block + `--seeds N` aggregation with `mean ± stderr`.
- 9 default tasks covering pick / stack / push / pour / drawer.
- Authoring-time reachability pre-flight check.

**Interaction**

- Browser live viewer — FastAPI + WebSocket + SPA; task dropdown, Run/Reset, Record, Inspector scrubber.
- Keyboard teleop — WASD/QE drives EE; Space toggles gripper.

**Loop closure**

- `LocalRecorder` — per-episode MP4 + events.jsonl + result.json.
- LeRobot v3 export — `robo-sandbox export-lerobot` CLI.
- `Policy` protocol — `ReplayTrajectoryPolicy`, `LeRobotPolicyAdapter`, `run_policy`.
- `RealRobotBackend` — satisfies `SimBackend` Protocol; SO-101 skeleton under `examples/so101_handoff/`.

## Open / deferred

### Pillar 2 — task diversity

- **`insert_peg` skill** — needs a peg-hole articulated primitive
  (prismatic-jointed hole with compliance). ~1 day slice.
- **More composites** — "put the apple in the drawer", "stack three
  cubes by colour", "tidy the table".
- **Richer randomization fields** — rgba, size, mass (currently only
  xy + yaw).

### Pillar 3 — interaction

- **Client-side orbit camera** — move viewer render from server-side
  MJPEG to client-side Three.js. ~2–3 days; needs its own spec.
- **Gamepad + continuous teleop** — extend keyboard teleop with
  gamepad axes → velocity integration.
- **Trajectory inspector** — post-run scrubber; requires in-RAM
  episode buffer.

### Pillar 4 — loop closure

- **First integration with a real policy checkpoint** — wire a
  public LeRobot/ACT/Diffusion ckpt into `load_policy`, validate
  against `pick_cube_franka`.
- **SO-101 reference backend** — first concrete
  `RealRobotBackend` subclass on LeRobot's SO-101 driver.

### Polish

- **Full-mesh Franka visuals** — `--meshes full` downloads
  menagerie's 33 MB of OBJ visuals on demand, caches locally.
- **PyPI release** — CI wheel+sdist on tag.

## v0.3 directions (not committed)

- Full-scene randomization (lighting, camera, domain randomization).
- Grasp evaluation plugin (AnyGrasp / GraspNet / anti-podal /
  learned).
- More robot arms (UR5, xArm, SO-100 variants, two-arm setups).
- Procedural kitchens / desks / shelves beyond `tabletop_clutter`.
- Soft-body objects (fabric, fluids).
- Multi-robot scenes + coordination skills.

