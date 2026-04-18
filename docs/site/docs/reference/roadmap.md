# Roadmap

Mission-aligned: a playground where someone can build and evaluate
manipulation agents. The four pillars are object diversity, task
diversity, interaction, and loop closure (record → train → deploy).

Full detail, history, and sprint sequencing live in the repo's
[`TODO.md`](https://github.com/amarrmb/robosandbox/blob/main/TODO.md).
This page is the shipped / in-flight snapshot.

## v0.1 — shipped

- MuJoCo backend + built-in 6-DOF arm + 4 core skills
  (`pick` / `place_on` / `push` / `home`).
- Stub planner + OpenAI-compatible VLM planner.
- 5-task starter benchmark.

## v0.2 (2026-04-18 session) — shipped

| Slice | Pillar | Notes |
|---|---|---|
| URDF import | robot | `Scene(robot_urdf=...)` + sidecar YAML; bundled Franka; `pick_cube_franka`. |
| Mesh object import | objects | `SceneObject(kind="mesh")` + sidecar; BYO CoACD + hull cache; `pick_ycb_mug`. |
| YCB object pack | objects | 10 bundled YCB items; `@ycb:<id>` shorthand; `list_builtin_ycb_objects()`. |
| Procedural scenes | objects | `scene.presets.tabletop_clutter(n, seed)`. |
| Randomized benchmark | tasks | `randomize:` YAML block + `--seeds N` with `mean ± stderr`. |
| 5 new skills | tasks | `pour`, `tap`, `open_drawer`, `close_drawer`, `stack`. |
| Drawer primitive | tasks | `SceneObject(kind="drawer")` — first articulated primitive. |
| Long-horizon composite | tasks | `pour_can_into_bowl` benchmark (pick → pour). |
| Browser live viewer | interaction | FastAPI + WS + SPA; task dropdown, Run/Reset, live MJPEG. |
| Viewer Record button | interaction | Toggle → LocalRecorder writes under `./runs/`. |
| Keyboard teleop | interaction | WASD/QE drives EE xy/z; Space toggles gripper. |
| VLM cassette | loop | Record/replay client + hand-authored red-cube cassette for CI. |
| LeRobot v3 export | loop | `export_episode()` + `robo-sandbox export-lerobot` CLI. |
| Policy replay | loop | `Policy` protocol + `ReplayTrajectoryPolicy` + `robo-sandbox run --policy`. |
| Real-robot bridge stub | loop | `RealRobotBackend` satisfies `SimBackend` Protocol; subclass to wire hardware. |
| 9 runnable examples | polish | `examples/*.py` covering every feature. |

**State at end of 2026-04-18:** 135 tests, 8 default benchmarks,
all green.

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

## See also

- Full history + sprint sequencing:
  [`TODO.md`](https://github.com/amarrmb/robosandbox/blob/main/TODO.md).
- Prior-slice specs:
  [`docs/superpowers/specs/`](https://github.com/amarrmb/robosandbox/tree/main/docs/superpowers/specs).
