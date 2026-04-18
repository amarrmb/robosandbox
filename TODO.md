# RoboSandbox — Roadmap

## Mission

> **A playground where someone can build and evaluate manipulation agents.**

Today RoboSandbox is a library that runs one kind of task (pick a cube)
with one kind of arm. To become a playground we need four things:
**object diversity, task diversity, interaction, and a closing of the
loop (record → train → deploy).** Everything in this file is justified
against that mission — if a work item doesn't move one of those pillars,
cut it.

---

## Done

### v0.1 baseline
Agent loop, VLMPlanner, built-in 6-DOF arm, 4 skills (pick / place_on /
push / home), 4-task benchmark, stub + OpenAI-compatible VLM providers.

### v0.2 — 2026-04-18 session

| Slice | Pillar | Summary |
|---|---|---|
| URDF import | robot | `Scene(robot_urdf=...)` + sidecar YAML; bundled Franka Panda; `pick_cube_franka` task. Spec: `docs/superpowers/specs/2026-04-18-urdf-import-design.md`. |
| Browser live viewer | interaction | FastAPI + WS + SPA; dropdown task picker, live MJPEG, event stream. |
| Mesh object import | objects | `SceneObject(kind="mesh")` + per-object sidecar; BYO CoACD with `~/.cache/robosandbox/mesh_hulls` cache; `collision: hull` fallback; `pick_ycb_mug` 5×-smoke green. Spec: `docs/superpowers/specs/2026-04-18-mesh-import-design.md`. |
| YCB object pack | objects | 10 bundled YCB objects (box/can/bottle/banana/apple/bowl/mug/drill/wrench/baseball). `@ycb:<id>` shorthand + `list_builtin_ycb_objects()`. Pack LICENSE. |
| Procedural scenes (1.3) | objects | `scene.presets.tabletop_clutter(n, seed)` returns non-overlapping Franka + N YCB objects; seed 0 deterministic. |
| Randomized benchmark (2.2) | tasks | Task YAML `randomize: {xy_jitter, yaw_jitter}` + `--seeds N` with `mean ± stderr`. `pick_ycb_mug --seeds 50` = 50/50. |
| New skills (2.1, 5/6) | tasks | **Pour**, **Tap**, **OpenDrawer**, **CloseDrawer**, **Stack**. Each: unit + agent-loop tests, StubPlanner routing, entry-point registered. `insert_peg` deferred. |
| Drawer scene primitive | tasks | `SceneObject(kind="drawer")` — first articulated primitive; static U-cabinet + sliding body + handle. Uses existing `displaced` success criterion. |
| Long-horizon composite (2.3) | tasks | `pour_can_into_bowl` benchmark chains pick → pour. Planner decomposes the phrase, agent executes two skills. |
| Record button (3.3) | interaction | Viewer sidebar Record toggle → LocalRecorder under `./runs/<ts>-<id>/`. CLI: `--runs-dir <path>`. |
| Teleop — keyboard (3.2) | interaction | Viewer Teleop checkbox. WASD/QE drives ee xy/z; Space toggles gripper. 1.5 cm per keystroke. Unreachable-safe. |
| VLM cassette (4.1) | loop | `CassetteVLMClient` record/replay; hand-authored red-cube cassette in `tests/cassettes/`; CI-safe VLMPointer test. |
| LeRobot export (4.2) | loop | `robosandbox.export.lerobot.export_episode(runs_dir, dst)` writes LeRobot v3 parquet + `meta/` + `videos/`. CLI: `robo-sandbox export-lerobot`. Optional `[lerobot]` extra (pyarrow). |
| Policy replay (4.3) | loop | `Policy` Protocol + `ReplayTrajectoryPolicy` + `run_policy()` loop. CLI: `robo-sandbox run --policy <ckpt_dir>`. Real LeRobot/ACT integration is BYO — we ship the plumbing. |
| Real-robot bridge (4.4) | loop | `backends.RealRobotBackend` stub satisfying `SimBackend` Protocol. Subclass + fill hardware driver — skills/motion/grasp/agent all work unchanged. Example: `examples/real_robot_swap.py`. |
| Examples (5.1) | polish | 9 runnable scripts — list_ycb, spawn_ycb_scene, custom_robot, custom_task, custom_skill, record_demo, headless_eval, llm_guided, procedural_scene. |

**State at end of 2026-04-18:** 135 tests, 8 default benchmarks, all
green.

---

## Pillar 1 — Object diversity

Everything on this pillar shipped this session (1.1 mesh import, 1.2
YCB pack, 1.3 procedural scenes). Future ideas live in the **Backlog**
section below.

---

## Pillar 2 — Task diversity

### 2.1 — `insert_peg` skill **[deferred]**
Needs a peg-hole articulated scene primitive (prismatic-jointed hole
with compliance). Its own slice — parallel to the drawer primitive
already shipped. Effort: ~1 day.

### 2.3 — More long-horizon composites **[open]**
`pour_can_into_bowl` shipped. Easy follow-ups now that drawer + stack
exist:
- "put the apple in the drawer" (drawer open → pick apple → place in
  drawer → drawer close)
- "stack three cubes by colour" (stack_n orchestrator, Tap/Home mixed)
- "tidy the table" (push distractors off, pick target)

### 2.4 — Richer randomization fields **[open]**
2.2 currently randomizes xy + yaw. Add: rgba, size, mass. Needed before
training runs care about visual invariance.

---

## Pillar 3 — Interaction

### 3.1 — Client-side orbit camera **[deferred: dedicated slice]**
Move viewer render from server-side MJPEG to client-side Three.js. Pose
stream WS + MJCF/URDF loader for Three.js + camera controls. ~2–3 days;
needs its own spec.

### 3.2 — Gamepad + continuous teleop **[extension of 3.2 keyboard]**
Keyboard teleop shipped. Add: gamepad axes → velocity integration on
the server. Effort: ~½ day.

### 3.4 — Trajectory inspector **[open]**
After a run ends, the viewer shows a scrubber: drag to replay from any
step, inspect robot joints + object poses + camera view at that
instant. Requires an in-RAM episode buffer (currently recorder writes
straight to disk). Effort: ~1 day.

---

## Pillar 4 — Loop closure

All four items shipped this session (4.1 cassette, 4.2 LeRobot export,
4.3 policy replay, 4.4 real-robot stub).

### 4.5 — First integration with a real policy checkpoint **[open]**
Pick one public LeRobot/ACT/Diffusion checkpoint, wire it into
`load_policy()`, run against `pick_cube_franka`, report success rate.
Validates the replay API against a real (not mock) learner.

### 4.6 — SO-101 reference backend **[open]**
Build the first concrete `RealRobotBackend` subclass against LeRobot's
SO-101 driver. Mechanical work on top of 4.4. Effort: ~2 days.

---

## Pillar-agnostic polish

### 5.2 — Full-mesh Franka visuals **[open]**
`--meshes full` downloads menagerie's 33 MB of OBJ visuals on demand,
caches at `~/.cache/robosandbox/`, re-enables `class="visual"` geoms.
Default stays collision-only so install is lean. Effort: ~½ day.

### 5.3 — Docs site **[open]**
MkDocs-material or Starlight. Quickstart, core concepts, custom-arm
tutorial, API reference. Effort: ~1–2 days.

### 5.4 — PyPI release **[open]**
First public release. CI builds wheel + sdist, publishes on tag.
Effort: ~½ day.

---

## Sequencing recommendation

**Next up (any order; independent):**
1. **4.5** — integrate a real policy checkpoint to validate the 4.3 API
   against something trained.
2. **3.1** — orbit camera. #1 "feels like a toy" complaint; client-side
   render is the unlock.
3. **5.3 + 5.4** — docs site + PyPI release. RoboSandbox is useful now;
   time for it to be findable.

**Bigger strategic directions after that:**
- 4.6 SO-101 integration — first real hardware target.
- 2.3 composites + 2.4 rich randomization — grow the benchmark into a
  credible evaluation suite.
- 5.2 full-mesh Franka — renders stop looking like a tech demo.

---

## Backlog / ideas (not committed)

- Full-scene randomization (lighting, camera pose, domain randomization
  knobs for sim-to-real).
- Grasp evaluation plugin (AnyGrasp / GraspNet / anti-podal / learned).
- More robot arms: UR5, xArm, SO-100 variants, two-arm setups.
- Procedural kitchens / desks / shelves beyond `tabletop_clutter`.
- Soft-body objects (fabric, containers with fluid).
- Audio feedback in viewer (gripper click, contact thump).
- Multi-robot scenes + coordination skills.
