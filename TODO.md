# RoboSandbox — v0.1 → v0.2 TODO

What's pending for the repo to be actually usable, grouped by severity
and with concrete next actions. Estimates assume a focused afternoon
unless noted.

---

## Critical — blocks the "any arm, any object" headline

Without these two, the tagline is a lie.

### 1. URDF import for the robot

**Problem:** `Scene.robot_urdf` is a dataclass field, but
`scene/mjcf_builder.py` ignores it and always emits the hardcoded
6-DOF arm. Dropping in a Franka / UR5 / SO-101 URDF does nothing.

**Fix:**
- Detect `scene.robot_urdf is not None`; load via
  `mujoco.MjModel.from_xml_path` (MuJoCo accepts URDF natively) into a
  sub-tree.
- Wire the arm's joint names, end-effector site, and gripper actuator
  into `MuJoCoBackend`'s cached addresses. Either derive them via URDF
  metadata (tag the URDF with `<mujoco><compiler>` directives) or
  require a small YAML sidecar that maps roles to link/joint names.
- Verify SO-101, Franka Panda (menagerie), and UR5 all load + home.

**Where:** `packages/robosandbox-core/src/robosandbox/scene/mjcf_builder.py`,
`sim/mujoco_backend.py`.

**Effort:** ~1 day (mostly debugging URDF → MuJoCo quirks per arm).

**Done when:** `robo-sandbox run --urdf /path/to/panda.urdf "pick the cube"` works.

---

### 2. Mesh object import

**Problem:** `SceneObject(kind="mesh")` raises `NotImplementedError` in
`scene/mjcf_builder.py:_object_xml`. Only primitives (box/sphere/
cylinder) work.

**Fix:**
- Accept OBJ/STL via `mesh_path`.
- Convert collision mesh via V-HACD (convex decomposition) so MuJoCo
  contacts are stable.
- Emit `<mesh>` asset definition in the MJCF and a `<geom type="mesh"
  mesh="<name>">` in the body.
- Tool integration: `obj2mjcf` or hand-rolled conversion.

**Where:** `scene/mjcf_builder.py:_object_xml` + a new
`scene/mesh_conversion.py`.

**Effort:** ~1 day including convex decomposition.

**Done when:** `SceneObject(kind="mesh", mesh_path="bunny.obj")` spawns a
grippable bunny.

---

## Important — blocks believable demos

### 3. VLM path end-to-end verification

**Problem:** All VLM tests use stubs. The real path (OpenAI / Ollama)
has never been executed against a live endpoint in this project. The
projection math in `VLMPointer._pixel_to_world` was only validated
against the `top` camera, not the default `scene` camera.

**Fix:**
- Run `robo-sandbox run --vlm-provider openai "pick the red cube"` with
  a real key; verify correct plan + correct perception.
- Record the VLM request/response as a fixture cassette for a
  regression test.
- Validate projection math against both cameras for known cube
  positions.

**Where:** `perception/vlm_pointer.py`, new
`tests/test_vlm_pointer_camera_scene.py`.

**Effort:** ~half day.

**Done when:** there's a committed VCR cassette showing the full agent
loop succeeding with a real VLM, replayable by CI.

---

### 4. `place_on` verifies placement

**Problem:** `PlaceOn` returns `success=True` after opening the gripper
regardless of whether the cube actually landed on the target. Held
cube can fall mid-traverse and we claim success.

**Fix:**
- After release + retract, re-observe. Find the cube closest to the
  release pose (or, more principled: track the cube the agent last
  picked via `AgentContext.last_picked`).
- Check `final_cube.xy ≈ target.xy` within tolerance AND
  `final_cube.z > target.z`. If not, return
  `SkillResult(success=False, reason="misplaced")`.

**Where:** `skills/place.py`, possibly
`agent/context.py` (to thread `last_picked` forward).

**Effort:** half day.

**Done when:** misplaced stack attempts return failure → agent replans.

---

### 5. Force-controlled grip (fixes stack + pick reliability)

**Problem:** MuJoCo position-controlled grippers keep pressing toward
"fully closed" even when a cube is in the way. Fingers squeeze past
the cube, cube pops out mid-lift or mid-traverse. Stack fails
deterministically; Pick succeeds ~65% of the time.

**Fix path A (proper):** Force-control actuator. Replace
`<position>` with `<general>` or `<motor>` + a control law that
commands constant force until contact velocity = 0.

**Fix path B (hack that works):** Weld-on-grasp. When Pick's verify
passes, add a MuJoCo `<equality type="weld">` between the palm and the
cube. Remove on PlaceOn's release. Runtime equality management in
MuJoCo requires rebuilding MjModel which is expensive; a trick is to
have pre-declared "weld slots" in the MJCF, disabled at start, and
flip `eq_active` at runtime.

**Where:** `sim/mujoco_backend.py`, `scene/mjcf_builder.py`,
`skills/pick.py`, `skills/place.py`.

**Effort:** 1-2 days for (A); ~1 day for (B).

**Done when:** `robo-sandbox-bench --tasks _experimental_stack_two
--seeds 10` passes 10/10 and we re-enable it as a default task.

---

### 6. Projection-math fix for the scene camera

**Problem:** `_pixel_to_world` math was derived assuming MuJoCo
camera convention; verified against `top` camera in tests but the
default `scene` camera has a tilted orientation that may expose a
sign bug in my quat-to-rotation-matrix conversion.

**Fix:** regression test covering a cube at known world coords,
rendering from both `top` and `scene` cameras, asserting
`_pixel_to_world` returns world coords within 3cm.

**Where:** `perception/vlm_pointer.py`,
`tests/test_vlm_pointer_projection.py`.

**Effort:** ~2 hours.

---

## Usability — blocks `pip install` and "drop in and try"

### 7. Publish `robosandbox` to PyPI

**Problem:** README says `pip install robosandbox`. It's not on PyPI.

**Fix:**
- GitHub Actions workflow: on a tag push, build wheel + source dist,
  run tests, publish via trusted publisher (OIDC).
- Take the name before someone else does.

**Where:** `.github/workflows/publish.yml`.

**Effort:** half day.

---

### 8. GitHub Actions CI

**Problem:** No CI. Tests pass on my laptop; no guarantee on Mac or
Windows, no regression protection on PRs.

**Fix:**
- Matrix job: Linux + macOS, Python 3.10/3.11/3.12.
- Steps: `uv sync`, `uv run pytest`, `uv run ruff check`.
- Cache the uv venv.
- Add a CI badge to the README.

**Where:** `.github/workflows/ci.yml`.

**Effort:** half day.

---

### 9. `examples/` directory

**Problem:** No entry point for "how do I add a skill / task / arm?"
The code is the only documentation.

**Fix:** three short example files:
- `examples/custom_skill.py` — add a `Wave` skill that waves the arm.
- `examples/custom_task.yaml` — new benchmark task with a
  `moved_above` success.
- `examples/custom_arm.py` — load a Franka URDF and pick a cube
  (depends on #1 landing first).

**Where:** new top-level `examples/`.

**Effort:** few hours each once the underlying capability exists.

---

### 10. Docs site

**Problem:** Just the README. No per-page docs, no tutorial, no
architecture deep-dive.

**Fix:** mkdocs-material skeleton:
- `Overview`, `Quickstart`, `Architecture`, `Custom skills`, `Custom
  arms`, `Benchmark`, `Roadmap`, `API reference` (pdoc).
- Deploy via GitHub Pages.

**Where:** new top-level `docs/`.

**Effort:** half day setup + 1 day content.

---

## Strategic — blocks the "why this and why DN" story

### 11. `robosandbox-dn` plugin (the closed loop)

**Problem:** Without this, RoboSandbox is a cute sandbox, not a pipe
into DeviceNexus. The commercial argument requires record → train →
deploy-back working.

**Fix:**
- `packages/robosandbox-dn/`:
  - `DNRecordSink` (`RecordSink` protocol): writes MCAP locally, then
    uploads via existing Nexus data-ingestion endpoint (same as
    Loopback). Batch on `end_episode`. Tags: `source=robosandbox`,
    task_prompt, plan, success, episode_id.
  - `DNLearnedSkill` (`Skill` protocol): pulls a checkpoint from a
    Nexus tenant (REST + signed URL, same as Forge download), caches
    locally, runs via LeRobot / ACT / SmolVLA inference. Formats
    observation, rolls out N steps, verifies like the scripted skill.
- Auth via env var (API key or JWT) referenced in config. Placeholder
  key for local no-auth endpoints (same pattern as current ollama
  support).
- CLI: `robo-sandbox run --recorder dn --skill pick=dn_learned ...`.

**Where:** new `packages/robosandbox-dn/` with its own pyproject,
separate publishable package.

**Effort:** ~2-3 days.

**Done when:** the 8-line demo from the design spec works:
```
# Day 1: collect
robo-sandbox run --recorder dn --tenant $DN_TENANT --seeds 50 pick_cube
# (train in NCC console)
# Day 2: deploy
robo-sandbox run --skill pick=dn_learned --checkpoint nexus://... pick_cube
```

---

### 12. Collision-aware motion

**Problem:** `plan_linear_cartesian` interpolates in a straight line
through whatever happens to be in the way. In multi-object scenes the
arm plows through other cubes during traverses.

**Fix path A:** Optional `robosandbox-curobo` plugin registering a
`curobo` MotionPlanner that does GPU-accelerated collision-aware
planning.

**Fix path B:** Integrate MoveIt2 via ROS2 (heavier, for customers
already in ROS).

**Where:** new `packages/robosandbox-curobo/`.

**Effort:** 1 day for (A) integration once cuRobo is installed.

---

### 13. Real-robot bridge

**Problem:** "Sim-first" is currently "sim-only." No path from a
working sandbox episode to a real robot.

**Fix:** Port DeviceNexus's LeRobot adapter (device-side serial bus
control) and ROS2 adapter into a
`packages/robosandbox-real-lerobot/` plugin that registers as a
`SimBackend` (reusing the interface). `Observation.depth` becomes
optional (most real cameras don't have it) and perception falls back
to VLMPointer + RGB-only projection using table-plane assumption.

**Where:** new `packages/robosandbox-real-lerobot/` +
`packages/robosandbox-real-ros2/`.

**Effort:** ~1-2 weeks given the existing DN code to port.

---

## Ordering recommendation

**Week 1 — "actually usable":**
1. URDF import (#1) ⬅ start here
2. Mesh import (#2)
3. VLM path E2E verify (#3)
4. `place_on` verify (#4)
5. CI + PyPI (#7, #8)

**Week 2 — "demos don't embarrass me":**
6. Force-controlled grip (#5)
7. Projection math fix (#6)
8. Examples directory (#9)

**Week 3+ — "commercial story":**
9. `robosandbox-dn` (#11)
10. `robosandbox-curobo` (#12)
11. Docs site (#10)

**Backlog:** real-robot bridge (#13), Molmo plugin, Contact-GraspNet
plugin, AnyGrasp contrib package.

---

*Last updated: 2026-04-17 after the initial v0.1 push to
`github.com/amarrmb/robosandbox`.*
