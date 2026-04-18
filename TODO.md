# RoboSandbox — Roadmap

## Mission

> **A playground where someone can build and evaluate manipulation agents.**

To become a playground we need four things: **object diversity, task
diversity, interaction, and a closing of the loop (record → train →
deploy).** Everything here is justified against that mission.

---

## State (end of 2026-04-18 session)

- **165 tests** passing.
- **9 benchmark tasks** passing (home, open_drawer, pick_cube,
  pick_cube_franka, pick_cube_scrambled, pick_from_three,
  pick_ycb_mug, pour_can_into_bowl, push_forward).
- Docs site builds clean (`mkdocs build --strict`).
- CI + release workflows present. PyPI Trusted Publishing ready.
- `uv build` produces a 2.5 MB wheel with all bundled assets.

---

## Done

### v0.1
Agent loop, VLMPlanner, built-in 6-DOF arm, 4 skills, 4-task benchmark,
stub + OpenAI-compatible VLM providers.

### v0.2 (2026-04-18)

| # | Pillar | Slice |
|---|---|---|
| 1.1 | objects | Mesh object import — `SceneObject(kind="mesh")` + BYO CoACD. Spec. |
| 1.2 | objects | YCB pack (10 objects) + `@ycb:<id>` shorthand. |
| 1.3 | objects | `tabletop_clutter(n, seed)` procedural scene preset. |
| 2.1 | tasks | 5 new skills: **Pour**, **Tap**, **OpenDrawer**, **CloseDrawer**, **Stack**. |
| 2.1+ | tasks | Articulated drawer scene primitive (cabinet + sliding + handle). |
| 2.2 | tasks | `randomize:` YAML block + `--seeds N` with mean ± stderr. |
| 2.3 | tasks | First long-horizon composite: `pour_can_into_bowl`. |
| 2.4 | tasks | rgba / size / mass jitter (kind-aware skip rules for mesh/drawer). |
| 3.2 | interaction | Keyboard teleop + gamepad analog-stick teleop. |
| 3.3 | interaction | Viewer record button → `LocalRecorder` episode under `runs/`. |
| 3.4 | interaction | Trajectory inspector — scrubber over in-RAM frames after a run. |
| 4.1 | loop | CassetteVLMClient — CI-safe VLMPointer tests. |
| 4.2 | loop | LeRobot v3 dataset exporter + `robo-sandbox export-lerobot`. |
| 4.3 | loop | `Policy` protocol + `run_policy` + ReplayTrajectoryPolicy. |
| 4.4 | loop | `RealRobotBackend` stub satisfying SimBackend. |
| 4.5 | loop | `LeRobotPolicyAdapter` — drop-in wrap for LeRobot checkpoints. |
| 5.1 | polish | 9 runnable example scripts. |
| 5.2 | polish | On-demand Franka full-mesh fetcher + `robo-sandbox download-franka-visuals`. |
| 5.3 | polish | MkDocs-material docs site (15 pages; builds strict). |
| 5.4 | polish | CI workflow (Py 3.11/3.12/3.13 matrix) + PyPI release workflow (Trusted Publishing). |
| 5.4+ | polish | `pyproject.toml` — keywords, classifiers, URLs, optional extras. `uv build` green. |

---

## Open (ordered by leverage)

### 4.6 — SO-101 reference real-robot backend
First concrete `RealRobotBackend` subclass on top of 4.4. Uses
LeRobot's SO-101 driver. Effort: ~2 days. Unlocks the first real
hardware target.

### 3.1 — Client-side orbit camera (dedicated slice)
Move viewer render from server-side MJPEG to client-side Three.js.
Pose-stream WS + MJCF/URDF Three.js loader + camera controls.
Effort: ~2–3 days. Dominant "feels less like a toy" UX win.

### 4.5+ — Validate against a real checkpoint
The `LeRobotPolicyAdapter` is plumbed; user-side activity: pick a
public LeRobot/ACT/Diffusion checkpoint, wire it via the adapter, run
against `pick_cube_franka`, report success rate.

### 5.2+ — Auto-augment panda.xml with cached visual meshes
The fetcher + cache are in place (5.2). The follow-up: when visuals
are cached, `load_robot()` programmatically injects `class="visual"`
mesh geoms into the MjSpec. Sidecar schema gains a `meshes: full`
option. Requires care around MjSpec meshdir semantics.

### 2.1 — `insert_peg` skill
Needs a peg-hole articulated scene primitive (prismatic hole with
compliance). Its own slice. Effort: ~1 day.

### 3.4+ — Trajectory inspector: persist to disk + re-open
Currently the scrubber is in-RAM. Add "open recorded run" so the
inspector can replay episodes from `runs/<id>/` without a sim.
Effort: ~1 day.

### 5.3+ — Tutorial cassette
Hand-authored VLM cassette for the LLM-guided tutorial path so the
docs page runs end-to-end in CI without API keys. Builds on 4.1's
cassette infra.

---

## Backlog / ideas (not committed)

- **Policy training loop**: go beyond replay — integrate LeRobot's
  training entrypoint against the exporter's parquet output.
- **Grasp evaluation plugin**: AnyGrasp / GraspNet / antipodal /
  learned grasp score heads as drop-in replacements for
  `AnalyticTopDown`.
- **More arms**: UR5, xArm, SO-100 variants, two-arm setups.
- **Procedural scene kinds**: kitchen / desk / shelf beyond
  `tabletop_clutter`.
- **Soft-body objects**: fabric, containers with fluid, ropes.
- **Audio feedback**: gripper click, contact thump in the viewer.
- **Multi-robot scenes**: coordination skills, shared workspace.
- **Sim-to-real domain randomization knobs**: lighting, camera pose,
  noise injection on observation.state.
