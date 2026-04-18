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

- **v0.1** — agent loop, VLMPlanner, built-in 6-DOF arm, 4 skills (pick /
  place_on / push / home), 4-task benchmark, stub + VLM providers.
- **v0.2 slice 1 — URDF import** (2026-04-18). `Scene(robot_urdf=Path)`
  loads any URDF/MJCF via MjSpec + sidecar YAML. Bundled Franka Panda
  (160 KB collision-only), `pick_cube_franka` acceptance test passes.
  Design spec: `docs/superpowers/specs/2026-04-18-urdf-import-design.md`.
- **v0.2 slice 2 — browser live viewer** (2026-04-18). `robo-sandbox
  viewer` → FastAPI + WebSocket + SPA client. Watch the agent live,
  pick tasks from a dropdown, see events stream.

---

## Pillar 1 — Object diversity (so users can grasp real things)

> The "any object" tagline is a lie until we support meshes.

### 1.1 — Mesh object import  **[next up]**
`SceneObject(kind="mesh")` raises NotImplementedError today. We need
OBJ/STL → V-HACD convex decomp → MuJoCo `<mesh>` asset + `<geom
type="mesh">` in the body. Tool: `obj2mjcf` or a hand-rolled helper.
- **Where:** `scene/mjcf_builder.py:_object_xml`,
  `scene/robot_loader.py:inject_scene_objects`, new
  `scene/mesh_conversion.py`.
- **Done when:** `SceneObject(kind="mesh", mesh_path="bunny.obj")` spawns
  a grippable bunny. Convex decomp stable enough that the agent's pick
  skill succeeds ≥80 % on the YCB mug.
- **Effort:** ~1 day.

### 1.2 — YCB object pack
Ship a small YCB subset (10 objects: mug, can, box, bowl, apple, banana,
bottle, drill, wrench, ball). Vendored under `assets/objects/ycb/` with
sidecar metadata (recommended grasp hints).
- **Why:** single biggest credibility signal — people know YCB.
- **Done when:** `SceneObject(kind="ycb", id="003_cracker_box")` resolves.
- **Effort:** ~½ day (mostly licensing + trimming meshes).

### 1.3 — Procedural scene generator
`ScenePresets.tabletop_clutter(n_objects=5, seed=0)` → randomized scene
with YCB distractors at feasible poses. Same API for `kitchen_drawer`,
`desk_push`, etc.
- **Done when:** benchmark tasks can request a preset + seed; same task
  at seeds 0..50 gives 50 distinct but solvable layouts.
- **Effort:** ~1 day.

---

## Pillar 2 — Task diversity (so evaluation means something)

> 4 tasks × 1 seed is not a benchmark. It's a smoke test.

### 2.1 — More skills
Implement: `open_drawer`, `close`, `pour`, `insert_peg`, `stack_n`,
`button_press`. Each is a constrained trajectory + verification criterion.
- **Done when:** each skill has a unit test and an agent-loop test.
- **Effort:** ~½ day per skill.

### 2.2 — Randomized benchmark suite (target: 20 tasks × 50 seeds)
Extend `tasks/loader.py` with a `randomize:` block: per-field jitter
distributions (position, rotation, color, size, mass). Runner loops
seeds automatically.
- **Done when:** `robo-sandbox-bench --seeds 50` reports mean success
  rate + standard error per task.
- **Effort:** ~1 day core + ~½ day per new task YAML.

### 2.3 — Long-horizon composites
Tasks that chain multiple skills: "put the can in the drawer," "stack
the three cubes by size." Exercise the planner's decomposition, not
just single-skill dispatch.
- **Effort:** ~½ day once skills + benchmark framework are in place.

---

## Pillar 3 — Interaction (so the viewer stops being a slideshow)

> Read-only viewer is barely more than an MP4. Users need to drive.

### 3.1 — Orbit camera  **[high leverage]**
Move the viewer's render from server-side MJPEG to client-side Three.js:
server streams joint states + object poses at 60 Hz via WS; browser
reconstructs the scene and renders with a draggable camera.
- **Why:** the #1 "feels like a toy" complaint.
- **Tradeoff:** URDF/MJCF → Three.js model sync is a real engineering
  task. Alternative: stay on MJPEG but ship a URDF loader for Three.js
  that reads the same `robot_urdf` path.
- **Effort:** ~2–3 days.

### 3.2 — Teleop (keyboard + gamepad)
`robo-sandbox viewer --teleop` → WASD drives end-effector xy, QE drives z,
arrow keys rotate, space toggles gripper. Gamepad axes map to the same.
- **Why:** demo collection. Without teleop, no human data.
- **Effort:** ~1 day. Most of it is IK-for-teleop-rate safety.

### 3.3 — Record button
Viewer exposes start/stop record. Produces an MCAP + MP4 + episode JSON
on disk. The existing `LocalRecorder` is almost enough — wire it into
the SimThread's step loop and add WS actions `{"action":"record_start/stop"}`.
- **Effort:** ~½ day.

### 3.4 — Trajectory inspector
After a run ends, the viewer shows a scrubber: drag to replay from any
step, inspect robot joints + object poses + camera view at that instant.
- **Why:** debugging skills + demos needs more than a final success/fail.
- **Effort:** ~1 day (requires recording into RAM, not just on disk).

---

## Pillar 4 — Loop closure (so users can ship, not just watch)

> "Sim-first" is fine as a starting point. "Sim-only" is a dead end.

### 4.1 — Real VLM path verified with cassette
Memory note says no VLM test has hit a real endpoint. One recording
against OpenAI or Ollama, saved as a fixture, played back in CI.
- **Where:** `tests/test_vlm_pointer.py` + new cassette.
- **Effort:** ~½ day.

### 4.2 — Data export: MCAP → LeRobot
Recorded episodes can be converted to LeRobot v3 format with a single
command. Users can then train ACT / Diffusion / GR00T on the output.
- **Why:** closes "record → train" half of the loop.
- **Effort:** ~1 day (schema mapping mostly done by LeRobot side).

### 4.3 — Policy-in-the-loop replay
Given a LeRobot checkpoint path, replace the agent loop with a policy
that runs on observations. `robo-sandbox run --policy /path/to/ckpt`.
Starts empty — users bring checkpoints. Validates the "deploy" half
without owning the training.
- **Effort:** ~1 day.

### 4.4 — Real-robot bridge (stub)
Define `SimBackend` vs `RealBackend` interface. `RealBackend` stub
talks to a hardware driver (ROS2 / serial / LeRobot). Users can swap
backends without changing skill code.
- **Why:** without this, RoboSandbox is sim-only forever.
- **Effort:** ~2 days for the stub + SO-101 reference impl.

---

## Pillar-agnostic polish

### 5.1 — Examples folder
`examples/` with ≥6 runnable scripts: custom_robot, custom_task,
custom_skill, record_demo, headless_eval, llm_guided, procedural_scene.
Each has a 10-line README and runs on a clean checkout.
- **Effort:** ~1 day.

### 5.2 — Full-mesh Franka (visual quality)
Add `--meshes full` flag to the bundled Franka sidecar. Downloads
menagerie's 33 MB of OBJ visuals on demand (cached at
`~/.cache/robosandbox/`), re-enables `class="visual"` geoms. Default
stays collision-only so install is lean.
- **Effort:** ~½ day.

### 5.3 — Docs site
MkDocs-material or Starlight. Hosted at robosandbox.dev or
`<github-pages>`. Content: quickstart, core concepts (Scene/Skill/
Agent/Perception), tutorial (custom arm + custom task), API reference.
- **Effort:** ~1–2 days depending on ambition.

### 5.4 — PyPI release
First public release. CI pipeline builds wheel + sdist, uploads on tag.
- **Effort:** ~½ day.

---

## Sequencing recommendation

**Sprint A — object diversity (1 week)**: 1.1 mesh import → 1.2 YCB pack
→ 5.1 examples (mesh-specific). Unblocks credible task content.

**Sprint B — task diversity (1 week)**: 2.1 skills (pick 3 most-used:
open_drawer, stack_n, pour) → 2.2 randomized benchmark. Now we have
real evaluation signal.

**Sprint C — interaction (1.5 weeks)**: 3.3 record button (easy win) →
3.2 teleop → 3.1 orbit camera. Turns the viewer into a workstation.

**Sprint D — loop closure (1.5 weeks)**: 4.1 VLM cassette → 4.2 LeRobot
export → 4.3 policy replay → 4.4 real-robot stub. Closes the loop.

After Sprint D, RoboSandbox is no longer a demo — it's a tool.
