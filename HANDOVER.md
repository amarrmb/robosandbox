# Handover Prompt — RoboSandbox "playground" pivot

Paste the block below into a fresh Claude Code session opened at
`/home/amar/robosandbox`. It is self-contained — the other session
needs no prior context from this one.

---

## Prompt

> You are continuing work on **RoboSandbox**, a standalone public
> repo at `/home/amar/robosandbox` (GitHub:
> `https://github.com/amarrmb/robosandbox`, private). It is a
> sim-first manipulation sandbox built on MuJoCo.
>
> **Mission (this is the north star — anchor every decision against it):**
>
> > RoboSandbox is a playground where someone can build and evaluate
> > manipulation agents. Today it's a library that runs one kind of
> > task (pick a cube) with one kind of arm. The pivot is:
> > **object diversity, task diversity, interaction, closing the loop
> > (record → train → deploy).**
>
> **What's already shipped** (read these before planning anything):
> - `TODO.md` — mission-aligned roadmap with 4 pillars + sprint
>   sequencing. Read it end to end.
> - `docs/superpowers/specs/2026-04-18-urdf-import-design.md` — design
>   spec for the URDF import slice that just landed. Mirrors the style
>   you should follow for future design docs.
> - `README.md` — user-facing narrative; last two sections ("Bundled
>   assets", "Browser live viewer") describe the current surface.
> - Recent commits (`git log --oneline -8`): URDF import + Franka
>   assets + `pick_cube_franka` benchmark + browser viewer. 36 tests
>   pass, 5/5 benchmark tasks pass.
>
> **Your first task (TODO item 1.1 — mesh object import):**
> `SceneObject(kind="mesh")` raises NotImplementedError today. Fix it.
> - Spec to follow in the design doc you write: OBJ/STL mesh files
>   spawnable as free-body graspable objects, convex-decomposed via
>   V-HACD (or equivalent) so MuJoCo contacts are stable, bundled YCB
>   mug or similar as the acceptance test.
> - Files to touch:
>   `packages/robosandbox-core/src/robosandbox/scene/mjcf_builder.py:_object_xml`
>   (built-in-arm path), `scene/robot_loader.py:inject_scene_objects`
>   (URDF path), new `scene/mesh_conversion.py` (decomp helper). Add
>   a benchmark task `tasks/definitions/pick_ycb_mug.yaml` as the
>   acceptance test (same structure as `pick_cube_franka.yaml`).
> - Done when: `robo-sandbox-bench --tasks pick_ycb_mug` succeeds
>   reliably on the bundled Franka, and the 36 existing tests +
>   5 existing benchmark tasks stay green.
>
> **Working style conventions** (from the last session, carry these
> forward):
> - Use the `superpowers:brainstorming` skill to align on the design
>   BEFORE coding. Present options with tradeoffs, get user approval
>   per section, then write the design to
>   `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`.
> - Commit in small, reviewable steps (the URDF slice shipped in 6
>   commits). Each commit must leave `pytest` + `robo-sandbox-bench`
>   green.
> - Physics is sensitive to floating-point order — a 1-ulp change in
>   gripper control compounded over 1000 steps and flipped a grasp
>   from success to failure during the URDF refactor. When refactoring
>   inner-loop math, use the form that preserves bit-exact output
>   (`open*(1-t) + closed*t`, not `open + t*(closed-open)`).
> - The outer `~/dn` directory is a personal local-only monorepo.
>   Don't commit or push at that level. This repo lives on its own.
> - Sidecar YAML + MjSpec is the pattern for "bring your own robot."
>   New robot support = `assets/robots/<name>/<name>.xml` +
>   `<name>.robosandbox.yaml`. Don't pollute `Scene` with
>   robot-specific fields.
> - Stay in scope. The mission is the mission; if a task doesn't move
>   one of the four pillars, push back or defer.
>
> **First actions** (do these in order):
> 1. Read `TODO.md` end to end.
> 2. Read `docs/superpowers/specs/2026-04-18-urdf-import-design.md` to
>    see the design-doc style.
> 3. Read the "Recent commits" via `git log --oneline -10` and glance
>    at the three most recent diffs to absorb the patterns.
> 4. Invoke `superpowers:brainstorming` to align on the mesh-import
>    design with me. Propose 2–3 approaches (V-HACD vs CoACD vs
>    manual convex hull) and a recommendation. Present sidecar schema
>    for mesh objects (mesh_path, scale, collision_decomp_strategy,
>    friction overrides), the acceptance test task shape, and
>    integration with both built-in-arm and URDF paths.
> 5. After I approve the design, write the spec doc, then implement.
>
> **Do not** start coding before we've agreed on the design.
> **Do not** skip the brainstorming skill (a previous session tried
> — it cost us time).

---

## Files the next session will need to understand

| Purpose | Path |
|---|---|
| Mission + roadmap | `TODO.md` |
| Prior design spec (style reference) | `docs/superpowers/specs/2026-04-18-urdf-import-design.md` |
| Scene composition (built-in arm path) | `packages/robosandbox-core/src/robosandbox/scene/mjcf_builder.py` |
| Scene composition (URDF path — where mesh injection lands) | `packages/robosandbox-core/src/robosandbox/scene/robot_loader.py` |
| Types (`SceneObject.kind == "mesh"` lives here) | `packages/robosandbox-core/src/robosandbox/types.py` |
| Sim backend (consumes the compiled model) | `packages/robosandbox-core/src/robosandbox/sim/mujoco_backend.py` |
| Task loader (benchmark YAMLs resolved here) | `packages/robosandbox-core/src/robosandbox/tasks/loader.py` |
| Existing Franka task (acceptance-test template) | `packages/robosandbox-core/src/robosandbox/tasks/definitions/pick_cube_franka.yaml` |
| Franka sidecar (YAML schema example) | `packages/robosandbox-core/src/robosandbox/assets/robots/franka_panda/panda.robosandbox.yaml` |
| Viewer (if you want to watch mesh imports live) | `packages/robosandbox-core/src/robosandbox/viewer/server.py` |

## Commands the next session will need

```bash
cd /home/amar/robosandbox
source .venv/bin/activate

pytest packages/robosandbox-core/tests/ -q       # 36 tests; must stay green
robo-sandbox-bench                               # 5 tasks; must stay green
robo-sandbox viewer --port 8011                  # browser viewer for live verification

git log --oneline -10                            # see what just shipped
```

## Current state at handover

- Branch: `main`, 7 commits ahead of `origin/main` (not pushed — ask user)
- Tests: 36 / 36 passing
- Benchmarks: 5 / 5 passing (home, pick_cube, pick_cube_franka, pick_from_three, push_forward)
- Python 3.13 via `.venv/` (uv-managed)
- Known blocker for "actually useful" (from the prior session's audit):
  mesh objects + object diversity. That's why item 1.1 is the next task.
