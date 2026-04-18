# Mesh Object Import — Design (Lightweight)

**Date:** 2026-04-18
**Target:** `robosandbox` v0.2 / TODO 1.1
**Status:** Decisions locked from brainstorming Q&A. Implementation can begin.

## Scope

`SceneObject(kind="mesh")` spawns a grippable free-body from OBJ/STL on **both** the built-in-arm and URDF paths. Bundled YCB mug (pre-decomposed) is the acceptance object. BYO meshes decompose on import via CoACD (optional dep) with a convex-hull fallback.

## Non-goals

- YCB pack of 10 (TODO 1.2).
- Seed randomization / success-rate reporting (TODO 2.2).
- Auto grasp annotations.
- Articulated / deformable meshes.
- Full built-in-arm MJCF→MjSpec migration (only object injection is unified).

## Decisions (from Q&A)

| # | Decision |
|---|---|
| Q1 | **Hybrid decomp policy:** bundled objects ship pre-decomposed; BYO cached on import under `~/.cache/robosandbox/mesh_hulls/<sha256>/`. |
| Q2 | **CoACD** for BYO (optional extra `robosandbox[meshes]`); convex-hull fallback via `collision: hull`. |
| Q3 | **One bundled object:** YCB 025_mug only. |
| Q4 | **Per-object sidecar** `<name>.robosandbox.yaml`, mirroring the robot pattern. |
| Q5 | **Both paths.** Unify *object injection* via MjSpec; keep built-in-arm body XML as-is. |
| Q6 | **Separate visual + collision meshes** for bundled; BYO defaults to single mesh reused for both. |
| Q7 | **Single-shot acceptance** + in-process 5× smoke test (≥4/5). |

## Schemas

**Object sidecar** (`assets/objects/ycb/025_mug/mug.robosandbox.yaml`):

```yaml
visual_mesh: mug_visual.obj            # relative to sidecar
collision_meshes:                      # pre-decomposed convex hulls
  - mug_hull_0.obj
  - mug_hull_1.obj
scale: 1.0                             # applied to all mesh files
mass: 0.15
friction: [1.5, 0.1, 0.01]
rgba: [0.9, 0.9, 0.9, 1.0]
```

**Task YAML (mesh object):**

```yaml
# Bundled sidecar
- id: mug
  kind: mesh
  mesh: "@builtin:objects/ycb/025_mug/mug.robosandbox.yaml"
  pose:
    xyz: [0.45, 0.0, 0.06]

# Bring-your-own
- id: widget
  kind: mesh
  mesh_path: /abs/path/widget.obj
  collision: coacd                     # or "hull"
  pose: {xyz: [0.4, 0.0, 0.05]}
  mass: 0.1
```

## Files

**Create:**
- `scene/mesh_conversion.py` — `MeshAsset` type, `load_bundled_mesh`, `load_byo_mesh`, `coacd_decompose`, `hull_decompose`, cache.
- `scene/mesh_injection.py` — `inject_mesh_object(spec, obj, asset)` shared by both paths.
- `assets/objects/ycb/025_mug/` — `mug_visual.obj`, `mug_hull_*.obj`, `mug.robosandbox.yaml`, `LICENSE`.
- `tasks/definitions/pick_ycb_mug.yaml`.
- `scripts/decompose_mesh.py` — offline dev tool (wraps CoACD) for creating the bundled hulls.
- `tests/test_mesh_conversion.py`, `tests/test_mesh_injection.py`, `tests/test_pick_ycb_mug.py`.

**Modify:**
- `types.py` — extend `SceneObject` (mesh sidecar ref + collision mode + scale); keep `mesh_path` field.
- `scene/mjcf_builder.build_model` — when scene has any mesh object, route objects through MjSpec (shared path with URDF) instead of string-template.
- `scene/robot_loader.inject_scene_objects` — add `mesh` branch using `inject_mesh_object`.
- `tasks/loader._object_from_dict` — parse `mesh` (bundled sidecar ref) or `mesh_path` + `collision`.
- `README.md` — "Bundled objects" section, YCB attribution.

## Implementation sequence (small commits, each keeps green)

1. `MeshAsset` + sidecar loader. Fixture test on a tiny hand-written sidecar.
2. `inject_mesh_object` — adds `<mesh>` assets + multi-geom body via MjSpec. Test against the URDF path.
3. Unify built-in-arm object injection through MjSpec when any object exists. All 36 tests stay green.
4. Pre-decompose YCB mug offline with `scripts/decompose_mesh.py`; commit assets + sidecar + LICENSE.
5. `tasks/loader` parses mesh objects; `pick_ycb_mug.yaml` + `test_pick_ycb_mug.py` (single-shot + 5× smoke).
6. BYO path: `coacd_decompose` + hash-keyed cache; hull fallback. Unit test with a tiny procedural OBJ (optional CoACD skipped if not installed).
7. README update.

## Acceptance

- `pytest` → 36 existing + new tests green.
- `robo-sandbox-bench` → 5 existing tasks green.
- `robo-sandbox-bench --tasks pick_ycb_mug` → passes single-shot.
- `test_pick_ycb_mug.py::test_smoke_5x` → ≥4/5 success.

## Risks

- **Pick skill on a mug** — handle vs rim approach axis may need a deliberately chosen pose. If the pick flakes at acceptance, tune pose before touching the skill.
- **CoACD optional dep** — import guarded; missing-dep error only fires when user asks for `collision: coacd`.
- **Bit-exact physics** — routing the built-in arm through MjSpec for object injection changes the compile path. 5 existing benchmarks must stay green; if they regress by rounding, switch to "MjSpec path only when a mesh is present; primitive-only objects keep the string-template path."
- **MjSpec meshdir** — bundled mesh files live next to their sidecar, not under the robot's meshdir. Use absolute paths when calling `spec.add_mesh`.
