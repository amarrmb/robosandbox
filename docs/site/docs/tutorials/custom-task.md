# Tutorial: custom task

Task YAMLs bundle a `Scene` + natural-language `prompt` + declarative
success criterion. No executable check code — diffable, versionable,
shippable.

## Minimal task

```yaml
name: my_pick_apple
prompt: "pick up the apple"
seed_note: "Demo task authored by hand."
scene:
  robot_urdf: "@builtin:robots/franka_panda/panda.xml"
  robot_config: "@builtin:robots/franka_panda/panda.robosandbox.yaml"
  objects:
    - id: apple
      kind: mesh
      mesh: "@ycb:013_apple"
      pose:
        xyz: [0.42, 0.0, 0.05]
success:
  kind: lifted
  object: apple
  min_mm: 50
```

Save as `my_pick_apple.yaml`. Load + run:

```python
from pathlib import Path
from robosandbox.tasks.loader import load_task

task = load_task(Path("my_pick_apple.yaml"))
print(task.name, task.prompt, len(task.scene.objects))
```

Full Python wrapper — identical shape to the custom-arm tutorial, just
drive `agent.run(task.prompt)` instead of a hard-coded string. See
`examples/custom_task.py` for a runnable reference.

## Resolving `@` prefixes

Task YAMLs support two shortcuts:

- `@builtin:<path>` — resolved inside the installed
  `robosandbox.assets` tree (`robots/franka_panda/panda.xml`,
  `objects/ycb/013_apple/`, …).
- `@ycb:<id>` — resolved to the bundled YCB sidecar
  `robosandbox/assets/objects/ycb/<id>/<short>.robosandbox.yaml`.

Both work anywhere a path is expected (`robot_urdf`, `robot_config`,
`mesh`). Outside those two, use an absolute filesystem path.

## Randomize block

Add a `randomize` block to enable per-seed jitter. Fields shipped in
v0.1:

```yaml
randomize:
  xy_jitter: 0.03      # meters: ±3 cm in x and ±3 cm in y
  yaw_jitter: 1.57     # radians: ±90° about z
```

Every `SceneObject` pose is perturbed by the seeded RNG at load time.

- **Seed 0** is always the deterministic base layout (bit-exact with
  `--seeds 1`).
- **Seeds ≥ 1** sample uniform perturbations keyed on the seed.

Run:

```bash
uv run robo-sandbox-bench --seeds 50
```

Aggregated `mean ± stderr` per task appears at the end. More fields
(rgba, size, mass) are on the roadmap —
see [roadmap 2.4](../reference/roadmap.md#pillar-2-task-diversity).

## Success criteria

Declarative shapes the runner understands:

### `lifted`

```yaml
success:
  kind: lifted
  object: red_cube
  min_mm: 50
```

Initial object Z → final object Z. Fails if either is missing.

### `moved_above`

```yaml
success:
  kind: moved_above
  object: tomato_soup_can
  target: bowl
  xy_tol: 0.10       # can xy within 10 cm of bowl xy
  min_dz: 0.06       # can at least 6 cm above bowl
```

Used by the long-horizon `pour_can_into_bowl` task.

### `displaced`

```yaml
success:
  kind: displaced
  object: red_cube
  direction: forward         # forward | back | left | right
  min_mm: 30
```

Directional horizontal motion. Used by `push_forward` and the drawer
open/close tasks.

### `all` / `any`

Compose sub-criteria:

```yaml
success:
  kind: all
  checks:
    - {kind: lifted, object: red_cube, min_mm: 50}
    - {kind: moved_above, object: red_cube, target: green_cube, xy_tol: 0.03, min_dz: 0.015}
```

## Worked example — push task

```yaml
name: push_forward
prompt: "push the red cube forward"
seed_note: "Tests push primitive in isolation."
scene:
  objects:
    - id: red_cube
      kind: box
      size: [0.014, 0.014, 0.014]
      pose:
        xyz: [0.03, 0.0, 0.07]
      rgba: [0.85, 0.2, 0.2, 1.0]
success:
  kind: displaced
  object: red_cube
  direction: forward
  min_mm: 30
```

No `robot_urdf` → uses the built-in 6-DOF arm. Runs under the
`push_forward` name in `robo-sandbox-bench`.

## Adding it to the benchmark

Drop the YAML under
`packages/robosandbox-core/src/robosandbox/tasks/definitions/` and
the runner auto-discovers it:

```bash
uv run robo-sandbox-bench
# ... your_task now appears in the summary
```

To exclude an in-progress task from the default run, prefix the
filename with `_experimental_` — the runner skips those unless
explicitly named.

```bash
uv run robo-sandbox-bench _experimental_stack_two   # explicit opt-in
```

## See also

- [Scenes & objects](../concepts/scenes.md) — full `SceneObject`
  shape.
- `examples/custom_task.py` — hands-on, runnable.
- [CLI: `robo-sandbox-bench`](../reference/cli.md#robo-sandbox-bench)
  for all runner flags.
