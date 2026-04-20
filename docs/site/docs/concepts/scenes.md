# Scenes & objects

A `Scene` is the full world description that gets handed to a
`SimBackend`: robot, objects, and workspace bounds.

## `Scene`

```python
from robosandbox.types import Scene, SceneObject, Pose
from pathlib import Path

scene = Scene(
    robot_urdf=Path("/path/to/robot.urdf"),   # None → built-in 6-DOF arm
    robot_config=Path("/path/to/robot.yaml"), # sibling auto-discovered if None
    objects=(
        SceneObject(id="red_cube", kind="box", size=(0.02, 0.02, 0.02),
                    pose=Pose(xyz=(0.4, 0.0, 0.05)),
                    rgba=(0.85, 0.2, 0.2, 1.0), mass=0.05),
    ),
    workspace_aabb=((-0.5, -0.5, 0.0), (0.5, 0.5, 0.8)),
    table_height=0.0,
    gravity=(0.0, 0.0, -9.81),
)
```

It is a frozen dataclass. `sim.load(scene)` turns it into MuJoCo XML.

## `SceneObject` kinds

| `kind` | `size` semantics | Notes |
|---|---|---|
| `box` | `(sx, sy, sz)` half-extents (m) | Free body with freejoint. |
| `sphere` | `(radius,)` | — |
| `cylinder` | `(radius, half_height)` | — |
| `mesh` | unused — dims from the mesh | Needs `mesh_sidecar` **or** `mesh_path`. |
| `drawer` | `(width_y, depth_x, height_z)` inner drawer | Articulated: static cabinet + prismatic inner body + handle. |

All primitives spawn with a freejoint unless kind says otherwise.
`pose` is the initial pose; physics takes over from there.

### Meshes

There are two mutually exclusive mesh paths:

**Bundled YCB** — set `mesh_sidecar` to a sidecar YAML file, or use
the YAML task shortcut `mesh: "@ycb:<id>"`:

```yaml
objects:
  - id: mug
    kind: mesh
    mesh: "@ycb:025_mug"
    pose: {xyz: [0.42, 0.0, 0.045]}
```

**Bring-your-own** — set `mesh_path` to an OBJ/STL. CoACD decomposes it
into convex hulls; hulls are cached at
`~/.cache/robosandbox/mesh_hulls/`. Needs the `meshes` extra:
`uv pip install -e 'packages/robosandbox-core[meshes]'`.

```python
SceneObject(
    id="widget",
    kind="mesh",
    size=(0.0,),                           # unused for mesh
    mesh_path=Path("/abs/path/to/widget.obj"),
    collision="coacd",                     # or "hull" if already convex
    pose=Pose(xyz=(0.4, 0.0, 0.05)),
    mass=0.1,
)
```

`collision="hull"` skips CoACD — the sandbox trusts the mesh is
already convex. Use `"coacd"` for concave objects.

If you want to pre-decompose a mesh once and keep the result:

```bash
python scripts/decompose_mesh.py \
  --input /path/to/drill.obj \
  --out-dir assets/objects/custom/drill \
  --name drill --mass 0.3 --center-bottom
```

### Drawer primitive

This is the only articulated primitive in v0.1.

```yaml
- id: drawer_a
  kind: drawer
  size: [0.15, 0.12, 0.05]    # (width_y, depth_x, height_z) of inner drawer
  pose:
    xyz: [0.42, 0.0, 0.08]
  rgba: [0.55, 0.35, 0.2, 1.0]
  drawer_max_open: 0.12       # cap on slide travel (m)
```

The `SceneObject.id` names the sliding body (observable as
`obs.scene_objects["drawer_a"]`). A sibling `<id>_handle` body is also
tracked so skills can grasp it.

See the [`open_drawer` / `close_drawer` skills](skills-and-agents.md#skills-shipped-in-v01)
and the [`open_drawer` task](../tutorials/custom-task.md) for how the
`displaced` success criterion reads the slide motion.

## Bundled YCB catalog

Ten pre-decomposed objects ship with core:

| YCB id | Description | Mass (kg) |
|---|---|---|
| `003_cracker_box` | cracker box | 0.411 |
| `005_tomato_soup_can` | tomato soup can | 0.349 |
| `006_mustard_bottle` | mustard bottle | 0.603 |
| `011_banana` | banana | 0.066 |
| `013_apple` | apple | 0.068 |
| `024_bowl` | bowl (hollow; 11 hulls) | 0.147 |
| `025_mug` | mug (handled; 15 hulls) | 0.118 |
| `035_power_drill` | power drill | 0.895 |
| `042_adjustable_wrench` | adjustable wrench | 0.252 |
| `055_baseball` | baseball | 0.148 |

From Python:

```python
from robosandbox.tasks.loader import list_builtin_ycb_objects
list_builtin_ycb_objects()
# ['003_cracker_box', '005_tomato_soup_can', ..., '055_baseball']
```

From a YAML task, use the `@ycb:<id>` shorthand. The loader resolves
it to the bundled sidecar.

## Robots

The default built-in robot is a simple 6-DOF arm defined directly in
MJCF. To use your own robot, set
`robot_urdf`:

```python
Scene(
    robot_urdf=Path("/path/to/ur5.urdf"),            # .urdf or .xml
    robot_config=Path("/path/to/ur5.robosandbox.yaml"),  # optional sibling
    objects=(...),
)
```

`robot_config` is a sidecar YAML mapping RoboSandbox roles (arm joint
list, gripper primary joint, end-effector TCP, home pose) onto the
robot's element names. See
[custom-arm tutorial](../tutorials/custom-arm.md) for the full
schema.

The bundled Franka Panda lives at
`packages/robosandbox-core/src/robosandbox/assets/robots/franka_panda/`
— its `panda.robosandbox.yaml` is the reference schema.

## Procedural scenes

```python
from robosandbox.scene.presets import tabletop_clutter

scene = tabletop_clutter(n_objects=5, seed=0)
# Franka + 5 non-overlapping YCB objects. seed 0 is bit-exact deterministic.
```

See `examples/procedural_scene.py` for a runnable end-to-end flow.

## Full YAML task schema

Used by every file under
`packages/robosandbox-core/src/robosandbox/tasks/definitions/`.

```yaml
name: pick_ycb_mug
prompt: "pick up the mug"
seed_note: "Optional free-text context for the task author."

scene:
  robot_urdf: "@builtin:robots/franka_panda/panda.xml"          # or a path
  robot_config: "@builtin:robots/franka_panda/panda.robosandbox.yaml"
  objects:
    - id: mug
      kind: mesh
      mesh: "@ycb:025_mug"
      pose:
        xyz: [0.42, 0.0, 0.045]
        # quat_xyzw: [0, 0, 0, 1]   (optional)
      # mass: 0.12         (optional; mesh mass defaults to sidecar)
      # rgba: [r, g, b, a] (optional)

success:
  kind: lifted               # lifted | moved_above | displaced | all | any
  object: mug
  min_mm: 50

randomize:                    # optional — per-seed jitter
  xy_jitter: 0.03             # ±3 cm
  yaw_jitter: 1.57            # ±90°
```

`@builtin:` resolves inside the installed package. `@ycb:<id>`
resolves to a bundled YCB sidecar. See
[recording & export](recording-and-export.md) for what the runner
does with the success criterion.

## Related

- [Skills & agents](skills-and-agents.md) — what consumes a Scene.
- [Custom arm](../tutorials/custom-arm.md) — dropping in a URDF.
- [Custom task](../tutorials/custom-task.md) — authoring a YAML task.
