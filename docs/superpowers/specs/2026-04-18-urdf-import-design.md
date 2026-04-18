# URDF Import for the Robot — Design

**Date:** 2026-04-18
**Target:** `robosandbox` v0.2 slice #1
**Status:** Approved, ready for implementation plan

---

## Problem

`Scene.robot_urdf` is a field on the Scene dataclass but `scene/mjcf_builder.py` ignores it and always emits the hardcoded self-contained 6-DOF arm. The "any arm" tagline is a lie: dropping in a Franka, UR5, or SO-101 URDF does nothing.

This slice makes `Scene(robot_urdf=Path("panda.xml"), objects=...)` actually load the referenced robot into MuJoCo, wire up its joints/site/gripper in `MuJoCoBackend`, and leave a clean extension point for arbitrary URDFs — without breaking the built-in zero-setup arm.

## Goals

- `Scene(robot_urdf=Path("panda.xml"))` loads a real Franka Panda into MuJoCo and completes a pick.
- `Scene(robot_urdf=None)` continues to use the built-in 6-DOF arm unchanged. 21 existing tests + 4 existing benchmark tasks still pass.
- One new benchmark task (`pick_cube_franka`) is the E2E acceptance test.
- `sim/` module contains zero URDF parsing; all scene-authoring logic lives in `scene/`.
- Architecture admits future arbitrary URDFs (UR5, SO-101, user-provided) without refactoring the boundary.

## Non-goals

- Automatic actuator injection for raw URDFs (URDFs without MuJoCo-specific markup). Deferred to a later slice. In v0.2 the sidecar YAML must name actuators that already exist in the file.
- Shipping UR5 or SO-101 assets. Franka is the single validation target. The loader must be usable for them (clean extension point), but we don't test them in CI.
- Mesh-object support (`SceneObject(kind="mesh")`). That's a separate TODO item.
- Renaming `Scene.robot_urdf`. Field accepts both `.urdf` and `.xml` because `MjSpec.from_file` handles both; the name is a minor wart we tolerate.

## Architecture

Four existing files change, two new files, one new assets directory:

```
packages/robosandbox-core/src/robosandbox/
  scene/
    __init__.py         (re-export build_model, RobotSpec, load_robot)
    robot_spec.py       NEW — frozen RobotSpec dataclass
    robot_loader.py     NEW — load_robot(urdf_path, config_path) -> (MjSpec, RobotSpec)
    mjcf_builder.py     MODIFIED — keeps build_mjcf() for built-in arm;
                        adds build_model(scene) -> (MjModel, RobotSpec) top-level entry
                        adds BUILTIN_ROBOT_SPEC module constant
  sim/
    mujoco_backend.py   MODIFIED — consumes RobotSpec; drops module-level ARM_JOINTS/
                        GRIPPER_JOINT/EE_SITE constants and hardcoded neutral pose
  tasks/
    loader.py           MODIFIED — plumbs robot_urdf + robot_config from YAML;
                        resolves @builtin: prefix to packaged assets
    definitions/
      pick_cube_franka.yaml  NEW — acceptance test task
  assets/
    robots/
      franka_panda/
        panda.xml              (from mujoco_menagerie, ~50 KB)
        panda.robosandbox.yaml NEW — sidecar
        assets/                (meshes, ~4 MB total)
        LICENSE                (menagerie attribution)

packages/robosandbox-core/tests/
  test_robot_loader.py    NEW — unit tests for loader
  test_franka_pick.py     NEW — integration test with bundled Franka
```

**Split rationale:** `sim/` consumes a `RobotSpec` produced by `scene/`. No hardcoded joint names, no YAML parsing, no URDF logic in `sim/`. This is the boundary that preserves "sim backend agnostic to robot" — replacing MuJoCo with PyBullet later doesn't re-derive joint names.

## Core Types

### `scene/robot_spec.py`

```python
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class RobotSpec:
    """Everything a SimBackend needs to drive a specific robot.

    Produced by scene/robot_loader.py (URDF path) or scene/mjcf_builder.py
    (built-in arm). Names refer to elements in the compiled MjModel.
    """
    arm_joint_names: tuple[str, ...]
    arm_actuator_names: tuple[str, ...]
    gripper_joint_names: tuple[str, ...]   # primary + mimic siblings
    gripper_primary_joint: str
    gripper_actuator_name: str
    ee_site_name: str
    home_qpos: tuple[float, ...]            # length == len(arm_joint_names)
    gripper_open_qpos: float
    gripper_closed_qpos: float

    def __post_init__(self) -> None:
        if len(self.home_qpos) != len(self.arm_joint_names):
            raise ValueError(
                f"RobotSpec.home_qpos length {len(self.home_qpos)} != "
                f"arm_joint_names length {len(self.arm_joint_names)}"
            )
        if self.gripper_primary_joint not in self.gripper_joint_names:
            raise ValueError(
                f"gripper_primary_joint {self.gripper_primary_joint!r} not in "
                f"gripper_joint_names {self.gripper_joint_names}"
            )
```

### Built-in arm spec

Defined as a module constant in `scene/mjcf_builder.py` so the built-in path produces the same shape:

```python
BUILTIN_ROBOT_SPEC = RobotSpec(
    arm_joint_names=("j1", "j2", "j3", "j4", "j5", "j6"),
    arm_actuator_names=("a1", "a2", "a3", "a4", "a5", "a6"),
    gripper_joint_names=("left_finger_joint", "right_finger_joint"),
    gripper_primary_joint="left_finger_joint",
    gripper_actuator_name="a_gripper",
    ee_site_name="ee_site",
    home_qpos=(0.0, -0.4, 1.2, -0.8, 0.0, 0.0),
    gripper_open_qpos=-0.035,
    gripper_closed_qpos=0.0,
)
```

## Sidecar YAML Schema

```yaml
# panda.robosandbox.yaml (sibling to panda.xml, or pointed at explicitly)

arm:
  joints: [joint1, joint2, joint3, joint4, joint5, joint6, joint7]
  actuators: [actuator1, actuator2, actuator3, actuator4, actuator5, actuator6, actuator7]
  home_qpos: [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

gripper:
  joints: [finger_joint1, finger_joint2]        # all gripper joints including mimics
  primary_joint: finger_joint1                   # the one we control via ctrl
  actuator: actuator8
  open_qpos: 0.04
  closed_qpos: 0.0

ee_site:
  # Exactly one of the two modes:
  name: existing_site_name                       # mode A: reference an existing site
  # --- or ---
  inject:                                        # mode B: create a site at an existing body
    attach_body: hand
    xyz: [0.0, 0.0, 0.1034]
    quat_xyzw: [0.0, 0.0, 0.0, 1.0]              # optional; defaults to identity

base_pose:                                       # where to place the robot root in world
  xyz: [-0.32, 0.0, 0.04]
  quat_xyzw: [0.0, 0.0, 0.0, 1.0]                # optional; defaults to identity
```

**Required fields:** `arm.joints`, `arm.actuators`, `arm.home_qpos`, `gripper.joints`, `gripper.primary_joint`, `gripper.actuator`, `gripper.open_qpos`, `gripper.closed_qpos`, `ee_site` (one of `name` / `inject`), `base_pose.xyz`.

**Loader contract:**
- Exactly one of `ee_site.name` / `ee_site.inject` — error otherwise.
- `len(arm.actuators) == len(arm.joints)` — error otherwise.
- `len(arm.home_qpos) == len(arm.joints)` — error otherwise.
- `gripper.primary_joint ∈ gripper.joints` — error otherwise.
- All names referenced must exist in the compiled model (validated after compile).

## Data Flow

```
Scene.robot_urdf is None
  → mjcf_builder.build_model(scene):
      mjcf_str = build_mjcf(scene)                   (existing, unchanged)
      model = MjModel.from_xml_string(mjcf_str)
      return (model, BUILTIN_ROBOT_SPEC)

Scene.robot_urdf = Path("panda.xml")
  → mjcf_builder.build_model(scene):
      spec, robot_spec = robot_loader.load_robot(
          scene.robot_urdf, scene.robot_config
      )
      # spec is an MjSpec with the robot loaded; objects not yet added
      _inject_scene_objects(spec, scene)               (adds free-bodies to worldbody)
      _inject_scene_decor(spec, scene)                 (floor, table, default cameras)
      model = spec.compile()
      validate_names(model, robot_spec)                (raises if a name is missing)
      return (model, robot_spec)

  → MuJoCoBackend.load(scene):
      model, robot_spec = build_model(scene)
      self._model = model
      self._robot = robot_spec
      self._arm_qpos_adr = [model.joint(n).qposadr[0] for n in robot_spec.arm_joint_names]
      self._gripper_qpos_adr = model.joint(robot_spec.gripper_primary_joint).qposadr[0]
      self._arm_ctrl_adr = [model.actuator(n).id for n in robot_spec.arm_actuator_names]
      self._gripper_ctrl_adr = model.actuator(robot_spec.gripper_actuator_name).id
      self._ee_site_id = model.site(robot_spec.ee_site_name).id
      # ...rest of load is unchanged
      self.reset()
```

### `scene/robot_loader.py` internals

```
load_robot(urdf_path, config_path):
    sidecar_path = resolve_sidecar(urdf_path, config_path)
        # explicit config_path wins; else look for <stem>.robosandbox.yaml next to urdf
        # raise RobotConfigNotFoundError listing tried paths if neither found
    sidecar = yaml.safe_load(sidecar_path)
    validate_sidecar_shape(sidecar)   # required fields, types, mutually-exclusive ee_site modes

    spec = mujoco.MjSpec.from_file(str(urdf_path))   # handles both .urdf and .xml

    if "inject" in sidecar["ee_site"]:
        inj = sidecar["ee_site"]["inject"]
        attach_body = spec.body(inj["attach_body"])   # raises if missing
        site = attach_body.add_site(
            name="robosandbox_ee_site",
            pos=inj["xyz"],
            quat=_quat_wxyz_from_xyzw(inj.get("quat_xyzw", [0, 0, 0, 1])),
        )
        ee_site_name = "robosandbox_ee_site"
    else:
        ee_site_name = sidecar["ee_site"]["name"]

    # Apply base_pose by translating the root body of the robot.
    # MJCF robots typically have a single top-level body under worldbody; URDFs
    # likewise get a single root body after MuJoCo conversion. We set its pos/quat.
    root = _find_robot_root_body(spec)
    root.pos = sidecar["base_pose"]["xyz"]
    root.quat = _quat_wxyz_from_xyzw(sidecar["base_pose"].get("quat_xyzw", [0, 0, 0, 1]))

    robot_spec = RobotSpec(
        arm_joint_names=tuple(sidecar["arm"]["joints"]),
        arm_actuator_names=tuple(sidecar["arm"]["actuators"]),
        gripper_joint_names=tuple(sidecar["gripper"]["joints"]),
        gripper_primary_joint=sidecar["gripper"]["primary_joint"],
        gripper_actuator_name=sidecar["gripper"]["actuator"],
        ee_site_name=ee_site_name,
        home_qpos=tuple(sidecar["arm"]["home_qpos"]),
        gripper_open_qpos=float(sidecar["gripper"]["open_qpos"]),
        gripper_closed_qpos=float(sidecar["gripper"]["closed_qpos"]),
    )
    return spec, robot_spec
```

### Scene decor for URDF path

The built-in arm's MJCF includes a floor, a table, two cameras, and lights. For the URDF path, `_inject_scene_decor(spec, scene)` adds the same elements programmatically via `MjSpec`:
- A plane floor geom at z=0 on worldbody
- A box table at `(0.1, 0, 0.02)` sized `(0.4, 0.4, 0.02)` (matches built-in positioning)
- `scene` camera at `(0.9, -0.9, 0.9)` pointing at the workspace
- `top` camera at `(0, 0, 1.2)` looking down
- Two directional lights

This matches the built-in arm's layout so the existing `"scene"` camera name works identically for both paths, and existing perception/projection code is robot-agnostic.

Future: make decor configurable via sidecar or Scene; out of scope for this slice.

### MuJoCoBackend refactor

```python
# Remove module-level constants:
# ARM_JOINTS, GRIPPER_JOINT, EE_SITE  --> deleted

class MuJoCoBackend:
    def __init__(self, render_size=(480, 640), camera="scene"):
        # unchanged setup, plus:
        self._robot: RobotSpec | None = None

    def load(self, scene: Scene) -> None:
        self._model, self._robot = build_model(scene)
        self._data = mujoco.MjData(self._model)
        # ...renderers unchanged
        # cache addresses from self._robot (see Data Flow)

    def reset(self) -> None:
        mujoco.mj_resetData(self._model, self._data)
        for adr, q in zip(self._arm_qpos_adr, self._robot.home_qpos):
            self._data.qpos[adr] = q
        self._data.qpos[self._gripper_qpos_adr] = self._robot.gripper_open_qpos
        for adr, q in zip(self._arm_ctrl_adr, self._robot.home_qpos):
            self._data.ctrl[adr] = q
        self._data.ctrl[self._gripper_ctrl_adr] = self._robot.gripper_open_qpos
        mujoco.mj_forward(self._model, self._data)
        self._t = 0.0

    def step(self, target_joints=None, gripper=None) -> None:
        if target_joints is not None:
            arr = np.asarray(target_joints, dtype=np.float64).ravel()
            if arr.shape != (len(self._robot.arm_joint_names),):
                raise ValueError(...)
            # write ctrl...
        if gripper is not None:
            # Linear interp from open->closed qpos via 0..1 semantic input
            t = float(np.clip(gripper, 0.0, 1.0))
            ctrl = self._robot.gripper_open_qpos + t * (
                self._robot.gripper_closed_qpos - self._robot.gripper_open_qpos
            )
            self._data.ctrl[self._gripper_ctrl_adr] = ctrl
        mujoco.mj_step(self._model, self._data)
        self._t += self._model.opt.timestep

    @property
    def n_dof(self) -> int:
        return len(self._robot.arm_joint_names)

    @property
    def joint_names(self) -> list[str]:
        return list(self._robot.arm_joint_names)
```

**Gripper semantic change:** old code mapped `ctrl = -0.035 * (1 - closed)`, which assumes the built-in arm's specific open/closed qpos values. New code linearly interpolates between the spec's named values, which works for any robot. The built-in arm's open/closed values are preserved in `BUILTIN_ROBOT_SPEC`, so behavior is identical on that path.

## Error Handling

All loader errors are subclasses of `RobotConfigError` (new exception in `scene/robot_loader.py`):

- `RobotConfigNotFoundError(urdf_path, tried_paths)` — no sidecar found; lists paths checked.
- `RobotConfigValidationError(field, reason)` — YAML parsed but fails schema (missing field, wrong type, `ee_site` has both/neither mode, length mismatches).
- `RobotConfigMismatchError(kind, name, available)` — sidecar references a joint/actuator/body/site name not in the compiled model. `available` lists what's actually there.
- `RobotModelCompileError(urdf_path, cause)` — MuJoCo's `FatalError`/`ValueError` at compile time, wrapped with the file path.

The built-in arm path never raises these; they only fire on the URDF path.

## Bundled Franka Assets

**Source:** [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) `franka_emika_panda/` (Apache 2.0).

**Copied into repo** at `packages/robosandbox-core/src/robosandbox/assets/robots/franka_panda/`:
- `panda.xml` — the MJCF scene
- `assets/` — STL meshes referenced by `panda.xml`
- `LICENSE` — menagerie's LICENSE file
- `panda.robosandbox.yaml` — new, written for this slice

**Size estimate:** ~4 MB of meshes. Acceptable given the wheel already force-includes `assets/` and the repo is not size-constrained.

**Attribution** added to `README.md` under a "Bundled assets" section, crediting mujoco_menagerie + referencing their LICENSE.

**Menagerie scene trimming:** menagerie ships two files — `panda.xml` (scene with floor/lights) and `mjx_panda.xml` / similar variants. We want the robot-only file. If menagerie's `panda.xml` inlines scene decor (floor, skybox, lights, `scene` camera), strip those elements so the checked-in `panda.xml` contains only: `<compiler>`, `<option>` (minimal), `<asset>` (robot meshes), `<worldbody>` with the `panda_link0` root body hierarchy, `<actuator>`, and any `<equality>` for the finger mimic. Decor (floor, table, `scene`/`top` cameras, lights) is added by `_inject_scene_decor` at load time. Rationale: robosandbox owns the scene layout so it's consistent across robots.

**`panda.robosandbox.yaml` content:**

Joint/actuator names come from menagerie's `panda.xml` directly. The ee_site uses `inject` mode because menagerie doesn't ship a named TCP site — we attach one at the `hand` body with the standard 0.1034 m TCP offset.

```yaml
arm:
  joints: [joint1, joint2, joint3, joint4, joint5, joint6, joint7]
  actuators: [actuator1, actuator2, actuator3, actuator4, actuator5, actuator6, actuator7]
  home_qpos: [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

gripper:
  joints: [finger_joint1, finger_joint2]
  primary_joint: finger_joint1
  actuator: actuator8
  open_qpos: 0.04
  closed_qpos: 0.0

ee_site:
  inject:
    attach_body: hand
    xyz: [0.0, 0.0, 0.1034]

base_pose:
  xyz: [-0.32, 0.0, 0.04]
```

**Verification step during implementation:** before writing the YAML, run `python3 -c "import mujoco; m = mujoco.MjModel.from_xml_path('panda.xml'); print([m.joint(i).name for i in range(m.njnt)]); print([m.actuator(i).name for i in range(m.nu)])"` to confirm actual names. Don't trust the YAML schema-writing from memory.

## New Benchmark Task

`tasks/definitions/pick_cube_franka.yaml`:

```yaml
name: pick_cube_franka
prompt: "pick up the red cube"
seed_note: "Franka Panda from bundled menagerie assets; validates URDF/MJCF import path."
scene:
  robot_urdf: "@builtin:robots/franka_panda/panda.xml"
  robot_config: "@builtin:robots/franka_panda/panda.robosandbox.yaml"
  objects:
    - id: red_cube
      kind: box
      size: [0.012, 0.012, 0.012]
      pose:
        xyz: [0.5, 0.0, 0.04]
      rgba: [0.85, 0.2, 0.2, 1.0]
      mass: 0.05
success:
  kind: lifted
  object: red_cube
  min_mm: 50
```

**The `@builtin:` prefix** is a new resolver in `tasks/loader.py._resolve_asset_path()`:
```
@builtin:robots/franka_panda/panda.xml
  -> importlib.resources.files("robosandbox") / "assets" / "robots/franka_panda/panda.xml"
```
Absolute paths and relative paths (to the YAML file) continue to work as-is.

**`tasks/loader.py` changes:** `_scene_from_dict` gains a `base_dir` parameter (the directory of the task YAML) so relative paths resolve correctly. `load_task` passes `path.parent`.

```python
def load_task(path: Path) -> Task:
    with path.open() as fh:
        raw = yaml.safe_load(fh)
    scene = _scene_from_dict(raw["scene"], base_dir=path.parent)    # CHANGED: pass base_dir
    return Task(...)

def _scene_from_dict(d, base_dir: Path):
    objs = tuple(_object_from_dict(o) for o in d.get("objects", []))
    robot_urdf = _resolve_asset_path(d.get("robot_urdf"), base_dir) if d.get("robot_urdf") else None
    robot_config = _resolve_asset_path(d.get("robot_config"), base_dir) if d.get("robot_config") else None
    return Scene(
        robot_urdf=robot_urdf,
        robot_config=robot_config,
        objects=objs,
        table_height=float(d.get("table_height", 0.04)),
    )

def _resolve_asset_path(raw: str, base_dir: Path) -> Path:
    """Resolve a task-YAML asset reference in this order:
      1. "@builtin:<rel>"  -> importlib.resources.files("robosandbox") / "assets" / <rel>
      2. absolute path     -> Path(raw) unchanged
      3. relative path     -> (base_dir / raw).resolve()
    Raises FileNotFoundError if the resolved path doesn't exist.
    """
```

## Scene Type Change

Add one field:
```python
@dataclass(frozen=True)
class Scene:
    robot_urdf: Path | None = None
    robot_config: Path | None = None        # NEW
    objects: tuple[SceneObject, ...] = ()
    workspace_aabb: tuple[XYZ, XYZ] = ((-0.5, -0.5, 0.0), (0.5, 0.5, 0.8))
    table_height: float = 0.0
    gravity: XYZ = (0.0, 0.0, -9.81)
```

Default-None so every existing `Scene(...)` call in tests and elsewhere keeps working without changes.

## Testing Plan

### Unit: `tests/test_robot_loader.py` (NEW)
- `test_load_bundled_franka_returns_robot_spec` — sidecar fields match RobotSpec fields
- `test_sidecar_explicit_path_overrides_sibling` — explicit config wins
- `test_sidecar_sibling_fallback` — `panda.xml` alone finds `panda.robosandbox.yaml` next to it
- `test_sidecar_missing_raises_not_found` — no sibling, no explicit → RobotConfigNotFoundError with tried paths
- `test_sidecar_missing_required_field_raises` — sidecar without `arm.joints` → RobotConfigValidationError
- `test_sidecar_ee_site_both_modes_raises` — sidecar with both `name` and `inject` → RobotConfigValidationError
- `test_sidecar_ee_site_neither_mode_raises` — sidecar with neither → RobotConfigValidationError
- `test_sidecar_length_mismatch_raises` — `home_qpos` wrong length → RobotConfigValidationError
- `test_sidecar_unknown_joint_name_raises` — name not in compiled model → RobotConfigMismatchError, `available` list non-empty
- `test_ee_site_inject_produces_site_at_correct_world_pos` — load Franka, reset, query `data.site_xpos[ee_site_id]`, verify it's near `hand_body_pos + [0, 0, 0.1034]`

### Unit: `tests/test_mjcf_builder.py` (extend existing or create)
- `test_builtin_path_returns_builtin_spec` — `build_model(Scene())` returns `BUILTIN_ROBOT_SPEC`
- `test_builtin_path_unchanged_xml` — `build_mjcf(Scene())` output string unchanged (regression)

### Integration: `tests/test_franka_pick.py` (NEW)
- `test_franka_loads_and_homes` — Franka scene loads, `n_dof == 7`, `joint_names == [joint1..joint7]`, home pose error < 1 cm after settling
- `test_franka_pick_lifts_cube` — Pick skill on cube at (0.5, 0, 0.04) lifts ≥5 cm (same pattern as `test_pick_smoke.py`)

### Regression
- All 21 existing tests pass unchanged
- `robo-sandbox-bench` — all 4 existing tasks succeed single-shot
- `robo-sandbox-bench --task pick_cube_franka` — new task passes single-shot. This is the acceptance criterion.

### Manual verification checklist (in implementation plan)
Before calling it done:
1. `python3 -c "import mujoco; m = mujoco.MjModel.from_xml_path('packages/robosandbox-core/src/robosandbox/assets/robots/franka_panda/panda.xml'); print(m.njnt, m.nu, m.nsite)"` — confirm file parses
2. `python3 -c "from robosandbox.scene.robot_loader import load_robot; from pathlib import Path; print(load_robot(Path('.../panda.xml'), None))"` — confirm loader returns RobotSpec
3. `pytest packages/robosandbox-core/tests/ -v` — 21+new tests pass
4. `robo-sandbox-bench` — 4 existing tasks pass
5. `robo-sandbox-bench --task pick_cube_franka` — passes

## Implementation Sequence

Small commits, in order:
1. `scene/robot_spec.py` + `BUILTIN_ROBOT_SPEC` in `mjcf_builder.py`; `Scene.robot_config` field added. No behavior change.
2. Refactor `MuJoCoBackend` to consume `RobotSpec` via a new `build_model(scene)` that still only handles the built-in path. All 21 tests pass.
3. `scene/robot_loader.py` (sidecar resolution + MjSpec load + ee_site injection + base_pose). Unit tests.
4. Bundle Franka assets into `assets/robots/franka_panda/`. Add `.robosandbox.yaml`.
5. Wire URDF path through `build_model()` — now handles both paths. Decor injection.
6. `tasks/loader.py` `@builtin:` resolver + `robot_urdf`/`robot_config` plumbing.
7. `pick_cube_franka.yaml` + integration test. Acceptance.
8. README attribution update for bundled menagerie assets.

Each commit keeps the repo green.

## Open Questions (all resolved)

- [x] Which validation robot? → Franka Panda (bundled from mujoco_menagerie).
- [x] Sidecar location? → Explicit `Scene.robot_config` field with sibling-file fallback.
- [x] URDF vs MJCF in `robot_urdf`? → Both — `MjSpec.from_file` handles both; field name is a wart we tolerate.
- [x] `MjSpec` vs XML string splicing? → `MjSpec` (clean, programmatic, 3.2+ is already a hard dep).
- [x] Actuator auto-injection for raw URDFs? → Deferred. v0.2 requires actuators to already exist in the file and be named in the sidecar.

## Risks

1. **Menagerie `panda.xml` root-body pose manipulation.** Setting `root.pos` on the Franka's root body might clash with menagerie's existing base frame. Mitigation: during step 5, verify with `data.xpos[root_id]` matches the sidecar `base_pose`. If menagerie's base is not a direct child of worldbody, we may need to wrap it or skip base_pose override.
2. **`MjSpec.from_file` with assets subdir.** Mesh references use `meshdir` in the MJCF `<compiler>` tag. Menagerie's `panda.xml` expects `assets/` relative to itself. We must load by path (not string) so the compiler resolves meshes from the right directory.
3. **Ee_site placement for Franka.** The `hand` body's local frame orientation may not match "fingers point down". The 0.1034 m offset is standard for Panda's TCP; verify visually (render an observation, check where the red site dot lands) during implementation.
4. **Pick skill sensitivity to 7-DOF.** `motion/ik.py` uses DLS IK which is DOF-agnostic, but orientation targets may be harder to reach with Franka's kinematics vs the 6-DOF arm. The pick_cube_franka task places the cube in-reach and should work, but if it fails at acceptance we may need to tune approach heights or orientation targets — not a design issue, an implementation detail.

## Success Criterion

`robo-sandbox-bench --task pick_cube_franka` reports success on first try from a clean checkout, while `robo-sandbox-bench` (all default tasks) remains 4/4 green and `pytest` remains green.
