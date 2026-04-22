# Newton Integration Design

**Date:** 2026-04-22  
**Status:** Approved

## Context

RoboSandbox is an inspectable experimentation layer — same task contract, different backends. Newton unlocks two things MuJoCo cannot do: GPU-parallel physics and deformable body simulation (cloth). This design integrates Newton as a first-class backend while leaving the MuJoCo path completely unchanged.

---

## Goals

1. Default path (MuJoCo) works as-is — no changes visible to existing users.
2. Users with a capable GPU (RTX, DGX Spark) can launch with `--sim-backend newton`.
3. Showcase: 1 robot folding cloth (`cloth_fold_franka`) and 10 parallel worlds doing the same (`cloth_fold_franka_x10`).
4. Newton tasks appear in the existing task dropdown only when Newton is the active backend.

---

## Architecture

### What does not change

- WebSocket protocol (binary JPEG frames + JSON events)
- All MuJoCo tasks, scene loading, agent runs, recording, teleop
- `server.py` FastAPI app structure
- `index.html` layout, controls, events log, keyboard shortcuts

### Phase 1 — Server + BackendThread wiring

**`SimThread` → backend-agnostic**

`SimThread.__init__` gains two params:
```python
backend: str = "mujoco"
backend_kwargs: dict = {}
```

`_load_task` replaces the hardcoded `MuJoCoBackend(...)` call with:
```python
self._sim = create_sim_backend(self._backend, **self._backend_kwargs)
```

For Newton, `backend_kwargs` includes `viewer="viser"` and `port=<viser_port>`. `NewtonBackend` starts Viser internally when `load()` is called.

**Frame publishing — conditional**

MuJoCo: unchanged — pushes JPEGs to the queue after every step.  
Newton: frame publishing is skipped entirely. Viser owns the 3D render; the JPEG queue stays empty. WebSocket events (`loaded`, `running`, `done`, `error`) still flow through unchanged for both backends.

**New module-level vars in `server.py`**

```python
_SIM_BACKEND: str = "mujoco"
_VISER_URL: str | None = None  # e.g. "http://127.0.0.1:8090" for Newton
```

Set at startup via CLI args, same pattern as existing `_INITIAL_TASK`.

**New endpoint**

```
GET /config
→ {"backend": "mujoco", "viser_url": null}
→ {"backend": "newton", "viser_url": "http://127.0.0.1:8090"}
```

**New CLI args**

```
robo-sandbox viewer --sim-backend newton --viewer-port 8090
```

`--sim-backend` defaults to `"mujoco"`. `--viewer-port` only matters for Newton.

### Phase 2 — UI + Newton-exclusive tasks

**`index.html` — what changes**

1. Fetch `/config` on startup.
2. Set `data-backend="newton"` on `<body>` if Newton — CSS handles the conditional render.
3. Task fetch uses `/tasks?backend=<backend>` — server filters by `supported_backends`.
4. Center panel (`#sim-main`): MuJoCo → `<img id="frame">` (existing); Newton → `<iframe id="viser-frame" src="{viser_url}">`.
5. Backend pill in header: purple `⚡ newton` badge when Newton, no badge for MuJoCo.
6. Inspector (frame scrubber) grayed out for Newton — Viser owns playback.
7. fps counter hidden for Newton (no JPEG stream to measure).

**Task registry — `supported_backends` field**

Each `Task` gets `supported_backends: list[str]`, defaulting to `["mujoco"]`. Existing tasks need no changes.

`list_builtin_tasks(backend: str | None = None)` gains an optional filter:
```python
# server passes ?backend=newton → only Newton tasks returned
```

**New Newton-exclusive tasks**

`cloth_fold_franka` — 1 Franka arm, deformable cloth mesh (Warp cloth primitive), flat table. Newton only. This is the "Newton unlocks a new task class" demo.

`cloth_fold_franka_x10` — same scene, `world_count=10`. Newton spawns 10 independent GPU worlds; Viser renders them in a single aggregated 3D view. This is the evaluation-at-scale story: same task contract, 10 simultaneous rollouts.

**Note:** The existing `NewtonBackend` is single-world. The x10 task will use Newton's multi-world API directly (as demonstrated in `examples/newton_cloth_franka_showcase.py`). Implementation will determine whether `NewtonBackend` grows a `world_count` param or the x10 task uses a thin parallel wrapper. This decision belongs in the implementation plan, not here.

**Constraint:** Both cloth tasks ship with `success_criterion: null`. Cloth state evaluation (did the cloth actually fold?) requires a deformable-body metric that does not exist yet. Rigid-body tasks on Newton (e.g., `pick_cube_franka`) inherit MuJoCo's success logic — the observation schema is identical.

---

## Deploy paths

```bash
# Default — MuJoCo, no GPU, unchanged
robo-sandbox viewer

# Newton local — RTX GPU, warp + newton installed
robo-sandbox viewer --sim-backend newton --viewer-port 8090
# Browser: http://127.0.0.1:8000 (Viser at :8090 auto-embedded as iframe)

# Newton on DGX Spark — SSH tunnel to laptop
bash scripts/newton_probe_pick_cube_franka.sh
# Update script: add --task param so cloth tasks work too
```

The `newton_probe_pick_cube_franka.sh` script already supports a `TASK` env var (defaults to `pick_cube_franka`). The only change needed: wire a `--task` CLI arg through as `TASK` so callers don't need to set env vars manually.

---

## Files to create or modify

**Modify:**
- `packages/robosandbox-core/src/robosandbox/viewer/server.py` — BackendThread generalization, `/config` endpoint, CLI args
- `packages/robosandbox-core/src/robosandbox/viewer/index.html` — /config fetch, conditional render, backend badge, task filter
- `packages/robosandbox-core/src/robosandbox/tasks/loader.py` — `supported_backends` field, `backend` filter on `list_builtin_tasks`
- `packages/robosandbox-core/src/robosandbox/cli.py` — `--sim-backend`, `--viewer-port` args for `viewer` subcommand
- `scripts/newton_probe_pick_cube_franka.sh` — add `--task` forwarding

**Create:**
- `packages/robosandbox-core/src/robosandbox/tasks/cloth_fold_franka.py` — scene + task definition
- `packages/robosandbox-core/src/robosandbox/tasks/cloth_fold_franka_x10.py` — parallel worlds variant
- `packages/robosandbox-core/tests/test_newton_viewer_integration.py` — `/config` endpoint, task filtering

**Already done (commit these):**
- `packages/robosandbox-core/src/robosandbox/sim/newton_backend.py`
- `packages/robosandbox-core/src/robosandbox/sim/factory.py`
- `packages/robosandbox-core/src/robosandbox/sim/__init__.py`
- `packages/robosandbox-core/tests/test_sim_factory.py`
- `examples/newton_pick_cube_franka.py`
- `examples/newton_cloth_franka_showcase.py`
- `scripts/newton_probe_pick_cube_franka.sh`
- `scripts/newton_dgxspark_showcase.sh`
- `packages/robosandbox-core/src/robosandbox/viewer/showcase.html`

---

## What this is not

- Not a toggle: backend is a launch-time decision, not a runtime switch.
- Not a replacement: MuJoCo remains the default and the path for all existing workflows.
- Not a parallel renderer: Newton's visual output is Viser; we embed it, not replicate it.
- Not a cloth success metric: deformable-body evaluation is out of scope for this iteration.
