# Quickstart

This gets you from a fresh checkout to a working run, a benchmark, a
viewer session, and a recorded episode.

!!! note "Supported platform"
    v0.1 is **Linux-first** — developed and CI-tested on
    Ubuntu 22.04/24.04 with Python 3.11/3.12/3.13. macOS and Windows
    are not regression-gated; they may work but platform-specific
    issues (headless GL, Apple Silicon MuJoCo wheels, Windows paths)
    are not tracked.

## Install

```bash
git clone https://github.com/amarrmb/robosandbox
cd robosandbox
uv sync
uv pip install -e packages/robosandbox-core
```

You need Python 3.10+. MuJoCo 3.2+ comes in as a dependency, and the
built-in demos do not need a GPU.

Headless rendering (viewer + any test that renders a frame) needs an
OpenGL backend. On Ubuntu:

```bash
sudo apt-get install -y libosmesa6 libosmesa6-dev libgl1-mesa-dri
export MUJOCO_GL=osmesa    # or `egl` when a GPU is available
```

Install extras only if you need them:

```bash
uv pip install -e 'packages/robosandbox-core[viewer]'     # FastAPI viewer
uv pip install -e 'packages/robosandbox-core[meshes]'     # CoACD for BYO meshes
uv pip install -e 'packages/robosandbox-core[lerobot]'    # pyarrow for export
uv pip install -e 'packages/robosandbox-core[docs]'       # this site
uv pip install -e 'packages/robosandbox-core[dev]'        # pytest, ruff, mypy
```

## 1. Run the simplest demo

```bash
uv run robo-sandbox run "pick up the red cube"
```

You should see MuJoCo open, the built-in arm pick the cube, and the CLI
print the plan and final reason. No API key required.

## 2. Run the benchmark

The benchmark uses the stub planner and ground-truth perception, so
what you are measuring here is mostly sim reliability rather than model
quality.

```bash
uv run robo-sandbox-bench
```

Typical output looks like this:

```
TASK                 SEED  RESULT   SECS  REPLANS DETAIL
------------------------------------------------------------------------------------------
home                 0     OK        0.0        0
open_drawer          0     OK        2.0        0  displacement_mm=62.7, min_mm=50.0
pick_cube            0     OK        1.1        0  dz_mm=159.8, min_mm=50.0
pick_cube_franka     0     OK        1.5        0  dz_mm=166.9, min_mm=50.0
pick_cube_scrambled  0     OK        1.5        0  dz_mm=166.9, min_mm=50.0
pick_from_three      0     OK        1.1        0  dz_mm=160.0, min_mm=50.0
pick_ycb_mug         0     OK        1.7        0  dz_mm=105.3, min_mm=50.0
pour_can_into_bowl   0     OK        2.8        0  xy=0.004, dz=0.133
push_forward         0     OK        1.0        0  displacement_mm=76.5, min_mm=30.0

SUMMARY: 9/9 successful
```

Results append to `benchmark_results.json`. Use `--out` if you want a
different path.

If you want something closer to a regression run than a smoke test:

```bash
uv run robo-sandbox-bench --seeds 50
```

See the [CLI reference](reference/cli.md#robo-sandbox-bench) for all
flags.

## 3. Open the browser viewer

Install the viewer extra once:

```bash
uv pip install -e 'packages/robosandbox-core[viewer]'
```

Start it:

```bash
robo-sandbox viewer
# → open http://localhost:8000
```

Pick a task from the dropdown and hit **Run**. Frames stream in the
browser, events show up in the sidebar, and if you enable **Record**
first you'll get a run directory under `./runs/<ts>-<id>/`. **Teleop**
lets you drive the arm with WASD/QE + Space.

## 4. Record a scripted episode

If you want a headless run that writes the usual artifacts to disk:

```bash
uv run python examples/record_demo.py --out-dir runs
```

Expected output:

```
recorded episode 1a2b3c4d
  runs/20260418-094533-1a2b3c4d
    episode.json (204 B)
    events.jsonl (84512 B)
    result.json (178 B)
    video.mp4 (42980 B)
```

If you want the file layout and schema details, see
[recording & export](concepts/recording-and-export.md).

## 5. Export to LeRobot v3

```bash
uv pip install -e 'packages/robosandbox-core[lerobot]'
robo-sandbox export-lerobot runs/20260418-094533-1a2b3c4d /tmp/my_dataset
```

At that point you have a LeRobot v3 dataset on disk. You can inspect it,
train on it, or use it as the starting point for the policy replay
workflow.

## Next steps

- **[Scenes & objects](concepts/scenes.md)** — what `Scene` and
  `SceneObject` mean; YCB catalog.
- **[Skills & agents](concepts/skills-and-agents.md)** — how the agent
  loop plans + executes.
- **[Custom arm tutorial](tutorials/custom-arm.md)** — swap the robot.
