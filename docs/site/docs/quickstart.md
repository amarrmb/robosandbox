# Quickstart

Five minutes from clone to a recorded episode in the browser.

## Install

**Step 1 — install `uv`** (Python package manager, replaces pip + venv):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env   # or open a new terminal
```

**Step 2 — clone and install:**

```bash
git clone https://github.com/amarrmb/robosandbox
cd robosandbox
uv sync
uv pip install -e 'packages/robosandbox-core[viewer]'
```

MuJoCo 3.2+ and the browser viewer come in as dependencies. No GPU required.

=== "macOS (Apple Silicon or Intel)"

    Nothing extra needed — MuJoCo uses the system OpenGL automatically.

=== "Linux (Ubuntu 22.04 / 24.04)"

    Install an OSMesa headless GL backend for rendering without a display:

    ```bash
    sudo apt-get install -y libosmesa6 libosmesa6-dev libgl1-mesa-dri
    export MUJOCO_GL=osmesa    # or `egl` if a GPU is available
    ```

=== "Windows"

    Not currently supported. WSL2 running Ubuntu 22.04 works; follow the Linux tab inside WSL.

## 1. Open the viewer

```bash
uv run robo-sandbox viewer
```

Then open **http://localhost:8000** in your browser.

!!! tip "Running on a remote machine?"
    The viewer binds to `127.0.0.1` by default. To reach it from another machine, either:

    **SSH tunnel (recommended)** — run this on your laptop, then open `http://localhost:8000`:
    ```bash
    ssh -L 8000:127.0.0.1:8000 user@remote-host
    ```

    **Bind to all interfaces** — use on trusted/private networks only:
    ```bash
    uv run robo-sandbox viewer --host 0.0.0.0
    # → open http://<remote-ip>:8000
    ``` You'll see a live sim window, a task input field, and a log panel.

![Viewer running a pick task](assets/demos/franka_pick.gif)

Type `pick up the red cube` into the task field and hit **Run**. The arm plans and executes the pick while frames stream into the browser. The log panel shows each step and prints the outcome when the episode ends.

## 2. Record an episode

Enable **Record** in the viewer toolbar, then hit **Run** again. The sim runs the same task and writes a run directory to disk. When it finishes the log prints the exact path:

```
✓ Succeeded · 1 skill
  Next: uv run robo-sandbox export-lerobot runs/20260421-…-…/ datasets/my_demo
```

Your `runs/` folder now has a timestamped directory with `video.mp4`, `events.jsonl`, and `result.json`.

## 3. Export to LeRobot

Copy the path from the log and run:

```bash
uv pip install -e 'packages/robosandbox-core[lerobot]'
uv run robo-sandbox export-lerobot runs/<your-episode-dir> datasets/my_demo
```

You now have a LeRobot v3 dataset on disk, ready to inspect or train on.

## Next steps

- **[Guides](guides/how-it-works.md)** — how the agent loop, skills, and replan work
- **[Bring your own robot](guides/bring-your-own-robot.md)** — swap the arm
- **[Bring your own task](guides/bring-your-own-task.md)** — write a custom task
- **[LeRobot workflow](tutorials/lerobot-export.md)** — full sim-to-real pipeline

!!! tip "Smoke test"
    Want to verify the sim runs correctly across all built-in tasks?
    ```bash
    uv run robo-sandbox-bench
    ```
    This runs the stub planner headlessly and prints a pass/fail table. Useful after changing environment or dependencies.
