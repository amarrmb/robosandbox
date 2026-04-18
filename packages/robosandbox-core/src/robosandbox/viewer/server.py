"""FastAPI server exposing the sim as a browser-viewable live stream.

Architecture:
  - SimThread: owns the MuJoCoBackend in a single dedicated thread
    (MuJoCo renderer is not thread-safe, so all sim access is serialized
    through this thread's command queue). Publishes JPEG frames + status
    events to thread-safe queues.
  - Async FastAPI app: serves the SPA, broadcasts frames + events to
    connected WebSocket clients, relays client commands to SimThread.

Client protocol (WebSocket JSON + binary):
  client -> server : {"action":"load", "task":"pick_cube_franka"}
                     {"action":"run",  "prompt":"pick up the red cube"}
  server -> client : binary frames (JPEG blobs)
                     text {"type":"event", "event":{...}} — status updates
"""

from __future__ import annotations

import asyncio
import io
import queue
import threading
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Viewer requires fastapi + uvicorn — install via "
        "`pip install 'robosandbox[viewer]'`"
    ) from e

from robosandbox.agent.agent import Agent
from robosandbox.agent.context import AgentContext
from robosandbox.agent.planner import StubPlanner
from robosandbox.grasp.analytic import AnalyticTopDown
from robosandbox.motion.ik import DLSMotionPlanner
from robosandbox.perception.ground_truth import GroundTruthPerception
from robosandbox.sim.mujoco_backend import MuJoCoBackend
from robosandbox.skills.home import Home
from robosandbox.skills.pick import Pick
from robosandbox.skills.place import PlaceOn
from robosandbox.skills.push import Push
from robosandbox.tasks.loader import list_builtin_tasks, load_builtin_task


_RENDER_SIZE = (360, 480)  # h, w
_IDLE_HZ = 15
_AGENT_RENDER_EVERY_N_STEPS = 3  # Render every 3rd on_step to cap CPU
_JPEG_QUALITY = 80


def _encode_jpeg(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    iio.imwrite(buf, rgb, extension=".jpg", quality=_JPEG_QUALITY)
    return buf.getvalue()


class SimThread(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True, name="robosandbox-viewer-sim")
        self._cmds: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.frames: queue.Queue[bytes] = queue.Queue(maxsize=2)
        self.events: queue.Queue[dict] = queue.Queue()
        self._sim: MuJoCoBackend | None = None
        self._task_name: str | None = None
        self._task_prompt: str | None = None
        self._stop = threading.Event()

    def run(self) -> None:  # noqa: D401
        idle_dt = 1.0 / _IDLE_HZ
        while not self._stop.is_set():
            try:
                cmd = self._cmds.get(timeout=idle_dt)
            except queue.Empty:
                cmd = None
            if cmd is None:
                # Idle: publish current frame so the browser sees the scene
                # even when nothing's happening. Cheap (no mj_step).
                if self._sim is not None:
                    self._publish_frame()
                continue
            kind, args = cmd
            try:
                if kind == "load":
                    self._load_task(args)
                elif kind == "run":
                    self._run_agent(args)
            except Exception as e:  # pragma: no cover - surfaced to client
                self._emit({"type": "error", "message": str(e)})

    # -- commands ------------------------------------------------------------

    def submit(self, kind: str, args: Any = None) -> None:
        self._cmds.put((kind, args))

    def stop(self) -> None:
        self._stop.set()

    def _load_task(self, task_name: str) -> None:
        if self._sim is not None:
            self._sim.close()
            self._sim = None
        task = load_builtin_task(task_name)
        sim = MuJoCoBackend(render_size=_RENDER_SIZE)
        sim.load(task.scene)
        for _ in range(60):
            sim.step()
        self._sim = sim
        self._task_name = task_name
        self._task_prompt = task.prompt
        self._publish_frame()
        self._emit(
            {
                "type": "loaded",
                "task": task_name,
                "prompt": task.prompt,
                "n_dof": sim.n_dof,
                "joint_names": sim.joint_names,
            }
        )

    def _run_agent(self, prompt: str | None) -> None:
        if self._sim is None:
            self._emit({"type": "error", "message": "no task loaded"})
            return
        sim = self._sim
        effective_prompt = prompt or self._task_prompt or ""
        self._emit({"type": "running", "prompt": effective_prompt})

        ctx = AgentContext(
            sim=sim,
            perception=GroundTruthPerception(),
            grasp=AnalyticTopDown(),
            motion=DLSMotionPlanner(n_waypoints=160, dt=0.005),
        )
        step_counter = {"n": 0}

        def on_step() -> None:
            step_counter["n"] += 1
            if step_counter["n"] % _AGENT_RENDER_EVERY_N_STEPS == 0:
                self._publish_frame()

        ctx.on_step = on_step

        skills = [Pick(), PlaceOn(), Push(), Home()]
        planner = StubPlanner(skills=skills)
        agent = Agent(ctx, skills, planner, max_replans=3)
        episode = agent.run(effective_prompt)
        self._publish_frame()
        self._emit(
            {
                "type": "done",
                "success": episode.success,
                "reason": episode.final_reason,
                "detail": episode.final_detail,
                "replans": episode.replans,
                "steps": [
                    {"skill": s.skill, "args": s.args, "success": s.result.success,
                     "reason": s.result.reason}
                    for s in episode.steps
                ],
            }
        )

    # -- helpers -------------------------------------------------------------

    def _publish_frame(self) -> None:
        assert self._sim is not None
        obs = self._sim.observe()
        jpg = _encode_jpeg(obs.rgb)
        # latest-wins: drop older frame if queue is full
        try:
            self.frames.put_nowait(jpg)
        except queue.Full:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                pass
            self.frames.put_nowait(jpg)

    def _emit(self, event: dict) -> None:
        self.events.put(event)


# -- FastAPI app -------------------------------------------------------------


app = FastAPI(title="RoboSandbox Viewer")

_state: SimThread | None = None
_connected: set[WebSocket] = set()
_connected_lock = asyncio.Lock()

_INDEX_PATH = Path(__file__).parent / "index.html"


@app.on_event("startup")
async def _startup() -> None:
    global _state
    _state = SimThread()
    _state.start()
    # Preload Franka for instant gratification.
    initial = _INITIAL_TASK or "pick_cube_franka"
    if initial in list_builtin_tasks():
        _state.submit("load", initial)
    asyncio.create_task(_frame_broadcaster())
    asyncio.create_task(_event_broadcaster())


@app.on_event("shutdown")
async def _shutdown() -> None:
    if _state is not None:
        _state.stop()


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(_INDEX_PATH.read_text())


@app.get("/tasks")
async def tasks_endpoint() -> JSONResponse:
    return JSONResponse({"tasks": list_builtin_tasks()})


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    async with _connected_lock:
        _connected.add(ws)
    try:
        while True:
            msg = await ws.receive_json()
            action = msg.get("action")
            if action == "load":
                assert _state is not None
                _state.submit("load", msg.get("task", "pick_cube_franka"))
            elif action == "run":
                assert _state is not None
                _state.submit("run", msg.get("prompt"))
            else:
                await ws.send_json({"type": "error", "message": f"unknown action {action!r}"})
    except WebSocketDisconnect:
        pass
    finally:
        async with _connected_lock:
            _connected.discard(ws)


async def _frame_broadcaster() -> None:
    """Poll the sim thread's frame queue; broadcast latest JPEG to all WSes."""
    assert _state is not None
    loop = asyncio.get_running_loop()
    while True:
        try:
            jpg = await loop.run_in_executor(None, _state.frames.get, True, 0.5)
        except queue.Empty:
            continue
        dead: list[WebSocket] = []
        for ws in list(_connected):
            try:
                await ws.send_bytes(jpg)
            except Exception:
                dead.append(ws)
        if dead:
            async with _connected_lock:
                for ws in dead:
                    _connected.discard(ws)


async def _event_broadcaster() -> None:
    assert _state is not None
    loop = asyncio.get_running_loop()
    while True:
        try:
            evt = await loop.run_in_executor(None, _state.events.get, True, 0.5)
        except queue.Empty:
            continue
        payload = {"type": "event", "event": evt}
        dead: list[WebSocket] = []
        for ws in list(_connected):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        if dead:
            async with _connected_lock:
                for ws in dead:
                    _connected.discard(ws)


_INITIAL_TASK: str | None = None


def run(host: str = "127.0.0.1", port: int = 8000, initial_task: str | None = None) -> None:
    """Start the viewer (blocking)."""
    try:
        import uvicorn
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Viewer requires uvicorn — install via `pip install 'robosandbox[viewer]'`"
        ) from e
    global _INITIAL_TASK
    _INITIAL_TASK = initial_task
    uvicorn.run(app, host=host, port=port, log_level="info")
