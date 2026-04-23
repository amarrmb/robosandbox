"""FastAPI server exposing the sim as a browser-viewable live stream.

Architecture:
  - SimThread: owns the sim backend in a single dedicated thread
    (MuJoCo renderer is not thread-safe, so all sim access is serialized
    through this thread's command queue). Publishes JPEG frames + status
    events to thread-safe queues.
  - Async FastAPI app: serves the SPA, broadcasts frames + events to
    connected WebSocket clients, relays client commands to SimThread.

Backend modes:
  mujoco (default): renders JPEG frames streamed to the browser <img>.
  newton: Viser owns the 3D render; JPEG queue stays empty. The browser
    embeds Viser as an <iframe> pointed at _VISER_URL.

Client protocol (WebSocket JSON + binary):
  client -> server : {"action":"load", "task":"pick_cube_franka"}
                     {"action":"run",  "prompt":"pick up the red cube"}
  server -> client : binary frames (JPEG blobs, MuJoCo only)
                     text {"type":"event", "event":{...}} — status updates
"""

from __future__ import annotations

import asyncio
import io
import queue
import threading
from collections import deque
from dataclasses import dataclass
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
from robosandbox.motion.ik import DLSMotionPlanner, UnreachableError, plan_linear_cartesian
from robosandbox.perception.ground_truth import GroundTruthPerception
from robosandbox.recorder.local import LocalRecorder
from robosandbox.sim import create_sim_backend
from robosandbox.skills.drawer import CloseDrawer, OpenDrawer
from robosandbox.skills.home import Home
from robosandbox.skills.pick import Pick
from robosandbox.skills.place import PlaceOn
from robosandbox.skills.pour import Pour
from robosandbox.skills.push import Push
from robosandbox.skills.stack import Stack
from robosandbox.skills.tap import Tap
from robosandbox.tasks.loader import list_builtin_tasks, load_builtin_task

_RENDER_SIZE = (720, 960)  # h, w — 2x linear resolution; scene camera fovy widened to 65° keeps the full arm in frame
_IDLE_HZ = 15
_AGENT_RENDER_EVERY_N_STEPS = 3  # Render every 3rd on_step to cap CPU
_JPEG_QUALITY = 80


def _encode_jpeg(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    iio.imwrite(buf, rgb, extension=".jpg", quality=_JPEG_QUALITY)
    return buf.getvalue()


@dataclass(frozen=True)
class FrameSnapshot:
    """One in-RAM trajectory frame for the post-hoc inspector.

    ``rgb_jpeg`` is the already-encoded JPEG (same bytes we sent to the
    client) so memory stays bounded — a 480x360 JPEG at quality 80 is
    typically 20-40 kB, giving a 900-frame ceiling of ~30 MB.
    """

    timestamp: float
    robot_joints: np.ndarray  # (n_dof,) float64
    ee_xyz: tuple[float, float, float]
    ee_quat_xyzw: tuple[float, float, float, float]
    gripper_width: float
    scene_objects: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]]
    rgb_jpeg: bytes


class SimThread(threading.Thread):
    # Cap the in-RAM trajectory buffer. 900 frames ≈ 60 s at the agent's
    # observed publish cadence (every 3rd sim step ~= 15 Hz wall-clock for a
    # 200 Hz physics sim); ~30 MB at JPEG quality 80. Drop-oldest on overflow.
    _TRAJ_MAX_FRAMES = 900

    def __init__(self, runs_dir: Path, backend: str = "mujoco", backend_kwargs: dict | None = None) -> None:
        super().__init__(daemon=True, name="robosandbox-viewer-sim")
        self._backend = backend
        self._backend_kwargs: dict = backend_kwargs or {}
        self._cmds: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.frames: queue.Queue[bytes] = queue.Queue(maxsize=2)
        self.events: queue.Queue[dict] = queue.Queue()
        self._sim: Any = None
        self._task_name: str | None = None
        self._task_prompt: str | None = None
        self._stop = threading.Event()
        # Recording state. LocalRecorder is reused across episodes; a fresh
        # start_episode call resets it. ``_recording`` is the source of truth.
        self._runs_dir = runs_dir
        self._recorder = LocalRecorder(root=runs_dir)
        self._recording = False
        self._rec_episode_id: str | None = None
        self._rec_episode_dir: str | None = None
        # Teleop persists a gripper state (0=open, 1=closed) between clicks.
        self._teleop_gripper = 0.0
        # In-RAM trajectory buffer for the inspector. Reset on each load_task.
        # A deque with maxlen naturally drops the oldest snapshot on overflow.
        self._trajectory: deque[FrameSnapshot] = deque(maxlen=self._TRAJ_MAX_FRAMES)
        # While inspecting a past frame we freeze the idle publisher so the
        # scrubbed frame doesn't get stomped by the live render.
        self._inspecting = False

    def run(self) -> None:
        idle_dt = 1.0 / _IDLE_HZ
        while not self._stop.is_set():
            try:
                cmd = self._cmds.get(timeout=idle_dt)
            except queue.Empty:
                cmd = None
            if cmd is None:
                # MuJoCo only: push a live frame at idle rate so the browser
                # sees the scene even when nothing is happening. Newton's
                # visual output is Viser — no JPEG to publish.
                if self._sim is not None and self._backend == "mujoco" and not self._inspecting:
                    self._publish_frame()
                continue
            kind, args = cmd
            try:
                if kind == "load":
                    self._load_task(args)
                elif kind == "run":
                    self._run_agent(args)
                elif kind == "record_start":
                    self._record_start()
                elif kind == "record_stop":
                    self._record_stop()
                elif kind == "teleop":
                    self._teleop(args)
                elif kind == "inspect_at":
                    self._inspect_at(args)
                elif kind == "inspect_clear":
                    self._inspect_clear()
            except Exception as e:  # pragma: no cover - surfaced to client
                self._emit({"type": "error", "message": str(e)})

    # -- commands ------------------------------------------------------------

    def submit(self, kind: str, args: Any = None) -> None:
        self._cmds.put((kind, args))

    def stop(self) -> None:
        self._stop.set()

    def _load_task(self, task_name: str) -> None:
        if self._recording:
            # Switching tasks mid-record closes the current episode so we
            # don't leak frames from task A into an episode labelled for B.
            self._record_stop(reason="task switched")
        if self._sim is not None:
            self._sim.close()
            self._sim = None
        # Fresh task → fresh trajectory. Any prior inspection is invalid now.
        self._trajectory.clear()
        self._inspecting = False
        task = load_builtin_task(task_name)
        kwargs = dict(self._backend_kwargs)
        if self._backend == "mujoco":
            kwargs.setdefault("render_size", _RENDER_SIZE)
        sim = create_sim_backend(self._backend, **kwargs)
        sim.load(task.scene)
        for _ in range(60):
            sim.step()
        self._sim = sim
        self._task_name = task_name
        self._task_prompt = task.prompt
        if self._backend == "mujoco":
            self._publish_frame()
        self._emit(
            {
                "type": "loaded",
                "task": task_name,
                "prompt": task.prompt,
                "n_dof": sim.n_dof,
                "joint_names": sim.joint_names,
                # Extra metadata so the client can render an "About this
                # task" panel without a second round-trip. Object ids
                # come from the Scene; success criterion + randomize
                # spec come straight from the loaded Task.
                "object_ids": [o.id for o in task.scene.objects],
                "object_kinds": {o.id: o.kind for o in task.scene.objects},
                "success_criterion": task.success.data,
                "randomize": task.randomize or {},
                "robot": (
                    "franka_panda" if task.scene.robot_urdf else "builtin_6dof"
                ),
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

        is_mujoco = self._backend == "mujoco"

        def on_step() -> None:
            step_counter["n"] += 1
            if is_mujoco:
                render_tick = step_counter["n"] % _AGENT_RENDER_EVERY_N_STEPS == 0
                if render_tick:
                    obs = sim.observe()
                    jpg = _encode_jpeg(obs.rgb)
                    self._enqueue_jpeg(jpg)
                    self._append_snapshot(obs, jpg)
            if self._recording:
                try:
                    action = sim.last_action() if hasattr(sim, "last_action") else None
                    self._recorder.write_frame(sim.observe(), action=action)
                except Exception:  # pragma: no cover
                    pass

        ctx.on_step = on_step

        skills = [
            Pick(), PlaceOn(), Push(), Home(), Pour(), Tap(),
            OpenDrawer(), CloseDrawer(), Stack(),
        ]
        planner = StubPlanner(skills=skills)
        agent = Agent(ctx, skills, planner, max_replans=3)
        episode = agent.run(effective_prompt)
        if is_mujoco:
            self._publish_frame()
        self._emit(
            {
                "type": "done",
                "success": episode.success,
                "reason": episode.final_reason,
                "detail": episode.final_detail,
                "replans": episode.replans,
                "trajectory_frames": len(self._trajectory),
                "steps": [
                    {"skill": s.skill, "args": s.args, "success": s.result.success,
                     "reason": s.result.reason}
                    for s in episode.steps
                ],
            }
        )

    # -- recording -----------------------------------------------------------

    def _record_start(self) -> None:
        if self._sim is None:
            self._emit({"type": "error", "message": "load a task before recording"})
            return
        if self._recording:
            self._emit({"type": "error", "message": "already recording"})
            return
        metadata = {
            "sim_dt": float(self._sim.model.opt.timestep),
            "task": self._task_name or "",
            "source": "viewer",
        }
        episode_id = self._recorder.start_episode(
            task=self._task_name or "unknown",
            metadata=metadata,
        )
        self._recording = True
        self._rec_episode_id = episode_id
        self._rec_episode_dir = str(self._recorder.current_episode_dir)
        # Write one immediate frame so the recorded video is not empty if the
        # user stops quickly with no agent running.
        try:
            sim = self._sim
            action = sim.last_action() if hasattr(sim, "last_action") else None
            self._recorder.write_frame(sim.observe(), action=action)
        except Exception:  # pragma: no cover
            pass
        self._emit(
            {
                "type": "recording_started",
                "episode_id": episode_id,
                "runs_dir": str(self._runs_dir.resolve()),
            }
        )

    # -- teleop --------------------------------------------------------------

    def _teleop(self, args: dict[str, Any]) -> None:
        """Nudge the end-effector by (dx, dy, dz) from its current pose.

        ``args`` keys (all optional):
          dx, dy, dz: Cartesian delta in metres. Tiny values (≤ 3 cm per
              call) keep the linear Cartesian planner reliably convergent.
          gripper: "toggle" | "open" | "close" | None.

        Discrete stepping keeps the physics stable: each call runs a short
        Cartesian plan, executes it, and publishes one frame. The viewer
        client rate-limits keystrokes to ~20 Hz.
        """
        if self._sim is None:
            return
        sim = self._sim
        obs = sim.observe()

        g_cmd = args.get("gripper")
        if g_cmd == "toggle":
            self._teleop_gripper = 1.0 - self._teleop_gripper
        elif g_cmd == "open":
            self._teleop_gripper = 0.0
        elif g_cmd == "close":
            self._teleop_gripper = 1.0

        dx = float(args.get("dx", 0.0))
        dy = float(args.get("dy", 0.0))
        dz = float(args.get("dz", 0.0))

        # Pure gripper toggle with no translation: just commit the gripper
        # change over a handful of ticks so the fingers actually move.
        if dx == 0.0 and dy == 0.0 and dz == 0.0:
            for _ in range(40):
                sim.step(target_joints=obs.robot_joints, gripper=self._teleop_gripper)
            if self._backend == "mujoco":
                self._publish_frame()
            if self._recording:
                try:
                    action = sim.last_action() if hasattr(sim, "last_action") else None
                    self._recorder.write_frame(sim.observe(), action=action)
                except Exception:  # pragma: no cover
                    pass
            return

        from robosandbox.types import Pose
        ex, ey, ez = obs.ee_pose.xyz
        target = Pose(
            xyz=(ex + dx, ey + dy, ez + dz),
            quat_xyzw=obs.ee_pose.quat_xyzw,
        )
        try:
            traj = plan_linear_cartesian(
                sim,
                start_joints=obs.robot_joints,
                target_pose=target,
                n_waypoints=12,
                dt=0.005,
                orientation="z_down",
            )
        except UnreachableError:
            self._emit({"type": "teleop_unreachable", "target_xyz": target.xyz})
            return

        for row in traj.waypoints:
            for _ in range(4):
                sim.step(target_joints=row, gripper=self._teleop_gripper)
        if self._backend == "mujoco":
            self._publish_frame()
        if self._recording:
            try:
                action = sim.last_action() if hasattr(sim, "last_action") else None
                self._recorder.write_frame(sim.observe(), action=action)
            except Exception:  # pragma: no cover
                pass

    def _record_stop(self, *, reason: str | None = None) -> None:
        if not self._recording:
            return
        self._recording = False
        ep_id = self._rec_episode_id or ""
        self._rec_episode_id = None
        self._recorder.end_episode(
            success=False,  # viewer recording is open-ended; not a task
            result={"reason": reason or "stopped_by_user"},
        )
        self._emit(
            {
                "type": "recording_stopped",
                "episode_id": ep_id,
                "episode_dir": self._rec_episode_dir or "",
                "runs_dir": str(self._runs_dir.resolve()),
            }
        )
        self._rec_episode_dir = None

    # -- inspector -----------------------------------------------------------

    def _inspect_at(self, args: dict[str, Any]) -> None:
        """Seek to ``frame_idx`` in the in-RAM trajectory, if available."""
        if not self._trajectory:
            self._emit({"type": "error", "message": "no trajectory to inspect"})
            return
        try:
            idx = int(args.get("frame_idx", 0))
        except (TypeError, ValueError):
            self._emit({"type": "error", "message": "frame_idx must be an int"})
            return
        total = len(self._trajectory)
        # Clamp instead of erroring — slider at the end is a normal case.
        idx = max(0, min(idx, total - 1))
        snap = self._trajectory[idx]
        self._inspecting = True
        self._enqueue_jpeg(snap.rgb_jpeg)
        self._emit(
            {
                "type": "inspect_frame",
                "frame_idx": idx,
                "total": total,
                "timestamp": snap.timestamp,
                "robot_joints": snap.robot_joints.tolist(),
                "ee_pose": {"xyz": list(snap.ee_xyz), "quat_xyzw": list(snap.ee_quat_xyzw)},
                "gripper_width": snap.gripper_width,
                "scene_objects": {
                    k: {"xyz": list(xyz), "quat_xyzw": list(q)}
                    for k, (xyz, q) in snap.scene_objects.items()
                },
            }
        )

    def _inspect_clear(self) -> None:
        """Exit inspection mode; resume live streaming."""
        self._inspecting = False
        self._emit({"type": "inspect_cleared"})
        if self._sim is not None and self._backend == "mujoco":
            self._publish_frame()

    # -- helpers -------------------------------------------------------------

    def _publish_frame(self) -> None:
        """Render current sim state and enqueue as the live JPEG."""
        assert self._sim is not None
        obs = self._sim.observe()
        self._enqueue_jpeg(_encode_jpeg(obs.rgb))

    def _enqueue_jpeg(self, jpg: bytes) -> None:
        """Latest-wins publish to the frame queue."""
        try:
            self.frames.put_nowait(jpg)
        except queue.Full:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                pass
            self.frames.put_nowait(jpg)

    def _append_snapshot(self, obs, jpg: bytes) -> None:
        """Add one trajectory snapshot (deque drops oldest at cap)."""
        snap = FrameSnapshot(
            timestamp=float(obs.timestamp),
            robot_joints=np.asarray(obs.robot_joints, dtype=np.float64).copy(),
            ee_xyz=tuple(obs.ee_pose.xyz),
            ee_quat_xyzw=tuple(obs.ee_pose.quat_xyzw),
            gripper_width=float(obs.gripper_width),
            scene_objects={
                k: (tuple(p.xyz), tuple(p.quat_xyzw))
                for k, p in obs.scene_objects.items()
            },
            rgb_jpeg=jpg,
        )
        self._trajectory.append(snap)

    def _emit(self, event: dict) -> None:
        self.events.put(event)


# -- FastAPI app -------------------------------------------------------------


app = FastAPI(title="RoboSandbox Viewer")

_state: SimThread | None = None
_connected: set[WebSocket] = set()
_connected_lock = asyncio.Lock()

_INDEX_PATH = Path(__file__).parent / "index.html"
_SHOWCASE_PATH = Path(__file__).parent / "showcase.html"


@app.on_event("startup")
async def _startup() -> None:
    global _state
    backend_kwargs: dict = {}
    if _SIM_BACKEND == "newton":
        backend_kwargs = {"viewer": "viser", "port": _VISER_PORT, "device": _DEVICE}
    _state = SimThread(runs_dir=_RUNS_DIR, backend=_SIM_BACKEND, backend_kwargs=backend_kwargs)
    _state.start()
    initial = _INITIAL_TASK or ("cloth_fold_franka" if _SIM_BACKEND == "newton" else "pick_cube_franka")
    if initial in list_builtin_tasks(backend=_SIM_BACKEND):
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


@app.get("/showcase")
async def showcase() -> HTMLResponse:
    return HTMLResponse(_SHOWCASE_PATH.read_text())


@app.get("/config")
async def config_endpoint() -> JSONResponse:
    viser_url = f"http://127.0.0.1:{_VISER_PORT}" if _SIM_BACKEND == "newton" else None
    return JSONResponse({"backend": _SIM_BACKEND, "viser_url": viser_url})


@app.get("/tasks")
async def tasks_endpoint(backend: str | None = None) -> JSONResponse:
    return JSONResponse({"tasks": list_builtin_tasks(backend=backend or _SIM_BACKEND)})


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
            elif action == "record_start":
                assert _state is not None
                _state.submit("record_start")
            elif action == "record_stop":
                assert _state is not None
                _state.submit("record_stop")
            elif action == "teleop":
                assert _state is not None
                _state.submit("teleop", {
                    "dx": float(msg.get("dx", 0.0)),
                    "dy": float(msg.get("dy", 0.0)),
                    "dz": float(msg.get("dz", 0.0)),
                    "gripper": msg.get("gripper"),
                })
            elif action == "inspect_at":
                assert _state is not None
                _state.submit("inspect_at", {"frame_idx": msg.get("frame_idx", 0)})
            elif action == "inspect_clear":
                assert _state is not None
                _state.submit("inspect_clear")
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
_RUNS_DIR: Path = Path("runs")
_SIM_BACKEND: str = "mujoco"
_VISER_PORT: int = 8090
_DEVICE: str = "cuda:0"


def run(
    host: str = "127.0.0.1",
    port: int = 8000,
    initial_task: str | None = None,
    runs_dir: Path | str = "runs",
    sim_backend: str = "mujoco",
    viser_port: int = 8090,
    device: str = "cuda:0",
) -> None:
    """Start the viewer (blocking)."""
    try:
        import uvicorn
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Viewer requires uvicorn — install via `pip install 'robosandbox[viewer]'`"
        ) from e
    global _INITIAL_TASK, _RUNS_DIR, _SIM_BACKEND, _VISER_PORT, _DEVICE
    _INITIAL_TASK = initial_task
    _RUNS_DIR = Path(runs_dir)
    _SIM_BACKEND = sim_backend
    _VISER_PORT = viser_port
    _DEVICE = device
    uvicorn.run(app, host=host, port=port, log_level="info")
