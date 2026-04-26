"""Threaded N-world MuJoCo wrapper for parallel PPO training.

Why this exists
---------------
NewtonBackend exposes ``observe_all`` / ``step_all`` / ``n_worlds`` so
``train_ppo`` can run N parallel rollouts on one GPU. MuJoCo's classical
sim is single-world, but its contact solver converges where Newton's
mujoco_warp solver does not (see the experimental/newton-eval session
notes — gripper-passes-through-cube is a solver-implementation gap, not
a parser config we can fix). For the "ACT prior → PPO refine" thesis
demo, we want PPO to refine on the sim where the policy *has* a reward
signal (MuJoCo, ~19% on the v2 distilled MLP), even if that means
giving up GPU-parallel training.

This wrapper holds N independent ``MuJoCoBackend`` instances and steps
them in a ``ThreadPoolExecutor``. ``mj_step`` releases the GIL, so
threads do parallelize across cores. With N=64 the loop is CPU-bound on
the Python coordination overhead per tick, but throughput is still
3–10× a single-world backend on a multi-core box.

Per-world scenes
----------------
``load(scene)`` loads the same scene into every world.
``load_per_world(scenes)`` (optional) loads one scene per world for
randomized eval / curriculum training. Both paths share a single
``RobotSpec`` derived from world 0 so the eval criterion can resolve
joint/object names without knowing which world.

API contract
------------
Implements the subset of NewtonBackend's parallel API that
``train_ppo`` consumes:

- ``n_worlds: int``
- ``n_dof: int``
- ``observe_all() -> list[Observation]``
- ``step_all(targets: (N, n_dof), grippers: (N,)) -> None``
- ``step() -> None``      (settle: broadcast no-op to all worlds)
- ``reset() -> None``
- ``get_object_pose(oid) -> Pose | None``  (world-0; matches the
  protocol the eval criterion uses)
- ``close() -> None``
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from robosandbox.types import Observation, Pose, Scene
from robosandbox.sim.mujoco_backend import MuJoCoBackend


class MuJoCoVecBackend:
    """Holds N ``MuJoCoBackend`` instances; ticks them in parallel threads."""

    def __init__(
        self,
        world_count: int = 8,
        render_size: tuple[int, int] = (240, 320),
        camera: str = "scene",
        max_workers: int | None = None,
    ) -> None:
        if world_count < 1:
            raise ValueError(f"world_count must be >= 1, got {world_count}")
        self._world_count = int(world_count)
        self._render_size = render_size
        self._camera = camera
        self._workers: list[MuJoCoBackend] = []
        # ThreadPoolExecutor: one worker thread per sim is the natural fit.
        # mj_step releases the GIL; the bottleneck is per-tick coordination,
        # not raw compute.
        self._pool = ThreadPoolExecutor(max_workers=max_workers or self._world_count)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, scene: Scene) -> None:
        """Load the same scene into every world."""
        self._workers = [
            MuJoCoBackend(render_size=self._render_size, camera=self._camera)
            for _ in range(self._world_count)
        ]
        for w in self._workers:
            w.load(scene)

    def load_per_world(self, scenes: list[Scene]) -> None:
        """Load one scene per world (for eval randomization or curriculum).

        ``scenes`` must have length == world_count and share a robot
        topology — ``n_dof`` and joint names come from world 0.
        """
        if len(scenes) != self._world_count:
            raise ValueError(
                f"len(scenes)={len(scenes)} != world_count={self._world_count}"
            )
        self._workers = [
            MuJoCoBackend(render_size=self._render_size, camera=self._camera)
            for _ in range(self._world_count)
        ]
        for w, s in zip(self._workers, scenes):
            w.load(s)

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def step(self) -> None:
        """No-op step: ticks every world with no command (used for settling)."""
        list(self._pool.map(lambda w: w.step(), self._workers))

    def step_all(self, targets: np.ndarray, grippers: np.ndarray) -> None:
        """Per-world joint targets + gripper command.

        Args:
            targets:  (N, n_dof) absolute joint positions per world
            grippers: (N,) gripper command in [0, 1] per world
        """
        targets = np.asarray(targets, dtype=np.float64)
        grippers_arr = np.asarray(grippers, dtype=np.float64)
        N = self._world_count
        n_dof = self.n_dof
        if targets.shape != (N, n_dof):
            raise ValueError(f"targets must be ({N}, {n_dof}), got {targets.shape}")
        if grippers_arr.shape != (N,):
            raise ValueError(f"grippers must be ({N},), got {grippers_arr.shape}")

        def _tick(args: tuple[MuJoCoBackend, np.ndarray, float]) -> None:
            w, t, g = args
            w.step(target_joints=t, gripper=float(g))

        list(self._pool.map(
            _tick,
            [(self._workers[w], targets[w], float(grippers_arr[w])) for w in range(N)],
        ))

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def observe(self) -> Observation:
        """World-0 observation (single-world compat). State-only — see
        :meth:`MuJoCoBackend.observe_state` for why."""
        return self._workers[0].observe_state()

    def observe_all(self) -> list[Observation]:
        """One state-only observation per world. Skips RGB/depth render.

        EGL contexts can't be shared across threads, so calling MuJoCoBackend
        .observe() (which renders) from the worker threadpool panics with
        ``EGL_BAD_ACCESS``. PPO uses ObsEncoder which only reads state
        fields, so the render is unnecessary anyway.
        """
        return list(self._pool.map(lambda w: w.observe_state(), self._workers))

    def get_object_pose(self, object_id: str) -> Pose | None:
        """World-0 object pose, matching the single-world eval-criterion API."""
        return self._workers[0].get_object_pose(object_id)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        list(self._pool.map(lambda w: w.reset(), self._workers))

    def close(self) -> None:
        for w in self._workers:
            try:
                w.close()
            except Exception:
                pass
        self._workers = []
        self._pool.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_worlds(self) -> int:
        return self._world_count

    @property
    def n_dof(self) -> int:
        return self._workers[0].n_dof if self._workers else 0
