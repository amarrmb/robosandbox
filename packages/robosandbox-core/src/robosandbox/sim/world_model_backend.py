"""WorldModelBackend — score policies against a learned dynamics model.

The world-model wave (World Labs, AMI Labs, NVIDIA Cosmos, V-JEPA-2) is
betting that learned dynamics replace classical physics for robotics
eval. This backend is the wedge: a SimBackend implementation whose
time-evolution is a pluggable WorldModelPredictor instead of a physics
solver.

Shape: a reference physics backend (MuJoCo by default) provides the
initial observation, scene metadata, and forward kinematics. After
load() / reset() the predictor takes over — every step() asks the
predictor for the next observation given the current one and the
commanded action. observe() returns whatever the predictor said.

That keeps the eval contract honest:

- Same scene, same initial state distribution → comparable to a
  classical-sim eval against the same policy.
- Different propagator → the difference between the two scores is
  attributable to the model.
- Provenance can carry the predictor's identity (model_sha256 etc.)
  alongside the policy's, so two evals against different world models
  are visibly non-comparable in the JSON.

The reference predictor shipped here is intentionally trivial
(IdentityPredictor — return the input unchanged). It exists to prove
the slot is wired and that `robo-sandbox eval --sim-backend
world_model` runs end to end. It will score 0% on every task because
nothing moves. Real world models — V-JEPA-2 latents, Cosmos frame
prediction, a state-space DreamerV3-style learned model — plug into
the same WorldModelPredictor protocol.

See docs/site/docs/concepts/eval-for-world-models.md for the framing
and docs/site/docs/tutorials/world-model-as-sim-backend.md for the
walkthrough.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from robosandbox.sim.factory import create_sim_backend
from robosandbox.types import Observation, Pose, Scene


@runtime_checkable
class WorldModelPredictor(Protocol):
    """Plug-in interface for a learned dynamics model.

    Given the current observation and the commanded action, return the
    predicted next observation. The predictor is responsible for
    populating whatever fields it cares about (robot_joints, ee_pose,
    gripper_width, scene_objects, rgb). Fields it leaves unchanged are
    carried over from the input — that lets a joint-state-only model
    avoid faking ee_pose / images it doesn't predict.

    Implementations should be deterministic given (obs, action). RNG
    state, batch buffers, and any cached compile/autograd state are
    the predictor's problem; the eval CLI's --reload-policy / per-trial
    reset semantics treat the backend as black-box.
    """

    def predict_next(
        self,
        obs: Observation,
        target_joints: np.ndarray | None,
        gripper: float | None,
    ) -> Observation:
        ...

    def reset(self) -> None:
        """Clear any per-episode state. Called from WorldModelBackend.reset()."""
        ...


class IdentityPredictor:
    """Reference predictor: next observation = current observation.

    Smoke-test implementation. Verifies the slot wiring without
    requiring a real model. Every eval against this predictor will
    score 0% — that's the point. Replace with a real predictor when
    one is worth integrating.
    """

    def predict_next(
        self,
        obs: Observation,
        target_joints: np.ndarray | None,
        gripper: float | None,
    ) -> Observation:
        return obs

    def reset(self) -> None:
        pass


class WorldModelBackend:
    """SimBackend that uses a WorldModelPredictor for time-evolution.

    Bootstraps from a reference SimBackend (MuJoCo by default) to get
    the initial observation, scene metadata, and joint/dof
    introspection. After load(), the reference backend's role is
    purely to answer "what was the initial state and what does the
    scene look like" — all time stepping goes through the predictor.

    Two things this means in practice:

    1. The eval task's success criterion still reads object positions
       off ``obs.scene_objects``. A predictor that doesn't update
       scene_objects will look like the cube never moved (which, for
       its prediction, is true). Real world models that care about
       manipulation eval need to predict object positions explicitly.

    2. ``get_object_pose(id)`` returns the *predicted* pose at the
       current step — read from the latest observation, not from the
       reference backend. Two backends running the same policy on the
       same scene will diverge after step 0; that divergence is the
       signal.
    """

    def __init__(
        self,
        predictor: WorldModelPredictor | None = None,
        bootstrap_backend: str = "mujoco",
        bootstrap_kwargs: dict | None = None,
    ) -> None:
        self._predictor: WorldModelPredictor = predictor or IdentityPredictor()
        self._bootstrap = create_sim_backend(
            bootstrap_backend, **(bootstrap_kwargs or {})
        )
        self._scene: Scene | None = None
        self._obs: Observation | None = None
        self._t: float = 0.0

    # ---- lifecycle -------------------------------------------------------
    def load(self, scene: Scene) -> None:
        self._bootstrap.load(scene)
        self._scene = scene
        self._predictor.reset()
        self._obs = self._bootstrap.observe()
        self._t = 0.0

    def reset(self) -> None:
        if self._scene is None:
            raise RuntimeError("WorldModelBackend.reset() before load()")
        self._bootstrap.reset()
        self._predictor.reset()
        self._obs = self._bootstrap.observe()
        self._t = 0.0

    def close(self) -> None:
        self._bootstrap.close()
        self._obs = None

    # ---- stepping --------------------------------------------------------
    def step(
        self,
        target_joints: np.ndarray | None = None,
        gripper: float | None = None,
    ) -> None:
        if self._obs is None:
            raise RuntimeError("WorldModelBackend.step() before load()")
        next_obs = self._predictor.predict_next(self._obs, target_joints, gripper)
        # Advance the timestamp by the bootstrap's nominal dt so eval
        # code that bins by time still works. World-model predictors
        # typically step at video-frame rate (4-30 Hz) rather than the
        # bootstrap's 200 Hz; if you care about that mismatch wire a
        # `--action-repeat` aware override here.
        dt = getattr(self._bootstrap, "dt", 0.005)
        self._obs = Observation(
            rgb=next_obs.rgb,
            depth=next_obs.depth,
            robot_joints=next_obs.robot_joints,
            ee_pose=next_obs.ee_pose,
            gripper_width=next_obs.gripper_width,
            scene_objects=next_obs.scene_objects,
            timestamp=self._t + dt,
            camera_intrinsics=next_obs.camera_intrinsics,
            camera_extrinsics=next_obs.camera_extrinsics,
        )
        self._t += dt

    def observe(self) -> Observation:
        if self._obs is None:
            raise RuntimeError("WorldModelBackend.observe() before load()")
        return self._obs

    # ---- scene queries ---------------------------------------------------
    def get_object_pose(self, object_id: str) -> Pose | None:
        if self._obs is None:
            return None
        return self._obs.scene_objects.get(object_id)

    def set_object_pose(self, object_id: str, pose: Pose) -> None:
        # Pre-step setters mutate the bootstrap (which seeds the next
        # observe) and the cached observation so the predictor sees the
        # updated scene on its next call.
        self._bootstrap.set_object_pose(object_id, pose)
        if self._obs is not None:
            new_objects = dict(self._obs.scene_objects)
            new_objects[object_id] = pose
            self._obs = Observation(
                rgb=self._obs.rgb,
                depth=self._obs.depth,
                robot_joints=self._obs.robot_joints,
                ee_pose=self._obs.ee_pose,
                gripper_width=self._obs.gripper_width,
                scene_objects=new_objects,
                timestamp=self._obs.timestamp,
                camera_intrinsics=self._obs.camera_intrinsics,
                camera_extrinsics=self._obs.camera_extrinsics,
            )

    # ---- introspection ---------------------------------------------------
    @property
    def n_dof(self) -> int:
        return int(self._bootstrap.n_dof)

    @property
    def joint_names(self) -> list[str]:
        return list(self._bootstrap.joint_names)
