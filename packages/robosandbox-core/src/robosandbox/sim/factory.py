"""Sim backend registry helpers.

This keeps backend selection in one place so CLI and future plugins do
not need to import concrete sim implementations directly.
"""

from __future__ import annotations

import inspect
from importlib import import_module
from importlib.metadata import entry_points
from typing import Any


_BUILTIN_BACKENDS: dict[str, str] = {
    "mujoco": "robosandbox.sim.mujoco_backend:MuJoCoBackend",
    "newton": "robosandbox.sim.newton_backend:NewtonBackend",
    # ThreadPool over N MuJoCoBackend instances; same observe_all/step_all
    # parallel API as Newton, used by `train --sim-backend mujoco_vec` for
    # PPO refinement on the sim where contact actually converges.
    "mujoco_vec": "robosandbox.sim.mujoco_vec_backend:MuJoCoVecBackend",
    # Learned-dynamics propagator. Bootstraps from a real backend for the
    # initial observation + scene introspection, then time-evolves via a
    # WorldModelPredictor (default IdentityPredictor — proves the slot,
    # scores 0% until you swap in a real model). See
    # docs/site/docs/concepts/eval-for-world-models.md.
    "world_model": "robosandbox.sim.world_model_backend:WorldModelBackend",
}


def _load_dotted(ref: str) -> type:
    module_name, _, attr = ref.partition(":")
    if not module_name or not attr:
        raise ValueError(f"invalid backend reference {ref!r}")
    module = import_module(module_name)
    return getattr(module, attr)


def _iter_entry_points():
    for ep in entry_points(group="robosandbox.sim"):
        yield ep


def list_sim_backends() -> list[str]:
    names = set(_BUILTIN_BACKENDS)
    for ep in _iter_entry_points():
        names.add(ep.name)
    return sorted(names)


def get_sim_backend_class(name: str) -> type:
    for ep in _iter_entry_points():
        if ep.name == name:
            return ep.load()
    ref = _BUILTIN_BACKENDS.get(name)
    if ref is None:
        available = ", ".join(list_sim_backends())
        raise ValueError(f"unknown sim backend {name!r}; available: {available}")
    return _load_dotted(ref)


def create_sim_backend(name: str, **kwargs: Any) -> Any:
    cls = get_sim_backend_class(name)
    params = inspect.signature(cls).parameters
    accepted = {
        key: value for key, value in kwargs.items() if key in params
    }
    return cls(**accepted)
