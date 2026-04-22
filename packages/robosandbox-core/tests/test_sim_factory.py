from __future__ import annotations

import argparse

import pytest

from robosandbox import cli
from robosandbox.sim.factory import (
    create_sim_backend,
    get_sim_backend_class,
    list_sim_backends,
)


def test_list_sim_backends_includes_builtin_names() -> None:
    names = list_sim_backends()
    assert "mujoco" in names
    assert "newton" in names


def test_get_sim_backend_class_returns_newton_without_importing_runtime() -> None:
    cls = get_sim_backend_class("newton")
    assert cls.__name__ == "NewtonBackend"


def test_create_sim_backend_filters_unknown_kwargs() -> None:
    backend = create_sim_backend("newton", viewer="null", bogus="ignored")
    assert backend.__class__.__name__ == "NewtonBackend"
    assert backend._viewer_kind == "null"


def test_unknown_backend_raises_helpful_error() -> None:
    with pytest.raises(ValueError, match="unknown sim backend"):
        get_sim_backend_class("does-not-exist")


def test_planner_run_rejects_newton_backend() -> None:
    args = argparse.Namespace(
        cmd="run",
        task="pick up the red cube",
        task_flag=None,
        policy=None,
        max_steps=1000,
        vlm_provider="stub",
        model=None,
        base_url=None,
        api_key_env=None,
        perception=None,
        max_replans=3,
        sim_backend="newton",
        sim_viewer="null",
        viewer_port=8080,
        device="cuda:0",
        log_level="INFO",
    )
    with pytest.raises(SystemExit):
        cli.main(
            [
                "run",
                "pick up the red cube",
                "--sim-backend",
                "newton",
            ]
        )
