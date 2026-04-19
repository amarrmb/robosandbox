"""VLMPointer uses a stub VLM to return known pixel coords.
Validates: projection from pixel + depth → 3D lands within a few cm of
the cube's ground-truth pose.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from robosandbox.perception.vlm_pointer import VLMPointer
from robosandbox.sim.mujoco_backend import MuJoCoBackend
from robosandbox.types import Pose, Scene, SceneObject


class StubVLM:
    """Returns a canned response with a given pixel for the only object."""

    def __init__(self, px: int, py: int, label: str = "cube") -> None:
        self._px = px
        self._py = py
        self._label = label

    def chat(self, messages, tools=None, tool_choice=None, **kw):
        return {
            "content": json.dumps(
                {
                    "objects": [
                        {
                            "label": self._label,
                            "point": [self._px, self._py],
                            "bbox": [
                                self._px - 10,
                                self._py - 10,
                                self._px + 10,
                                self._py + 10,
                            ],
                            "confidence": 0.9,
                        }
                    ]
                }
            ),
            "tool_calls": [],
            "finish_reason": "stop",
            "raw": None,
        }


def _cube_scene(cube_xyz: tuple[float, float, float]) -> Scene:
    return Scene(
        objects=(
            SceneObject(
                id="red_cube",
                kind="box",
                size=(0.012, 0.012, 0.012),
                pose=Pose(xyz=cube_xyz),
                mass=0.05,
                rgba=(0.85, 0.2, 0.2, 1.0),
            ),
        ),
    )


def _find_cube_pixel(obs) -> tuple[int, int]:
    """Scan the rendered RGB for the brightest red pixel — that's the cube."""
    rgb = obs.rgb.astype(np.int16)
    # "red-ness" = R - max(G, B)
    red_strength = rgb[..., 0] - np.maximum(rgb[..., 1], rgb[..., 2])
    y, x = np.unravel_index(int(np.argmax(red_strength)), red_strength.shape)
    return int(x), int(y)


def test_projection_lands_near_cube_for_various_positions() -> None:
    """VLMPointer + stub VLM should land within 3cm of the cube's true pose
    when the pixel is the actual cube centroid."""
    for cube_xyz in [(0.05, 0.0, 0.06), (0.08, 0.05, 0.06), (0.02, -0.07, 0.06)]:
        sim = MuJoCoBackend(render_size=(480, 640), camera="top")
        sim.load(_cube_scene(cube_xyz))
        try:
            for _ in range(80):
                sim.step()
            obs = sim.observe()
            px, py = _find_cube_pixel(obs)
            stub = StubVLM(px, py, label="red_cube")
            pointer = VLMPointer(vlm=stub)
            hits = pointer.locate("red cube", obs)
        finally:
            sim.close()

        assert len(hits) == 1, f"expected 1 hit for {cube_xyz}"
        assert hits[0].pose_3d is not None
        predicted = np.array(hits[0].pose_3d.xyz)
        # The cube at steady state settles at cube_center_z determined by sim;
        # we compare only X and Y because Z depends on cube resting pose.
        actual = np.array(cube_xyz)
        xy_err = float(np.linalg.norm(predicted[:2] - actual[:2]))
        assert xy_err < 0.03, (
            f"projection error {xy_err:.3f}m > 3cm for cube at {cube_xyz} — "
            f"predicted {predicted.tolist()}"
        )


def test_vlm_pointer_without_vlm_raises() -> None:
    pointer = VLMPointer(vlm=None)
    sim = MuJoCoBackend(render_size=(120, 160))
    sim.load(_cube_scene((0.05, 0.0, 0.06)))
    try:
        for _ in range(20):
            sim.step()
        obs = sim.observe()
        with pytest.raises(RuntimeError):
            pointer.locate("anything", obs)
    finally:
        sim.close()


def test_vlm_pointer_returns_empty_on_bad_json() -> None:
    class BrokenVLM:
        def chat(self, *a, **k):
            return {
                "content": "I'm not sure.",
                "tool_calls": [],
                "finish_reason": "stop",
                "raw": None,
            }

    sim = MuJoCoBackend(render_size=(240, 320), camera="top")
    sim.load(_cube_scene((0.05, 0.0, 0.06)))
    try:
        for _ in range(30):
            sim.step()
        obs = sim.observe()
        out = VLMPointer(vlm=BrokenVLM()).locate("anything", obs)
    finally:
        sim.close()
    assert out == []
