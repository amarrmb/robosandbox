"""Exercise the real VLMPointer code path against a checked-in cassette.

Unlike ``test_vlm_pointer.py`` — which stubs the VLM client in-line —
this test drives ``VLMPointer`` through ``CassetteVLMClient``, the same
wrapper a user would run in record mode against a live OpenAI-compatible
endpoint. That way the production plumbing (message construction, JSON
parsing, pixel-to-world projection) is verified with an response shape
matching what the real API returns.

Recording against a live endpoint is documented in the module docstring
at ``robosandbox/vlm/cassette.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from robosandbox.perception.vlm_pointer import VLMPointer
from robosandbox.sim.mujoco_backend import MuJoCoBackend
from robosandbox.types import DetectedObject, Pose, Scene, SceneObject
from robosandbox.vlm.cassette import CassetteMissError, CassetteVLMClient

CASSETTE_DIR = Path(__file__).parent / "cassettes"
RED_CUBE_CASSETTE = CASSETTE_DIR / "vlm_pointer_red_cube.json"


def _red_cube_scene() -> Scene:
    return Scene(
        objects=(
            SceneObject(
                id="red_cube",
                kind="box",
                size=(0.012, 0.012, 0.012),
                pose=Pose(xyz=(0.05, 0.0, 0.06)),
                mass=0.05,
                rgba=(0.85, 0.2, 0.2, 1.0),
            ),
        ),
    )


def test_cassette_file_exists_and_has_entries() -> None:
    """Guard against the cassette being accidentally deleted or emptied."""
    assert RED_CUBE_CASSETTE.exists(), (
        f"checked-in cassette missing: {RED_CUBE_CASSETTE}"
    )
    import json

    data = json.loads(RED_CUBE_CASSETTE.read_text())
    assert data.get("entries"), "cassette has no entries"


def test_vlm_pointer_replays_cassette_and_returns_detected_object() -> None:
    """End-to-end: VLMPointer -> CassetteVLMClient -> hand-authored response
    -> parsed into a typed DetectedObject with a finite 3D pose."""
    client = CassetteVLMClient(RED_CUBE_CASSETTE, model="gpt-4o-mini")
    pointer = VLMPointer(vlm=client)

    sim = MuJoCoBackend(render_size=(480, 640), camera="top")
    sim.load(_red_cube_scene())
    try:
        for _ in range(40):
            sim.step()
        obs = sim.observe()
        hits = pointer.locate("red cube", obs)
    finally:
        sim.close()

    assert len(hits) == 1, f"expected 1 detection from cassette, got {len(hits)}"
    hit = hits[0]
    assert isinstance(hit, DetectedObject)
    # Label and bbox are copied through from the cassette response.
    assert hit.label == "red cube"
    assert hit.bbox_2d == (310, 230, 330, 250)
    # Pixel is inside the rendered frame.
    px, py = hit.pixel_xy
    assert 0 <= px < 640 and 0 <= py < 480
    # Projection succeeded — pose_3d is finite. We deliberately do NOT
    # assert exact world coordinates: the cassette's pixel (320, 240)
    # is plausible but not the true cube centroid; this test is about
    # plumbing, not localization accuracy (see test_vlm_pointer.py for
    # that).
    assert hit.pose_3d is not None
    for v in hit.pose_3d.xyz:
        assert isinstance(v, float)
        assert v == v  # not NaN


def test_cassette_miss_raises_with_diagnostic_message() -> None:
    """A request not present in the cassette should fail loudly at
    replay time — silent empty responses would mask bugs."""
    client = CassetteVLMClient(RED_CUBE_CASSETTE, model="gpt-4o-mini")
    with pytest.raises(CassetteMissError) as excinfo:
        client.chat(
            messages=[
                {"role": "system", "content": "unrelated system prompt"},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Query: nothing"}],
                },
            ]
        )
    # Diagnostic should mention the searched text so a human can see why.
    assert "Query: nothing" in str(excinfo.value) or "nothing" in str(excinfo.value)
