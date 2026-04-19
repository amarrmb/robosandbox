"""parse_json_loose — forgiving JSON parser for VLM outputs."""

import pytest
from robosandbox.vlm.json_recovery import VLMOutputError, parse_json_loose


def test_clean_json() -> None:
    assert parse_json_loose('{"a": 1}') == {"a": 1}


def test_fenced_json() -> None:
    txt = "```json\n{\"objects\": []}\n```"
    assert parse_json_loose(txt) == {"objects": []}


def test_prose_wrapper() -> None:
    txt = 'Sure, here is the result:\n{"x": [1, 2, 3]}\nLet me know.'
    assert parse_json_loose(txt) == {"x": [1, 2, 3]}


def test_truncated_object_recovered() -> None:
    txt = '{"objects": [{"label": "cube", "confidence": 0.9}'
    out = parse_json_loose(txt)
    assert out == {"objects": [{"label": "cube", "confidence": 0.9}]}


def test_empty_raises() -> None:
    with pytest.raises(VLMOutputError):
        parse_json_loose("")


def test_pure_prose_raises() -> None:
    with pytest.raises(VLMOutputError):
        parse_json_loose("I don't know what you want.")
