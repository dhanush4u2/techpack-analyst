"""Sanity tests for utility helpers."""
from pathlib import Path

from service import utils


def test_normalize_text_for_compare_handles_whitespace():
    assert utils.normalize_text_for_compare("  Foo- Bar\n") == "foo bar"


def test_parse_mm_extracts_numbers():
    assert utils.parse_mm("12.5 mm seam") == 12.5
    assert utils.parse_mm("no measurement") is None


def test_detect_measurement_delta():
    assert utils.detect_measurement_delta("10 mm", "12 mm") == 2.0
    assert utils.detect_measurement_delta("text", "text") is None


def test_read_config_loads_defaults(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("text_similarity_threshold: 0.5\n")
    config = utils.read_config(cfg_path)
    assert config["text_similarity_threshold"] == 0.5
