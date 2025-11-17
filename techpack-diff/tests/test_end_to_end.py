"""Lightweight end-to-end test using synthesized PDFs."""
import json
from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from service import diff_engine, page_matching


def _create_pdf(path: Path, lines: list[str]) -> None:
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for line in lines:
        page.insert_text((72, y), line)
        y += 20
    doc.save(str(path))
    doc.close()


def test_run_comparison_smoke(tmp_path, monkeypatch):
    old_pdf = tmp_path / "old.pdf"
    new_pdf = tmp_path / "new.pdf"
    _create_pdf(old_pdf, ["Part 001", "Qty 5", "Zipper"])
    _create_pdf(new_pdf, ["Part 001", "Qty 7", "Zipper", "New hood panel"])

    out_dir = tmp_path / "out"

    monkeypatch.setattr(diff_engine, "extract_tables", lambda path: {})
    monkeypatch.setattr(page_matching, "extract_tables", lambda path: {})

    result = diff_engine.run_comparison(old_pdf, new_pdf, out_dir)
    assert Path(result["changes_json"]).exists()
    payload = json.loads(Path(result["changes_json"]).read_text())
    assert isinstance(payload, list)
    assert Path(result["diff_pdf"]).exists()
