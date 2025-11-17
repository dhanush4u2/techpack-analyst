"""Utilities dedicated to n8n webhook flows."""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Dict

from fastapi import UploadFile

from .diff_engine import run_comparison


async def _stream_to_path(upload: UploadFile, destination: Path) -> None:
    with destination.open("wb") as buffer:
        while chunk := await upload.read(1_048_576):
            buffer.write(chunk)
    await upload.close()


async def process_webhook(techpack1: UploadFile, techpack2: UploadFile,
                          output_root: Path) -> Dict:
    """Persist uploaded PDFs, run comparison, and return artifact paths."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        old_path = tmp / "techpack1.pdf"
        new_path = tmp / "techpack2.pdf"
        await asyncio.gather(
            _stream_to_path(techpack1, old_path),
            _stream_to_path(techpack2, new_path),
        )
        result = run_comparison(old_path, new_path, output_root)
    return {
        "job_id": result["job_id"],
        "changes_json_url": result["changes_json"],
        "diff_pdf_url": result["diff_pdf"],
        "page_mapping": result["page_mapping"],
    }
