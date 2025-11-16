"""FastAPI service exposing the techpack diff comparison pipeline."""
from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .diff_engine import run_comparison

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "out"
OUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Techpack Diff MVP", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _save_upload(upload: UploadFile, destination: Path) -> None:
    """Persist uploaded files to disk without loading into memory."""

    with destination.open("wb") as buffer:
        while chunk := await upload.read(1024 * 1024):
            buffer.write(chunk)
    await upload.close()


@app.post("/compare")
async def compare_endpoint(
    old: UploadFile = File(..., description="Legacy TechPack PDF"),
    new: UploadFile = File(..., description="Updated TechPack PDF"),
    review: UploadFile = File(..., description="Review Excel"),
) -> Dict[str, str]:
    """Execute the comparison pipeline and return output file paths."""

    if not old.filename.lower().endswith(".pdf") or not new.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Both tech packs must be PDF files")
    if not review.filename.lower().endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Review report must be an .xlsx workbook")

    job_dir = OUT_DIR / f"job_{int(time.time() * 1000)}"
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            old_path = tmp_path / "old.pdf"
            new_path = tmp_path / "new.pdf"
            review_path = tmp_path / "review.xlsx"

            await asyncio.gather(
                _save_upload(old, old_path),
                _save_upload(new, new_path),
                _save_upload(review, review_path),
            )

            result = run_comparison(old_path, new_path, review_path, job_dir)
    except Exception as exc:  # pragma: no cover - FastAPI error propagation path
        LOGGER.exception("Comparison failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result


@app.get("/health")
async def health() -> Dict[str, str]:
    """Simple health probe for orchestration tooling."""

    return {"status": "ok"}
