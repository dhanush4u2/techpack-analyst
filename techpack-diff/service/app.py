"""FastAPI surface for the TechPack diff service."""
from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .diff_engine import run_comparison
from .webhook_handler import process_webhook

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "out"
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="TechPack Diff", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _validate_pdf(upload: UploadFile) -> None:
    if not upload.filename or not upload.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are allowed")


async def _save_and_compare(old_file: UploadFile, new_file: UploadFile) -> Dict:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        old_path = tmp / "TechPack1.pdf"
        new_path = tmp / "TechPack2.pdf"
        await asyncio.gather(
            _stream_upload(old_file, old_path),
            _stream_upload(new_file, new_path),
        )
        return run_comparison(old_path, new_path, OUTPUT_DIR)


async def _stream_upload(upload: UploadFile, destination: Path) -> None:
    with destination.open("wb") as buffer:
        while chunk := await upload.read(1_048_576):
            buffer.write(chunk)
    await upload.close()


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/compare")
async def compare_endpoint(
    techpack1: UploadFile = File(..., description="Legacy TechPack PDF"),
    techpack2: UploadFile = File(..., description="Updated TechPack PDF"),
) -> Dict:
    _validate_pdf(techpack1)
    _validate_pdf(techpack2)
    try:
        result = await _save_and_compare(techpack1, techpack2)
    except Exception as exc:  # pragma: no cover - runtime errors bubble to FastAPI
        LOGGER.exception("Comparison failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result


@app.post("/webhook/compare")
async def webhook_compare(
    techpack1: UploadFile = File(...),
    techpack2: UploadFile = File(...),
) -> Dict:
    _validate_pdf(techpack1)
    _validate_pdf(techpack2)
    try:
        result = await process_webhook(techpack1, techpack2, OUTPUT_DIR)
    except Exception as exc:
        LOGGER.exception("Webhook comparison failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result


if __name__ == "__main__":  # pragma: no cover - manual entrypoint
    import uvicorn

    uvicorn.run("service.app:app", host="0.0.0.0", port=8000, reload=True)
