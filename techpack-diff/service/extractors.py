"""Input extraction utilities for PDF pages, tables, and images."""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import camelot
import fitz  # PyMuPDF
import pandas as pd
from PIL import Image

from .utils import hash_tables, normalize_text, read_config

LOGGER = logging.getLogger(__name__)


def open_document(path: Path) -> fitz.Document:
    """Open a PDF document via PyMuPDF with caching."""

    return fitz.open(path)


def extract_page_text(doc: fitz.Document, page_index: int, limit: int) -> str:
    """Return normalized text for the target page."""

    page = doc[page_index]
    text = page.get_text("text")
    return normalize_text(text, limit=limit)


def render_thumbnail(doc: fitz.Document, page_index: int, zoom: float) -> Image.Image:
    """Render a page to a PIL image using the provided zoom."""

    page = doc[page_index]
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def extract_tables(pdf_path: Path) -> Dict[int, List[pd.DataFrame]]:
    """Use Camelot to read tables from a PDF and group by page index."""

    tables_by_page: Dict[int, List[pd.DataFrame]] = {}
    try:
        result = camelot.read_pdf(str(pdf_path), pages="all", flavor="lattice", strip_text="\n")
    except Exception as exc:  # pragma: no cover - Camelot-specific runtime failures
        LOGGER.warning("Camelot failed to parse %s: %s", pdf_path, exc)
        return tables_by_page
    for table in result:
        df = table.df.replace({"\n": " "}, regex=True)
        page_zero_indexed = max(int(str(table.page).split(",")[0]) - 1, 0)
        tables_by_page.setdefault(page_zero_indexed, []).append(df)
    return tables_by_page


def table_signature_for_page(tables_by_page: Dict[int, List[pd.DataFrame]], page_index: int) -> str:
    """Combine table rows into a deterministic signature string."""

    frames = tables_by_page.get(page_index, [])
    if not frames:
        return ""
    rows = []
    for frame in frames:
        rows.extend(" ".join(frame.iloc[row].astype(str).tolist()) for row in range(len(frame)))
    return hash_tables(rows)
