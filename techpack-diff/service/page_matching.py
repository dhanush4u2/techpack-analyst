"""Page fingerprinting and semantic matching utilities."""
from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import fitz  # PyMuPDF

from .extractors import extract_page_text, extract_tables, render_thumbnail, table_signature_for_page
from .utils import PageFingerprint, hamming_similarity, perceptual_hash, read_config, text_similarity

LOGGER = logging.getLogger(__name__)


def build_page_fingerprints(pdf_path: Path, config: Dict | None = None) -> List[PageFingerprint]:
    """Generate multimodal fingerprints for each page of a PDF."""

    config = config or read_config()
    doc = fitz.open(pdf_path)
    tables_by_page = extract_tables(pdf_path)
    fingerprints: List[PageFingerprint] = []
    for page_index in range(doc.page_count):
        text_summary = extract_page_text(doc, page_index, config.get("max_text_summary_chars", 2500))
        table_signature = table_signature_for_page(tables_by_page, page_index)
        thumbnail = render_thumbnail(doc, page_index, config.get("scan_zoom", 2.0))
        thumbnail_hash = perceptual_hash(thumbnail)
        fingerprints.append(
            PageFingerprint(
                page_num=page_index,
                text_summary=text_summary,
                table_signature=table_signature,
                thumbnail_hash=thumbnail_hash,
            )
        )
    doc.close()
    return fingerprints


def combined_similarity(old_fp: PageFingerprint, new_fp: PageFingerprint) -> Tuple[float, float, float, float]:
    """Return (combined, text, table, thumbnail) similarity tuple."""

    text_score = text_similarity(old_fp.text_summary, new_fp.text_summary)
    table_score = 1.0 if old_fp.table_signature and old_fp.table_signature == new_fp.table_signature else 0.0
    thumb_score = hamming_similarity(old_fp.thumbnail_hash, new_fp.thumbnail_hash)
    combined = 0.7 * text_score + 0.2 * table_score + 0.1 * thumb_score
    return combined, text_score, table_score, thumb_score


def match_pages(old_fps: Iterable[PageFingerprint], new_fps: Iterable[PageFingerprint], config: Dict | None = None) -> Dict[int, List[Tuple[int, float]]]:
    """Produce similarity rankings for each new page relative to the legacy document."""

    config = config or read_config()
    threshold = config.get("page_combined_score_threshold", 0.78)
    old_list = list(old_fps)
    new_list = list(new_fps)
    mapping: Dict[int, List[Tuple[int, float]]] = {}
    for new_fp in new_list:
        scores: List[Tuple[int, float]] = []
        for old_fp in old_list:
            combined, *_ = combined_similarity(old_fp, new_fp)
            if combined > 0:
                scores.append((old_fp.page_num, combined))
        scores.sort(key=lambda item: item[1], reverse=True)
        mapping[new_fp.page_num] = [(idx, score) for idx, score in scores if score >= threshold]
        if not mapping[new_fp.page_num]:
            LOGGER.info("No semantic match for new page %s; will require fallback", new_fp.page_num)
    return mapping
