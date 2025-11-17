"""Visual sketch comparison utilities."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from .utils import read_config

LOGGER = logging.getLogger(__name__)


@dataclass
class VisualChange:
    id: str
    type: str
    page_old: int
    page_new: int
    old_value: str
    new_value: str
    bbox_old: Tuple[float, float, float, float]
    bbox_new: Tuple[float, float, float, float]
    confidence: float


def _callout_regions(page: fitz.Page) -> List[fitz.Rect]:
    """Extract candidate callout bounding boxes via text blocks containing digits."""

    blocks = page.get_text("blocks")
    regions: List[fitz.Rect] = []
    for block in blocks:
        if len(block) < 5:
            continue
        text = block[4].strip()
        if not text or len(text) > 25:
            continue
        if any(ch.isdigit() for ch in text):
            regions.append(fitz.Rect(block[:4]))
    return regions


def _iou(rect_a: fitz.Rect, rect_b: fitz.Rect) -> float:
    inter = rect_a & rect_b
    if inter.is_empty:
        return 0.0
    inter_area = inter.get_area()
    union = rect_a.get_area() + rect_b.get_area() - inter_area
    if union == 0:
        return 0.0
    return inter_area / union


def _pixmap_from_rect(page: fitz.Page, rect: fitz.Rect, zoom: float) -> Image.Image:
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, clip=rect)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def compare_visual_regions(doc_old: fitz.Document, doc_new: fitz.Document,
                           page_old: int, page_new: int,
                           config: dict | None = None) -> List[VisualChange]:
    """Compare overlapping callout regions across two pages."""

    config = config or read_config()
    zoom = config.get("scan_zoom", 2.0)
    threshold = config.get("ssim_threshold", 0.92)
    min_area = config.get("ssim_min_area", 5000)

    page_old_obj = doc_old[page_old]
    page_new_obj = doc_new[page_new]

    regions_old = _callout_regions(page_old_obj)
    regions_new = _callout_regions(page_new_obj)

    changes: List[VisualChange] = []
    change_counter = 0
    for rect_new in regions_new:
        best_rect = None
        best_iou = 0.0
        for rect_old in regions_old:
            overlap = _iou(rect_old, rect_new)
            if overlap > best_iou:
                best_iou = overlap
                best_rect = rect_old
        if not best_rect or best_iou < 0.05:
            continue
        if rect_new.get_area() < min_area:
            continue
        img_old = _pixmap_from_rect(page_old_obj, best_rect, zoom).convert("L")
        img_new = _pixmap_from_rect(page_new_obj, rect_new, zoom).convert("L")
        resized = img_old.resize(img_new.size)
        score = ssim(np.array(resized), np.array(img_new))
        if score >= threshold:
            continue
        change_counter += 1
        changes.append(
            VisualChange(
                id=f"visual_{page_new}_{change_counter}",
                type="visual_change",
                page_old=page_old,
                page_new=page_new,
                old_value="Sketch difference",
                new_value="Sketch difference",
                bbox_old=(best_rect.x0, best_rect.y0, best_rect.x1, best_rect.y1),
                bbox_new=(rect_new.x0, rect_new.y0, rect_new.x1, rect_new.y1),
                confidence=1 - score,
            )
        )
    return changes
