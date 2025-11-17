"""Core orchestration logic for TechPack semantic diffing."""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import fitz  # PyMuPDF

from .extractors import extract_tables
from .llm_verifier import verify_change
from .page_matching import build_page_fingerprints, match_pages
from .table_diff import TableChange, diff_tables_for_pages
from .utils import (
    ensure_directory,
    normalize_text,
    read_config,
    text_similarity,
)
from .visual_diff import VisualChange, compare_visual_regions


def _blocks_for_document(doc: fitz.Document) -> List[Dict]:
    blocks = []
    for page_index, page in enumerate(doc):
        for raw in page.get_text("blocks"):
            if len(raw) < 5:
                continue
            text = raw[4].strip()
            if not text:
                continue
            blocks.append(
                {
                    "page": page_index,
                    "text": text,
                    "norm": normalize_text(text),
                    "bbox": tuple(raw[:4]),
                }
            )
    return blocks


def _compare_text_blocks(old_page: fitz.Page, new_page: fitz.Page,
                          change_prefix: str, text_threshold: float,
                          fallback_threshold: float, counter_start: int) -> Tuple[List[Dict], int]:
    """Compare block-level text content on matched pages."""

    old_blocks = [
        {
            "bbox": tuple(raw[:4]),
            "text": raw[4].strip(),
            "norm": normalize_text(raw[4]),
        }
        for raw in old_page.get_text("blocks")
        if len(raw) >= 5 and raw[4].strip()
    ]
    new_blocks = [
        {
            "bbox": tuple(raw[:4]),
            "text": raw[4].strip(),
            "norm": normalize_text(raw[4]),
        }
        for raw in new_page.get_text("blocks")
        if len(raw) >= 5 and raw[4].strip()
    ]

    changes: List[Dict] = []
    used_old = set()
    counter = counter_start
    for block in new_blocks:
        best_idx = None
        best_score = 0.0
        for idx, old_block in enumerate(old_blocks):
            score = text_similarity(block["norm"], old_block["norm"])
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_score >= text_threshold:
            used_old.add(best_idx)
            continue
        change_type = "text_change"
        change_kind = "modified" if best_score >= fallback_threshold else "added"
        counter += 1
        old_value = old_blocks[best_idx]["text"] if best_idx is not None else ""
        old_bbox = old_blocks[best_idx]["bbox"] if best_idx is not None else None
        changes.append(
            {
                "id": f"{change_prefix}_{counter}",
                "type": change_type,
                "page_old": old_page.number if best_idx is not None else None,
                "page_new": new_page.number,
                "old_value": old_value,
                "new_value": block["text"],
                "bbox_old": old_bbox,
                "bbox_new": block["bbox"],
                "confidence": float(max(best_score, 0.4)),
                "status": change_kind,
            }
        )
        if best_idx is not None:
            used_old.add(best_idx)

    for idx, old_block in enumerate(old_blocks):
        if idx in used_old:
            continue
        counter += 1
        changes.append(
            {
                "id": f"{change_prefix}_{counter}",
                "type": "text_change",
                "page_old": old_page.number,
                "page_new": new_page.number,
                "old_value": old_block["text"],
                "new_value": "",
                "bbox_old": old_block["bbox"],
                "bbox_new": None,
                "confidence": 0.7,
                "status": "removed",
            }
        )
    return changes, counter


def global_block_search_and_compare(all_old_blocks: List[Dict], new_page: fitz.Page,
                                    change_prefix: str, threshold: float,
                                    counter_start: int) -> Tuple[List[Dict], int]:
    """Fallback search that scans the entire legacy TechPack for nearest matches."""

    counter = counter_start
    changes: List[Dict] = []
    for raw in new_page.get_text("blocks"):
        if len(raw) < 5 or not raw[4].strip():
            continue
        target_norm = normalize_text(raw[4])
        best_block = None
        best_score = 0.0
        for block in all_old_blocks:
            score = text_similarity(target_norm, block["norm"])
            if score > best_score:
                best_score = score
                best_block = block
        if best_score < threshold or best_block is None:
            continue
        counter += 1
        changes.append(
            {
                "id": f"{change_prefix}_{counter}",
                "type": "text_change",
                "page_old": best_block["page"],
                "page_new": new_page.number,
                "old_value": best_block["text"],
                "new_value": raw[4].strip(),
                "bbox_old": best_block["bbox"],
                "bbox_new": tuple(raw[:4]),
                "confidence": best_score,
                "status": "fallback",  # indicates global search
            }
        )
    return changes, counter


def annotate_diff_pdf(changes: Sequence[Dict], new_pdf: Path, output_pdf: Path, config: Dict) -> None:
    """Overlay colored rectangles for TP2 pages and append a sidebar change list."""

    doc = fitz.open(new_pdf)
    colors = config.get("annotation_colors", {})
    color_added = tuple(colors.get("added", [0.2, 0.8, 0.2]))
    color_removed = tuple(colors.get("removed", [0.9, 0.2, 0.2]))
    color_modified = tuple(colors.get("modified", [0.95, 0.8, 0.2]))

    by_page: Dict[int, List[Dict]] = {}
    for change in changes:
        page_new = change.get("page_new")
        if page_new is None or page_new >= doc.page_count:
            continue
        by_page.setdefault(page_new, []).append(change)

    for page_index, page in enumerate(doc):
        page_changes = by_page.get(page_index)
        if not page_changes:
            continue
        original_rect = page.rect
        sidebar_start = original_rect.width
        new_width = original_rect.width * 1.35
        page.set_media_box(fitz.Rect(0, 0, new_width, original_rect.height))
        # Sidebar background
        page.draw_rect(
            fitz.Rect(sidebar_start, 0, new_width, original_rect.height),
            color=(1, 1, 1),
            fill=(0.97, 0.97, 0.97),
        )
        page.draw_line(
            fitz.Point(sidebar_start, 0),
            fitz.Point(sidebar_start, original_rect.height),
            color=(0.7, 0.7, 0.7),
            width=0.5,
        )
        y = 20
        for change in page_changes:
            bbox = change.get("bbox_new") or change.get("bbox_old")
            if bbox:
                rect = fitz.Rect(*bbox)
                if change.get("status") == "added" and not change.get("old_value"):
                    color = color_added
                elif change.get("status") == "removed" or not change.get("new_value"):
                    color = color_removed
                else:
                    color = color_modified
                page.draw_rect(rect, color=color, width=1.4)
            text = f"{change['id']}: {change['type']} ({change.get('confidence', 0):.2f})"
            page.insert_text(
                fitz.Point(sidebar_start + 10, y),
                text,
                fontsize=8,
                color=(0.1, 0.1, 0.1),
            )
            y += 12
    doc.save(output_pdf)
    doc.close()


def run_comparison(old_pdf: Path, new_pdf: Path, output_root: Path,
                   config_path: Optional[Path] = None) -> Dict:
    """End-to-end orchestration returning artifact metadata."""

    config = read_config(config_path)
    output_root = ensure_directory(Path(output_root))
    job_id = f"job_{int(time.time() * 1000)}"
    job_dir = ensure_directory(output_root / job_id)

    fingerprints_old = build_page_fingerprints(old_pdf, config)
    fingerprints_new = build_page_fingerprints(new_pdf, config)
    page_mapping = match_pages(fingerprints_old, fingerprints_new, config)

    tables_old = extract_tables(old_pdf)
    tables_new = extract_tables(new_pdf)

    doc_old = fitz.open(old_pdf)
    doc_new = fitz.open(new_pdf)
    all_old_blocks = _blocks_for_document(doc_old)

    text_threshold = config.get("text_similarity_threshold", 0.82)
    fallback_threshold = config.get("block_fallback_threshold", 0.65)

    change_counter = 0
    change_objects: List[Dict] = []

    for new_page, ranked_matches in page_mapping.items():
        if ranked_matches:
            old_page = ranked_matches[0][0]
            old_page_obj = doc_old[old_page]
            new_page_obj = doc_new[new_page]
            page_changes, change_counter = _compare_text_blocks(
                old_page_obj,
                new_page_obj,
                f"change_{new_page}",
                text_threshold,
                fallback_threshold,
                change_counter,
            )
            change_objects.extend(page_changes)
            table_changes = diff_tables_for_pages(
                tables_old.get(old_page, []),
                tables_new.get(new_page, []),
                old_page,
                new_page,
                f"table_{new_page}",
            )
            for table_change in table_changes:
                change_counter += 1
                change_objects.append(
                    {
                        "id": f"table_{new_page}_{change_counter}",
                        "type": table_change.type,
                        "page_old": table_change.page_old,
                        "page_new": table_change.page_new,
                        "old_value": table_change.old_value,
                        "new_value": table_change.new_value,
                        "bbox_old": table_change.bbox_old,
                        "bbox_new": table_change.bbox_new,
                        "confidence": table_change.confidence,
                        "status": "table",
                    }
                )
            visual_changes = compare_visual_regions(doc_old, doc_new, old_page, new_page, config)
            for visual_change in visual_changes:
                change_counter += 1
                change_objects.append(
                    {
                        "id": f"visual_{new_page}_{change_counter}",
                        "type": visual_change.type,
                        "page_old": visual_change.page_old,
                        "page_new": visual_change.page_new,
                        "old_value": visual_change.old_value,
                        "new_value": visual_change.new_value,
                        "bbox_old": visual_change.bbox_old,
                        "bbox_new": visual_change.bbox_new,
                        "confidence": visual_change.confidence,
                        "status": "visual",
                    }
                )
        else:
            fallback_changes, change_counter = global_block_search_and_compare(
                all_old_blocks,
                doc_new[new_page],
                f"fallback_{new_page}",
                fallback_threshold,
                change_counter,
            )
            change_objects.extend(fallback_changes)

    doc_old.close()
    doc_new.close()

    verified_changes = [verify_change(change, config) for change in change_objects]

    changes_json_path = job_dir / "changes.json"
    changes_json_path.write_text(json.dumps(verified_changes, indent=2), encoding="utf-8")

    diff_pdf_path = job_dir / "diff.pdf"
    annotate_diff_pdf(verified_changes, new_pdf, diff_pdf_path, config)

    page_mapping_path = job_dir / "page_mapping.json"
    serialized_mapping = {
        str(new_page): [(old_page, score) for (old_page, score) in ranked]
        for new_page, ranked in page_mapping.items()
    }
    page_mapping_path.write_text(json.dumps(serialized_mapping, indent=2), encoding="utf-8")

    return {
        "job_id": job_id,
        "changes_json": str(changes_json_path),
        "diff_pdf": str(diff_pdf_path),
        "page_mapping": serialized_mapping,
        "page_mapping_file": str(page_mapping_path),
    }
