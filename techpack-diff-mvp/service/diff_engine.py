"""Diff engine that orchestrates PDF, table, text, and visual comparisons."""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import camelot
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from .utils import (
    TextBlock,
    detect_measurement_delta,
    detect_repeated_blocks,
    extract_part_ids,
    is_significant_text_change,
    normalize_text_for_compare,
    priority_from_change,
    read_config,
    text_similarity,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class PageData:
    """Container for text blocks on a page."""

    index: int
    blocks: List[TextBlock] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(block.text for block in self.blocks)


@dataclass
class TableData:
    """Metadata for extracted tables."""

    page: int
    df: pd.DataFrame
    label: str


@dataclass
class ChangeRecord:
    """Normalized change data passed through the pipeline."""

    change_id: str
    page: int
    section: str
    old: str
    new: str
    type: str
    priority: str
    confidence: float
    bbox: Optional[Tuple[float, float, float, float]]
    evidence_old: Optional[str]
    evidence_new: Optional[str]


def extract_page_blocks(pdf_path: Path, config: Dict) -> List[PageData]:
    """Extract page blocks, stripping repeated header/footer text."""

    doc = fitz.open(pdf_path)
    blocks_with_page: List[Tuple[int, TextBlock]] = []
    page_blocks: List[PageData] = []
    for page_index, page in enumerate(doc):
        raw_blocks = page.get_text("blocks")
        converted = []
        for block in raw_blocks:
            if len(block) < 5:
                continue
            x0, y0, x1, y1, text = block[:5]
            clean_text = text.strip()
            converted.append(TextBlock(text=clean_text, bbox=(x0, y0, x1, y1)))
            blocks_with_page.append((page_index, converted[-1]))
        page_blocks.append(PageData(index=page_index, blocks=converted))

    repeated = set(
        detect_repeated_blocks(
            blocks_with_page,
            config.get("header_repeat_threshold", 0.7),
        )
    )
    filtered_pages = []
    for page in page_blocks:
        filtered = [block for block in page.blocks if normalize_text_for_compare(block.text) not in repeated]
        filtered_pages.append(PageData(index=page.index, blocks=filtered))
    return filtered_pages


def match_pages_by_text(old_pages: Sequence[PageData], new_pages: Sequence[PageData], config: Dict) -> Dict[int, Optional[int]]:
    """Map old page indices to new page indices using similarity."""

    threshold = config.get("page_match_threshold", 0.4)
    mapping: Dict[int, Optional[int]] = {}
    used_new: set[int] = set()
    for old_page in old_pages:
        old_text = normalize_text_for_compare(old_page.text)
        best_idx = None
        best_score = 0.0
        for new_page in new_pages:
            if new_page.index in used_new:
                continue
            score = text_similarity(old_text, normalize_text_for_compare(new_page.text))
            if score > best_score:
                best_score = score
                best_idx = new_page.index
        if best_idx is not None and best_score >= threshold:
            mapping[old_page.index] = best_idx
            used_new.add(best_idx)
        else:
            mapping[old_page.index] = None
    return mapping


def extract_tables(pdf_path: Path) -> List[TableData]:
    """Extract tables from a PDF via Camelot."""

    tables: List[TableData] = []
    try:
        camelot_tables = camelot.read_pdf(str(pdf_path), pages="all", flavor="lattice", strip_text="\n")
    except Exception as exc:  # pragma: no cover - Camelot backend issues
        LOGGER.warning("Camelot failed to parse %s: %s", pdf_path, exc)
        return tables
    for table in camelot_tables:
        df = table.df.replace({"\n": " "}, regex=True)
        label = classify_table(df)
        try:
            page_number = int(str(table.page).split(",")[0]) - 1
        except ValueError:
            page_number = 0
        tables.append(TableData(page=page_number, df=df, label=label))
    return tables


def classify_table(df: pd.DataFrame) -> str:
    """Infer table section based on header keywords."""

    header = " ".join(df.iloc[0].astype(str).tolist()).lower() if not df.empty else ""
    if "bom" in header or "bill of" in header:
        return "BOM"
    if "measurement" in header:
        return "Measurements"
    if "construction" in header or "stitch" in header:
        return "Construction"
    return "General"


def diff_tables(old_tables: Sequence[TableData], new_tables: Sequence[TableData], config: Dict) -> List[ChangeRecord]:
    """Compare table collections and yield change records."""

    changes: List[ChangeRecord] = []
    change_counter = 0

    for old_table in old_tables:
        counterparts = [t for t in new_tables if t.label == old_table.label]
        if not counterparts:
            continue
        match = max(counterparts, key=lambda candidate: table_similarity(old_table.df, candidate.df))
        change_counter = analyze_table_pair(old_table, match, config, changes, change_counter)
    return changes


def table_similarity(df_a: pd.DataFrame, df_b: pd.DataFrame) -> float:
    """Very coarse table similarity metric based on flattened text."""

    text_a = " ".join(df_a.astype(str).fillna("").values.flatten())
    text_b = " ".join(df_b.astype(str).fillna("").values.flatten())
    return text_similarity(text_a, text_b)


def analyze_table_pair(old_table: TableData, new_table: TableData, config: Dict,
                       changes: List[ChangeRecord], counter: int) -> int:
    """Compare rows between two matching tables and append change records."""

    old_rows = old_table.df.iloc[1:].reset_index(drop=True)
    new_rows = new_table.df.iloc[1:].reset_index(drop=True)

    def row_signature(row: pd.Series) -> str:
        return " ".join(row.astype(str).tolist())

    old_signatures = [row_signature(row) for _, row in old_rows.iterrows()]
    new_signatures = [row_signature(row) for _, row in new_rows.iterrows()]

    matched_indices: Dict[int, int] = {}
    used_new = set()
    for old_idx, sig in enumerate(old_signatures):
        part_ids = extract_part_ids(sig)
        target_idx = None
        if part_ids:
            for pid in part_ids:
                for new_idx, new_sig in enumerate(new_signatures):
                    if new_idx in used_new:
                        continue
                    if pid in new_sig:
                        target_idx = new_idx
                        break
                if target_idx is not None:
                    break
        if target_idx is None:
            best_idx, score = best_fuzzy_match(sig, new_signatures, used_new)
            if score >= config.get("text_similarity_threshold", 0.9):
                target_idx = best_idx
        if target_idx is not None:
            matched_indices[old_idx] = target_idx
            used_new.add(target_idx)

    # Detect removed rows
    for old_idx, sig in enumerate(old_signatures):
        if old_idx not in matched_indices:
            counter += 1
            change = ChangeRecord(
                change_id=f"C{counter:03d}",
                page=old_table.page,
                section=old_table.label,
                old=sig,
                new="",
                type="Row Removed",
                priority="HIGH",
                confidence=0.95,
                bbox=None,
                evidence_old=None,
                evidence_new=None,
            )
            changes.append(change)

    # Detect added rows and cell deltas
    for new_idx, sig in enumerate(new_signatures):
        if new_idx not in matched_indices.values():
            counter += 1
            changes.append(
                ChangeRecord(
                    change_id=f"C{counter:03d}",
                    page=new_table.page,
                    section=new_table.label,
                    old="",
                    new=sig,
                    type="Row Added",
                    priority="MED",
                    confidence=0.9,
                    bbox=None,
                    evidence_old=None,
                    evidence_new=None,
                )
            )

    for old_idx, new_idx in matched_indices.items():
        old_row = old_rows.iloc[old_idx]
        new_row = new_rows.iloc[new_idx]
        for col, old_val in old_row.items():
            new_val = new_row.get(col)
            if pd.isna(old_val) and pd.isna(new_val):
                continue
            if normalize_text_for_compare(str(old_val)) == normalize_text_for_compare(str(new_val)):
                continue
            measurement_delta = detect_measurement_delta(str(old_val), str(new_val))
            severity_score = measurement_delta if measurement_delta is not None else text_similarity(str(old_val), str(new_val))
            contains_keyword = any(k in normalize_text_for_compare(str(new_val)) for k in config.get("domain_keywords", []))
            priority = priority_from_change("Table", severity_score or 0.0, contains_keyword, config)
            counter += 1
            changes.append(
                ChangeRecord(
                    change_id=f"C{counter:03d}",
                    page=new_table.page,
                    section=new_table.label,
                    old=f"{col}: {old_val}",
                    new=f"{col}: {new_val}",
                    type="Cell Updated",
                    priority=priority,
                    confidence=0.85,
                    bbox=None,
                    evidence_old=None,
                    evidence_new=None,
                )
            )
    return counter


def best_fuzzy_match(source: str, candidates: Sequence[str], used: set[int]) -> Tuple[int, float]:
    """Return the best fuzzy match index and its score."""

    best_index = -1
    best_score = 0.0
    for idx, candidate in enumerate(candidates):
        if idx in used:
            continue
        score = text_similarity(source, candidate)
        if score > best_score:
            best_index = idx
            best_score = score
    return best_index, best_score


def diff_page_text(old_pages: Sequence[PageData], new_pages: Sequence[PageData], config: Dict) -> List[ChangeRecord]:
    """Detect textual changes at the page level."""

    changes: List[ChangeRecord] = []
    counter = 0
    mapping = match_pages_by_text(old_pages, new_pages, config)
    old_lookup = {page.index: page for page in old_pages}
    new_lookup = {page.index: page for page in new_pages}
    for old_idx, new_idx in mapping.items():
        if new_idx is None:
            continue
        old_page = old_lookup[old_idx]
        new_page = new_lookup[new_idx]
        old_text = old_page.text
        new_text = new_page.text
        if is_significant_text_change(old_text, new_text, config.get("significant_text_ratio", 0.85),
                                      domain_keywords=config.get("domain_keywords")):
            counter += 1
            changes.append(
                ChangeRecord(
                    change_id=f"C{counter:03d}",
                    page=new_idx,
                    section="Construction",
                    old=old_text,
                    new=new_text,
                    type="Text",
                    priority="MED",
                    confidence=0.8,
                    bbox=None,
                    evidence_old=None,
                    evidence_new=None,
                )
            )
    return changes


def detect_visual_changes(old_pdf: Path, new_pdf: Path, config: Dict) -> List[ChangeRecord]:
    """Perform lightweight SSIM-based visual comparisons per matched pages."""

    changes: List[ChangeRecord] = []
    doc_old = fitz.open(old_pdf)
    doc_new = fitz.open(new_pdf)
    min_pages = min(len(doc_old), len(doc_new))
    counter = 0
    for page_idx in range(min_pages):
        img_old = render_page_image(doc_old, page_idx, config.get("default_zoom", 2.0))
        img_new = render_page_image(doc_new, page_idx, config.get("default_zoom", 2.0))
        if img_old.size != img_new.size:
            img_new = img_new.resize(img_old.size)
        arr_old = np.array(img_old.convert("L"))
        arr_new = np.array(img_new.convert("L"))
        score = ssim(arr_old, arr_new)
        if score >= config.get("image_ssim_threshold", 0.9):
            continue
        counter += 1
        changes.append(
            ChangeRecord(
                change_id=f"V{counter:03d}",
                page=page_idx,
                section="Sketch",
                old="Sketch changed",
                new="Sketch changed",
                type="Visual",
                priority="HIGH",
                confidence=1 - score,
                bbox=None,
                evidence_old=None,
                evidence_new=None,
            )
        )
    return changes


def render_page_image(doc: fitz.Document, page_index: int, zoom: float) -> Image.Image:
    """Render a page to a PIL image."""

    page = doc[page_index]
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


def generate_thumbnails(change: ChangeRecord, old_pdf: Path, new_pdf: Path, out_dir: Path, config: Dict) -> ChangeRecord:
    """Create before/after thumbnails for a change."""

    evidence_dir = out_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    zoom = config.get("default_zoom", 2.0)

    doc_old = fitz.open(old_pdf)
    doc_new = fitz.open(new_pdf)

    def crop_region(doc: fitz.Document, pdf_path: Path, page_index: int,
                    bbox: Optional[Tuple[float, float, float, float]]) -> Optional[Path]:
        if page_index >= doc.page_count:
            return None
        page = doc[page_index]
        rect = fitz.Rect(*bbox) if bbox else page.rect
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, clip=rect)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        max_width = config.get("evidence_max_width", 1200)
        if image.width > max_width:
            ratio = max_width / float(image.width)
            image = image.resize((max_width, int(image.height * ratio)))
        file_name = evidence_dir / f"{change.change_id}_{pdf_path.stem}.png"
        image.save(file_name)
        return file_name

    old_path = crop_region(doc_old, old_pdf, change.page, change.bbox)
    new_path = crop_region(doc_new, new_pdf, change.page, change.bbox)
    change.evidence_old = str(old_path) if old_path else None
    change.evidence_new = str(new_path) if new_path else None
    doc_old.close()
    doc_new.close()
    return change


def annotate_pdf(changes: Sequence[ChangeRecord], new_pdf: Path, output_pdf: Path, config: Dict) -> None:
    """Draw rectangles & callout numbers for priority changes."""

    allowed = set(config.get("show_priority_levels_in_pdf", []))
    doc = fitz.open(new_pdf)
    callout_counter = 1
    for change in changes:
        if change.priority not in allowed:
            continue
        if change.page >= doc.page_count:
            continue
        page = doc[change.page]
        rect = fitz.Rect(*change.bbox) if change.bbox else page.rect
        color = (1, 0, 0) if change.priority == "HIGH" else (1, 0.5, 0)
        page.draw_rect(rect, color=color, width=1)
        page.insert_text(
            rect.tl,
            f"{callout_counter}. {change.priority}",
            fontsize=10,
            color=color,
        )
        callout_counter += 1
    doc.save(output_pdf)


def map_review_rows_to_changes(changes: Sequence[ChangeRecord], review_excel: Path, config: Dict) -> List[Dict]:
    """Map review rows to detected change IDs with fuzzy logic."""

    try:
        df = pd.read_excel(review_excel)
    except Exception as exc:
        LOGGER.warning("Failed to read review workbook %s: %s", review_excel, exc)
        return []
    mapped_rows: List[Dict] = []
    for idx, row in df.iterrows():
        row_text = " ".join(map(str, row.values))
        part_ids = extract_part_ids(row_text)
        candidate_changes = []
        for change in changes:
            base_text = f"{change.old} {change.new}"
            if part_ids and any(pid in base_text for pid in part_ids):
                candidate_changes.append(change)
                continue
            similarity = text_similarity(row_text, base_text)
            if similarity >= 0.6:
                candidate_changes.append(change)
        status = derive_review_status(row_text, candidate_changes)
        mapped_rows.append(
            {
                "row_index": int(idx),
                "row_preview": row_text[:200],
                "matched_changes": [c.change_id for c in candidate_changes],
                "status": status,
            }
        )
    return mapped_rows


def derive_review_status(row_text: str, candidate_changes: Sequence[ChangeRecord]) -> str:
    """Infer review mapping status based on matches."""

    if not candidate_changes:
        return "NOT_IMPLEMENTED"
    priorities = {change.priority for change in candidate_changes}
    if "HIGH" in priorities and len(candidate_changes) == 1:
        return "IMPLEMENTED"
    if "HIGH" in priorities and len(candidate_changes) > 1:
        return "PARTIAL"
    if "MED" in priorities or "LOW" in priorities:
        return "PARTIAL"
    return "DIVERGENT"


def generate_change_log(changes: Sequence[ChangeRecord], csv_path: Path) -> None:
    """Persist all change records to CSV."""

    fieldnames = [
        "change_id",
        "page",
        "section",
        "type",
        "priority",
        "confidence",
        "old",
        "new",
        "bbox",
        "evidence_old",
        "evidence_new",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for change in changes:
            writer.writerow({
                "change_id": change.change_id,
                "page": change.page,
                "section": change.section,
                "type": change.type,
                "priority": change.priority,
                "confidence": f"{change.confidence:.2f}",
                "old": change.old,
                "new": change.new,
                "bbox": json.dumps(change.bbox) if change.bbox else "",
                "evidence_old": change.evidence_old or "",
                "evidence_new": change.evidence_new or "",
            })


def save_review_mapping(mapping: Sequence[Dict], output_path: Path) -> None:
    """Write review mapping JSON."""

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(mapping, handle, indent=2)


def run_comparison(old_pdf: Path, new_pdf: Path, review_excel: Path, output_dir: Path,
                   config_path: Optional[Path] = None) -> Dict[str, str]:
    """Execute the end-to-end diff pipeline."""

    config = read_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    old_pages = extract_page_blocks(old_pdf, config)
    new_pages = extract_page_blocks(new_pdf, config)

    table_changes = diff_tables(extract_tables(old_pdf), extract_tables(new_pdf), config)
    text_changes = diff_page_text(old_pages, new_pages, config)
    visual_changes = detect_visual_changes(old_pdf, new_pdf, config)

    all_changes = table_changes + text_changes + visual_changes
    for idx, change in enumerate(all_changes, start=1):
        change.change_id = f"C{idx:03d}"

    evidence_changes = [generate_thumbnails(change, old_pdf, new_pdf, output_dir, config) for change in all_changes]

    annotated_pdf = output_dir / "annotated_TechPack2.pdf"
    annotate_pdf(evidence_changes, new_pdf, annotated_pdf, config)

    change_log = output_dir / "change_log.csv"
    generate_change_log(evidence_changes, change_log)

    review_mapping = map_review_rows_to_changes(evidence_changes, review_excel, config)
    review_mapping_path = output_dir / "review_mapping.json"
    save_review_mapping(review_mapping, review_mapping_path)

    return {
        "annotated_pdf": str(annotated_pdf),
        "change_log": str(change_log),
        "review_mapping": str(review_mapping_path),
        "evidence_dir": str(output_dir / "evidence"),
    }
