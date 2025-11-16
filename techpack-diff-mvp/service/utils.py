"""Utility helpers for the techpack diff MVP.

These helpers intentionally favor readability over micro-optimizations so the
collision of PDF, table and Excel parsing logic remains approachable.
"""
from __future__ import annotations

import logging
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml
from dateutil import parser as date_parser
from rapidfuzz import fuzz

LOGGER = logging.getLogger(__name__)

# Precompile regex patterns once to keep call sites lean.
_WS_RE = re.compile(r"\s+", re.MULTILINE)
_PUNCT_TABLE = str.maketrans({ch: "" for ch in string.punctuation})
_PART_ID_RE = re.compile(r"\b(?:PART|PRT|COMP|ITEM|PC)\s*[-:_]?\s*\d{2,5}\b", re.IGNORECASE)
_MM_RE = re.compile(r"(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>mm|millimeter(?:s)?)", re.IGNORECASE)


@dataclass
class TextBlock:
    """Lightweight representation of a PyMuPDF block.

    PyMuPDF returns tuples, but a typed structure clarifies intent and improves
    testability.
    """

    text: str
    bbox: Tuple[float, float, float, float]


def read_config(config_path: Optional[str] = None) -> Dict:
    """Load YAML configuration, falling back to service/config.yaml."""

    if config_path is None:
        config_path = Path(__file__).with_name("config.yaml")
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found at {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_text_for_compare(text: str) -> str:
    """Aggressively normalize strings for similarity checks."""

    if not text:
        return ""
    lowered = text.lower()
    no_punct = lowered.translate(_PUNCT_TABLE)
    compact_ws = _WS_RE.sub(" ", no_punct)
    return compact_ws.strip()


def text_similarity(left: str, right: str) -> float:
    """Return a fuzzy token-set ratio in [0, 1]."""

    norm_left = normalize_text_for_compare(left)
    norm_right = normalize_text_for_compare(right)
    if not norm_left and not norm_right:
        return 1.0
    if not norm_left or not norm_right:
        return 0.0
    # RapidFuzz returns 0-100; normalize into 0-1.
    return fuzz.token_set_ratio(norm_left, norm_right) / 100.0


def detect_repeated_blocks(blocks_with_page: Sequence[Tuple[int, TextBlock]], threshold: float) -> List[str]:
    """Return normalized texts considered repeated headers/footers.

    The function groups block texts by page occurrence and compares the ratio of
    pages a block appears on relative to the total number of pages processed.
    Returning the normalized text keeps downstream removal lightweight.
    """

    if not blocks_with_page:
        return []

    page_count = {}
    total_pages = len({page for page, _ in blocks_with_page})
    for page_index, block in blocks_with_page:
        normalized = normalize_text_for_compare(block.text)
        if not normalized:
            continue
        page_count.setdefault(normalized, set()).add(page_index)

    repeated = [text for text, pages in page_count.items() if len(pages) / total_pages >= threshold]
    return repeated


def is_significant_text_change(old: str, new: str, threshold: float, keyword_weight: float = 0.1,
                               domain_keywords: Optional[Sequence[str]] = None) -> bool:
    """Determine if a text mutation matters enough to log."""

    similarity = text_similarity(old, new)
    if similarity >= threshold:
        return False
    if domain_keywords:
        tokens = set(tokenize_for_diff(old) + tokenize_for_diff(new))
        hits = sum(1 for kw in domain_keywords if kw.lower() in tokens)
        if hits:
            similarity -= min(keyword_weight * hits, 0.2)  # bias toward significance
    return similarity < threshold


def extract_part_ids(text: str) -> List[str]:
    """Return part identifiers commonly used in BOM tables."""

    if not text:
        return []
    return [match.group(0).upper().replace(" ", "") for match in _PART_ID_RE.finditer(text)]


def tokenize_for_diff(text: str) -> List[str]:
    """Tokenize text in a whitespace + punctuation agnostic manner."""

    normalized = normalize_text_for_compare(text)
    return normalized.split()


def priority_from_change(change_type: str, severity_score: float, contains_keyword: bool,
                         config: Dict) -> str:
    """Map heuristics onto the HIGH/MED/LOW priority scale."""

    high_threshold = config.get("measurement_high_tolerance_mm", 2.0)
    med_threshold = config.get("measurement_med_tolerance_mm", 1.0)

    if contains_keyword or severity_score >= high_threshold:
        return "HIGH"
    if severity_score >= med_threshold:
        return "MED"
    return "LOW"


def normalize_date(value: str) -> Optional[str]:
    """Normalize date strings to ISO 8601; return None on failure."""

    if not value:
        return None
    try:
        parsed = date_parser.parse(value, dayfirst=False, fuzzy=True)
    except (ValueError, TypeError):
        return None
    return parsed.date().isoformat()


def parse_mm(value: str) -> Optional[float]:
    """Extract a numeric millimeter measurement if present."""

    if value is None:
        return None
    match = _MM_RE.search(value)
    if not match:
        return None
    return float(match.group("value"))


def detect_measurement_delta(old: str, new: str) -> Optional[float]:
    """Return absolute difference between measurement values, if both parse."""

    old_mm = parse_mm(old)
    new_mm = parse_mm(new)
    if old_mm is None or new_mm is None:
        return None
    return abs(new_mm - old_mm)
