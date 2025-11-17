"""Shared helpers for the TechPack diff engine."""
from __future__ import annotations

import hashlib
import logging
import math
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from PIL import Image
from rapidfuzz import fuzz

LOGGER = logging.getLogger(__name__)

_PUNCT_TABLE = str.maketrans({ch: " " for ch in string.punctuation})
_WS_RE = re.compile(r"\s+", re.MULTILINE)
_PART_ID_RE = re.compile(r"\b(?:part|item|style|component)[-_ ]?(\w{2,})\b", re.IGNORECASE)


@dataclass
class PageFingerprint:
    """Container describing the multimodal fingerprint of a single page."""

    page_num: int
    text_summary: str
    table_signature: str
    thumbnail_hash: str


def read_config(config_path: Optional[Path] = None) -> Dict:
    """Load YAML configuration from disk."""

    if config_path is None:
        config_path = Path(__file__).with_name("config.yaml")
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_text(text: str, limit: Optional[int] = None) -> str:
    """Lowercase, strip punctuation, and collapse whitespace for comparison."""

    if not text:
        return ""
    lowered = text.lower().translate(_PUNCT_TABLE)
    flattened = _WS_RE.sub(" ", lowered).strip()
    if limit:
        return flattened[:limit]
    return flattened


def text_similarity(left: str, right: str) -> float:
    """Token-set ratio normalized to 0-1."""

    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if not left_norm and not right_norm:
        return 1.0
    if not left_norm or not right_norm:
        return 0.0
    return fuzz.token_set_ratio(left_norm, right_norm) / 100.0


def perceptual_hash(image: Image.Image, hash_size: int = 16) -> str:
    """Compute a simple perceptual hash by averaging grayscale values."""

    gray = image.convert("L").resize((hash_size, hash_size))
    pixels = np.array(gray)
    avg = pixels.mean()
    bits = pixels > avg
    return "".join("1" if bit else "0" for bit in bits.flatten())


def hash_tables(table_text: Sequence[str]) -> str:
    """Create a deterministic signature from table row strings."""

    digest = hashlib.sha256()
    for row in table_text:
        digest.update(normalize_text(row).encode("utf-8"))
    return digest.hexdigest()


def hamming_similarity(hash_a: str, hash_b: str) -> float:
    """Return similarity between equal-length bit strings."""

    if not hash_a or not hash_b or len(hash_a) != len(hash_b):
        return 0.0
    matches = sum(ch_a == ch_b for ch_a, ch_b in zip(hash_a, hash_b))
    return matches / len(hash_a)


def chunk_text(text: str, window: int = 300) -> List[str]:
    """Split text into manageable overlapping windows for search."""

    tokens = normalize_text(text).split()
    if not tokens:
        return []
    chunks = []
    for start in range(0, len(tokens), window // 2):
        chunk = tokens[start:start + window]
        if chunk:
            chunks.append(" ".join(chunk))
    return chunks


def detect_part_column(headers: Sequence[str]) -> Optional[int]:
    """Return column index likely holding part identifiers."""

    for idx, header in enumerate(headers):
        if _PART_ID_RE.search(header or ""):
            return idx
    return None


def extract_candidate_keys(row: Sequence[str]) -> List[str]:
    """Extract deterministic row keys from part columns or fallback fields."""

    keys = []
    joined = " ".join(str(value) for value in row)
    matches = _PART_ID_RE.findall(joined)
    for match in matches:
        keys.append(normalize_text(match))
    if not keys:
        keys.append(hashlib.sha1(joined.encode("utf-8")).hexdigest())
    return keys


def ensure_directory(path: Path) -> Path:
    """Create a directory path if it does not yet exist."""

    path.mkdir(parents=True, exist_ok=True)
    return path
