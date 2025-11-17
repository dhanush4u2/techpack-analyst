"""Key-based BOM and measurement table diffing."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from .utils import detect_part_column, extract_candidate_keys, normalize_text

LOGGER = logging.getLogger(__name__)
_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class TableChange:
    """Structured representation of a table change."""

    id: str
    type: str
    page_old: int | None
    page_new: int | None
    old_value: str
    new_value: str
    bbox_old: Tuple[float, float, float, float] | None
    bbox_new: Tuple[float, float, float, float] | None
    confidence: float


@dataclass
class TableRecord:
    key: str
    columns: Dict[str, str]


def _prepare_rows(df: pd.DataFrame) -> List[TableRecord]:
    """Convert a Camelot-style DataFrame into keyed records."""

    if df.empty:
        return []
    headers = df.iloc[0].astype(str).tolist()
    data_rows = df.iloc[1:].reset_index(drop=True)
    part_col_idx = detect_part_column(headers)
    records: List[TableRecord] = []
    for _, row in data_rows.iterrows():
        row_values = row.astype(str).tolist()
        if part_col_idx is not None:
            candidate = row_values[part_col_idx]
            keys = [normalize_text(candidate)] if candidate else []
        else:
            keys = extract_candidate_keys(row_values)
        key = keys[0] if keys else normalize_text(" ".join(row_values))
        columns = {normalize_text(headers[i]): row_values[i] for i in range(len(headers))}
        records.append(TableRecord(key=key, columns=columns))
    return records


def _records_by_key(df: pd.DataFrame) -> Dict[str, TableRecord]:
    return {record.key: record for record in _prepare_rows(df)}


def _numeric_delta(old: str, new: str) -> float | None:
    left = _NUMERIC_RE.search(old or "")
    right = _NUMERIC_RE.search(new or "")
    if not left or not right:
        return None
    try:
        return float(right.group()) - float(left.group())
    except ValueError:
        return None


def diff_table_pair(old_df: pd.DataFrame, new_df: pd.DataFrame,
                    page_old: int, page_new: int, id_prefix: str) -> List[TableChange]:
    """Diff a pair of tables ignoring column ordering."""

    changes: List[TableChange] = []
    old_records = _records_by_key(old_df)
    new_records = _records_by_key(new_df)

    counter = 0
    for key, record in old_records.items():
        if key not in new_records:
            counter += 1
            changes.append(
                TableChange(
                    id=f"{id_prefix}_removed_{counter}",
                    type="table_change",
                    page_old=page_old,
                    page_new=None,
                    old_value=str(record.columns),
                    new_value="",
                    bbox_old=None,
                    bbox_new=None,
                    confidence=0.9,
                )
            )

    for key, record in new_records.items():
        if key not in old_records:
            counter += 1
            changes.append(
                TableChange(
                    id=f"{id_prefix}_added_{counter}",
                    type="table_change",
                    page_old=None,
                    page_new=page_new,
                    old_value="",
                    new_value=str(record.columns),
                    bbox_old=None,
                    bbox_new=None,
                    confidence=0.85,
                )
            )

    for key in set(old_records).intersection(new_records):
        old_cols = old_records[key].columns
        new_cols = new_records[key].columns
        for column, old_val in old_cols.items():
            if column not in new_cols:
                continue
            new_val = new_cols[column]
            if normalize_text(old_val) == normalize_text(new_val):
                continue
            delta = _numeric_delta(old_val, new_val)
            confidence = 0.95 if delta else 0.8
            counter += 1
            changes.append(
                TableChange(
                    id=f"{id_prefix}_modified_{counter}",
                    type="table_change",
                    page_old=page_old,
                    page_new=page_new,
                    old_value=f"{column}: {old_val}",
                    new_value=f"{column}: {new_val}",
                    bbox_old=None,
                    bbox_new=None,
                    confidence=confidence,
                )
            )
    return changes


def diff_tables_for_pages(old_tables: List[pd.DataFrame], new_tables: List[pd.DataFrame],
                          page_old: int, page_new: int, id_prefix: str) -> List[TableChange]:
    """Diff lists of tables for matched pages by pairing on index and fallback to best effort."""

    changes: List[TableChange] = []
    pair_count = max(len(old_tables), len(new_tables))
    for idx in range(pair_count):
        old_df = old_tables[idx] if idx < len(old_tables) else pd.DataFrame()
        new_df = new_tables[idx] if idx < len(new_tables) else pd.DataFrame()
        if old_df.empty and new_df.empty:
            continue
        table_changes = diff_table_pair(old_df, new_df, page_old, page_new, f"{id_prefix}_{idx}")
        changes.extend(table_changes)
    return changes
