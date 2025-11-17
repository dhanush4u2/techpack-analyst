"""Tests for key-based table diffing ignoring column order."""
import pandas as pd

from service.table_diff import diff_table_pair


def test_table_diff_ignores_column_reorder():
    data = [["Part", "Size", "Qty"], ["P-01", "10", "5"]]
    df_old = pd.DataFrame(data)
    df_new = pd.DataFrame([["Qty", "Part", "Size"], ["5", "P-01", "10"]])
    changes = diff_table_pair(df_old, df_new, page_old=0, page_new=0, id_prefix="unit")
    assert changes == []
