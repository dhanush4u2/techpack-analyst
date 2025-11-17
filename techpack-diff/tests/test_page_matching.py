"""Unit tests for semantic page matching."""
from service.page_matching import match_pages
from service.utils import PageFingerprint


def test_page_mapping_handles_reorder():
    old_fps = [
        PageFingerprint(page_num=0, text_summary="front panel", table_signature="sigA", thumbnail_hash="1111"),
        PageFingerprint(page_num=1, text_summary="back panel", table_signature="sigB", thumbnail_hash="2222"),
    ]
    new_fps = [
        PageFingerprint(page_num=0, text_summary="back panel", table_signature="sigB", thumbnail_hash="2222"),
        PageFingerprint(page_num=1, text_summary="front panel", table_signature="sigA", thumbnail_hash="1111"),
    ]
    mapping = match_pages(old_fps, new_fps, {"page_combined_score_threshold": 0.2})
    assert mapping[0][0][0] == 1  # new page 0 matches old page 1
    assert mapping[1][0][0] == 0
