"""Ensure fallback block search surfaces matches above threshold."""
from types import SimpleNamespace

from service import diff_engine


class FakePage:
    def __init__(self, blocks):
        self._blocks = blocks
        self.number = 0

    def get_text(self, mode):  # pragma: no cover - signature compatibility only
        assert mode == "blocks"
        return self._blocks


def test_global_block_search_returns_matches():
    old_blocks = [
        {"page": 2, "text": "zipper pocket lined", "norm": "zipper pocket lined", "bbox": (0, 0, 10, 10)},
    ]
    new_page = FakePage([(0, 0, 10, 10, "Zipper pocket lined", 0, 0, 0, 0)])
    changes, counter = diff_engine.global_block_search_and_compare(
        old_blocks,
        new_page,
        change_prefix="fallback",
        threshold=0.5,
        counter_start=0,
    )
    assert counter == 1
    assert changes[0]["page_old"] == 2
    assert changes[0]["page_new"] == 0
