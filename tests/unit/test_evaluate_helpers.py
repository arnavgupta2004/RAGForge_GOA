import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

from evaluate import _recall_and_mrr  # noqa: E402


def test_recall_and_mrr_hit_at_rank_one():
    hit, rr = _recall_and_mrr(["a", "b", "c"], {"a"})
    assert hit is True
    assert rr == 1.0


def test_recall_and_mrr_hit_at_rank_three():
    hit, rr = _recall_and_mrr(["x", "y", "a"], {"a"})
    assert hit is True
    assert rr == pytest.approx(1 / 3)


def test_recall_and_mrr_no_hit():
    hit, rr = _recall_and_mrr(["x", "y", "z"], {"a"})
    assert hit is False
    assert rr == 0.0
