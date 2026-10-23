import math

import pytest

from rag.evaluate import retrieval_metrics


def test_perfect_ranking():
    assert retrieval_metrics(["a", "b"], {"a", "b"}, 2) == {
        "recall@2": 1.0,
        "hit@2": 1.0,
        "ndcg@2": 1.0,
    }


def test_partial_ranking_discount_and_cutoff():
    metrics = retrieval_metrics(["x", "a", "b"], {"a", "b"}, 2)
    assert metrics["recall@2"] == 0.5
    assert metrics["hit@2"] == 1.0
    assert metrics["ndcg@2"] == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))


def test_no_hits():
    assert all(value == 0 for value in retrieval_metrics(["x"], {"a"}, 8).values())


@pytest.mark.parametrize(("relevant", "k"), [(set(), 8), ({"a"}, 0)])
def test_rejects_undefined_metrics(relevant, k):
    with pytest.raises(ValueError):
        retrieval_metrics([], relevant, k)
