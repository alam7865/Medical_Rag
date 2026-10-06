import math

import pytest

from src.evaluation import metrics as M
from src.utils.helpers import safe_improvement_pct


def test_recall_at_k():
    assert M.recall_at_k([0, 0, 1], 1) == 0.0
    assert M.recall_at_k([0, 0, 1], 3) == 1.0
    assert M.recall_at_k([], 5) == 0.0


def test_precision_at_k():
    assert M.precision_at_k([1, 0, 1, 0, 0], 5) == 0.4
    assert M.precision_at_k([1], 3) == pytest.approx(1 / 3)  # short list: missing = non-relevant


def test_mrr():
    assert M.reciprocal_rank([0, 1, 1]) == 0.5
    assert M.reciprocal_rank([0, 0, 0]) == 0.0
    assert M.first_relevant_rank([0, 0, 1]) == 3
    assert M.first_relevant_rank([0]) is None


def test_ndcg():
    assert M.ndcg_at_k([1, 0, 0], 3, n_relevant=1) == 1.0
    assert M.ndcg_at_k([0, 1, 0], 3, n_relevant=1) == pytest.approx(1 / math.log2(3))
    # two relevant in corpus, only one retrieved at rank 1
    assert M.ndcg_at_k([1, 0, 0], 3, n_relevant=2) == pytest.approx(1 / (1 + 1 / math.log2(3)))
    assert M.ndcg_at_k([0, 0], 2, n_relevant=0) == 0.0


def test_invalid_k():
    with pytest.raises(ValueError):
        M.recall_at_k([1], 0)


def test_improvement_pct_safe():
    assert safe_improvement_pct(0.5, 0.75) == pytest.approx(50.0)
    assert safe_improvement_pct(0.0, 0.3) is None
