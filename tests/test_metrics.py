import pytest
from src.eval.metrics import compute_recall_at_k, compute_mrr, compute_ndcg_at_k

def test_recall_at_k_perfect():
    retrieved = ["c1", "c2", "c3"]
    expected = ["c1", "c2"]
    assert compute_recall_at_k(retrieved, expected, k=2) == 1.0

def test_recall_at_k_partial():
    retrieved = ["c1", "c3", "c4"]
    expected = ["c1", "c2"]
    assert compute_recall_at_k(retrieved, expected, k=2) == 0.5

def test_recall_at_k_miss():
    retrieved = ["c3", "c4", "c5"]
    expected = ["c1", "c2"]
    assert compute_recall_at_k(retrieved, expected, k=2) == 0.0

def test_recall_empty_inputs():
    assert compute_recall_at_k([], ["c1"], k=5) == 0.0
    assert compute_recall_at_k(["c1"], [], k=5) == 0.0
    assert compute_recall_at_k(["c1"], ["c1"], k=0) == 0.0

def test_mrr_ranks():
    expected = ["target"]
    assert compute_mrr(["target", "x", "y"], expected) == 1.0
    assert compute_mrr(["x", "target", "y"], expected) == 0.5
    assert compute_mrr(["x", "y", "target"], expected) == pytest.approx(0.3333, rel=1e-3)
    assert compute_mrr(["x", "y", "z"], expected) == 0.0

def test_ndcg_at_k():
    expected = ["c1", "c2"]
    perfect = ["c1", "c2", "x"]
    assert compute_ndcg_at_k(perfect, expected, k=2) == 1.0
    assert compute_ndcg_at_k(["x", "y", "z"], expected, k=3) == 0.0