import os
import json
import pytest

SUMMARY_PATH = os.path.join("data", "results", "evaluation_summary.json")


@pytest.fixture(scope="session")
def evaluation_results():
    if not os.path.exists(SUMMARY_PATH):
        pytest.fail(
            f"Run 'python -m src.eval.run_eval' first to generate {SUMMARY_PATH}"
        )
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_hybrid_beats_dense_mrr(evaluation_results):
    dense_mrr = evaluation_results["dense"]["MRR"]
    hybrid_mrr = evaluation_results["hybrid"]["MRR"]
    assert (
        hybrid_mrr > dense_mrr
    ), f"Expected Hybrid MRR ({hybrid_mrr}) to beat Dense MRR ({dense_mrr})"


def test_reranked_meets_quality_baseline(evaluation_results):
    reranked = evaluation_results["reranked"]
    assert (
        reranked["Recall@5"] >= 0.40
    ), f"Reranked Recall@5 below threshold: {reranked['Recall@5']}"
    assert reranked["MRR"] >= 0.35, f"Reranked MRR below threshold: {reranked['MRR']}"
    assert (
        reranked["nDCG@3"] >= 0.30
    ), f"Reranked nDCG@3 below threshold: {reranked['nDCG@3']}"
