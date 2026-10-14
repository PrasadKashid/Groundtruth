import math
from typing import List


def compute_recall_at_k(
    retrieved_ids: List[str], expected_ids: List[str], k: int
) -> float:
    """Computes Recall@k.

    Measures what fraction of the ground-truth relevant chunks were
    successfully captured within the top-k retrieved results.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        expected_ids: Ground-truth chunk IDs that contain the answer.
        k: Rank cutoff threshold.

    Returns:
        float: Score between 0.0 and 1.0.
    """
    if not expected_ids or k <= 0:
        return 0.0

    top_k_retrieved = set(retrieved_ids[:k])
    matched_count = sum(1 for cid in expected_ids if cid in top_k_retrieved)
    return matched_count / len(expected_ids)


def compute_mrr(retrieved_ids: List[str], expected_ids: List[str]) -> float:
    """Computes Reciprocal Rank (RR) for a single query.

    Evaluates the rank position of the FIRST relevant chunk found in the
    retrieved list. When averaged across all queries, this yields Mean
    Reciprocal Rank (MRR).

    Formula:
        RR = 1 / rank (where rank is 1-indexed)
        RR = 0.0 if no relevant chunk appears in retrieved_ids

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        expected_ids: Ground-truth chunk IDs.

    Returns:
        float: Score between 0.0 and 1.0 (e.g., 1.0 for rank 1, 0.5 for rank 2, 0.333 for rank 3).
    """
    if not expected_ids or not retrieved_ids:
        return 0.0

    expected_set = set(expected_ids)
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in expected_set:
            return 1.0 / rank

    return 0.0


def compute_ndcg_at_k(
    retrieved_ids: List[str], expected_ids: List[str], k: int
) -> float:
    """Computes Normalized Discounted Cumulative Gain at rank cutoff k (nDCG@k).

    Accounts for position decay using a logarithmic discount, rewarding
    retrieval pipelines that place relevant items at higher ranks.

    Formulas:
        DCG@k = sum_{i=1}^k [ rel_i / log2(i + 1) ]
        IDCG@k = sum_{i=1}^{min(k, |expected_ids|)} [ 1 / log2(i + 1) ]
        nDCG@k = DCG@k / IDCG@k

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        expected_ids: Ground-truth chunk IDs.
        k: Rank cutoff threshold.

    Returns:
        float: Normalized score between 0.0 and 1.0.
    """
    if not expected_ids or not retrieved_ids or k <= 0:
        return 0.0

    expected_set = set(expected_ids)
    top_k = retrieved_ids[:k]

    # Calculate Discounted Cumulative Gain (DCG@k)
    dcg = 0.0
    for i, chunk_id in enumerate(top_k, start=1):
        if chunk_id in expected_set:
            dcg += 1.0 / math.log2(i + 1)

    # Calculate Ideal Discounted Cumulative Gain (IDCG@k)
    ideal_hits = min(k, len(expected_ids))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


if __name__ == "__main__":
    # Unit smoke test to verify all formulas
    ground_truth = ["chunk_B", "chunk_C"]

    # Perfect ranking: both ground truth items at rank 1 and 2
    perfect_retrieval = ["chunk_B", "chunk_C", "chunk_X", "chunk_Y"]
    print("--- Perfect Ranking ---")
    print(
        f"Recall@2: {compute_recall_at_k(perfect_retrieval, ground_truth, k=2):.4f}"
    )  # 1.0
    print(f"MRR:      {compute_mrr(perfect_retrieval, ground_truth):.4f}")  # 1.0
    print(
        f"nDCG@2:   {compute_ndcg_at_k(perfect_retrieval, ground_truth, k=2):.4f}"
    )  # 1.0

    # Suboptimal ranking: first hit buried at rank 3
    delayed_retrieval = ["chunk_X", "chunk_Y", "chunk_B", "chunk_Z"]
    print("\n--- Suboptimal Ranking (Hit at Rank 3) ---")
    print(
        f"Recall@2: {compute_recall_at_k(delayed_retrieval, ground_truth, k=2):.4f}"
    )  # 0.0
    print(
        f"Recall@3: {compute_recall_at_k(delayed_retrieval, ground_truth, k=3):.4f}"
    )  # 0.5
    print(
        f"MRR:      {compute_mrr(delayed_retrieval, ground_truth):.4f}"
    )  # 1/3 = 0.3333
    print(
        f"nDCG@3:   {compute_ndcg_at_k(delayed_retrieval, ground_truth, k=3):.4f}"
    )  # 0.5
