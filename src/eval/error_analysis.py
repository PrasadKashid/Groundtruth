import os
import json
from src.retrieval.search_engine import RetrievalEngine

CHUNKS_PATH = os.path.join("data", "processed", "chunks.jsonl")
BENCHMARK_PATH = os.path.join("data", "benchmark", "ground_truth.jsonl")
if not os.path.exists(BENCHMARK_PATH):
    BENCHMARK_PATH = os.path.join("data", "processed", "ground_truth.jsonl")


def run_failure_diagnostics():
    engine = RetrievalEngine(CHUNKS_PATH)

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    print(f"\n{'='*80}")
    print("                    RETRIEVAL FAILURE CASE ANALYSIS (k=5)")
    print(f"{'='*80}\n")

    misses = []

    for item in queries:
        query = item["query"]
        expected_ids = item["expected_chunk_ids"]
        category = item.get("category", "unknown")

        dense_top5 = engine.search_dense(query, top_k=5)
        hybrid_top5 = engine.search_hybrid(query, top_k=5)
        rerank_top5 = engine.search_reranked(query, top_k=5)

        # Check if any pipeline retrieved at least one expected chunk
        dense_hit = any(cid in dense_top5 for cid in expected_ids)
        hybrid_hit = any(cid in hybrid_top5 for cid in expected_ids)
        rerank_hit = any(cid in rerank_top5 for cid in expected_ids)

        if not (dense_hit or hybrid_hit or rerank_hit):
            misses.append(
                {
                    "query": query,
                    "category": category,
                    "expected_ids": expected_ids,
                    "dense_retrieved": dense_top5,
                    "rerank_retrieved": rerank_top5,
                }
            )

    print(f"Total Queries: {len(queries)}")
    print(
        f"Total Complete Misses (Across All Pipelines): {len(misses)} / {len(queries)}\n"
    )

    for i, miss in enumerate(misses, start=1):
        print(f"[{i}] Query: {miss['query']}")
        print(f"    Category: {miss['category']}")
        print(f"    Expected: {miss['expected_ids']}")

        # Show expected chunk text snippet
        exp_id = miss["expected_ids"][0]
        exp_text = engine.chunk_map.get(exp_id, "NOT FOUND IN CORPUS")
        print(f"    Target Text Snippet: {exp_text[:140]}...")

        # Show what reranked pipeline actually brought back
        top_retrieved_id = miss["rerank_retrieved"][0]
        top_retrieved_text = engine.chunk_map.get(top_retrieved_id, "")
        print(
            f"    Rank 1 Retrieved:    {top_retrieved_id} -> {top_retrieved_text[:140]}..."
        )
        print("-" * 80)


if __name__ == "__main__":
    run_failure_diagnostics()
