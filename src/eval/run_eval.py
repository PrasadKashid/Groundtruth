import os
import json
import numpy as np
from typing import List, Dict
from src.retrieval.search_engine import RetrievalEngine
from src.eval.metrics import compute_recall_at_k, compute_mrr, compute_ndcg_at_k

CHUNKS_PATH = os.path.join("data", "processed", "chunks.jsonl")
BENCHMARK_PATH = os.path.join("data", "benchmark", "ground_truth.jsonl")
RESULTS_DIR = os.path.join("data", "results")


def load_benchmark(filepath: str) -> List[Dict]:
    benchmark = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                benchmark.append(json.loads(line))
    return benchmark


def evaluate_pipeline(
    engine: RetrievalEngine, benchmark: List[Dict], pipeline_type: str, top_k: int = 10
) -> Dict[str, float]:
    """Runs all benchmark queries through a specific pipeline and calculates aggregate metrics."""
    recalls_at_1 = []
    recalls_at_3 = []
    recalls_at_5 = []
    mrrs = []
    ndcgs_at_3 = []
    print(
        f"\nEvaluating [{pipeline_type.upper()}] across {len(benchmark)} test queries..."
    )

    for item in benchmark:
        query = item["query"]
        expected_ids = item["expected_chunk_ids"]

        if pipeline_type == "dense":
            retrieved = engine.search_dense(query, top_k=top_k)
        elif pipeline_type == "hybrid":
            retrieved = engine.search_hybrid(query, top_k=top_k)
        elif pipeline_type == "reranked":
            retrieved = engine.search_reranked(query, top_k=top_k)
        else:
            raise ValueError(f"Unknown pipeline: {pipeline_type}")

        # compute ranking metrics
        recalls_at_1.append(
            compute_recall_at_k(retrieved_ids=retrieved, expected_ids=expected_ids, k=1)
        )
        recalls_at_3.append(
            compute_recall_at_k(retrieved_ids=retrieved, expected_ids=expected_ids, k=3)
        )
        recalls_at_5.append(
            compute_recall_at_k(retrieved_ids=retrieved, expected_ids=expected_ids, k=5)
        )

        mrrs.append(compute_mrr(retrieved, expected_ids))
        ndcgs_at_3.append(compute_ndcg_at_k(retrieved, expected_ids, k=3))

    return {
        "Recall@1": float(np.mean(recalls_at_1)),
        "Recall@3": float(np.mean(recalls_at_3)),
        "Recall@5": float(np.mean(recalls_at_5)),
        "MRR": float(np.mean(mrrs)),
        "nDCG@3": float(np.mean(ndcgs_at_3)),
    }


def print_markdown_comparison(results: Dict[str, Dict[str, float]]):
    """Prints a clean ASCII comparison table."""
    headers = ["Pipeline", "Recall@1", "Recall@3", "Recall@5", "MRR", "nDCG@3"]
    row_format = "{:<16} | {:<10} | {:<10} | {:<10} | {:<8} | {:<8}"

    print("\n" + "=" * 75)
    print(" " * 22 + "RETRIEVAL BENCHMARK RESULTS")
    print("=" * 75)
    print(row_format.format(*headers))
    print("-" * 75)

    for pipeline_name, metrics in results.items():
        print(
            row_format.format(
                pipeline_name,
                f"{metrics['Recall@1']:.4f}",
                f"{metrics['Recall@3']:.4f}",
                f"{metrics['Recall@5']:.4f}",
                f"{metrics['MRR']:.4f}",
                f"{metrics['nDCG@3']:.4f}",
            )
        )
    print("=" * 75 + "\n")


if __name__ == "__main__":
    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(f"Missing corpus chunks at {CHUNKS_PATH}")
    if not os.path.exists(BENCHMARK_PATH):
        raise FileNotFoundError(f"Missing benchmark file at {BENCHMARK_PATH}")

    # 1. Initialize search engine once
    engine = RetrievalEngine(CHUNKS_PATH)

    # 2. Load ground truth questions
    benchmark_dataset = load_benchmark(BENCHMARK_PATH)

    # 3. Benchmark all three architectures
    pipelines = ["dense", "hybrid", "reranked"]
    all_results = {}

    for pipe in pipelines:
        all_results[pipe] = evaluate_pipeline(engine, benchmark_dataset, pipe)

    # 4. Print comparative report
    print_markdown_comparison(all_results)

    # 5. Persist results for regression gating / CI
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_file = os.path.join(RESULTS_DIR, "evaluation_summary.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Metrics saved to {out_file}")
