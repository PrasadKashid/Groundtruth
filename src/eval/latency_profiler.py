import os
import time
import json
import numpy as np
from typing import List, Dict
from src.retrieval.search_engine import RetrievalEngine

CHUNKS_PATH = os.path.join("data", "processed", "chunks.jsonl")
BENCHMARK_PATH = os.path.join("data", "benchmark", "ground_truth.jsonl")
RESULTS_DIR = os.path.join("data", "results")


def profile_pipeline_latency(
    engine: RetrievalEngine,
    queries: List[str],
    pipeline_type: str,
    runs_per_query: int = 3,
) -> Dict[str, float]:
    """Measures p50, p95, p99, and mean latency in milliseconds."""
    latencies = []

    # Warm-up run (to prime caches / GPU/CPU torch kernels)
    _ = getattr(engine, f"search_{pipeline_type}")(queries[0], top_k=5)

    print(
        f"Profiling [{pipeline_type.upper()}] latency across {len(queries)} queries ({runs_per_query} runs each)..."
    )

    for q in queries:
        for _ in range(runs_per_query):
            start = time.perf_counter()
            if pipeline_type == "dense":
                _ = engine.search_dense(q, top_k=5)
            elif pipeline_type == "hybrid":
                _ = engine.search_hybrid(q, top_k=5)
            elif pipeline_type == "reranked":
                _ = engine.search_reranked(q, top_k=5)
            duration_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(duration_ms)

    latencies_arr = np.array(latencies)
    return {
        "mean_ms": float(np.mean(latencies_arr)),
        "p50_ms": float(np.percentile(latencies_arr, 50)),
        "p95_ms": float(np.percentile(latencies_arr, 95)),
        "p99_ms": float(np.percentile(latencies_arr, 99)),
    }


def print_latency_report(latency_results: Dict[str, Dict[str, float]]):
    headers = ["Pipeline", "Mean (ms)", "p50 (ms)", "p95 (ms)", "p99 (ms)"]
    row_fmt = "{:<16} | {:<10} | {:<10} | {:<10} | {:<10}"

    print("\n" + "=" * 68)
    print(" " * 20 + "RETRIEVAL LATENCY BENCHMARK")
    print("=" * 68)
    print(row_fmt.format(*headers))
    print("-" * 68)

    for pipe, stats in latency_results.items():
        print(
            row_fmt.format(
                pipe,
                f"{stats['mean_ms']:.2f}",
                f"{stats['p50_ms']:.2f}",
                f"{stats['p95_ms']:.2f}",
                f"{stats['p99_ms']:.2f}",
            )
        )
    print("=" * 68 + "\n")


if __name__ == "__main__":
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark_queries = [json.loads(line)["query"] for line in f if line.strip()]

    engine = RetrievalEngine(CHUNKS_PATH)
    pipelines = ["dense", "hybrid", "reranked"]

    latency_summary = {}
    for p in pipelines:
        latency_summary[p] = profile_pipeline_latency(
            engine, benchmark_queries, p, runs_per_query=3
        )

    print_latency_report(latency_summary)

    # Save to disk
    out_path = os.path.join(RESULTS_DIR, "latency_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(latency_summary, f, indent=2)
    print(f"Latency profile saved to {out_path}")
