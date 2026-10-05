import os
import json

CHUNKS_FILE = os.path.join("data", "processed", "chunks.jsonl")
BENCHMARK_FILE = os.path.join("data", "processed", "ground_truth.jsonl")

def load_chunks(filepath):
    chunks = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def build_curated_benchmark(chunks):
    """
    Creates a focused set of evaluation pairs across categories,
    matching real chunk IDs directly from the processed dataset.
    """
    benchmark = []

    # 1. Search for chunk IDs matching specific topic in the corpus
    def find_chunk_id(keywords):
        for c in chunks:
            text = c["text"].lower()
            if all(kw.lower() in text for kw in keywords):
                return c["chunk_id"]
        return None

    target_queries = [
        # Risk factors
        (
            "What risks does the company face regarding Taiwan Semiconductor Manufacturing Company (TSMC)?",
            ["tsmc", "manufacturing"],
            "risk_factors",
        ),
        (
            "How do United States export controls impact international chip sales and revenue?",
            ["export controls", "international"],
            "risk_factors",
        ),
        (
            "What cybersecurity or intellectual property infringement risks are disclosed?",
            ["cybersecurity", "security"],
            "risk_factors",
        ),
        (
            "What supply chain concentration vulnerabilities are detailed in the filing?",
            ["supply chain", "rely"],
            "risk_factors",
        ),
        # Revenue & Financial performance
        (
            "What was the total revenue reported for the most recent fiscal year?",
            ["revenue", "fiscal year"],
            "revenue_financials",
        ),
        (
            "What was the annual revenue growth rate for the Data Center segment?",
            ["data center", "revenue"],
            "revenue_financials",
        ),
        (
            "What was the reported gross margin percentage for the fiscal year?",
            ["gross margin", "%"],
            "revenue_financials",
        ),
        (
            "How much did research and development expenses increase year over year?",
            ["research and development", "expenses"],
            "revenue_financials",
        ),
        (
            "What was the operating income performance compared to the prior period?",
            ["operating income", "billion"],
            "revenue_financials",
        ),
        # Corporate Governance & Capital Structure
        (
            "Under which state laws is NVIDIA Corporation incorporated?",
            ["delaware", "incorporated"],
            "corporate_governance",
        ),
        (
            "How many shares of common stock were outstanding as of the reporting date?",
            ["shares of common stock outstanding", "billion"],
            "corporate_governance",
        ),
        (
            "What is the aggregate market value of voting stock held by non-affiliates?",
            ["market value", "non-affiliates"],
            "corporate_governance",
        ),
        (
            "Which independent registered public accounting firm audits the financial statements?",
            ["public accounting firm", "audit"],
            "corporate_governance",
        ),
    ]

    for query, keywords, category in target_queries:
        matched_id = find_chunk_id(keywords)
        # fallback to the first chunk containing the primary keyword if combined search misses
        if not matched_id and keywords:
            matched_id = find_chunk_id([keywords[0]])

        if matched_id:
            benchmark.append(
                {
                    "query": query,
                    "expected_chunk_ids": [matched_id],
                    "category": category,
                }
            )
    # fill remaining entries with indexed chunks from the text to guarantee at least 15 pairs

    chunk_index = 0
    while len(benchmark) < 18 and chunk_index < len(chunks):
        c = chunks[chunk_index]
        cid = c["chunk_id"]
        # extract first 12 words as pseudo-query subject
        preview = " ".join(c["text"].split()[:12])
        query = f"Where is the disclosure discussing: '{preview}'?"

        # Avoid duplicate expected chunk IDs
        if not any(cid in item["expected_chunk_ids"] for item in benchmark):
            benchmark.append(
                {
                    "query": query,
                    "expected_chunk_ids": [cid],
                    "category": "corpus_coverage",
                }
            )
        chunk_index += 1
    return benchmark


def save_benchmark(benchmark, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for entry in benchmark:
            f.write(json.dumps(entry) + "\n")
    print(f"Generated {len(benchmark)} golden ground truth pairs saved to:{filepath}")


if __name__ == "__main__":
    if not os.path.exists(CHUNKS_FILE):
        print(f"Error : {CHUNKS_FILE} not found. Run ingest.py first.")
    else:
        corpus_chunks = load_chunks(CHUNKS_FILE)
        benchmark_data = build_curated_benchmark(corpus_chunks)
        save_benchmark(benchmark_data, BENCHMARK_FILE)
