import os
import json

BENCHMARK_PATH = os.path.join("data", "benchmark", "ground_truth.jsonl")

# Clean, factual ground-truth pairings verified directly against NVDA 10-K text
curated_ground_truth = [
    # Supply chain & manufacturing
    {
        "query": "Where does NVIDIA state that its supply chain is concentrated in Asia?",
        "expected_chunk_ids": ["nvda_10k_chunk_0018"],
        "category": "supply_chain",
    },
    {
        "query": "What risks are detailed regarding component shortages and delays in product shipments from suppliers?",
        "expected_chunk_ids": ["nvda_10k_chunk_0041"],
        "category": "supply_chain",
    },
    # Financial results & MD&A
    {
        "query": "How is Gross Profit and Gross Margin calculated relative to cost of semiconductors and packaging?",
        "expected_chunk_ids": ["nvda_10k_chunk_0102"],
        "category": "revenue_financials",
    },
    {
        "query": "What drove Gaming revenue growth in fiscal year 2026 and what was the impact of Blackwell demand?",
        "expected_chunk_ids": ["nvda_10k_chunk_0095"],
        "category": "revenue_financials",
    },
    {
        "query": "Where does NVIDIA discuss customer concentration among CSPs, AI model makers, and direct versus indirect customers?",
        "expected_chunk_ids": ["nvda_10k_chunk_0101"],
        "category": "revenue_financials",
    },
    {
        "query": "Which segment demand drove revenue increases between fiscal year 2026 and 2025 regarding Compute & Networking?",
        "expected_chunk_ids": ["nvda_10k_chunk_0162"],
        "category": "revenue_financials",
    },
    # International & Risk factors
    {
        "query": "What risks are disclosed regarding international sales and impairment charges in investments?",
        "expected_chunk_ids": ["nvda_10k_chunk_0045"],
        "category": "risk_factors",
    },
    {
        "query": "What manufacturing risks are highlighted regarding long lead times and uncertain capacity?",
        "expected_chunk_ids": ["nvda_10k_chunk_0029"],
        "category": "risk_factors",
    },
    # Corporate Governance / Legal
    {
        "query": "Where does the report reference Note 17 and Note 12 for Legal Proceedings incorporated by reference?",
        "expected_chunk_ids": ["nvda_10k_chunk_0088"],
        "category": "corporate_governance",
    },
    {
        "query": "Where is the report table of contents and Item 7 Management's Discussion and Analysis index located?",
        "expected_chunk_ids": ["nvda_10k_chunk_0091"],
        "category": "corporate_governance",
    },
    # Coverage Checks
    {
        "query": "Where are creative and engineering GPU use cases such as visual effects in movies described?",
        "expected_chunk_ids": ["nvda_10k_chunk_0009"],
        "category": "corpus_coverage",
    },
    {
        "query": "Where does the filing discuss developer programs and marketing personnel engaging with software developers?",
        "expected_chunk_ids": ["nvda_10k_chunk_0017"],
        "category": "corpus_coverage",
    },
]

os.makedirs(os.path.dirname(BENCHMARK_PATH), exist_ok=True)
with open(BENCHMARK_PATH, "w", encoding="utf-8") as f:
    for item in curated_ground_truth:
        f.write(json.dumps(item) + "\n")

print(
    f"Successfully wrote {len(curated_ground_truth)} curated ground-truth records to {BENCHMARK_PATH}"
)
