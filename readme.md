# Groundtruth: Retrieval Evaluation & Regression CI Harness

An automated evaluation harness for financial document retrieval (SEC 10-K filings).

Instead of qualitative spot-checking ("vibes-based evaluation"), this repository provides a mathematically rigorous testing suite that measures **Recall@k**, **MRR**, and **nDCG** across competing retrieval architectures, with CI gates that block retrieval degradation.

## Architecture

- **Layer 1:** SEC 10-K parsing, semantic chunking, and 100-pair Golden Ground Truth benchmark.
- **Layer 2:** Retrieval pipelines under test (Dense Vector vs. Hybrid BM25+Dense vs. Cross-Encoder Reranker).
- **Layer 3:** Ranking metric scorer and GitHub Actions automated regression gate (`Recall@10` regression threshold: < 1%).

## Setup

```bash
python -m venv venv
source venv\Scripts\activate
pip install -r requirements.txt
```
