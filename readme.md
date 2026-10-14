# SEC 10-K RAG Retrieval Evaluation Framework

An enterprise-grade retrieval benchmarking and evaluation harness built on NVIDIA's Form 10-K SEC filing. Benchmarks three retrieval paradigms (**Dense**, **Hybrid RRF**, and **Cross-Encoder Reranking**) using pure-Python information retrieval metrics (**Recall@k**, **MRR**, **nDCG@k**), latency profiling, and automated CI regression gates.

---

## 🏗 Architecture Overview

```
                                 ┌─── Dense Retriever (FAISS + all-MiniLM-L6-v2) ───┐
User Query ──► Query Pipeline ──┤                                                   ├──► Reciprocal Rank Fusion (RRF) ──► Cross-Encoder (ms-marco-MiniLM-L-6-v2) ──► Top-K Chunks
                                 └─── Lexical Retriever (BM25) ─────────────────────┘
```

1. **Dense Retrieval:** Semantic dense embeddings via `sentence-transformers/all-MiniLM-L6-v2` indexed in an inner-product FAISS vector index.
2. **Lexical Retrieval:** BM25 sparse index using tokenized n-grams to match exact financial nomenclature, legal clauses, and accounting codes.
3. **Hybrid RRF:** Multi-source rank aggregation via Reciprocal Rank Fusion ($k=60$) combining semantic and lexical signals without score calibration dependencies.
4. **Cross-Encoder Reranker:** Second-stage deep re-ranking using `cross-encoder/ms-marco-MiniLM-L-6-v2` to score joint `(query, document)` token interactions, accelerated via an $\mathcal{O}(1)$ chunk hash map lookup.

---

## 📊 Benchmark Results

Evaluated across a curated golden ground-truth benchmark of 10-K disclosures covering risk factors, MD&A financials, corporate governance, and operational reporting:

| Pipeline                     |  Recall@1  |  Recall@3   |  Recall@5   |    MRR     |   nDCG@3   |  Latency (p50)  |
| :--------------------------- | :--------: | :---------: | :---------: | :--------: | :--------: | :-------------: |
| **Dense (FAISS)**            |   41.67%   |   58.33%    |   58.33%    |   0.5119   |   0.5218   |  **12.19 ms**   |
| **Hybrid (Dense + BM25)**    |   50.00%   |   83.33%    |   83.33%    |   0.6591   |   0.6885   |  **13.14 ms**   |
| **Reranked (Cross-Encoder)** | **91.67%** | **100.00%** | **100.00%** | **0.9583** | **0.9692** | **2,441.62 ms** |

### Key Empirical Findings

- **Lexical Lift:** Adding BM25 to Dense via RRF improved **Recall@3 by +25.0%** (58.33% $\rightarrow$ 83.33%) with negligible latency overhead (**+0.95 ms**). Exact financial terminology (e.g., "Blackwell", "Note 17", "CSPs") is reliably recovered.
- **Cross-Encoder Precision:** The reranker pushed **Recall@1 to 91.67%** and **Recall@3 to 100.00%** with an **MRR of 0.9583**, ensuring the correct chunk is ranked #1 in 11 out of 12 test queries.
- **Production Trade-off:** The Cross-Encoder is ~180x slower on CPU due to cross-attention complexity ($2,441.62\text{ ms}$ vs. $13.14\text{ ms}$), establishing the Hybrid pipeline as the optimal first-stage retrieval filter before routing candidate subsets to the reranker.

---

## 📐 Implemented Metrics

All ranking formulas are implemented from scratch in pure Python (`src/eval/metrics.py`):

- **Recall@k:** Fraction of ground-truth chunks captured within top-$k$ results:
  $$\text{Recall@k} = \frac{|\text{Retrieved}_{:k} \cap \text{Expected}|}{|\text{Expected}|}$$

- **Mean Reciprocal Rank (MRR):** Measures position of the first relevant document:
  $$\text{RR} = \begin{cases} \frac{1}{\text{rank}} & \text{where rank is 1-indexed position of 1st hit} \\ 0.0 & \text{if not found} \end{cases}$$

- **Normalized Discounted Cumulative Gain (nDCG@k):** Evaluates ranking quality with logarithmic position discount:
  $$\text{DCG@k} = \sum_{i=1}^k \frac{\text{rel}_i}{\log_2(i + 1)}, \quad \text{IDCG@k} = \sum_{i=1}^{\min(k, |\text{Expected}|)} \frac{1}{\log_2(i + 1)}$$
  $$\text{nDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}$$

---

## 📁 Repository Structure

```
.
├── .github/
│   └── workflows/
│       └── eval_regression.yml       # Automated CI regression gate
├── data/
│   ├── benchmark/
│   │   └── ground_truth.jsonl        # Curated SEC 10-K evaluation pairs
│   ├── processed/
│   │   └── chunks.jsonl              # 170 parsed filing chunks
│   └── results/
│       ├── evaluation_summary.json   # Precision & recall benchmark outputs
│       └── latency_summary.json      # p50/p95/p99 latency measurements
├── src/
│   ├── eval/
│   │   ├── error_analysis.py         # Failure diagnostic script
│   │   ├── latency_profiler.py       # Multi-query latency harness
│   │   ├── metrics.py                # Pure-Python ranking formulas
│   │   └── run_eval.py               # Main benchmark orchestrator
│   └── retrieval/
│       ├── search_engine.py          # Dense, BM25, RRF & Cross-Encoder engine
│       └── write_curated_benchmark.py # Golden dataset builder
├── tests/
│   ├── test_metrics.py               # Unit tests for ranking metrics
│   └── test_retrieval_regression.py  # CI regression quality assertions
├── pytest.ini                        # Pytest configuration
├── requirements.txt                  # Python dependencies
└── README.md                         # Technical documentation
```

---

## 🚀 Quickstart

### 1. Environment Setup

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv venv

# Windows (PowerShell):
venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Evaluation Benchmark

Executes the evaluation loop across Dense, Hybrid, and Reranked pipelines and updates the results summary:

```bash
python -m src.eval.run_eval
```

### 3. Run Latency Profiler

Profiles round-trip latency across all pipelines with warm-up cycles and multi-run sampling:

```bash
python -m src.eval.latency_profiler
```

### 4. Run Retrieval Failure Diagnostics

Analyzes individual query misses against ground-truth snippets:

```bash
python -m src.eval.error_analysis
```

### 5. Run Pytest Regression Suite

```bash
python -m pytest -v
```

---

## 🛡 CI/CD Quality Gates

Automated regression assertions enforce minimum retrieval quality on every commit via GitHub Actions (`.github/workflows/eval_regression.yml`):

- `test_hybrid_beats_dense_mrr`: Enforces that Hybrid lexical-semantic search outperforms pure Dense retrieval on MRR.
- `test_reranked_meets_quality_baseline`: Blocks merges if Cross-Encoder performance drops below:
  - $\text{Recall@5} \ge 0.40$
  - $\text{MRR} \ge 0.35$
  - $\text{nDCG@3} \ge 0.30$