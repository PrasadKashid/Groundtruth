import json
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict, Tuple

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RetrievalEngine:
    def __init__(self, chunks_path: str):
        # Loading the chunks 
        print(f"Loading chunks from {chunks_path}...")
        self.chunks: List[Dict] = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.chunks.append(json.loads(line))
        self.chunk_ids = [c["chunk_id"] for c in self.chunks]
        self.corpus_text = [c["text"] for c in self.chunks]
        print(f"Loaded {len(self.chunks)} chunks")

        # 1. initializing Dense Embeddings Model & FAISS
        print("Initializing Desne Embedding Model (all-MiniLM-L6-v2)...")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)
        self._build_dense_index()

        # 2. Initialzing BM25 Keyword Search
        print("Building BM25 Lexical Index...")
        tokenized_corpus = [doc.lower().split() for doc in self.corpus_text]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 3. Initialize Cross-Encoder Reranker (Lazy loaded when Pipeline C is called)
        self.reranker = None

    def _build_dense_index(self):
        """Encodes all chunks and populates a normalized FAISS cosine index."""
        embeddings = self.embed_model.encode(
            self.corpus_text,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        dim = embeddings.shape[1]
        self.dense_index = faiss.IndexFlatIP(
            dim
        )  # Inner Product on normalized vectors = Cosine Similarity
        self.dense_index.add(embeddings.astype(np.float32))

    # Pipeline A: Desne Search
    def search_dense(self, query: str, top_k: int = 10) -> List[str]:
        """Returns top_k chunk_ids using dense semantic vector search."""
        query_vec = self.embed_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )

        _, indices = self.dense_index.search(query_vec.astype(np.float32), top_k)

        return [self.chunk_ids[idx] for idx in indices[0] if idx < len(self.chunk_ids)]

    # Pipeline B: Hybrid Seach (BM25 + Dense with Reciprocal Rank Fusion)
    def search_bm25(self, query: str, top_k=10) -> List[str]:
        """Returns top_k chunk_ids using BM25 token matching."""

        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.chunk_ids[idx] for idx in top_indices]

    def search_hybrid(self, query: str, top_k: int = 10, rrf_k: int = 60) -> List[str]:
        dense_results = self.search_dense(query, top_k=top_k * 2)
        bm25_results = self.search_bm25(query, top_k * 2)

        rrf_scores = {}

        # Score dense Ranks
        for rank, chunk_id in enumerate(dense_results):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (
                1.0 / (rrf_k + rank + 1)
            )

        # Scores BM25 Ranks
        for rank, chunk_id in enumerate(bm25_results):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (
                1.0 / (rrf_k + rank + 1)
            )

        sorted_chunks = sorted(
            rrf_scores.items(), key=lambda item: item[1], reverse=True
        )
      
        return [chunk_id for chunk_id, _ in sorted_chunks[:top_k]]
    
    # Pipeline C: Hybrid + Cross-Encoder Reranker
    def search_reranked(
        self, query: str, top_k: int = 10, initial_k: int = 25
    ) -> List[str]:
        """Fetches initial candidates via Hybrid search, then reranks with a Cross-Encoder."""
        if self.reranker is None:
            print("Loading Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)...")
            self.reranker = CrossEncoder(CROSS_ENCODER)

        # 1. Candidate Generation
        candidate_ids = self.search_hybrid(query, top_k=initial_k)
        candidate_texts = [
            self.chunks[self.chunk_ids.index(cid)]["text"] for cid in candidate_ids
        ]

        # 2. Pairwise Scoring
        pairs = [[query, text] for text in candidate_texts]
        scores = self.reranker.predict(pairs)

        # 3. Sort by reranker cross-attention score
        ranked_indices = np.argsort(scores)[::-1]
        return [candidate_ids[idx] for idx in ranked_indices[:top_k]]


if __name__ == "__main__":
    # Smoke test the engine
    import os

    chunks_file = os.path.join("data", "processed", "chunks.jsonl")

    engine = RetrievalEngine(chunks_file)
    test_query = "What are the primary risk factors regarding Taiwan Semiconductor Manufacturing Company?"

    print("\n--- Testing Pipeline A (Dense) ---")
    print(engine.search_dense(test_query, top_k=3))

    print("\n--- Testing Pipeline B (Hybrid RRF) ---")
    print(engine.search_hybrid(test_query, top_k=3))

    print("\n--- Testing Pipeline C (Hybrid + Reranker) ---")
    print(engine.search_reranked(test_query, top_k=3))
