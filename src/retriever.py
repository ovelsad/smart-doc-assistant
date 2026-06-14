"""Hybrid retriever: semantic search + BM25 fused via Reciprocal Rank Fusion."""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from .embedder import Embedder
from .vector_store import VectorStore


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _reciprocal_rank_fusion(ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Merge multiple ranked lists via RRF. Each list item must have 'text' key."""
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = doc["text"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in items:
                items[key] = doc

    merged = sorted(scores.keys(), key=lambda key: scores[key], reverse=True)
    result = []
    for key in merged:
        item = dict(items[key])
        item["rrf_score"] = scores[key]
        item["retriever"] = "hybrid"
        result.append(item)
    return result


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, embedder: Embedder) -> None:
        self._vector_store = vector_store
        self._embedder = embedder
        self._bm25: BM25Okapi | None = None
        self._corpus: list[str] = []

    def index_bm25(self) -> None:
        self._corpus = self._vector_store.get_all_texts()
        if self._corpus:
            tokenized = [_tokenize(doc) for doc in self._corpus]
            self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        query_emb = self._embedder.embed_single(query)
        semantic_results = self._vector_store.search(query_emb, k=k)

        if self._bm25 and self._corpus:
            bm25_scores = self._bm25.get_scores(_tokenize(query))
            top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]
            bm25_results = [
                {"text": self._corpus[i], "metadata": {}, "score": float(bm25_scores[i]), "retriever": "bm25"}
                for i in top_indices
            ]
            return _reciprocal_rank_fusion([semantic_results, bm25_results])[:k]

        return semantic_results
