"""Tests for Embedder module."""

import numpy as np
import pytest

from src.embedder import Embedder


@pytest.fixture(scope="module")
def embedder():
    return Embedder()


class TestEmbedder:
    def test_embed_single(self, embedder):
        emb = embedder.embed_single("hello world")
        assert isinstance(emb, np.ndarray)
        assert emb.ndim == 1
        assert emb.shape[0] == embedder.dim

    def test_embed_batch(self, embedder):
        texts = ["first sentence", "second sentence", "third sentence"]
        embs = embedder.embed(texts)
        assert embs.shape == (3, embedder.dim)

    def test_normalized_embeddings(self, embedder):
        emb = embedder.embed_single("test sentence for normalization")
        norm = float(np.linalg.norm(emb))
        assert abs(norm - 1.0) < 1e-5

    def test_empty_list_returns_empty(self, embedder):
        result = embedder.embed([])
        assert result.shape[0] == 0

    def test_semantic_similarity(self, embedder):
        emb1 = embedder.embed_single("The cat sat on the mat")
        emb2 = embedder.embed_single("A cat rested on a rug")
        emb3 = embedder.embed_single("Python is a programming language")
        sim_related = float(np.dot(emb1, emb2))
        sim_unrelated = float(np.dot(emb1, emb3))
        assert sim_related > sim_unrelated, "Related texts should have higher cosine similarity"
