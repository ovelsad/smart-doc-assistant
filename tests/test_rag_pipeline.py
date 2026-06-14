"""Integration tests for the RAG pipeline (without LLM generation)."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.document_processor import Chunk
from src.embedder import Embedder
from src.rag_pipeline import RAGPipeline
from src.vector_store import VectorStore


@pytest.fixture
def tmp_vector_store(tmp_path):
    return VectorStore(persist_dir=tmp_path / "chroma", collection_name="test_collection")


@pytest.fixture
def embedder():
    return Embedder()


@pytest.fixture
def mock_generator():
    gen = MagicMock()
    gen.generate.return_value = "This is a mocked answer."
    return gen


@pytest.fixture
def pipeline(embedder, tmp_vector_store, mock_generator):
    p = RAGPipeline(embedder, tmp_vector_store, mock_generator, top_k=3)
    return p


class TestRAGPipelineIngestion:
    def test_ingest_text(self, pipeline):
        count = pipeline.ingest_text("Machine learning is a subset of AI.", source_name="test")
        assert count == 1
        assert pipeline._vector_store.count() == 1

    def test_ingest_txt_file(self, pipeline, tmp_path):
        doc = tmp_path / "test.txt"
        doc.write_text(
            "Natural language processing helps computers understand text. "
            "It is widely used in chatbots and search engines. "
            "Transformers revolutionized NLP.",
            encoding="utf-8",
        )
        added = pipeline.ingest_file(doc)
        assert added >= 1
        assert pipeline._vector_store.count() >= 1

    def test_reset_clears_store(self, pipeline):
        pipeline.ingest_text("Some content to be cleared.")
        assert pipeline._vector_store.count() > 0
        pipeline.reset()
        assert pipeline._vector_store.count() == 0


class TestRAGPipelineQuery:
    def test_query_returns_response(self, pipeline):
        pipeline.ingest_text(
            "The Eiffel Tower is located in Paris, France. It was built in 1889.",
            source_name="facts",
        )
        response = pipeline.query("Where is the Eiffel Tower?")
        assert response.answer == "This is a mocked answer."
        assert len(response.source_chunks) >= 1
        assert response.latency_ms > 0
        assert response.question == "Where is the Eiffel Tower?"

    def test_retrieval_finds_relevant_chunks(self, pipeline):
        pipeline.ingest_text("Python is a high-level programming language.", source_name="cs")
        pipeline.ingest_text("The Amazon River is the largest river in South America.", source_name="geo")

        response = pipeline.query("Tell me about programming languages.")
        texts = [c["text"] for c in response.source_chunks]
        assert any("Python" in t or "programming" in t.lower() for t in texts)
