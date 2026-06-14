"""Orchestrates the full RAG pipeline: ingest → embed → retrieve → generate."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from .document_processor import Chunk, process_document
from .embedder import Embedder
from .generator import Generator
from .retriever import HybridRetriever
from .vector_store import VectorStore


@dataclass
class RAGResponse:
    answer: str
    source_chunks: list[dict]
    question: str
    latency_ms: float
    metadata: dict = field(default_factory=dict)


class RAGPipeline:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        generator: Generator,
        top_k: int = 5,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._generator = generator
        self._retriever = HybridRetriever(vector_store, embedder)
        self.top_k = top_k

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_file(
        self,
        file_path: str | Path,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> int:
        chunks = process_document(file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            return 0
        embeddings = self._embedder.embed([c.text for c in chunks], show_progress=True)
        self._vector_store.add(chunks, embeddings)
        self._retriever.index_bm25()
        return len(chunks)

    def ingest_text(self, text: str, source_name: str = "direct_input") -> int:
        chunk = Chunk(text=text.strip(), metadata={"source": source_name})
        embedding = self._embedder.embed([chunk.text])
        self._vector_store.add([chunk], embedding)
        self._retriever.index_bm25()
        return 1

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, question: str) -> RAGResponse:
        t0 = time.perf_counter()

        context_chunks = self._retriever.retrieve(question, k=self.top_k)
        answer = self._generator.generate(question, context_chunks)

        latency_ms = (time.perf_counter() - t0) * 1000
        return RAGResponse(
            answer=answer,
            source_chunks=context_chunks,
            question=question,
            latency_ms=latency_ms,
            metadata={
                "num_chunks_retrieved": len(context_chunks),
                "docs_in_store": self._vector_store.count(),
            },
        )

    def reset(self) -> None:
        self._vector_store.reset()
        self._retriever.index_bm25()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        persist_dir: str | Path = "./chroma_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "Qwen/Qwen2.5-7B-Instruct",
        top_k: int = 5,
    ) -> "RAGPipeline":
        embedder = Embedder(model_name=embedding_model)
        vector_store = VectorStore(persist_dir=persist_dir)
        generator = Generator(model=llm_model)
        pipeline = cls(embedder, vector_store, generator, top_k=top_k)
        pipeline._retriever.index_bm25()
        return pipeline
