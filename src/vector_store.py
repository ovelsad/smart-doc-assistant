"""ChromaDB-backed persistent vector store."""

from __future__ import annotations

import uuid
from pathlib import Path

import chromadb
import numpy as np

from .document_processor import Chunk


class VectorStore:
    def __init__(self, persist_dir: str | Path = "./chroma_db", collection_name: str = "documents") -> None:
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> list[str]:
        if not chunks:
            return []
        ids = [str(uuid.uuid4()) for _ in chunks]
        self._collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )
        return ids

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[dict]:
        count = self._collection.count()
        if count == 0:
            return []
        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(k, count),
            include=["documents", "metadatas", "distances"],
        )
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        return [
            {
                "text": doc,
                "metadata": meta,
                "score": 1.0 - dist,  # cosine similarity from distance
                "retriever": "semantic",
            }
            for doc, meta, dist in zip(docs, metas, distances)
        ]

    def get_all_texts(self) -> list[str]:
        result = self._collection.get(include=["documents"])
        return result["documents"] or []

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name,
            metadata={"hnsw:space": "cosine"},
        )
