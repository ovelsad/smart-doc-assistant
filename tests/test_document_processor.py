"""Tests for document_processor module."""

import tempfile
from pathlib import Path

import pytest

from src.document_processor import Chunk, process_document, _split_text


class TestSplitText:
    def test_basic_split(self):
        text = "First sentence. Second sentence. Third sentence."
        chunks = _split_text(text, chunk_size=30, chunk_overlap=5)
        assert len(chunks) >= 1
        assert all(isinstance(c, str) for c in chunks)

    def test_empty_text(self):
        assert _split_text("", 512, 64) == []

    def test_overlap_not_exceeding_chunk(self):
        text = " ".join(f"Word{i}." for i in range(100))
        chunks = _split_text(text, chunk_size=100, chunk_overlap=20)
        for chunk in chunks:
            assert len(chunk) > 0


class TestProcessDocument:
    def test_txt_file(self, tmp_path):
        doc = tmp_path / "test.txt"
        doc.write_text("This is a test document. It has multiple sentences. Let us process it.", encoding="utf-8")
        chunks = process_document(doc)
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.metadata["source"] == "test.txt" for c in chunks)

    def test_chunk_metadata(self, tmp_path):
        doc = tmp_path / "sample.txt"
        doc.write_text("Hello world. This is content.", encoding="utf-8")
        chunks = process_document(doc)
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i
            assert "file_path" in chunk.metadata

    def test_unsupported_format(self, tmp_path):
        doc = tmp_path / "file.csv"
        doc.write_text("a,b,c")
        with pytest.raises(ValueError, match="Unsupported"):
            process_document(doc)

    def test_fixed_strategy(self, tmp_path):
        doc = tmp_path / "fixed.txt"
        doc.write_text("A" * 1000, encoding="utf-8")
        chunks = process_document(doc, chunk_size=200, chunk_overlap=20, strategy="fixed")
        assert len(chunks) >= 4


class TestChunk:
    def test_default_metadata(self):
        chunk = Chunk(text="hello")
        assert chunk.metadata == {}
        assert chunk.text == "hello"
