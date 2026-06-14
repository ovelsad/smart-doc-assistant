"""Document loading and chunking with configurable strategies."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current, current_len = [], [], 0

    for sentence in sentences:
        sen_len = len(sentence)
        if current_len + sen_len > chunk_size and current:
            chunks.append(" ".join(current))
            overlap_tokens = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) <= chunk_overlap:
                    overlap_tokens.insert(0, s)
                    overlap_len += len(s)
                else:
                    break
            current = overlap_tokens
            current_len = overlap_len
        current.append(sentence)
        current_len += sen_len

    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if c.strip()]


def _load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _load_docx(path: Path) -> str:
    from docx import Document
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


_LOADERS = {".txt": _load_txt, ".pdf": _load_pdf, ".docx": _load_docx}


def process_document(
    file_path: str | Path,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    strategy: Literal["sentence", "fixed"] = "sentence",
) -> list[Chunk]:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix not in _LOADERS:
        raise ValueError(f"Unsupported format: {suffix}. Supported: {list(_LOADERS)}")

    raw_text = _LOADERS[suffix](path)
    raw_text = re.sub(r"\s+", " ", raw_text).strip()

    if strategy == "fixed":
        texts = [
            raw_text[i : i + chunk_size]
            for i in range(0, len(raw_text), chunk_size - chunk_overlap)
        ]
    else:
        texts = _split_text(raw_text, chunk_size, chunk_overlap)

    base_meta = {"source": path.name, "file_path": str(path), "strategy": strategy}
    return [
        Chunk(text=t, metadata={**base_meta, "chunk_index": i})
        for i, t in enumerate(texts)
        if t.strip()
    ]
