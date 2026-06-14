"""FastAPI backend for DocuMind RAG system."""

from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.embedder import Embedder
from src.generator import Generator
from src.rag_pipeline import RAGPipeline, RAGResponse
from src.vector_store import VectorStore

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("WARNING: HF_TOKEN not set — generation endpoint will be unavailable.")
    try:
        embedder = Embedder()
        vector_store = VectorStore(persist_dir=Path("./chroma_db"))
        generator = Generator(hf_token=hf_token) if hf_token else None
        pipeline = RAGPipeline(
            embedder=embedder,
            vector_store=vector_store,
            generator=generator,
            top_k=5,
        ) if generator else None
        if pipeline:
            pipeline._retriever.index_bm25()
    except Exception as exc:
        print(f"Pipeline init error: {exc}")
    yield


app = FastAPI(
    title="DocuMind API",
    description="RAG-powered document Q&A system — semantic + BM25 hybrid retrieval",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    latency_ms: float
    docs_in_store: int


class IngestResponse(BaseModel):
    chunks_added: int
    total_docs: int
    source: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {
        "status": "ok",
        "pipeline_ready": pipeline is not None,
        "docs_in_store": pipeline._vector_store.count() if pipeline else 0,
    }


@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    chunk_size: int = Query(default=512, ge=128, le=2048),
    chunk_overlap: int = Query(default=64, ge=0, le=256),
):
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialized. Check HF_TOKEN.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".txt", ".pdf", ".docx"}:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        added = pipeline.ingest_file(tmp_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return IngestResponse(
        chunks_added=added,
        total_docs=pipeline._vector_store.count(),
        source=file.filename,
    )


@app.post("/ingest/text", response_model=IngestResponse)
def ingest_text(text: str, source_name: str = "manual_input"):
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialized.")
    added = pipeline.ingest_text(text, source_name=source_name)
    return IngestResponse(
        chunks_added=added,
        total_docs=pipeline._vector_store.count(),
        source=source_name,
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialized. Check HF_TOKEN.")
    if pipeline._vector_store.count() == 0:
        raise HTTPException(400, "No documents ingested yet. Upload documents first.")

    response: RAGResponse = pipeline.query(request.question)
    return QueryResponse(
        answer=response.answer,
        sources=[
            {"text": c["text"][:300], "metadata": c.get("metadata", {}), "score": c.get("rrf_score", c.get("score", 0))}
            for c in response.source_chunks
        ],
        latency_ms=round(response.latency_ms, 1),
        docs_in_store=response.metadata.get("docs_in_store", 0),
    )


@app.delete("/reset")
def reset():
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialized.")
    pipeline.reset()
    return {"status": "reset", "docs_in_store": 0}
