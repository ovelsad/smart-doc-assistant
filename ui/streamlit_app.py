"""Streamlit UI for DocuMind — RAG Document Q&A."""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedder import Embedder
from src.generator import Generator
from src.rag_pipeline import RAGPipeline
from src.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DocuMind — RAG Q&A",
    page_icon="📚",
    layout="wide",
)

st.title("📚 DocuMind")
st.caption("Intelligent document Q&A powered by hybrid RAG (semantic + BM25 retrieval)")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------------------------------------------------------------------------
# Sidebar: configuration & ingestion
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")

    hf_token = st.text_input(
        "HuggingFace Token",
        value=os.getenv("HF_TOKEN", ""),
        type="password",
        help="Get your free token at https://huggingface.co/settings/tokens",
    )

    llm_model = st.selectbox(
        "LLM Model",
        options=[
            "Qwen/Qwen2.5-7B-Instruct",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "HuggingFaceH4/zephyr-7b-beta",
            "microsoft/Phi-3-mini-4k-instruct",
        ],
    )

    top_k = st.slider("Retrieved chunks (top-k)", min_value=2, max_value=10, value=5)

    if st.button("Initialize Pipeline", type="primary"):
        if not hf_token:
            st.error("Please enter your HuggingFace token.")
        else:
            with st.spinner("Loading embedding model..."):
                try:
                    embedder = Embedder()
                    vector_store = VectorStore(persist_dir="./chroma_db")
                    generator = Generator(model=llm_model, hf_token=hf_token)
                    pipeline = RAGPipeline(embedder, vector_store, generator, top_k=top_k)
                    pipeline._retriever.index_bm25()
                    st.session_state.pipeline = pipeline
                    st.success(f"Pipeline ready! ({pipeline._vector_store.count()} chunks in store)")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()
    st.header("Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or DOCX files",
        accept_multiple_files=True,
        type=["pdf", "txt", "docx"],
    )

    chunk_size = st.slider("Chunk size (chars)", 256, 2048, 512, step=64)
    chunk_overlap = st.slider("Chunk overlap (chars)", 0, 256, 64, step=16)

    if st.button("Ingest Documents") and uploaded_files:
        if st.session_state.pipeline is None:
            st.error("Initialize the pipeline first.")
        else:
            pipeline = st.session_state.pipeline
            total_added = 0
            progress = st.progress(0)
            for i, file in enumerate(uploaded_files):
                suffix = Path(file.name).suffix.lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name
                try:
                    added = pipeline.ingest_file(tmp_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    total_added += added
                    st.write(f"✓ {file.name}: {added} chunks")
                except Exception as e:
                    st.error(f"✗ {file.name}: {e}")
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
                progress.progress((i + 1) / len(uploaded_files))

            st.success(f"Ingested {total_added} chunks. Total: {pipeline._vector_store.count()}")

    st.divider()
    if st.session_state.pipeline:
        docs = st.session_state.pipeline._vector_store.count()
        st.metric("Chunks in store", docs)

    if st.button("Reset Knowledge Base"):
        if st.session_state.pipeline:
            st.session_state.pipeline.reset()
            st.session_state.chat_history = []
            st.success("Knowledge base cleared.")


# ---------------------------------------------------------------------------
# Main: chat interface
# ---------------------------------------------------------------------------

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander(f"Sources ({len(msg['sources'])} chunks)", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Chunk {i}** | Score: `{src.get('rrf_score', src.get('score', 0)):.4f}`")
                    st.caption(src.get("metadata", {}).get("source", "unknown"))
                    st.text(src["text"][:400] + "..." if len(src["text"]) > 400 else src["text"])
                    st.divider()

if question := st.chat_input("Ask a question about your documents..."):
    if st.session_state.pipeline is None:
        st.error("Initialize the pipeline first using the sidebar.")
    elif st.session_state.pipeline._vector_store.count() == 0:
        st.error("Upload and ingest documents first.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.pipeline.query(question)
                    st.write(response.answer)
                    st.caption(f"Latency: {response.latency_ms:.0f}ms | Chunks retrieved: {len(response.source_chunks)}")

                    with st.expander(f"Sources ({len(response.source_chunks)} chunks)", expanded=False):
                        for i, src in enumerate(response.source_chunks, 1):
                            st.markdown(f"**Chunk {i}** | Score: `{src.get('rrf_score', src.get('score', 0)):.4f}`")
                            st.caption(src.get("metadata", {}).get("source", "unknown"))
                            st.text(src["text"][:400] + "..." if len(src["text"]) > 400 else src["text"])
                            st.divider()

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response.answer,
                        "sources": response.source_chunks,
                    })
                except Exception as e:
                    st.error(f"Error: {e}")
