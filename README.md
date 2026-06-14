# DocuMind — Intelligent RAG Document Q&A

> A production-ready Retrieval-Augmented Generation (RAG) system built from scratch — no LangChain abstractions.  
> Upload PDF, TXT, or DOCX files and ask questions in natural language.

---

## Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  Documents  │───▶│ Document         │───▶│ Embedder            │
│ PDF/TXT/DOCX│    │ Processor        │    │ (all-MiniLM-L6-v2)  │
└─────────────┘    │ (sentence chunker│    │ CPU · 384-dim       │
                   │  + fixed-size)   │    └──────────┬──────────┘
                   └──────────────────┘               │
                                                      ▼
                                          ┌─────────────────────┐
                                          │   ChromaDB          │
                                          │   Vector Store      │
                                          │   (persistent)      │
                                          └──────────┬──────────┘
                                                     │
┌─────────────┐    ┌──────────────────┐              │
│   Question  │───▶│ Hybrid Retriever │◀─────────────┘
└─────────────┘    │  ┌────────────┐  │
                   │  │  Semantic  │  │   BM25Okapi
                   │  │  Search    │  │   ┌──────────┐
                   │  └────────────┘  │◀──│  BM25    │
                   │  ┌────────────┐  │   └──────────┘
                   │  │   RRF      │  │
                   │  │  Fusion    │  │
                   │  └────────────┘  │
                   └──────────┬───────┘
                              │  top-k chunks
                              ▼
                   ┌──────────────────┐    ┌─────────────────────┐
                   │  HF Inference    │───▶│  Qwen2.5-7B /       │
                   │  API Generator   │    │  Llama-3.1-8B / ... │
                   └──────────┬───────┘    └─────────────────────┘
                              │
                              ▼
                         ┌─────────┐
                         │ Answer  │
                         │ + Sources│
                         └─────────┘
```

### Key Design Decisions

| Component | Choice | Why |
|-----------|--------|-----|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Fast on CPU, strong quality |
| Vector DB | ChromaDB | Persistent, zero-infrastructure |
| Retrieval | Semantic + BM25 via **RRF** | Best of both worlds — dense + sparse |
| Generation | HF `InferenceClient` (chat API) | No local GPU required, free tier |
| Backend | FastAPI | Async, auto-docs, production-grade |
| Frontend | Streamlit | Rapid demo with file upload |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get a free HuggingFace token

Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and create a **Read** token.

```bash
cp .env.example .env
# Edit .env and add your HF_TOKEN
```

### 3. Run Streamlit UI (recommended)

```bash
HF_TOKEN=hf_your_token_here streamlit run ui/streamlit_app.py
```

Or on Windows:
```powershell
$env:HF_TOKEN="hf_your_token_here"
python -m streamlit run ui/streamlit_app.py
```

### 4. Supported LLM models

| Model | Notes |
|-------|-------|
| `Qwen/Qwen2.5-7B-Instruct` | **Default** — fast and accurate |
| `meta-llama/Meta-Llama-3.1-8B-Instruct` | Strong instruction following |
| `HuggingFaceH4/zephyr-7b-beta` | Lightweight alternative |
| `microsoft/Phi-3-mini-4k-instruct` | Compact, low latency |

All models are served via the [HuggingFace Inference API](https://huggingface.co/docs/api-inference) — no local GPU needed.

---

### 5. Run FastAPI backend

```bash
HF_TOKEN=hf_your_token_here uvicorn api.main:app --reload
```

API docs available at `http://localhost:8000/docs`

---

## API Reference

### `POST /ingest/file`
Upload a PDF, TXT, or DOCX file for indexing.

```bash
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@your_document.pdf"
```

### `POST /query`
Ask a question against ingested documents.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?"}'
```

### `GET /health`
Check pipeline status and document count.

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Colab Notebook

Open `notebooks/rag_demo.ipynb` in Google Colab for a GPU-accelerated demo with larger models.

---

## Project Structure

```
smart-doc-assistant/
├── src/
│   ├── document_processor.py  # PDF/TXT/DOCX → chunks (sentence + fixed strategies)
│   ├── embedder.py            # Sentence-transformer wrapper
│   ├── vector_store.py        # ChromaDB CRUD operations
│   ├── retriever.py           # Hybrid BM25 + semantic retrieval with RRF
│   ├── generator.py           # HuggingFace Inference API wrapper
│   └── rag_pipeline.py        # Orchestrator (ingest + query)
├── api/
│   └── main.py                # FastAPI REST API
├── ui/
│   └── streamlit_app.py       # Streamlit web UI
├── notebooks/
│   └── rag_demo.ipynb         # Colab-ready experiment notebook
├── tests/                     # pytest test suite
├── requirements.txt
└── .env.example
```

---

## Author

**Oleg Sadykhov** — M.Sc. student, Faculty of AI Technologies (AI in Industry track), ITMO University, Saint Petersburg, Russia
[ovelsad23@gmail.com](mailto:ovelsad23@gmail.com) · [github.com/ovelsad](https://github.com/ovelsad)

