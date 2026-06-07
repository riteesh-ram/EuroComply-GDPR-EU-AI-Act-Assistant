---
title: EuroComply
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# EuroComply — GDPR & EU AI Act Compliance Assistant

A production-grade Retrieval Augmented Generation (RAG) system for querying GDPR and EU AI Act source material in natural language. Built for compliance, legal, and security teams who need fast, grounded answers from regulatory documents.

Live on Hugging Face Spaces · Powered by Groq + Cohere + ChromaDB

---

## How It Works

A user question flows through four stages before an answer is returned:

```
User Question
     │
     ▼
1. METADATA DETECTION
   LLM identifies whether the query is about GDPR, EU AI Act, or neither.
   Used to apply policy-specific filters on retrieval.
     │
     ▼
2. RETRIEVAL
   Hybrid mode:  Semantic search (ChromaDB) + BM25 keyword search → Ensemble → Cohere Rerank
   Advance mode: LLM generates 5 sub-queries → parallel retrieval → Cohere Rerank
     │
     ▼
3. GENERATION
   Top-ranked context passages are fed to a Groq-hosted LLM.
   The model answers strictly from retrieved context.
     │
     ▼
4. EVALUATION
   Two lightweight LLM checks score the response:
   - Relevance Score (1–10): Does the answer address the question?
   - Faithfulness Score (1–10): Is the answer grounded in the retrieved context?
```

---

## Architecture

### Data Pipeline (Build Time)

Source documents are preprocessed once during Docker image build and stored in ChromaDB.

| Source | Format | Preprocessing |
|---|---|---|
| GDPR | JSONL (Kaggle corpus) | Parsed per article/recital, prefixed with article number + title, stored with chapter/policy metadata |
| EU AI Act | CSV | Parsed per article/recital/annex, prefixed with article number + title, stored with chapter/policy metadata |

Each chunk is labelled (e.g. `"GDPR Article 5: Principles relating to processing of personal data"`) before embedding, so keyword and semantic retrieval both find article-specific content accurately.

### Vector Store

- **Embedding model**: `all-MiniLM-L6-v2` (SentenceTransformers, 384-dim) — loaded as a singleton to avoid duplicate model loads
- **Vector store**: ChromaDB (persisted, single collection for custom-preprocessed data)
- **Metadata filters**: Each chunk carries `policy`, `type`, `number`, `title`, `chapter_number` — queries filtered by policy at retrieval time

### RAG Modes

**Hybrid RAG** (recommended for specific article queries)
- Semantic retrieval from ChromaDB (top 10) + BM25 keyword retrieval over the same candidate set
- Ensemble weighted 70% semantic / 30% BM25
- Cohere `rerank-english-v3.0` reranks the ensemble to top 5 passages
- Best for: "What does GDPR Article 5 say?", exact article lookups

**Advance RAG** (recommended for complex, multi-part questions)
- LLM generates 5 distinct sub-queries from the original question
- All sub-queries retrieved in parallel from ChromaDB
- Cohere reranks the merged result set to top 5 passages
- Best for: "How do conformity assessment obligations differ between biometric and non-biometric high-risk AI systems?"

### Generation & Evaluation

- **LLMs**: Groq-hosted `llama-3.3-70b-versatile` (quality) or `llama-3.1-8b-instant` (speed, lower token usage)
- The same model selected for generation is also used for evaluation, keeping token usage predictable
- Relevance and faithfulness scores are computed per response and displayed below each answer

### Deployment

- **Platform**: Hugging Face Spaces (Docker SDK, port 7860)
- **Build**: `run_pipeline.py` runs at image build time — ChromaDB is fully populated before the app starts, no cold-start ingest
- **UI**: Streamlit in standalone mode, calling the service layer directly (no FastAPI overhead)
- **API layer**: FastAPI routes remain available for programmatic access if needed

---

## Project Structure

```
├── main_ui.py                        # Streamlit chat UI (standalone mode)
├── app.py                            # FastAPI entrypoint (optional API access)
├── run_pipeline.py                   # Orchestrates preprocessing + ChromaDB ingestion
├── Dockerfile                        # Builds image, runs pipeline, starts Streamlit on :7860
├── requirements.txt                  # Pinned dependency stack
│
├── RAG/
│   └── rag_chains.py                 # HybridRag, AdvanceRag, Evaluations, Prompts
│
├── API/
│   ├── routes/appRoutes.py           # POST /complaince/bot/ask
│   ├── controller/appController.py   # Request validation and response formatting
│   └── service/appService.py         # Dispatches to HybridRag or AdvanceRag
│
├── config/
│   ├── chroma_client.py              # Singleton embeddings, ChromaDB connection, retrievers
│   └── groq_client.py                # Groq LLM wrapper
│
└── pipeline/
    ├── preprocess_pdfs.py            # GDPR JSONL + EU AI Act CSV preprocessing logic
    └── pdfs/                         # Source corpora (JSONL, CSV, PDFs via Git LFS)
```

---

## Quickstart (Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
# GROQ_API_KEY=...
# COHERE_API_KEY=...

# 3. Build the vector store
python run_pipeline.py

# 4. Launch the UI
streamlit run main_ui.py
```

---

## Key Design Decisions

**Article-prefixed chunks** — Article number and title are prepended to every chunk before embedding. This ensures both semantic and BM25 retrieval can find article-specific content even when the query only mentions an article number.

**Singleton embedding model** — The `all-MiniLM-L6-v2` model is loaded once globally and reused across all retrieval calls, avoiding duplicate 80 MB loads in a single-process container.

**Build-time vector store** — ChromaDB is populated during Docker image build so the app starts fully ready. There is no ingest step at runtime.

**Standalone Streamlit mode** — The UI calls the async service layer directly, bypassing FastAPI. This reduces container memory usage and removes inter-process overhead.

**Policy-aware metadata filtering** — Queries mentioning GDPR or EU AI Act are automatically routed to policy-filtered retrieval, reducing noise from cross-policy chunks.

**Shared evaluation model** — Relevance and faithfulness scoring use the same model selected for generation, so switching to the smaller model reduces total token usage across all three LLM calls per query.
