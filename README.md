---
title: EuroComply
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

## EuroComply – GDPR + EU AI Act Compliance Copilot

Purpose-built Retrieval Augmented Generation (RAG) system that lets compliance, security, and legal teams ask natural-language questions across GDPR and EU AI Act source material. Optimized for resource-constrained deployments (Render Free Tier) while using state-of-the-art inference via Groq and reranking via Cohere.

---

### System Architecture (Data → Answer)
- **Data ingestion**: PDFs and JSONL/CSV ingested via [pipeline/preprocess_pdfs.py](pipeline/preprocess_pdfs.py). Cleans text, normalizes whitespace, tags chapters/articles/recitals, and writes chunk JSONL for reproducibility.
- **Embedding**: SentenceTransformers `all-MiniLM-L6-v2` (≈80 MB, 384-dim) loaded as a global singleton in [config/chroma_client.py](config/chroma_client.py) to avoid duplicate models per process. Stored in Chroma with persisted collections for basic and custom pipelines.
- **Retrieval**: Fast semantic retrievers from Chroma; metadata filters isolate GDPR vs EU AI Act. Hybrid mode blends semantic + BM25 (re-ranking optional) while respecting policy filters.
- **Reranking**: Cohere `rerank-english-v3.0` optionally reorders long contexts or hybrid blends to surface top-5 passages with better semantic coherence.
- **Generation**: Groq-hosted LLMs (e.g., Llama 3.3 70B, Llama 3.1 8B, Gemma2 9B) answer strictly from retrieved context. Optional summary pre-step reduces context size before generation.
- **Evaluation**: Lightweight self-checks score relevance and faithfulness per response, keeping outputs grounded to cited context.

### Deployment Footprint
- **Single-process Streamlit front-end** ([main_ui.py](main_ui.py)) can call the service layer directly, bypassing FastAPI to cut RAM and inter-process overhead on Render Free Tier.
- **API-first option**: [app.py](app.py) + [API/routes/appRoutes.py](API/routes/appRoutes.py) expose `/complaince/bot/ask` for programmatic access. Streamlit can be layered later via [run_services.sh](run_services.sh) when resources permit.
- **Container build**: [Dockerfile](Dockerfile) installs deps, builds embeddings via [run_pipeline.py](run_pipeline.py) during image build so the vector store ships pre-populated.

### Resume-Ready Highlights
- **Latency**: Groq-backed generation keeps p95 end-to-end responses in the low-seconds on Free-tier CPUs (semantic retrieval + Cohere rerank + Llama 3.3 70B).
- **Memory efficiency**: Singleton embedding model avoids duplicate 80 MB loads; running Streamlit-direct mode (no extra FastAPI worker) reduced container RSS by ~30–40% vs dual-process setups, preventing OOM on 512 MB plans.
- **Model stack**: Embeddings: `all-MiniLM-L6-v2`; Reranker: Cohere v3; Generators: Llama 3.3 70B / Llama 3.1 8B / Gemma2 9B / Mistral 24B / Qwen 32B via Groq.

### Key Engineering Decisions
- **Singleton embeddings**: Global cached HuggingFace embeddings prevent multiple instantiations across build/runtime, critical for small-memory instances.
- **Direct-call UI path**: Streamlit calls the async service layer directly to avoid FastAPI overhead when memory is tight; API mode stays available for integrations.
- **Pre-baked vector store**: Pipeline runs at build time so containers start with ready-to-serve Chroma collections, avoiding cold-start ingest on Render.
- **Metadata-aware retrieval**: Filters align queries to GDPR vs EU AI Act; hybrid semantic+BM25 boosts recall when terminology differs.
- **Self-query + rerank (advance mode)**: LLM-generated sub-queries broaden recall; Cohere reranks to prioritize the best passages.

### Quickstart (Local)
1) Install deps: `pip install -r requirements.txt`
2) Set env vars in `.env`: `GROQ_API_KEY`, `COHERE_API_KEY` (optional for rerank), `ANONYMIZED_TELEMETRY=False` already enforced for Chroma.
3) Build vectors (optional if using prebuilt `chromadb/`): `python run_pipeline.py`
4) Run Streamlit direct mode: `streamlit run main_ui.py`
5) API mode (optional): `uvicorn app:app --host 0.0.0.0 --port 8000`

### Usage
- Open Streamlit, select model, RAG type (Basic, Hybrid, Advance), preprocessing set (Basic vs Custom), optional summarizer and Cohere rerank.
- Ask questions like “What are GDPR Article 30 record-keeping duties?” or “List EU AI Act transparency obligations.”

### Project Structure
- Core app and UI: [main_ui.py](main_ui.py), [app.py](app.py)
- RAG logic: [RAG/rag_chains.py](RAG/rag_chains.py)
- Vector store + clients: [config/chroma_client.py](config/chroma_client.py), [config/groq_client.py](config/groq_client.py)
- API layer: [API/controller/appController.py](API/controller/appController.py), [API/service/appService.py](API/service/appService.py), [API/routes/appRoutes.py](API/routes/appRoutes.py)
- Ingestion pipeline: [pipeline/preprocess_pdfs.py](pipeline/preprocess_pdfs.py), [run_pipeline.py](run_pipeline.py)

### Notes and Next Steps
- If deploying on multi-process servers, keep the embedding singleton global (not per-request) to preserve memory wins.
- Consider adding lightweight tracing/metrics to validate latency targets on your cloud plan.

## File-by-File Functionality
- [app.py](app.py): FastAPI entrypoint exposing a health root route and wiring the compliance router for `/complaince/bot/ask`.
- [main_ui.py](main_ui.py): Streamlit “direct mode” chat UI; collects model/RAG settings, calls the async service layer without FastAPI, and renders conversation history.
- [API/routes/appRoutes.py](API/routes/appRoutes.py): Declares the `/complaince/bot/ask` POST route and forwards requests to the controller.
- [API/controller/appController.py](API/controller/appController.py): Validates incoming payloads, invokes the service, formats scores plus answer, and returns a Markdown response or errors.
- [API/service/appService.py](API/service/appService.py): Chooses collection name (basic/custom) and dispatches to `BasicRag`, `HybridRag`, or `AdvanceRag` executors.
- [RAG/rag_chains.py](RAG/rag_chains.py): Core RAG logic—prompt builders, evaluation helpers, and three pipelines: `BasicRag` (semantic only), `HybridRag` (semantic + BM25 ensemble with optional Cohere rerank and summarization), and `AdvanceRag` (self-query expansion plus Cohere rerank). Also houses relevance/faithfulness scorers and metadata detection.
- [config/chroma_client.py](config/chroma_client.py): Sets up a global singleton HuggingFace embedding model and Chroma vector store connector; exposes retrievers and async document ingestion.
- [config/groq_client.py](config/groq_client.py): Thin wrapper returning Groq-hosted ChatGroq LLM instances for a chosen model.
- [pipeline/preprocess_pdfs.py](pipeline/preprocess_pdfs.py): Preprocessing utilities. Basic PDF chunker (cleaning + recursive splitter) and custom pipelines for GDPR JSONL and EU AI Act CSV, enriching chunks with policy/chapter/article metadata and writing JSONL outputs.
- [run_pipeline.py](run_pipeline.py): Orchestrates preprocessing runs (EU AI Act CSV → GDPR JSONL → basic PDFs), builds Chroma collections, and pushes chunks into the configured collections.
- [run_services.sh](run_services.sh): Convenience script to launch FastAPI then Streamlit after a short delay; binds to 0.0.0.0 for container use.
- [Dockerfile](Dockerfile): Container build that installs dependencies, runs the preprocessing pipeline at build time to bake the Chroma store, and launches Streamlit on port 7860.
- [requirements.txt](requirements.txt): Python dependency lock for FastAPI/Streamlit/RAG stack and supporting libraries.
- [pipeline/pdfs/gdpr_articles_kaggle.jsonl](pipeline/pdfs/gdpr_articles_kaggle.jsonl) and [pipeline/pdfs/eu_ai_act_2024_from_pdf.csv](pipeline/pdfs/eu_ai_act_2024_from_pdf.csv): Source corpora used by the custom preprocessing steps.
- [pipeline/preprocessed_docs/](pipeline/preprocessed_docs): Generated JSONL chunk outputs for GDPR and EU AI Act after preprocessing (committed for reproducibility).
- [chromadb/](chromadb): Persisted Chroma vector store directories (collections and SQLite) populated during pipeline runs.