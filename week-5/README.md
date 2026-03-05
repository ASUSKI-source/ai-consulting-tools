# Week 5 — Document Q&A (RAG) + Market Analyzer

**What it does:** Two systems in one app. **(1) Market Analyzer:** Same as Week 4—stocks, crypto, compare, history, watchlist—all backed by the same FastAPI + SQLAlchemy stack. **(2) Document Q&A (RAG):** Upload PDF or text files; they are chunked, embedded with sentence-transformers, and stored in ChromaDB. Ask questions in natural language; Claude answers using only retrieved passages, with source citations and confidence. The UI includes a document upload area, document list, and a Q&A panel with history.

**Technical skills demonstrated:** RAG (retrieval-augmented generation), ChromaDB, sentence-transformers (embeddings), text chunking (overlap, PDF via PyPDF2), FastAPI file upload, REST design, PostgreSQL + SQLAlchemy (analyses, documents, QA history), frontend integration for both market analysis and document Q&A.

---

## Run locally

1. Create a `.env` in this folder or repo root with:
   ```bash
   ANTHROPIC_API_KEY=your_key_here
   ```
   Omit `DATABASE_URL` for local SQLite.

2. From the `week-5` folder (or repo root with a single `.venv`):
   ```bash
   pip install -r requirements.txt
   pip install sentence-transformers chromadb PyPDF2   # if not in requirements
   uvicorn main:app --reload --port 8000
   ```
3. Open **http://localhost:8000**. Use the main page for market analysis; scroll to **Document Q&A** to upload files and ask questions. ChromaDB data is stored in `./chroma_db`; the DB in `./analyzer.db` (or your `DATABASE_URL`).
