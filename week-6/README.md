# Production Document Q&A System

A production-ready retrieval-augmented generation (RAG) system for teams who need to search and query their documents in natural language. Users upload PDFs and text files into named collection groups, ask questions within a single document, across a whole collection, or across all documents, and receive answers grounded in retrieved passages with source attribution and confidence scoring. It is built for deployability (e.g. Railway + PostgreSQL), clear retrieval quality (sentence-aware chunking, token budgeting, keyword re-ranking), and real business use—legal, HR, and financial workflows.

**Live demo:** [Your Railway URL]

---

## Technical Improvements Over Week 5 (Portfolio Story)

Each improvement is framed as a problem discovered in the earlier design, the solution implemented in Week 6, and the result.

| Problem | Solution | Result |
|--------|----------|--------|
| Fixed word-count chunking created mid-sentence cuts, hurting retrieval quality. | Sentence-aware chunking with configurable target size, minimum size, and overlap (in sentences). | Zero mid-sentence cuts; measurably better retrieval and context coherence. |
| All documents were isolated; users couldn’t search related files together. | Named collection groups with metadata-filtered search (Chroma collection = group; multiple docs per group via `source_file`). | Clients can organize documents by project or topic and query “this document,” “whole collection,” or “all documents.” |
| Cross-document queries could exceed model context limits silently. | Token budgeting with explicit caps and logging (e.g. `[RAG] collection=X \| chunks=Y/Z \| tokens=~N \| cost=$...`). | No silent failures; visible context utilization and cost per request. |
| Vector similarity alone missed exact technical terms (e.g. clause numbers, tickers). | Keyword-overlap re-ranking on top of vector results. | More accurate results for domain-specific and exact-match-style queries. |

---

## Tech Stack

| Library / Component | Role |
|---------------------|------|
| **FastAPI** | REST API, file upload, CORS, static frontend, dependency injection. |
| **SQLAlchemy** | ORM and migrations; `Document`, `QAHistory`, `CollectionGroup` tables. |
| **PostgreSQL** (e.g. Railway) | Persistent storage for document metadata, collection groups, Q&A history. |
| **ChromaDB** | Vector store for document chunks; one Chroma collection per collection group. |
| **Voyage AI** (voyage-2) | 1024-d embeddings for chunks and queries. |
| **Anthropic Claude** | Answer generation from retrieved context (Haiku/Sonnet). |
| **PyPDF2** | PDF text extraction. |
| **smart_chunker** (custom) | Sentence-aware chunking with overlap; used by RAG pipeline. |
| **python-dotenv** | Environment variables (API keys, `DATABASE_URL`, `PORT`). |
| **yfinance / requests** | Market data for the bundled Financial Market Analyzer. |
| **Frontend** | Vanilla HTML/CSS/JS: collection tabs, upload dropdown, search-scope toggle, source attributions. |

---

## Setup

1. **Clone and enter the project:**
   ```bash
   cd week-6
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Environment variables** (e.g. `.env` in repo root or in `week-6`):
   - `ANTHROPIC_API_KEY` — Claude API.
   - `VOYAGE_API_KEY` — Voyage embeddings.
   - `DATABASE_URL` — PostgreSQL connection string (or omit for SQLite in dev).
   - `PORT` — Optional; default 8000.

4. **Run the app:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```
   Open `http://localhost:8000`. Use the Document Q&A section to create collections, upload files, and ask questions with “This document only,” “Whole collection,” or “All documents.”

5. **Railway (or similar):** Set root or `--app-dir` to `week-6`, add PostgreSQL and the env vars above, and deploy. ChromaDB data is ephemeral per container; for persistent vectors, consider a hosted vector DB or pgvector.

---

## Business Use Cases

1. **Law firm — contract search**  
   Associates upload contracts into collection groups by matter or client (e.g. “Acme_M&A”, “Employment_Agreements”). They ask “What are the termination clauses?” within one contract, across a matter, or across all agreements. Source attributions (e.g. `acme_nda.pdf (Acme_M&A)`) link back to the exact document for review.

2. **HR department — policy Q&A**  
   HR uploads handbooks, codes of conduct, and benefit docs into collections like “HR_Documents” and “Policies”. Employees ask “What is the remote work policy?” or “How many PTO days in year one?” and get answers with citations. “Whole collection” or “All documents” covers cross-policy questions without opening multiple PDFs.

3. **Financial analyst — earnings and reports**  
   An analyst uploads quarterly earnings PDFs and reports into collections (e.g. “Q3_Reports”, “Company_A”). They ask “What was said about margins in Q3?” or “Compare capex guidance across these reports” using “All documents.” Token budgeting keeps answers within context limits; re-ranking helps with exact figures and ticker mentions.
