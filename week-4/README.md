# Week 4 — Market Analyzer + Database Persistence

**What it does:** Builds on Week 3’s web app by adding **persistence**. Analyses are saved to a database (SQLite locally or PostgreSQL on Railway). Users get an **analysis history** (load past analyses, filter by ticker/type), **notes** on saved analyses, and a **watchlist**. Same stock/crypto/compare and Claude analysis as before, now with full CRUD and relational data.

**Technical skills demonstrated:** SQLAlchemy ORM, PostgreSQL (or SQLite), database design (analyses, watchlist), FastAPI dependency injection (`get_db`), REST endpoints for history and watchlist, frontend state and list UIs.

---

## Run locally

1. Create a `.env` in this folder (or repo root) with:
   ```bash
   ANTHROPIC_API_KEY=your_key_here
   ```
   For local dev you can omit `DATABASE_URL`; the app uses SQLite (`./analyzer.db`) by default.

2. From the `week-4` folder:
   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   uvicorn main:app --reload --port 8000
   ```
3. Open **http://localhost:8000**. Tables are created on first run.
