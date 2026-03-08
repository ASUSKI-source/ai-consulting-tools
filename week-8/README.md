# Week 7: Trustworthy RAG

Week 7 transforms a working RAG system into a **trustworthy** one. The core insight: users adopt AI tools when they understand what the AI is doing and how confident it is.

This document describes four design decisions and the reasoning behind each.

---

## 1. Confidence scoring

**Decision:** Show a 0–100 score derived from ChromaDB retrieval distances.

**Reasoning:** Users need to know when to verify an answer vs trust it. A single “answer” with no signal leaves them guessing. A score and label (e.g. “High Confidence” or “Very low — answer may be unreliable”) set the right expectation and encourage checking sources when confidence is low.

**Implementation:** Average distance across the top search results is converted to a normalized 0–100 score (`(1 - avg_distance/2) * 100`). The score drives a color-coded bar (green ≥75, yellow ≥50, orange ≥25, red &lt;25) and a short hint (e.g. “Strong match — the answer is well-supported by the document”). See `rag_pipeline.calculate_confidence()` and the Document Q&A confidence UI in `static/app.js` and `static/index.html`.

---

## 2. Categorized error messages

**Decision:** Map exception types and message content to specific, user-facing messages and suggestions.

**Reasoning:** “Something went wrong” is useless. “The ticker AAPL was not found — check the symbol” is actionable. Users and support both benefit when errors explain what happened and what to do next (e.g. “Wait 30 seconds and try again” for rate limits, “Check your API key” for auth failures).

**Implementation:** `error_messages.py` defines an `ErrorCategory` enum (e.g. `NOT_FOUND`, `RATE_LIMIT`, `INVALID_INPUT`) and `get_user_message(error, context)`, which inspects the error and returns a dict with `user_message`, `suggestion`, and `category`. All API error responses use this dict so the frontend can show a clear message and optional suggestion. HTTP status codes are derived from category (e.g. 404 for NOT_FOUND, 429 for RATE_LIMIT).

---

## 3. Phased loading states

**Decision:** Show explicit phases during Document Q&A: “Searching document…”, “Selecting best passages…”, “Generating answer with Claude…”.

**Reasoning:** Long waits (e.g. 10–15 seconds) feel shorter when users see progress instead of a single spinner. Phases communicate what the system is doing and reduce the sense of a “black box.” Stock and crypto flows keep their simpler loading because those calls are shorter and more predictable.

**Implementation:** A `showLoadingPhases()` function in `static/app.js` drives the shared loading div. Each phase has a duration; the last phase has duration 0 so the final message stays until the request completes. Phases advance with `setTimeout`; `hideLoadingPhases()` clears the timeout and hides the loader when the answer (or error) returns.

---

## 4. Database migrations with Alembic

**Decision:** Use versioned migrations (Alembic) instead of drop-and-recreate or ad-hoc `CREATE TABLE` on startup.

**Reasoning:** Production databases contain real data that cannot be lost. Recreating tables on deploy would wipe analyses, documents, and watchlist data. Migrations allow schema changes (new columns, indexes, tables) to be applied safely and repeatably across dev, staging, and production.

**Implementation:** `alembic upgrade head` runs on every deployment (e.g. via Procfile or release command). Migrations live in `alembic/versions/` and are applied in order. `alembic/env.py` loads `DATABASE_URL` from the environment (with a SQLite default locally and a `postgres://` → `postgresql://` fix for Railway), so the same migration flow works for SQLite in development and PostgreSQL in production.
