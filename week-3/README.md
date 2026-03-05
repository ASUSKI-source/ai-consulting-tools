# Week 3 — Market Analyzer (Web API + Frontend)

**What it does:** The same market analysis engine as Week 2 (stocks, crypto, technical indicators, Claude) exposed as a **FastAPI** backend with a **web frontend**. Users can analyze stocks or crypto, compare two assets, and see results in the browser. No database yet—each request is stateless.

**Technical skills demonstrated:** FastAPI, REST API design, CORS, static file serving, Pydantic request/response models, frontend (HTML/CSS/JS) calling APIs, Claude API integration, yfinance & CoinGecko pipelines.

---

## Run locally

1. Create a `.env` in this folder (or repo root) with:
   ```bash
   ANTHROPIC_API_KEY=your_key_here
   ```
2. From the `week-3` folder, create a venv and install deps:
   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```
3. Start the server:
   ```bash
   uvicorn main:app --reload --port 8000
   ```
4. Open **http://localhost:8000** in your browser.
