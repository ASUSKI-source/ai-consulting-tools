# AI Applications Portfolio
## Drew Suski — Full-Stack AI Developer

I'm a full-stack developer building production-ready AI applications for real business use. This repository showcases two end-to-end systems: an **AI Financial Market Analyzer** that pulls live data and delivers Claude-powered technical analysis with persistence and a modern web UI, and a **Document Q&A (RAG) system** that indexes PDFs and text, runs semantic search with ChromaDB, and answers questions with source attribution and confidence scoring. I'm looking for roles where I can ship AI-powered products—from API design and data pipelines to RAG and deployment—and grow with a team that values clear code and user-facing impact.

## Live Demos

| Project | Description | Live URL | Tech Stack |
|--------|-------------|----------|------------|
| AI Financial Market Analyzer | Live stock and crypto analysis with technical indicators, AI summaries, comparison mode, and saved history. | [Railway URL] | FastAPI, Claude AI, yfinance, CoinGecko, PostgreSQL, ChromaDB |
| Document Q&A System | Upload documents, ask questions in natural language, get answers grounded in your files with cited sources. | Coming soon | RAG, ChromaDB, FastAPI, Claude AI |

## Projects

### 1. AI Financial Market Analyzer

A full-stack web app that fetches real-time stock (yfinance) and crypto (CoinGecko) data, computes technical indicators (RSI, SMA, momentum, support/resistance), and returns structured AI analysis via Claude. Users can compare two assets, save analyses to a database, maintain a watchlist, and filter history by ticker and type.

**Key technical features:**
- Real-time data pipeline from yfinance and CoinGecko APIs
- Technical indicators: RSI, SMA(5/20), price momentum, support/resistance
- AI analysis and comparison via Claude with system prompts tuned for market context
- PostgreSQL (or SQLite) persistence for analyses, notes, and watchlist
- Full-stack web interface: search, tabs, history panel, watchlist chips

**Screenshot:** [screenshot]

**Code:** [week-3](week-3/) · [week-4](week-4/)

---

### 2. Document Q&A System (RAG)

A retrieval-augmented generation (RAG) pipeline that lets users upload PDF or text files, chunk and embed them with sentence-transformers, store vectors in ChromaDB, and ask questions in natural language. Claude answers using only retrieved passages, with source attribution and a confidence indicator when context is weak.

**Key technical features:**
- Semantic search with ChromaDB and sentence-transformers (all-MiniLM-L6-v2)
- Overlapping text chunking (configurable size/overlap) for PDF and .txt
- Source attribution: answers cite which passages were used; sources are expandable in the UI
- PDF + text support via PyPDF2 and a unified chunker
- Confidence scoring and cost display per query; Q&A history persisted in PostgreSQL

**Code:** [week-5](week-5/)

## Technical Skills Demonstrated

| Skill | Demonstrated In |
|-------|------------------|
| FastAPI | Both projects: REST API, file upload, dependency injection, CORS |
| SQLAlchemy + PostgreSQL | Analyses, watchlist, documents, QA history tables |
| RAG / ChromaDB | Document Q&A: vector store, semantic search, collection lifecycle |
| Claude AI API | Market analysis prompts, RAG system/user prompts, token usage and cost |
| HTML / CSS / JavaScript | Single-page UI: tabs, history, watchlist, document upload and Q&A panel |
| REST API design | Structured request/response models, error handling, clear endpoints |
| Real-time financial data pipelines | yfinance, CoinGecko, indicator computation, caching considerations |
| Sentence Transformers | Local embeddings for documents and queries (MiniLM) |
| Railway deployment | Production deployment (Market Analyzer); Document Q&A deployable same way |
| Git version control | Repo structure, commits, and project organization |

## About These Projects

These applications were built as part of an intensive, self-directed learning program over several weeks, starting from no professional programming background. The work is grounded in a consulting context: they are real tools designed for real business use cases—market research, document understanding, and decision support—and are built to be deployable, maintainable, and portfolio-ready.

## Contact

- **Email:** drewsuski@gmail.com  
- **LinkedIn:** [linkedin.com/in/drew-suski-129775284](https://www.linkedin.com/in/drew-suski-129775284)
