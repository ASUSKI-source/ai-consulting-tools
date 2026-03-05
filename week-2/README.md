# Week 2 — Market Analyzer (CLI)

**What it does:** A command-line market analyzer that fetches live stock data (yfinance) and crypto data (CoinGecko), computes technical indicators (RSI, SMA, momentum, support/resistance), and sends a formatted prompt to Claude for written analysis. Runs interactively in the terminal; no web server. Good foundation for the later web app.

**Technical skills demonstrated:** REST-style data APIs (yfinance, CoinGecko), technical indicator logic, prompt engineering, Claude API, cost tracking, structured output to files.

---

## Run locally

1. Create a `.env` in this folder (or repo root) with:
   ```bash
   ANTHROPIC_API_KEY=your_key_here
   ```
2. Install dependencies:
   ```bash
   pip install anthropic python-dotenv yfinance requests
   ```
3. From the `week-2` folder:
   ```bash
   python market_analyzer.py
   ```
   Follow the prompts to analyze a stock or crypto asset.
