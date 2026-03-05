# Week 1 — Claude API & Text Summarization

**What it does:** Introductory scripts that call the Anthropic Claude API. `summarizer.py` lets you paste or type text and get AI-generated summaries (short, medium, or long). `hello_claude.py` is a minimal “hello world” style call. Output is printed to the console and optionally saved to timestamped files.

**Technical skills demonstrated:** Python, Claude API integration, environment variables (`.env`), prompt design, basic I/O and user input.

---

## Run locally

1. Create a `.env` in this folder (or repo root) with:
   ```bash
   ANTHROPIC_API_KEY=your_key_here
   ```
2. Install dependencies:
   ```bash
   pip install anthropic python-dotenv
   ```
3. From the `week-1` folder:
   ```bash
   python summarizer.py
   ```
   Or run `hello_claude.py` for a simple one-shot API test.
