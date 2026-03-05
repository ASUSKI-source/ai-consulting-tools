"""
Core Retrieval-Augmented Generation (RAG) pipeline.

RAG = retrieve relevant chunks from your documents, then ask an LLM to
answer using only those chunks. This avoids hallucination and keeps
answers grounded in your data.

Flow for a typical "ask a question" request:
  1. index_document(file_path)  -> chunks file, embeds chunks, stores in ChromaDB
  2. ask_document(question, collection_name)  -> search_documents() finds top
     chunks, build_rag_context() formats them, Claude generates answer from
     that context only

This module wires together:
- chunker: splits PDF/text into overlapping chunks
- SentenceTransformer: turns text into 384-d vectors (same model for docs and query)
- ChromaDB: stores vectors, runs nearest-neighbor search
- Anthropic Claude: generates the final answer from retrieved context

All public functions have Google-style docstrings; inline comments explain
non-obvious steps so a junior developer can follow the pipeline from this file alone.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import chromadb
from sentence_transformers import SentenceTransformer
from chunker import chunk_file, chunk_text
import anthropic
from dotenv import load_dotenv


# Load environment variables (e.g., ANTHROPIC_API_KEY from .env)
load_dotenv()

# Claude model used for generating answers. Haiku is fast and cost-effective.
CURRENT_MODEL = "claude-haiku-4-5"

#
# Global model + client singletons
# --------------------------------
# These are expensive to construct (embedding model loads ~80MB, ChromaDB
# opens a connection), so we create them once at import time and reuse
# them for every RAG operation instead of re-loading per request.
#
print("Loading embedding model...")
# all-MiniLM-L6-v2 produces 384-dimensional vectors; same model must be
# used for both indexing and querying so distances are comparable.
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded.")

# PersistentClient writes to disk (./chroma_db) so data survives restarts.
# Contrast with EphemeralClient, which is in-memory only.
CHROMA_CLIENT = chromadb.PersistentClient(path="./chroma_db")

ANTHROPIC_CLIENT = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)


def get_or_create_collection(collection_name: str):
    """Return an existing ChromaDB collection or create a new one.

    A "collection" in ChromaDB is like a table: it holds one set of document
    chunks and their embeddings. This helper is used when you want to add to
    or query a collection without caring whether it already exists. For
    full re-indexing, index_document() instead deletes and recreates the
    collection so there is no stale data.

    Args:
        collection_name: Unique name for the collection (e.g. "company_policy").

    Returns:
        The ChromaDB collection object (existing or newly created).
    """
    existing_collections = CHROMA_CLIENT.list_collections()
    for coll in existing_collections:
        if coll.name == collection_name:
            return coll
    # No match: create a new empty collection with this name.
    return CHROMA_CLIENT.create_collection(collection_name)


def index_document(file_path: str, collection_name: str | None = None) -> Dict[str, Any]:
    """Chunk a document file, embed the chunks, and store them in ChromaDB.

    This is the "indexing" step of RAG: we turn a PDF or text file into
    many small overlapping chunks, convert each chunk to a vector (embedding),
    and save those vectors in ChromaDB so we can later find the most relevant
    chunks for any question. If you re-call this with the same collection
    name, the old collection is deleted first so you never have duplicate
    or stale chunks.

    Args:
        file_path: Path to a .txt or .pdf file (see chunker.chunk_file).
        collection_name: Optional. If None, derived from filename without
            extension (e.g. "report.pdf" -> "report").

    Returns:
        Dict with keys: collection_name, chunks_indexed (int), source_file.

    Raises:
        FileNotFoundError: If file_path does not exist.
        ValueError: If file type is not supported (raised by chunk_file).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")

    if collection_name is None:
        base = os.path.basename(file_path)
        collection_name = os.path.splitext(base)[0]

    # Delete existing collection if present. This way re-uploading the same
    # document fully replaces old chunks instead of appending duplicates.
    try:
        CHROMA_CLIENT.delete_collection(collection_name)
    except Exception:
        # Collection may not exist yet on first index; ignore.
        pass

    collection = CHROMA_CLIENT.create_collection(collection_name)

    # chunk_file() reads the file, splits by .txt/.pdf, and returns
    # (chunk_text, metadata) tuples with e.g. source_file, chunk_index.
    chunks_with_meta: List[Tuple[str, Dict[str, Any]]] = chunk_file(file_path)
    if not chunks_with_meta:
        return {
            "collection_name": collection_name,
            "chunks_indexed": 0,
            "source_file": os.path.basename(file_path),
        }

    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for text, metadata in chunks_with_meta:
        texts.append(text)
        # Add collection_name to metadata so we can filter or debug by
        # collection later without storing it separately.
        enriched = dict(metadata)
        enriched["collection_name"] = collection_name
        metadatas.append(enriched)

    # Single batch encode: one GPU/CPU call for all chunks. Much faster
    # than encoding one chunk at a time in a loop.
    embeddings = EMBEDDING_MODEL.encode(texts)

    # ChromaDB requires a unique string id per document (chunk_0, chunk_1, ...).
    ids = [f"chunk_{i}" for i in range(len(texts))]

    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        ids=ids,
        metadatas=metadatas,
    )

    return {
        "collection_name": collection_name,
        "chunks_indexed": len(texts),
        "source_file": os.path.basename(file_path),
    }


def _get_collection_or_raise(collection_name: str):
    """Return a ChromaDB collection by name, or raise if it does not exist.

    Used internally by search_documents and ask_document so we fail fast with
    a clear message instead of a cryptic ChromaDB error.

    Args:
        collection_name: Name of the collection to fetch.

    Returns:
        The ChromaDB collection object.

    Raises:
        ValueError: If no collection with that name exists.
    """
    existing = {c.name: c for c in CHROMA_CLIENT.list_collections()}
    if collection_name not in existing:
        raise ValueError(
            f"Collection '{collection_name}' does not exist. "
            "Index a document first using index_document()."
        )
    return existing[collection_name]


def search_documents(
    query: str,
    collection_name: str,
    n_results: int = 4,
    distance_threshold: float = 1.4,
) -> List[Dict[str, Any]]:
    """Run semantic search: find the most relevant document chunks for a query.

    We embed the query with the same model used for indexing so that query
    and document vectors live in the same space. ChromaDB returns the
    nearest neighbors by distance (lower = more similar). We then filter
    out any result whose distance is above the threshold (low relevance)
    and attach a simple 0–1 relevance score for the UI.

    Args:
        query: The user's question or search phrase.
        collection_name: Which indexed document collection to search.
        n_results: Maximum number of chunks to return (top-k).
        distance_threshold: Chunks with distance above this are excluded.
            Default 1.4 works well for L2 distance with our embedding model;
            tune down for stricter relevance, up for more recall.

    Returns:
        List of dicts, each with keys: text, metadata, distance,
        relevance_score. Empty list if query is blank or no results pass
        the threshold.
    """
    if not query.strip():
        return []

    collection = _get_collection_or_raise(collection_name)

    # Encode query the same way we encoded chunks: same model, same dimensions.
    # encode([query]) returns shape (1, 384); we take [0] and convert to list.
    query_embedding = EMBEDDING_MODEL.encode([query])[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    # ChromaDB returns dict of lists-of-lists: one inner list per query.
    # We sent one query, so we take index [0] to get the first (and only) set.
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    filtered: List[Dict[str, Any]] = []

    for text, meta, dist in zip(docs, metadatas, distances):
        if dist is None:
            continue

        # Drop results that are too far from the query (not relevant enough).
        if dist > distance_threshold:
            continue

        # Turn distance into a 0–1 score for UI: closer => higher score.
        # Heuristic: score = 1 - dist/2, clamped. Not calibrated; just for display.
        raw_score = 1.0 - dist / 2.0
        relevance_score = max(0.0, min(1.0, raw_score))

        filtered.append(
            {
                "text": text,
                "metadata": meta,
                "distance": float(dist),
                "relevance_score": relevance_score,
            }
        )

    return filtered


def build_rag_context(search_results: List[Dict[str, Any]]) -> str:
    """Format search results into a single context string for the LLM.

    The LLM is instructed to answer only from this context and to cite
    passages by number. By labeling each chunk as [Passage 1], [Passage 2],
    etc., we make it easy for the model to say "According to Passage 2, ..."
    and for us to map that back to the actual source text.

    Args:
        search_results: List of dicts from search_documents(), each with
            at least a "text" key.

    Returns:
        A single string: "CONTEXT FROM DOCUMENT:" followed by numbered
        passages and their text, or a fallback message if search_results
        is empty.
    """
    if not search_results:
        return "CONTEXT FROM DOCUMENT:\n\n(No relevant passages found.)"

    lines: List[str] = ["CONTEXT FROM DOCUMENT:", ""]

    for idx, result in enumerate(search_results, start=1):
        lines.append(f"[Passage {idx}]")
        lines.append(result["text"])
        lines.append("")  # blank line between passages for readability

    return "\n".join(lines)


def _estimate_claude_cost(input_tokens: int, output_tokens: int) -> float:
    """Estimate the cost in USD for a Claude API call.

    Uses approximate per-million-token rates. Update the constants when
    Anthropic changes pricing or when switching to a different model.

    Args:
        input_tokens: Number of tokens in the request (system + user message).
        output_tokens: Number of tokens in the model's response.

    Returns:
        Estimated cost in dollars (e.g. 0.00234).
    """
    # Dollars per million tokens; adjust for your model and region.
    INPUT_PER_MILLION = 3.0
    OUTPUT_PER_MILLION = 15.0

    input_cost = (input_tokens / 1_000_000) * INPUT_PER_MILLION
    output_cost = (output_tokens / 1_000_000) * OUTPUT_PER_MILLION
    return input_cost + output_cost


def ask_document(
    question: str,
    collection_name: str,
    n_results: int = 4,
) -> Dict[str, Any]:
    """Answer a question using only the content of an indexed document (RAG).

    This is the main RAG workflow: (1) search for relevant chunks, (2) if
    none pass the relevance threshold, return a "no context" message without
    calling Claude; (3) otherwise build a context string and prompt Claude
    to answer only from that context, then return the answer plus metadata.

    Args:
        question: The user's question in natural language.
        collection_name: Name of the ChromaDB collection (from index_document).
        n_results: How many top chunks to retrieve and pass to the LLM (default 4).

    Returns:
        Dict with: answer (str), sources (list of passage texts),
        found_relevant_context (bool), input_tokens, output_tokens,
        estimated_cost (float). If no relevant context, answer is a
        fallback message and sources is empty.
    """
    search_results = search_documents(
        query=question,
        collection_name=collection_name,
        n_results=n_results,
    )

    # No chunks passed the distance threshold: don't call Claude.
    # Return a fixed message so the UI can show "low confidence" or similar.
    if not search_results:
        return {
            "answer": (
                "I could not find relevant information in the document "
                "to answer this question."
            ),
            "sources": [],
            "found_relevant_context": False,
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost": 0.0,
        }

    context = build_rag_context(search_results)

    # Strict system prompt: answer ONLY from context. Reduces hallucination
    # and keeps answers grounded in the actual document.
    system_prompt = (
        "You are a precise document assistant. Answer questions using ONLY "
        "the context passages provided below. Do not use any outside "
        "knowledge. If the answer is not clearly present in the context, "
        "say exactly: I cannot find the answer to this question in the "
        "provided document. Always cite which passage(s) your answer comes from."
    )

    # User message = context + question. Claude sees the passages and the question.
    user_prompt = f"{context}\n\nQuestion: {question}"

    response = ANTHROPIC_CLIENT.messages.create(
        model=CURRENT_MODEL,
        max_tokens=500,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
    )

    # response.content is a list of blocks (e.g. text, tool_use). We only
    # want the text blocks; the API returns objects with .type and .text.
    answer_parts: List[str] = []
    for block in response.content:
        if hasattr(block, "type") and block.type == "text":
            answer_parts.append(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            answer_parts.append(block.get("text", ""))

    answer_text = "\n".join(part for part in answer_parts if part.strip())

    usage = response.usage
    input_tokens = getattr(usage, "input_tokens", 0)
    output_tokens = getattr(usage, "output_tokens", 0)
    estimated_cost = _estimate_claude_cost(input_tokens, output_tokens)

    # Return the exact passage texts we sent as context so the UI can
    # show "Sources used" or let the user expand them.
    sources = [r["text"] for r in search_results]

    return {
        "answer": answer_text,
        "sources": sources,
        "found_relevant_context": True,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost": estimated_cost,
    }


def list_collections() -> List[Dict[str, Any]]:
    """List all document collections and how many chunks each contains.

    Useful for dashboards or debugging: see which documents have been
    indexed and how large each collection is.

    Returns:
        List of dicts with keys "name" (str) and "count" (int). Count
        is -1 if the collection exists but count could not be read.
    """
    collections = CHROMA_CLIENT.list_collections()
    summary: List[Dict[str, Any]] = []

    for coll in collections:
        try:
            count = coll.count()
        except Exception:
            # Backend might fail on count(); still expose the collection name.
            count = -1
        summary.append({"name": coll.name, "count": count})

    return summary


# ---------------------------------------------------------------------------
# Demo: run this file directly to test the full pipeline (python rag_pipeline.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Write a sample document to disk (simulates a user uploading a file).
    # 2. Index it: chunk -> embed -> store in ChromaDB.
    # 3. Ask three questions (answerable, partial, unanswerable) and print
    #    answers, source previews, and token cost so you can verify behavior.
    #

    test_doc_text = """
    In the third quarter, Fictional Systems Inc. delivered solid financial
    performance despite ongoing macroeconomic uncertainty. Total Q3 revenue
    reached $4.8 billion, an increase of 11 percent year over year, driven
    by continued demand for the company's cloud infrastructure platform and
    strong adoption of its data analytics solutions. Management highlighted
    particularly robust growth in the mid-market customer segment, where
    companies are modernizing legacy workloads and consolidating vendors
    onto more efficient, scalable architectures.

    Subscription and recurring revenue represented 68 percent of total
    revenue in the quarter, up from 63 percent a year ago, underscoring the
    increasing durability and visibility of the business model. Gross margin
    expanded by 120 basis points year over year to 71.2 percent, reflecting
    a richer mix of software and services, as well as ongoing improvements
    in data center utilization. Operating income grew faster than revenue,
    with operating margin increasing to 29.5 percent as the company balanced
    disciplined expense management with targeted investments in research and
    development, go-to-market capacity, and customer success programs.

    The company ended the quarter with $9.3 billion in cash, cash
    equivalents, and short-term investments on the balance sheet and no
    long-term debt. During the period, Fictional Systems Inc. returned
    approximately $600 million to shareholders through a combination of
    share repurchases and a regular quarterly dividend. The board approved a
    new $2.0 billion repurchase authorization, reflecting confidence in the
    long-term outlook and the strength of the company's free-cash-flow
    generation.

    Management described demand trends as stable but noted that some
    customers are taking longer to approve large transformational projects,
    especially in interest-rate-sensitive industries such as financial
    services and real estate. Even so, expansion within the existing
    customer base remained healthy, and churn rates stayed near historic
    lows. The company continued to invest in artificial intelligence
    capabilities that help customers automate workflows, detect anomalies,
    and derive more actionable insights from their data.

    Looking ahead to the fourth quarter, leadership provided guidance for
    mid- to high-single-digit sequential revenue growth, assuming no
    material change in the macro environment. They emphasized that while
    they do not manage the business to any specific quarter, they remain
    focused on executing their long-term strategy: deepening customer
    relationships, expanding the product portfolio, and compounding free
    cash flow over time. The company did not disclose specific headcount
    figures but noted that hiring has become more targeted, with priority
    given to roles in engineering, security, and customer-facing functions.
    """

    test_file_path = "test_doc.txt"
    with open(test_file_path, "w", encoding="utf-8") as f:
        f.write(test_doc_text.strip() + "\n")

    print(f"Created sample document at: {test_file_path}")

    index_info = index_document(test_file_path)
    print(
        f"Indexed {index_info['chunks_indexed']} chunks into collection "
        f"'{index_info['collection_name']}' from file '{index_info['source_file']}'."
    )

    collection_name = index_info["collection_name"]

    test_questions = [
        # Clearly answerable: revenue and year-over-year growth.
        "What was the company's total Q3 revenue and year-over-year growth rate?",
        # Partially answerable: outlook/guidance and management commentary.
        "How did management describe the outlook and guidance for the next quarter?",
        # Unanswerable: details not present in the document.
        "What was the exact employee headcount at the end of Q3?",
    ]

    for q in test_questions:
        print("\n" + "=" * 80)
        print(f"QUESTION: {q}")

        result = ask_document(q, collection_name=collection_name, n_results=4)

        print("\nANSWER:")
        print(result["answer"])

        print("\nSOURCES (first 160 characters of each passage):")
        for i, src in enumerate(result["sources"], start=1):
            preview = src.replace("\n", " ")[:160]
            print(f"[Passage {i}] {preview}...")

        print(
            "\nCost estimate: "
            f"${result['estimated_cost']:.6f} "
            f"(input_tokens={result['input_tokens']}, "
            f"output_tokens={result['output_tokens']})"
        )

