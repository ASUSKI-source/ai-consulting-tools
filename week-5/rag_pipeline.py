"""
Core Retrieval-Augmented Generation (RAG) pipeline.

This module wires together:
- Our local text chunker
- ChromaDB for vector storage + search
- SentenceTransformer for embeddings
- Claude via the anthropic client for final answers

Interviewers: this file is intentionally verbose and documented to show
how a production-style RAG flow is structured end-to-end.
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

CURRENT_MODEL = "claude-haiku-4-5"
#
# Global model + client singletons
# --------------------------------
# These are expensive to construct, so we do it once at import time and
# reuse them across all RAG operations.
#
print("Loading embedding model...")
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded.")

CHROMA_CLIENT = chromadb.PersistentClient(path="./chroma_db")

ANTHROPIC_CLIENT = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)


def get_or_create_collection(collection_name: str):
    """
    Return an existing ChromaDB collection or create a new one.

    This helper prevents errors when indexing into a collection name that may
    or may not already exist. For indexing flows that need a clean slate,
    index_document will explicitly delete before recreating.
    """
    existing_collections = CHROMA_CLIENT.list_collections()
    for coll in existing_collections:
        if coll.name == collection_name:
            return coll
    return CHROMA_CLIENT.create_collection(collection_name)


def index_document(file_path: str, collection_name: str | None = None) -> Dict[str, Any]:
    """
    Chunk a document file, embed the chunks, and store them in ChromaDB.

    - If collection_name is None, we derive it from the filename (without
      extension), e.g. 'company_policy.pdf' -> 'company_policy'.
    - If a collection with that name already exists, it is deleted and
      recreated so that re-indexing fully replaces old content.
    - Chunks are embedded in a single batch for performance.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")

    if collection_name is None:
        base = os.path.basename(file_path)
        collection_name = os.path.splitext(base)[0]

    # Start from a clean collection for this document so we do not mix
    # stale and fresh content for the same logical source.
    try:
        CHROMA_CLIENT.delete_collection(collection_name)
    except Exception:
        # It's fine if the collection does not exist yet.
        pass

    collection = CHROMA_CLIENT.create_collection(collection_name)

    # Chunk the file into overlapping passages with metadata.
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
        # Record which logical collection this chunk belongs to so we can
        # inspect or debug later.
        enriched = dict(metadata)
        enriched["collection_name"] = collection_name
        metadatas.append(enriched)

    # Batch-encode all chunks in one go; this is much faster than encoding
    # each passage individually.
    embeddings = EMBEDDING_MODEL.encode(texts)

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
    """
    Retrieve a collection by name, raising a clear error if it does not exist.
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
    """
    Run a semantic search for the query against a specific collection.

    - Embeds the query with the shared embedding model.
    - Queries ChromaDB for the top-k nearest passages.
    - Filters out low-relevance hits based on a distance threshold.
    - Returns a list of results with text, metadata, distance, and
      a simple normalized relevance score in [0, 1].
    """
    if not query.strip():
        return []

    collection = _get_collection_or_raise(collection_name)

    query_embedding = EMBEDDING_MODEL.encode([query])[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    filtered: List[Dict[str, Any]] = []

    for text, meta, dist in zip(docs, metadatas, distances):
        # Some backends may return None; we defensively skip those.
        if dist is None:
            continue

        if dist > distance_threshold:
            # Consider this low relevance and skip it from downstream RAG.
            continue

        # A simple relevance score: smaller distance => higher score.
        # We map distances into [0, 1] with a linear heuristic
        # relevance = 1 - dist / 2, then clamp.
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
    """
    Turn search results into a structured context string for the LLM.

    The format is:

    CONTEXT FROM DOCUMENT:

    [Passage 1]
    <text>

    [Passage 2]
    <text>

    This makes it easy for the model to cite which passage(s) it used.
    """
    if not search_results:
        return "CONTEXT FROM DOCUMENT:\n\n(No relevant passages found.)"

    lines: List[str] = ["CONTEXT FROM DOCUMENT:", ""]

    for idx, result in enumerate(search_results, start=1):
        lines.append(f"[Passage {idx}]")
        lines.append(result["text"])
        lines.append("")  # blank line between passages

    return "\n".join(lines)


def _estimate_claude_cost(input_tokens: int, output_tokens: int) -> float:
    """
    Rough cost estimate for Claude 3.5 Sonnet usage.

    Pricing changes over time; these constants are based on public pricing
    at the time of writing and are easy to adjust later.
    """
    # Dollars per million tokens (approximate).
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
    """
    High-level RAG entry point: answer a question about a specific document.

    Steps:
    1. Retrieve the top-k most relevant passages via semantic search.
    2. If none are relevant, return a graceful "no context" answer.
    3. Build a strict system + user prompt that forces Claude to ground
       answers in the provided context only.
    4. Call Claude and return the answer, sources, and cost estimate.
    """
    search_results = search_documents(
        query=question,
        collection_name=collection_name,
        n_results=n_results,
    )

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

    system_prompt = (
        "You are a precise document assistant. Answer questions using ONLY "
        "the context passages provided below. Do not use any outside "
        "knowledge. If the answer is not clearly present in the context, "
        "say exactly: I cannot find the answer to this question in the "
        "provided document. Always cite which passage(s) your answer comes from."
    )

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

    # Extract plain text from Claude's structured content response.
    answer_parts: List[str] = []
    for block in response.content:
        if hasattr(block, "type") and block.type == "text":
            answer_parts.append(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            # For robustness in case of dict-like blocks.
            answer_parts.append(block.get("text", ""))

    answer_text = "\n".join(part for part in answer_parts if part.strip())

    usage = response.usage
    input_tokens = getattr(usage, "input_tokens", 0)
    output_tokens = getattr(usage, "output_tokens", 0)
    estimated_cost = _estimate_claude_cost(input_tokens, output_tokens)

    # For transparency, we expose the raw passage texts used as sources.
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
    """
    Return a summary of all collections in the ChromaDB client.

    Each entry includes:
    - name: collection name
    - count: total number of stored chunks
    """
    collections = CHROMA_CLIENT.list_collections()
    summary: List[Dict[str, Any]] = []

    for coll in collections:
        try:
            count = coll.count()
        except Exception:
            # In case of backend issues, we still want to list the name.
            count = -1
        summary.append({"name": coll.name, "count": count})

    return summary


if __name__ == "__main__":
    #
    # End-to-end smoke test for the RAG pipeline.
    #
    # We:
    # 1. Create a small synthetic Q3 earnings-style document.
    # 2. Save it as test_doc.txt.
    # 3. Index it into ChromaDB.
    # 4. Ask three questions: clearly answerable, partially answerable, and
    #    completely unanswerable, printing answers and cost for each.
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

