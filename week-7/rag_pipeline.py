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
- smart_chunker: splits PDF/text at paragraph/sentence boundaries (no mid-sentence cuts)
- Voyage AI: turns text into 1024-d vectors (voyage-2; document vs query input_type)
- ChromaDB: stores vectors, runs nearest-neighbor search
- Anthropic Claude: generates the final answer from retrieved context

All public functions have Google-style docstrings; inline comments explain
non-obvious steps so a junior developer can follow the pipeline from this file alone.
"""

from __future__ import annotations

import os
import difflib
from typing import Any, Dict, List, Optional

import chromadb
import voyageai
from smart_chunker import chunk_file_smart
import anthropic
from dotenv import load_dotenv


# Load environment variables (ANTHROPIC_API_KEY, VOYAGE_API_KEY, etc.)
load_dotenv()

# Claude model used for generating answers. Haiku is fast and cost-effective.
CURRENT_MODEL = "claude-haiku-4-5"

# Voyage-2 produces 1024-dimensional embeddings. ChromaDB infers dimension from
# the first add(); collections created with the old 384-d model must be
# re-indexed (index_document deletes and recreates the collection, so this is automatic).
VOYAGE_EMBEDDING_DIMENSION = 1024
VOYAGE_MODEL = "voyage-2"

# Token budgeting for RAG context (avoids exceeding model context window).
MAX_CONTEXT_TOKENS = 8000  # Max tokens to use for retrieved context
TOKENS_PER_WORD = 1.3  # Approximate tokens per word (conservative estimate)

#
# Global client singletons
# -------------------------
# Voyage client is lightweight (API-based); ChromaDB keeps a connection.
# We create them once at import and reuse for every RAG operation.
#
VOYAGE_CLIENT = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

# PersistentClient writes to disk (./chroma_db) so data survives restarts.
# Contrast with EphemeralClient, which is in-memory only.
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
CHROMA_CLIENT = chromadb.PersistentClient(path=CHROMA_DIR)
print(f"ChromaDB persistent storage: {CHROMA_DIR}")

ANTHROPIC_CLIENT = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)


def get_or_create_collection(collection_name: str):
    """Return an existing ChromaDB collection or create a new one.

    A "collection" in ChromaDB is like a table: it holds one set of document
    chunks and their embeddings. This helper is used when you want to add to
    or query a collection without caring whether it already exists.

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


# Common stopwords to exclude from keyword overlap in reranking.
_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "that", "this", "from", "have", "been",
        "are", "was", "were", "will", "would", "could", "should", "about",
        "into", "through", "during", "before", "after", "above", "below",
        "between", "under", "again", "further", "then", "once", "here", "there",
        "when", "where", "why", "how", "all", "each", "both", "few", "more",
        "most", "other", "some", "such", "only", "same", "than", "too", "just",
    }
)


def rerank_results(
    query: str,
    results: List[Dict[str, Any]],
    keyword_weight: float = 0.3,
) -> List[Dict[str, Any]]:
    """Re-sort results by semantic score plus keyword overlap bonus.

    For each result: base_score = 1 - (distance/2); keyword_bonus from
    important query words (len > 4, not stopwords) appearing in chunk;
    final_score = base_score + keyword_bonus. Results are sorted by
    final_score descending and each gets a 'final_score' key.
    """
    if not results:
        return results

    # Important words: length > 4, not stopwords (case-insensitive).
    words = query.split()
    important_words = [
        w for w in words
        if len(w) > 4 and w.lower() not in _STOPWORDS
    ]
    n_important = max(len(important_words), 1)

    scored: List[Dict[str, Any]] = []
    for r in results:
        base_score = 1.0 - (r.get("distance", 0) / 2.0)
        base_score = max(0.0, min(1.0, base_score))

        chunk_text = (r.get("text") or "").lower()
        matches = sum(1 for w in important_words if w.lower() in chunk_text)
        keyword_bonus = (matches / n_important) * keyword_weight
        final_score = base_score + keyword_bonus

        r = dict(r)
        r["final_score"] = final_score
        scored.append(r)

    scored.sort(key=lambda x: x["final_score"], reverse=True)
    return scored


def estimate_tokens(text: str) -> int:
    """Approximate token count for a string.

    Uses word count * TOKENS_PER_WORD. Real tokenization is model-specific:
    punctuation and subword tokenization mean many words are split into
    multiple tokens, so ~1.3 tokens per word is a conservative estimate
    that errs on the high side for budgeting.
    """
    return int(len(text.split()) * TOKENS_PER_WORD)


def budget_context(
    search_results: List[Dict[str, Any]],
    max_tokens: int = MAX_CONTEXT_TOKENS,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Trim search results to fit within a token budget (best-first).

    Adds results one by one until adding the next would exceed max_tokens.
    Returns the trimmed list and a stats dict for logging.
    """
    if not search_results:
        return [], {
            "results_included": 0,
            "results_excluded": 0,
            "estimated_tokens": 0,
            "budget_used_pct": 0.0,
        }
    included: List[Dict[str, Any]] = []
    total_tokens = 0
    for r in search_results:
        text = r.get("text", "")
        tokens = estimate_tokens(text)
        if total_tokens + tokens > max_tokens:
            break
        included.append(r)
        total_tokens += tokens
    n_included = len(included)
    n_excluded = len(search_results) - n_included
    budget_used_pct = (total_tokens / max_tokens * 100.0) if max_tokens else 0.0
    return included, {
        "results_included": n_included,
        "results_excluded": n_excluded,
        "estimated_tokens": total_tokens,
        "budget_used_pct": budget_used_pct,
    }


def index_document(file_path: str, collection_group: str = "default") -> Dict[str, Any]:
    """Chunk a document file, embed the chunks, and store them in ChromaDB.

    This is the "indexing" step of RAG: we turn a PDF or text file into
    many small overlapping chunks, convert each chunk to a vector (embedding),
    and save those vectors in ChromaDB so we can later find the most relevant
    chunks for any question.

    A *collection group* is a logical bucket of related documents that all
    share a single ChromaDB collection. Each document in the group is
    distinguished by its source_file metadata.

    Args:
        file_path: Path to a .txt, .pdf, or .md file (see smart_chunker.chunk_file_smart).
        collection_group: Logical group name; this becomes the ChromaDB
            collection name. All documents in the same group share one
            collection.

    Returns:
        Dict with keys: collection_name, collection_group, chunks_indexed (int),
        source_file.

    Raises:
        FileNotFoundError: If file_path does not exist.
        ValueError: If file type is not supported (raised by chunk_file_smart).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")

    # Collection name in ChromaDB is the logical collection_group.
    collection_name = collection_group
    filename = os.path.basename(file_path)

    # Get or create the shared collection for this group without
    # deleting other documents that belong to the same group.
    collection = get_or_create_collection(collection_name)

    # Remove any existing chunks for this specific document from the group.
    collection.delete(where={"source_file": filename})

    # chunk_file_smart() reads the file, handles .txt/.pdf/.md, and returns
    # a list of dicts with "text" and "metadata" (word_count, char_count,
    # starts_with, etc.) for debugging and threshold_test output.
    chunks_with_meta: List[Dict[str, Any]] = chunk_file_smart(file_path)
    if not chunks_with_meta:
        return {
            "collection_name": collection_name,
            "collection_group": collection_group,
            "chunks_indexed": 0,
            "source_file": filename,
        }

    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for i, item in enumerate(chunks_with_meta):
        text = item.get("text", "")
        metadata = item.get("metadata", {}) or {}
        if not text.strip():
            continue

        texts.append(text)

        # Preserve smart_chunker metadata (word_count, char_count, starts_with),
        # and ensure required fields for collection groups are present.
        enriched = dict(metadata)
        enriched["source_file"] = filename
        enriched["collection_name"] = collection_name
        enriched["collection_group"] = collection_group
        enriched["chunk_index"] = i
        # Ensure word_count is present even if upstream changes.
        if "word_count" not in enriched:
            enriched["word_count"] = len(text.split())
        metadatas.append(enriched)

    # Single batch embed via Voyage AI (document input_type for indexing).
    result = VOYAGE_CLIENT.embed(
        texts,
        model=VOYAGE_MODEL,
        input_type="document",
    )
    embeddings = result.embeddings

    # ChromaDB requires a unique string id per document (chunk_0, chunk_1, ...).
    ids = [f"{filename}chunk_{i}" for i in range(len(texts))]

    collection.add(
        documents=texts,
        embeddings=embeddings,
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
    source_file: Optional[str] = None,
) -> tuple[List[Dict[str, Any]], str]:
    """Run semantic search: find the most relevant document chunks for a query.

    We embed the query with the same model used for indexing so that query
    and document vectors live in the same space. ChromaDB returns the
    nearest neighbors by distance (lower = more similar). We then filter
    out any result whose distance is above the threshold (low relevance)
    and attach a simple 0–1 relevance score for the UI.

    Args:
        query: The user's question or search phrase.
        collection_name: Which indexed document collection (group) to search.
        n_results: Maximum number of chunks to return (top-k).
        distance_threshold: Chunks with distance above this are excluded.
            Default 1.4 works well for L2 distance with our embedding model;
            tune down for stricter relevance, up for more recall.
        source_file: Optional filename filter. If provided, restricts search
            to chunks whose metadata.source_file matches this value.

    Returns:
        Tuple of (list of result dicts, top_chunk_before_rerank_snippet).
        Each result has text, source_file, distance, chunk_index,
        relevance_score, final_score (after rerank). If no results,
        returns ([], "").
    """
    if not query.strip():
        return [], ""

    collection = _get_collection_or_raise(collection_name)

    # Embed query with same model as chunks; use input_type="query" for search.
    result = VOYAGE_CLIENT.embed(
        [query],
        model=VOYAGE_MODEL,
        input_type="query",
    )
    query_embedding = result.embeddings[0]

    where = {"source_file": source_file} if source_file else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
        where=where,
    )

    # Guard against empty results (small collection, no chunks found).
    if not results.get("documents") or not results["documents"][0]:
        return [], ""

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
                "source_file": (meta or {}).get("source_file", "unknown"),
                "distance": float(dist),
                "chunk_index": (meta or {}).get("chunk_index", 0),
                "relevance_score": relevance_score,
            }
        )

    # Capture top chunk before rerank for logging.
    top_before_rerank = ""
    if filtered:
        top_before_rerank = (filtered[0].get("text") or "").replace("\n", " ").strip()[:40]

    reranked = rerank_results(query, filtered)
    return reranked, top_before_rerank


def search_all_collections(
    query: str,
    n_results_per_collection: int = 3,
    distance_threshold: float = 1.4,
) -> List[Dict[str, Any]]:
    """Search across all ChromaDB collections (groups) for a query.

    This helper fans out the search to every existing collection, merges
    the results, deduplicates highly similar chunks, and returns a
    distance-sorted list.

    Args:
        query: The user's question or search phrase.
        n_results_per_collection: How many top chunks to retrieve from each
            collection before merging.
        distance_threshold: Passed through to search_documents().

    Returns:
        List of result dicts (same shape as search_documents) with at most
        n_results_per_collection * 2 items overall. Each result's metadata
        includes a collection_group key so callers can see which group
        (collection) it came from.
    """
    all_results: List[Dict[str, Any]] = []

    collections_summary = list_collections()
    if not collections_summary or not query.strip():
        return []

    def is_duplicate(existing: List[Dict[str, Any]], candidate_text: str) -> bool:
        """Return True if candidate_text is >90% similar to any existing text."""
        for r in existing:
            existing_text = r.get("text", "")
            if not existing_text:
                continue
            ratio = difflib.SequenceMatcher(None, candidate_text, existing_text).ratio()
            if ratio >= 0.9:
                return True
        return False

    for coll in collections_summary:
        coll_name = coll["name"]
        per_coll_results, _ = search_documents(
            query=query,
            collection_name=coll_name,
            n_results=n_results_per_collection,
            distance_threshold=distance_threshold,
        )

        for r in per_coll_results:
            text = r.get("text", "")
            if not text or is_duplicate(all_results, text):
                continue

            # Add collection_group at top level for downstream attribution.
            r = dict(r)
            r["collection_group"] = coll_name
            all_results.append(r)

    # Sort globally by distance (ascending = more similar).
    all_results.sort(key=lambda r: r.get("distance", float("inf")))

    limit = n_results_per_collection * 2
    return all_results[:limit]


def build_rag_context(
    search_results: List[Dict[str, Any]],
    max_tokens: int = MAX_CONTEXT_TOKENS,
) -> tuple[str, Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Format search results into a single context string for the LLM.

    Applies token budgeting first, then builds the context from the
    budgeted list. The LLM is instructed to answer only from this context
    and to cite passages by number.

    Args:
        search_results: List of dicts from search_documents(), each with
            at least a "text" key (best-first).
        max_tokens: Token budget for the context (default MAX_CONTEXT_TOKENS).

    Returns:
        Tuple of (context_string, budget_info_dict, budgeted_results).
        If search_results is empty, returns (fallback_message, None, []).
    """
    if not search_results:
        return (
            "CONTEXT FROM DOCUMENT:\n\n(No relevant passages found.)",
            None,
            [],
        )

    budgeted_results, budget_info = budget_context(search_results, max_tokens=max_tokens)
    if not budgeted_results:
        return (
            "CONTEXT FROM DOCUMENT:\n\n(No relevant passages found.)",
            budget_info,
            [],
        )

    lines: List[str] = ["CONTEXT FROM DOCUMENT:", ""]
    for idx, result in enumerate(budgeted_results, start=1):
        lines.append(f"[Passage {idx}]")
        lines.append(result["text"])
        lines.append("")  # blank line between passages for readability

    return "\n".join(lines), budget_info, budgeted_results


def calculate_confidence(search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute a 0–100 confidence score and label from search result distances.

    Uses average and best distance from the (best-first) search results to
    produce a score, label, and color for the UI. Empty results return
    a fixed "No Match" confidence.

    Args:
        search_results: List of result dicts from search_documents(), each
            with at least a "distance" key (sorted best-first).

    Returns:
        Dict with: score (0–100), label (str), color (str),
        results_count, best_distance, avg_distance.
    """
    if not search_results:
        return {
            "score": 0,
            "label": "No Match",
            "color": "red",
            "results_count": 0,
            "best_distance": 0.0,
            "avg_distance": 0.0,
        }

    avg_distance = sum(r["distance"] for r in search_results) / len(search_results)
    best_distance = search_results[0]["distance"]

    # Convert to 0-100 score
    score = max(0, min(100, round((1 - avg_distance / 2) * 100)))

    # Label based on score
    if score >= 75:
        label, color = "High Confidence", "green"
    elif score >= 50:
        label, color = "Moderate Confidence", "yellow"
    elif score >= 25:
        label, color = "Low Confidence", "orange"
    else:
        label, color = "Very Low — Answer May Be Unreliable", "red"

    return {
        "score": score,
        "label": label,
        "color": color,
        "results_count": len(search_results),
        "best_distance": round(best_distance, 3),
        "avg_distance": round(avg_distance, 3),
    }


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
    source_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Answer a question using only the content of an indexed document (RAG).

    This is the main RAG workflow: (1) search for relevant chunks, (2) if
    none pass the relevance threshold, return a "no context" message without
    calling Claude; (3) otherwise build a context string and prompt Claude
    to answer only from that context, then return the answer plus metadata.

    Args:
        question: The user's question in natural language.
        collection_name: Name of the ChromaDB collection/group (from index_document).
        n_results: How many top chunks to retrieve and pass to the LLM (default 4).
        source_file: Optional filename to restrict search to a single document
            within the collection group.

    Returns:
        Dict with: answer (str), sources (list of dicts with text,
        source_file, distance, chunk_index), found_relevant_context (bool),
        input_tokens, output_tokens, estimated_cost (float). If no relevant
        context, answer is a fallback message and sources is empty.
    """
    search_results, top_before_rerank = search_documents(
        query=question,
        collection_name=collection_name,
        n_results=n_results,
        source_file=source_file,
    )

    # No chunks passed the distance threshold: don't call Claude.
    # Return a fixed message so the UI can show "low confidence" or similar.
    if not search_results:
        confidence = calculate_confidence(search_results)
        return {
            "answer": (
                "I could not find relevant information in the document "
                "to answer this question."
            ),
            "sources": [],
            "sources_text": [],
            "found_relevant_context": False,
            "confidence": confidence,
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost": 0.0,
        }

    context, budget_info, budgeted_results = build_rag_context(search_results)
    num_retrieved = len(search_results)
    num_included = len(budgeted_results)

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

    # Return the budgeted source dicts (what we actually sent as context).
    sources = budgeted_results
    sources_text = [r.get("text", "") for r in budgeted_results]
    confidence = calculate_confidence(search_results)

    # One-line summary for development and debugging.
    est_tokens = budget_info.get("estimated_tokens", 0) if budget_info else 0
    top_after_rerank = (
        (search_results[0].get("text") or "").replace("\n", " ").strip()[:40]
        if search_results else ""
    )
    print(
        f"[RAG] collection={collection_name} | chunks={num_included}/{num_retrieved} | "
        f"tokens=~{est_tokens} | cost=${estimated_cost:.4f}"
    )
    print(
        f"      | top_chunk_before_rerank: {top_before_rerank!r}"
    )
    print(
        f"      | top_chunk_after_rerank: {top_after_rerank!r}"
    )

    return {
        "answer": answer_text,
        "sources": sources,
        "sources_text": sources_text,
        "found_relevant_context": True,
        "confidence": confidence,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost": estimated_cost,
    }


def ask_across_collections(
    question: str,
    n_results_per: int = 3,
    distance_threshold: float = 1.4,
) -> Dict[str, Any]:
    """Answer a question using chunks drawn from all indexed collections.

    This variant of ask_document() searches every ChromaDB collection (each
    representing a collection_group), merges the best chunks, and builds a
    context that clearly labels which collection and source file each
    passage came from.

    Args:
        question: The user's question in natural language.
        n_results_per: How many top chunks to pull from each collection
            before merging (default 3).
        distance_threshold: Maximum distance to accept when searching.

    Returns:
        Dict with: answer (str), sources (list of result dicts including
        metadata with collection_group and source_file), found_relevant_context
        (bool), input_tokens, output_tokens, estimated_cost (float).
    """
    search_results = search_all_collections(
        query=question,
        n_results_per_collection=n_results_per,
        distance_threshold=distance_threshold,
    )

    if not search_results:
        return {
            "answer": (
                "I could not find relevant information in any indexed "
                "collection to answer this question."
            ),
            "sources": [],
            "found_relevant_context": False,
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost": 0.0,
        }

    # Build a context that labels each passage with its collection_group
    # and source_file so the model can attribute correctly.
    lines: List[str] = ["CONTEXT FROM MULTIPLE DOCUMENTS:", ""]

    for idx, result in enumerate(search_results, start=1):
        collection_group = result.get("collection_group", "unknown_group")
        source_file = result.get("source_file", "unknown_file")
        lines.append(f"[Passage {idx}] (Collection: {collection_group}, File: {source_file})")
        lines.append(result.get("text", ""))
        lines.append("")

    context = "\n".join(lines)

    system_prompt = (
        "You are a precise document assistant. You are given passages drawn "
        "from multiple document collections. Answer questions using ONLY "
        "these passages. If the answer is not clearly present, say exactly: "
        "I cannot find the answer to this question in the provided documents. "
        "Always cite which passage(s) and which collection/file your answer "
        "comes from."
    )

    user_prompt = f"{context}\n\nQuestion: {question}"

    response = ANTHROPIC_CLIENT.messages.create(
        model=CURRENT_MODEL,
        max_tokens=800,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
    )

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

    return {
        "answer": answer_text,
        "sources": search_results,
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
            text = src.get("text", "") if isinstance(src, dict) else str(src)
            preview = text.replace("\n", " ")[:160]
            fname = src.get("source_file", "") if isinstance(src, dict) else ""
            label = f" (from {fname})" if fname else ""
            print(f"[Passage {i}]{label} {preview}...")

        print(
            "\nCost estimate: "
            f"${result['estimated_cost']:.6f} "
            f"(input_tokens={result['input_tokens']}, "
            f"output_tokens={result['output_tokens']})"
        )

