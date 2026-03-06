"""
Threshold calibration tool for RAG distance thresholds.

This script helps you find the optimal distance_threshold for search_documents()
when deploying a new RAG system with different content. Different document types
and embedding models can have very different optimal thresholds. Run this script
every time you add new content or change your embedding model to recalibrate.

Run interactively:
  python threshold_test.py

You'll pick a collection, enter test questions, and see how different thresholds
affect which chunks are returned. Use the feedback to tune search_documents(...,
distance_threshold=X.XX) in your actual queries.
"""

import os
import sys
from typing import Optional

# Add parent directory to path so we can import rag_pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline import VOYAGE_CLIENT, VOYAGE_MODEL, CHROMA_CLIENT, list_collections, _get_collection_or_raise


def classify_distance(distance: float) -> str:
    """Return a relevance label based on distance thresholds.
    
    Args:
        distance: L2 distance from query embedding to document embedding.
    
    Returns:
        A short label string: [RELEVANT], [BORDERLINE], or [LOW].
    """
    if distance < 1.2:
        return "[RELEVANT]  "
    elif distance <= 1.5:
        return "[BORDERLINE]"
    else:
        return "[LOW]       "


def truncate_text(text: str, max_chars: int = 80) -> str:
    """Truncate text to max_chars for display, replacing newlines with spaces.
    
    Args:
        text: Text to truncate.
        max_chars: Maximum number of characters to display.
    
    Returns:
        Truncated text with "..." appended if it was longer than max_chars.
    """
    text = text.replace('\n', ' ').replace('\r', ' ')
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def show_all_results(query: str, collection_name: str) -> list:
    """Query collection with n_results=10 and display all results with distances.
    
    Uses Voyage AI to embed the query, retrieves the top 10 chunks from ChromaDB,
    and prints each result with its distance and a relevance label (RELEVANT,
    BORDERLINE, or LOW). This gives you a full view of what's available so you
    can decide what threshold makes sense.
    
    Args:
        query: User's question or search phrase.
        collection_name: Name of the ChromaDB collection to query.
    
    Returns:
        List of result dicts with keys: text, distance, label.
    """
    collection = _get_collection_or_raise(collection_name)
    
    # Embed the query using Voyage AI with input_type="query"
    result = VOYAGE_CLIENT.embed(
        [query],
        model=VOYAGE_MODEL,
        input_type="query",
    )
    query_embedding = result.embeddings[0]
    
    # Query ChromaDB for top 10 results (more than we'll likely use)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10,
        include=["documents", "distances"],
    )
    
    # Extract docs and distances
    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    
    if not docs:
        print("  → No results found in collection.")
        return []
    
    # Display all results with labels
    all_results = []
    print("\n  All 10 results (sorted by distance):\n")
    for i, (text, dist) in enumerate(zip(docs, distances), start=1):
        label = classify_distance(dist)
        truncated = truncate_text(text)
        print(f"  {i:2d}. Distance: {dist:.2f}  | {label} | {truncated}")
        all_results.append({
            "text": text,
            "distance": dist,
            "label": label,
        })
    
    return all_results


def filter_by_threshold(all_results: list, threshold: float) -> list:
    """Filter results by the given threshold.
    
    Args:
        all_results: List of result dicts from show_all_results().
        threshold: Distance threshold (lower distances = more similar).
    
    Returns:
        Subset of all_results where distance <= threshold.
    """
    filtered = [r for r in all_results if r["distance"] <= threshold]
    return filtered


def main():
    print("\n" + "="*80)
    print("RAG Distance Threshold Calibration Tool")
    print("="*80)
    print("\nThis tool helps you find the optimal distance_threshold for your RAG queries.")
    print("It shows you all top-10 results for a query, labeled by relevance.\n")
    
    # Step 1: List available collections
    collections = list_collections()
    if not collections:
        print("ERROR: No indexed collections found. Index a document first with index_document().")
        return
    
    print("Available collections:")
    for i, coll in enumerate(collections, start=1):
        print(f"  {i}. {coll['name']} ({coll['count']} chunks)")
    
    # Step 2: User picks a collection
    while True:
        try:
            choice = int(input("\nEnter collection number (or 0 to exit): "))
            if choice == 0:
                print("Exiting.")
                return
            if 1 <= choice <= len(collections):
                collection_name = collections[choice - 1]["name"]
                break
            else:
                print("Invalid choice. Try again.")
        except ValueError:
            print("Please enter a number.")
    
    print(f"\nUsing collection: '{collection_name}'")
    
    # Step 3: Test loop — let user query until they exit
    while True:
        query = input("\n\nEnter a test question (or 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            print("Exiting.")
            return
        
        if not query:
            print("Question cannot be empty.")
            continue
        
        print(f"\nQuerying: {query}")
        
        # Get all 10 results with distances
        all_results = show_all_results(query, collection_name)
        
        if not all_results:
            continue
        
        # Step 4: Ask user what threshold they'd prefer
        print("\n  Relevance labels (current thresholds):")
        print("    < 1.2   = RELEVANT")
        print("    1.2-1.5 = BORDERLINE")
        print("    > 1.5   = LOW")
        
        threshold_input = input("\n  Based on these results, what threshold would you set? (e.g. 1.3): ").strip()
        
        try:
            threshold = float(threshold_input)
        except ValueError:
            print("Invalid threshold. Using default 1.4.")
            threshold = 1.4
        
        # Step 5: Show filtered results
        filtered = filter_by_threshold(all_results, threshold)
        print(f"\n  With threshold={threshold}, {len(filtered)} result(s) pass:")
        for r in filtered:
            truncated = truncate_text(r["text"])
            print(f"    Distance: {r['distance']:.2f} | {truncated}")
        
        if not filtered:
            print("    (No results pass this threshold.)")


if __name__ == "__main__":
    main()
