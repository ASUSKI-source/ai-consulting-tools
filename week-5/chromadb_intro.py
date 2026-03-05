"""
ChromaDB Introduction Script

Demonstrates a complete workflow:
- Create a persistent ChromaDB collection
- Store financial document chunks with custom embeddings
- Run semantic search queries using SentenceTransformer embeddings
"""

import chromadb
from sentence_transformers import SentenceTransformer


def main() -> None:
    # Load the embedding model once and reuse it for all documents and queries.
    # This is the same MiniLM model used in the embeddings intro script.
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model loaded.\n")

    # Initialize a persistent ChromaDB client.
    # PersistentClient stores data on disk under ./chroma_db so that
    # your collections and embeddings survive script restarts.
    print("Initializing ChromaDB PersistentClient at './chroma_db'...")
    client = chromadb.PersistentClient(path="./chroma_db")

    # Start from a clean slate for this demo by deleting the collection if it exists.
    collection_name = "financial_docs"
    try:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection '{collection_name}' for a clean run.")
    except Exception:
        print(f"No existing collection named '{collection_name}' to delete.")

    collection = client.create_collection(collection_name)
    print(f"Created collection '{collection_name}'.\n")

    # SECTION 1 — STORE DOCUMENTS
    print("=" * 80)
    print("SECTION 1 — STORING DOCUMENT CHUNKS")
    print("=" * 80)

    # Eight short financial document chunks that we will embed and store.
    documents = [
        "Q3 revenue increased 12% year over year to $4.2 billion driven by strong iPhone sales and services growth",
        "The company repurchased $25 billion in shares during the quarter and increased the dividend by 4%",
        "Operating margin improved to 31% from 28% in the prior year period reflecting operational efficiency gains",
        "Bitcoin mining difficulty reached an all time high as network hash rate surpassed 500 exahashes per second",
        "The Federal Reserve signaled two additional rate cuts in 2025 citing progress on inflation targets",
        "Ethereum completed its transition to proof of stake reducing energy consumption by approximately 99 percent",
        "Gold prices fell 2% as the dollar strengthened following better than expected jobs data",
        "The company issued $5 billion in corporate bonds at a 4.8% coupon rate to fund the acquisition",
    ]

    print("Encoding document chunks into embeddings...")
    embeddings = model.encode(documents)
    print(f"Embeddings shape for documents: {embeddings.shape} (chunks, dimensions)\n")

    # Add all documents to ChromaDB at once.
    # We store:
    # - raw text in 'documents'
    # - numeric embeddings as a Python list of lists
    # - stable string IDs (chunk_0, chunk_1, ...)
    # - simple metadata so we can see where each chunk came from
    collection.add(
        documents=documents,
        embeddings=embeddings.tolist(),
        ids=[f"chunk_{i}" for i in range(len(documents))],
        metadatas=[{"source": "test_doc", "chunk_index": i} for i in range(len(documents))],
    )
    print(f"Stored {len(documents)} document chunks in ChromaDB.\n")

    # SECTION 2 — SEARCH
    print("=" * 80)
    print("SECTION 2 — SEMANTIC SEARCH QUERIES")
    print("=" * 80)
    print("Distances measure how far each document embedding is from the query embedding.")
    print("Lower distance ⇒ more similar meaning. Higher distance (e.g. > 1.5) ⇒ likely low relevance.\n")

    queries = [
        "What were the company earnings results?",
        "Tell me about cryptocurrency network activity",
        "What happened with interest rates?",
        "What is the best pizza recipe?",  # no good match expected
    ]

    for query in queries:
        print("-" * 80)
        print(f"QUERY: {query}")

        # Embed the query using the same model so that query and document
        # vectors live in the same semantic space.
        query_embedding = model.encode([query])[0].tolist()

        # Ask ChromaDB for the 3 closest document chunks by distance.
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
        )

        # Results are grouped per query; we only have one query so we use index 0.
        docs = results.get("documents", [[]])[0]
        dists = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        print("\nTop 3 results (lower distance = closer semantic match):")
        for rank, (doc, dist, doc_id, meta) in enumerate(
            zip(docs, dists, ids, metadatas),
            start=1,
        ):
            low_relevance_flag = " [LOW RELEVANCE]" if dist > 1.5 else ""
            chunk_index = meta.get("chunk_index")
            source = meta.get("source")

            print(f"{rank}. id={doc_id}, distance={dist:.4f}{low_relevance_flag}")
            print(f"   source={source}, chunk_index={chunk_index}")
            print(f"   text: {doc}")

        print()  # blank line between queries

    print("=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print(
        "Remember: embeddings turn text into high-dimensional vectors, and ChromaDB\n"
        "stores those vectors so you can run fast semantic search over your documents."
    )


if __name__ == "__main__":
    main()

