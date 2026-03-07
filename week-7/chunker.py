"""
Text chunking utilities for production-style document processing.

Two main entry points:
- chunk_text: split a large string into overlapping word chunks
- chunk_file: load .txt or .pdf and then apply chunk_text
"""

from __future__ import annotations

import os
from typing import List, Tuple, Dict

import PyPDF2


def chunk_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 50,
    min_chunk_words: int = 50,
) -> List[str]:
    """
    Split text into overlapping chunks of approximately `chunk_size` words.

    Each consecutive chunk shares `overlap` words with the previous chunk.
    Very short trailing fragments (fewer than `min_chunk_words` words) are
    skipped to avoid low-quality context.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if overlap < 0:
        raise ValueError("overlap must be non-negative.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")
    if min_chunk_words <= 0:
        raise ValueError("min_chunk_words must be a positive integer.")

    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    step = chunk_size - overlap

    # We advance by `step` words each time, but include `overlap` words from
    # the previous chunk. This keeps semantic context across boundaries so
    # that queries that span a boundary still find relevant information.
    start = 0
    total_words = len(words)

    while start < total_words:
        end = start + chunk_size
        chunk_words = words[start:end]

        # If the remaining fragment is too small, discard it instead of
        # creating a tiny low-signal chunk at the end.
        if len(chunk_words) < min_chunk_words:
            # If this is the very first chunk and even it is too small,
            # we return an empty list.
            if not chunks:
                return []
            break

        chunks.append(" ".join(chunk_words))
        start += step

    return chunks


def _read_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _read_pdf(file_path: str) -> str:
    # Extract text from all pages of a PDF using PyPDF2.
    text_parts: List[str] = []
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts)


def chunk_file(
    file_path: str,
    chunk_size: int = 400,
    overlap: int = 50,
) -> List[Tuple[str, Dict[str, int | str]]]:
    """
    Load a file (.txt or .pdf), chunk its text, and attach metadata.

    Returns:
        List of (chunk_text, metadata) tuples where metadata includes:
        - source_file: original file path
        - chunk_index: index of this chunk (0-based)
        - total_chunks: total number of chunks for this file
        - word_count: number of words in this chunk
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".txt":
        content = _read_txt(file_path)
    elif ext == ".pdf":
        content = _read_pdf(file_path)
    else:
        raise ValueError(
            f"Unsupported file type: {ext}. Supported types are: .txt, .pdf"
        )

    chunks = chunk_text(content, chunk_size=chunk_size, overlap=overlap)
    total_chunks = len(chunks)

    results: List[Tuple[str, Dict[str, int | str]]] = []
    for idx, chunk in enumerate(chunks):
        word_count = len(chunk.split())
        metadata: Dict[str, int | str] = {
            "source_file": file_path,
            "chunk_index": idx,
            "total_chunks": total_chunks,
            "word_count": word_count,
        }
        results.append((chunk, metadata))

    return results


if __name__ == "__main__":
    # Sample long-form text for demonstration.
    # In a real application this would come from an earnings call transcript,
    # 10-K filing, or research report. Here we generate a realistic-style
    # narrative of at least ~600 words about financial markets and earnings.
    sample_text = """
    The third quarter marked a period of solid execution for the company
    against a complex macroeconomic backdrop. Revenue grew at a healthy pace
    as demand remained resilient across both consumer and enterprise
    segments, even while customers continued to exercise greater scrutiny
    over large discretionary projects. Management highlighted that the
    company entered the quarter with a cautious outlook, but ongoing
    strength in core subscription services and better-than-expected hardware
    sell-through enabled performance to land firmly at the upper end of
    guidance.

    Total revenue increased at a double-digit rate, driven primarily by
    strength in the premium smartphone lineup, continued expansion in
    wearables, and ongoing momentum in payments and cloud-based services.
    The company noted that the installed base of active devices reached a
    new all-time high, which provides a durable foundation for recurring
    revenue streams. Services once again outpaced the broader business,
    benefiting from higher engagement in digital media, app marketplace
    activity, and financial services offerings. Management emphasized that
    the flywheel between hardware and services remains a core competitive
    advantage, as each new device sold typically drives multiple new service
    subscriptions over time.

    On the profitability front, gross margins expanded modestly year over
    year, reflecting a more favorable product mix, easing component costs,
    and early benefits from ongoing supply-chain optimization initiatives.
    Operating expenses grew at a measured pace as the company continued to
    invest in research and development, particularly in areas related to
    artificial intelligence, on-device machine learning, and custom silicon.
    At the same time, management maintained disciplined control over sales
    and marketing spend, reallocating dollars toward the highest-return
    channels and away from lower-yield experimental campaigns. As a result,
    operating income grew faster than revenue, and operating margin
    improved, underscoring the inherent leverage in the business model.

    The company also provided detailed commentary on broader financial
    markets. Management acknowledged that volatility in equity indices,
    persistent uncertainty around the pace and timing of central bank rate
    cuts, and mixed economic data have all contributed to a more cautious
    sentiment among both institutional and retail investors. However, they
    pointed to signs of stabilization in credit spreads, improving liquidity
    conditions in commercial paper markets, and renewed issuance in the
    investment-grade bond segment as indicators that risk appetite is
    gradually normalizing. Within this context, the firm successfully
    executed a multi-billion-dollar debt issuance at attractive coupon
    rates, extending its weighted-average maturity profile while preserving
    balance-sheet flexibility.

    Foreign exchange remained a headwind during the quarter, as the
    sustained strength of the U.S. dollar weighed on reported results in
    several key international markets. Management estimated that currency
    movements reduced total revenue growth by several percentage points and
    clipped gross margin expansion by a modest amount. To mitigate this
    impact, the company has continued to refine its hedging program and
    adjust local pricing where appropriate, while remaining mindful of
    consumer purchasing power and competitive dynamics in each region.

    Looking ahead, executives reiterated their commitment to a disciplined
    capital-allocation framework that balances reinvestment in the business,
    strategic acquisitions, shareholder returns, and the maintenance of a
    strong liquidity position. The board authorized an additional share
    repurchase program and approved a modest increase in the quarterly
    dividend, reflecting confidence in the durability of cash flows. At the
    same time, management cautioned that they will continue to monitor the
    macro backdrop closely, including labor-market trends, inflation
    readings, and developments in monetary policy across major economies.

    Overall, the quarter reinforced the view that the company is well
    positioned to navigate a wide range of economic scenarios. A diversified
    revenue base, strong balance sheet, and ongoing innovation pipeline in
    both hardware and services provide meaningful resilience. While
    management does not attempt to forecast short-term market movements, they
    remain focused on delivering products and experiences that deepen
    customer loyalty, expand ecosystem engagement, and create long-term
    shareholder value, regardless of near-term fluctuations in financial
    markets.
    """

    # Chunk the sample text using default parameters.
    chunks = chunk_text(sample_text)

    print(f"Total chunks created: {len(chunks)}")
    if not chunks:
        raise SystemExit("No chunks were created from the sample text.")

    first_chunk = chunks[0]
    last_chunk = chunks[-1]

    # Preview first and last chunks.
    print("\nFirst chunk preview (first 100 characters):")
    print(first_chunk[:100].replace("\n", " "))

    print("\nLast chunk preview (first 100 characters):")
    print(last_chunk[:100].replace("\n", " "))

    # Demonstrate why overlap matters by showing that the boundary between
    # chunk 0 and chunk 1 shares context. The last 20 words of chunk 0
    # should overlap with the first 20 words of chunk 1.
    if len(chunks) > 1:
        c0_words = chunks[0].split()
        c1_words = chunks[1].split()

        last_20_c0 = " ".join(c0_words[-20:])
        first_20_c1 = " ".join(c1_words[:20])

        print("\nLast 20 words of chunk 0:")
        print(last_20_c0)

        print("\nFirst 20 words of chunk 1:")
        print(first_20_c1)

        print(
            "\nNotice that these sequences overlap. This overlap keeps sentences "
            "and ideas intact across chunk boundaries, "
            "so downstream search or QA systems do not lose important context "
            "just because it falls near the edge of a chunk."
        )

