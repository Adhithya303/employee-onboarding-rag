"""
Stage 3: Interactive CPU-only semantic retrieval demo for the
Tulane University Faculty Onboarding Assistant.

Important Notes:
- This script loads the pre-built FAISS index and chunk metadata from `data/index/`.
  It does NOT rebuild the index on query.
- Cosine similarity is a nearest-neighbor RETRIEVAL SCORE, NOT an answer confidence score.
  Nearest-neighbor search always returns the top-k closest chunks in vector space even when
  a question is unrelated to the Tulane University Faculty Handbook.
- No LLM answer generation or arbitrary similarity thresholding is performed at this stage.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from backend.build_index import MODEL_ID, load_index_and_metadata
    from backend.retriever import HandbookRetriever
except ImportError:
    from build_index import MODEL_ID, load_index_and_metadata  # type: ignore
    from retriever import HandbookRetriever  # type: ignore

TOP_K = 5


def retrieve_top_k(
    query: str,
    model: SentenceTransformer,
    index: Any,
    metadata: list[dict[str, Any]],
    top_k: int = TOP_K,
) -> list[dict[str, Any]]:
    """Encode the query on CPU with unit normalization and retrieve the top-k chunks via FAISS."""
    retriever = HandbookRetriever(
        model=model,
        index=index,
        metadata=metadata,
        manifest={"model_id": MODEL_ID, "num_chunks": len(metadata), "embedding_dimension": index.d},
    )
    return retriever.retrieve(query=query, top_k=top_k, max_context_tokens=None)


def print_retrieval_results(query: str, results: list[dict[str, Any]]) -> None:
    """Print formatted top-k retrieval results with explicit retrieval-score labeling."""
    print("\n" + "=" * 88)
    print(f"QUERY: {query}")
    print(
        "NOTE: Cosine similarity is a nearest-neighbor RETRIEVAL SCORE (vector similarity),\n"
        "      NOT an answer confidence score or proof that the handbook answers the query."
    )
    print("=" * 88)

    for item in results:
        pages_str = ", ".join(str(p) for p in item["pdf_pages"])
        heading_str = item["heading"] if item["heading"] else "None"
        print(
            f"\n[Rank #{item['rank']}] "
            f"Retrieval Score (Cosine Similarity): {item['retrieval_cosine_similarity']:.4f}"
        )
        print(f"  Document Title   : {item['document_title']}")
        print(f"  Document Section : {item['document_section']}")
        print(f"  PDF Page(s)      : {pages_str}")
        print(f"  Detected Heading : {heading_str}")
        print(f"  Chunk ID         : {item['chunk_id']} ({item['token_count']} content tokens)")
        print("  Chunk Text       :")
        for line in item["text"].split("\n"):
            print(f"    {line}")
        print("-" * 88)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tulane University Faculty Onboarding Assistant - CPU Retrieval Demo"
    )
    parser.add_argument(
        "--query",
        "-q",
        action="append",
        help="Optional query to run non-interactively (can be passed multiple times). "
        "If omitted, runs in interactive mode until 'exit'.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"Number of chunks to retrieve (default: {TOP_K}).",
    )
    args = parser.parse_args()

    try:
        index, metadata, manifest = load_index_and_metadata()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print("Tulane University Faculty Onboarding Assistant — Retrieval Demo (CPU)")
    print(
        f"Loaded FAISS index ({index.ntotal} chunks, dim={index.d}) "
        f"built with '{manifest['model_id']}' at {manifest['build_timestamp']}."
    )
    print("Policy Scope: Tulane University Faculty Handbook & Appended Research Misconduct Policy.")
    print("              (Faculty policies only — never present as general staff policies.)")
    print(f"Loading embedding model '{MODEL_ID}' on CPU...")
    model = SentenceTransformer(MODEL_ID, device="cpu")

    if args.query:
        for q in args.query:
            if not q.strip():
                continue
            results = retrieve_top_k(q, model, index, metadata, top_k=args.top_k)
            print_retrieval_results(q, results)
        return 0

    print("\nEnter a faculty onboarding / policy question, or type 'exit' to quit.")
    while True:
        try:
            user_input = input("\nQuestion (type 'exit' to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting retrieval demo.")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Exiting retrieval demo.")
            break

        results = retrieve_top_k(user_input, model, index, metadata, top_k=args.top_k)
        print_retrieval_results(user_input, results)

    return 0


if __name__ == "__main__":
    sys.exit(main())

