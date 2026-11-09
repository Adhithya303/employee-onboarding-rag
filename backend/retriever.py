"""
Reusable CPU-only retrieval module for the Tulane University Faculty Onboarding Assistant.

Loads the cached SentenceTransformer embedding model and pre-built FAISS index once per session,
enforces index compatibility and staleness checks, retrieves candidate chunks, deduplicates by
chunk_id, and enforces a bounded token context budget.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from backend.build_index import (
        FAISS_INDEX_PATH,
        MANIFEST_JSON_PATH,
        METADATA_JSON_PATH,
        MODEL_ID,
        load_index_and_metadata,
    )
    from backend.preprocess_chunk import CHUNKS_JSON_PATH, PAGES_JSON_PATH
except ImportError:
    from build_index import (  # type: ignore
        FAISS_INDEX_PATH,
        MANIFEST_JSON_PATH,
        METADATA_JSON_PATH,
        MODEL_ID,
        load_index_and_metadata,
    )
    from preprocess_chunk import CHUNKS_JSON_PATH, PAGES_JSON_PATH  # type: ignore

DEFAULT_RAG_CANDIDATES = 8
DEFAULT_MAX_CONTEXT_TOKENS = 1500


def make_short_excerpt(text: str, max_chars: int = 220) -> str:
    """Create a clean single-line excerpt for citation display."""
    compact = " ".join(line.strip() for line in text.strip().splitlines() if line.strip())
    if len(compact) <= max_chars:
        return compact
    clipped = compact[: max_chars - 3].rsplit(" ", 1)[0]
    return f"{clipped}..."


class HandbookRetriever:
    """
    Session-scoped CPU retriever that loads the FAISS index and SentenceTransformer model once.
    Never rebuilds the index on query.
    """

    def __init__(
        self,
        model: SentenceTransformer | None = None,
        index: Any | None = None,
        metadata: list[dict[str, Any]] | None = None,
        manifest: dict[str, Any] | None = None,
    ) -> None:
        if index is None or metadata is None or manifest is None:
            index, metadata, manifest = load_index_and_metadata(
                faiss_index_path=FAISS_INDEX_PATH,
                metadata_json_path=METADATA_JSON_PATH,
                manifest_json_path=MANIFEST_JSON_PATH,
                pages_json_path=PAGES_JSON_PATH,
                chunks_json_path=CHUNKS_JSON_PATH,
                expected_model_id=MODEL_ID,
            )
        self.index = index
        self.metadata = metadata
        self.manifest = manifest
        self.model = model if model is not None else SentenceTransformer(MODEL_ID, device="cpu")

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_RAG_CANDIDATES,
        max_context_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve up to `top_k` unique chunks ordered by cosine similarity (inner product of
        L2-normalized embeddings). Deduplicates by `chunk_id` and optionally caps total content
        tokens to `max_context_tokens`.

        Note: `retrieval_cosine_similarity` is a vector retrieval score, never answer confidence.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        query_vec = self.model.encode(
            [cleaned_query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_vec = np.asarray(query_vec, dtype=np.float32)

        # Fetch slightly more than top_k from FAISS in case of any duplicate chunk_ids
        search_k = min(max(top_k * 2, top_k), self.index.ntotal)
        scores, indices = self.index.search(query_vec, search_k)

        results: list[dict[str, Any]] = []
        seen_chunk_ids: set[str] = set()
        accumulated_tokens = 0

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            chunk = self.metadata[int(idx)]
            cid = chunk["chunk_id"]
            if cid in seen_chunk_ids:
                continue

            chunk_tokens = int(chunk.get("token_count", 0))
            if (
                max_context_tokens is not None
                and results
                and (accumulated_tokens + chunk_tokens > max_context_tokens)
            ):
                continue

            seen_chunk_ids.add(cid)
            accumulated_tokens += chunk_tokens
            rank = len(results) + 1
            label = f"S{rank}"

            results.append(
                {
                    "rank": rank,
                    "evidence_label": label,
                    "retrieval_cosine_similarity": float(score),
                    "chunk_id": cid,
                    "document_title": chunk["document_title"],
                    "document_section": chunk["document_section"],
                    "pdf_pages": list(chunk["pdf_pages"]),
                    "heading": chunk.get("heading"),
                    "token_count": chunk_tokens,
                    "text": chunk["text"],
                    "excerpt": make_short_excerpt(chunk["text"]),
                }
            )

            if len(results) >= top_k:
                break

        return results

    def retrieve_for_rag(
        self,
        query: str,
        max_candidates: int = DEFAULT_RAG_CANDIDATES,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    ) -> list[dict[str, Any]]:
        """Retrieve up to 8 deduplicated candidate chunks within a bounded context token budget."""
        return self.retrieve(
            query=query,
            top_k=max_candidates,
            max_context_tokens=max_context_tokens,
        )

