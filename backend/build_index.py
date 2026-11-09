"""
Stage 2B: CPU-only embedding generation and FAISS IndexFlatIP builder/loader for the
Tulane University Faculty Onboarding Assistant.

Note on Model Caching:
The first execution requires an internet connection to download
`sentence-transformers/all-MiniLM-L6-v2` from Hugging Face into the local cache.
Subsequent runs and searches load the cached model locally on CPU.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from backend.preprocess_chunk import (
        CHUNKS_JSON_PATH,
        DOCUMENT_TITLE,
        EXPECTED_APPENDIX_START_PAGE,
        INGESTION_REPORT_PATH,
        MAX_CONTENT_TOKENS,
        MODEL_ID,
        OVERLAP_TOKENS,
        PAGES_JSON_PATH,
        PROJECT_ROOT,
        SECTION_APPENDED_RMP,
        SECTION_MAIN_HANDBOOK,
        TARGET_CONTENT_TOKENS,
        format_embedding_input,
        run_preprocessing_and_chunking,
    )
except ImportError:
    from preprocess_chunk import (  # type: ignore
        CHUNKS_JSON_PATH,
        DOCUMENT_TITLE,
        EXPECTED_APPENDIX_START_PAGE,
        INGESTION_REPORT_PATH,
        MAX_CONTENT_TOKENS,
        MODEL_ID,
        OVERLAP_TOKENS,
        PAGES_JSON_PATH,
        PROJECT_ROOT,
        SECTION_APPENDED_RMP,
        SECTION_MAIN_HANDBOOK,
        TARGET_CONTENT_TOKENS,
        format_embedding_input,
        run_preprocessing_and_chunking,
    )

INDEX_DIR = PROJECT_ROOT / "data" / "index"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_JSON_PATH = INDEX_DIR / "chunks_metadata.json"
MANIFEST_JSON_PATH = INDEX_DIR / "manifest.json"

ENCODING_BATCH_SIZE = 32
REBUILD_CMD = r".\.venv\Scripts\python backend\build_index.py"


def compute_file_sha256(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()


def validate_chunks_for_indexing(
    chunks: list[dict[str, Any]],
    tokenizer: Any,
    model_max_seq_length: int,
) -> tuple[int, int]:
    """
    Validate all chunks before embedding:
    - Non-empty text and valid 1-based PDF page references
    - Unique chunk_id
    - No chunk crosses the main handbook / appended policy boundary (page 192)
    - Every complete embedding input (including heading prefix and [CLS]/[SEP]) fits within model_max_seq_length
    Returns (max_content_tokens, max_embedding_input_tokens).
    """
    if not chunks:
        raise ValueError("Cannot build index: chunks list is empty.")

    seen_ids: set[str] = set()
    max_content_toks = 0
    max_emb_toks = 0

    try:
        from backend.preprocess_chunk import count_tokens
    except ImportError:
        from preprocess_chunk import count_tokens  # type: ignore

    for idx, chunk in enumerate(chunks):
        cid = chunk.get("chunk_id")
        if not cid or cid in seen_ids:
            raise ValueError(f"Duplicate or missing chunk_id at index {idx}: {cid}")
        seen_ids.add(cid)

        text = str(chunk.get("text", "")).strip()
        if not text:
            raise ValueError(f"Chunk {cid} has empty text.")

        pages = chunk.get("pdf_pages")
        if not isinstance(pages, list) or not pages or not all(isinstance(p, int) and 1 <= p <= 238 for p in pages):
            raise ValueError(f"Chunk {cid} has invalid pdf_pages: {pages}")

        section = chunk.get("document_section")
        if section == SECTION_MAIN_HANDBOOK and any(p >= EXPECTED_APPENDIX_START_PAGE for p in pages):
            raise ValueError(
                f"Chunk {cid} in '{SECTION_MAIN_HANDBOOK}' crosses boundary into appendix pages: {pages}"
            )
        if section == SECTION_APPENDED_RMP and any(p < EXPECTED_APPENDIX_START_PAGE for p in pages):
            raise ValueError(
                f"Chunk {cid} in '{SECTION_APPENDED_RMP}' crosses boundary into main handbook pages: {pages}"
            )
        if any(p < EXPECTED_APPENDIX_START_PAGE for p in pages) and any(
            p >= EXPECTED_APPENDIX_START_PAGE for p in pages
        ):
            raise ValueError(f"Chunk {cid} crosses the handbook/appendix boundary (page 192): {pages}")

        emb_input = format_embedding_input(
            chunk["document_title"],
            chunk["document_section"],
            chunk.get("heading"),
            text,
        )
        content_toks = count_tokens(tokenizer, text, add_special_tokens=False)
        emb_toks = count_tokens(tokenizer, emb_input, add_special_tokens=True)

        if emb_toks > model_max_seq_length:
            raise ValueError(
                f"Chunk {cid} embedding input is {emb_toks} tokens, which exceeds "
                f"model max_seq_length ({model_max_seq_length})."
            )

        max_content_toks = max(max_content_toks, content_toks)
        max_emb_toks = max(max_emb_toks, emb_toks)

    return max_content_toks, max_emb_toks


def build_and_save_index(
    pages_json_path: Path = PAGES_JSON_PATH,
    chunks_json_path: Path = CHUNKS_JSON_PATH,
    ingestion_report_path: Path = INGESTION_REPORT_PATH,
    faiss_index_path: Path = FAISS_INDEX_PATH,
    metadata_json_path: Path = METADATA_JSON_PATH,
    manifest_json_path: Path = MANIFEST_JSON_PATH,
) -> dict[str, Any]:
    """
    Run preprocessing/chunking from pages.json, compute normalized CPU embeddings in batches,
    build a FAISS IndexFlatIP cosine similarity index, and save index + metadata + manifest.
    """
    print(f"Loading SentenceTransformer model '{MODEL_ID}' on CPU...")
    model = SentenceTransformer(MODEL_ID, device="cpu")
    tokenizer = model.tokenizer
    model_max_seq_length = int(model.max_seq_length)
    embedding_dim = int(model.get_embedding_dimension())

    chunks, report = run_preprocessing_and_chunking(
        pages_json_path=pages_json_path,
        chunks_json_path=chunks_json_path,
        ingestion_report_path=ingestion_report_path,
        model=model,
    )

    max_content_toks, max_emb_toks = validate_chunks_for_indexing(
        chunks, tokenizer=tokenizer, model_max_seq_length=model_max_seq_length
    )

    embedding_inputs = [
        format_embedding_input(
            c["document_title"],
            c["document_section"],
            c.get("heading"),
            c["text"],
        )
        for c in chunks
    ]

    print(
        f"Encoding {len(chunks)} chunks on CPU (batch_size={ENCODING_BATCH_SIZE}, "
        f"normalize_embeddings=True)..."
    )
    embeddings = model.encode(
        embedding_inputs,
        batch_size=ENCODING_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.shape != (len(chunks), embedding_dim):
        raise RuntimeError(
            f"Unexpected embeddings shape {embeddings.shape}; expected {(len(chunks), embedding_dim)}."
        )

    # Verify unit-norm normalization for cosine similarity via inner product
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise RuntimeError("Embeddings are not properly L2-normalized for IndexFlatIP cosine search.")

    # Build FAISS IndexFlatIP
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)

    if index.ntotal != len(chunks):
        raise RuntimeError(
            f"FAISS index count ({index.ntotal}) does not match chunks count ({len(chunks)})."
        )

    # Save index and metadata
    faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_index_path))

    with metadata_json_path.open("w", encoding="utf-8") as f_meta:
        json.dump(chunks, f_meta, indent=2, ensure_ascii=False)

    pages_sha256 = compute_file_sha256(pages_json_path)
    chunks_sha256 = compute_file_sha256(chunks_json_path)
    metadata_sha256 = compute_file_sha256(metadata_json_path)
    faiss_sha256 = compute_file_sha256(faiss_index_path)

    manifest = {
        "project": "Tulane University Faculty Onboarding Assistant",
        "document_title": DOCUMENT_TITLE,
        "model_id": MODEL_ID,
        "device": "cpu",
        "index_type": "IndexFlatIP",
        "similarity_metric": "cosine_similarity_via_normalized_inner_product",
        "embedding_dimension": embedding_dim,
        "model_max_seq_length": model_max_seq_length,
        "longest_content_tokens": max_content_toks,
        "longest_embedding_input_tokens": max_emb_toks,
        "num_chunks": len(chunks),
        "indexed_pages_count": report["indexed_pages_count"],
        "skipped_pages_count": report["skipped_pages_count"],
        "skipped_pages": report["skipped_pages"],
        "chunking_settings": {
            "target_content_tokens": TARGET_CONTENT_TOKENS,
            "max_content_tokens": MAX_CONTENT_TOKENS,
            "overlap_tokens": OVERLAP_TOKENS,
            "verified_appendix_boundary_page": EXPECTED_APPENDIX_START_PAGE,
        },
        "input_fingerprint": {
            "pages_json_sha256": pages_sha256,
            "chunks_json_sha256": chunks_sha256,
        },
        "artifacts_fingerprint": {
            "faiss_index_sha256": faiss_sha256,
            "chunks_metadata_sha256": metadata_sha256,
        },
        "build_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    with manifest_json_path.open("w", encoding="utf-8") as f_man:
        json.dump(manifest, f_man, indent=2, ensure_ascii=False)

    return manifest


def load_index_and_metadata(
    faiss_index_path: Path = FAISS_INDEX_PATH,
    metadata_json_path: Path = METADATA_JSON_PATH,
    manifest_json_path: Path = MANIFEST_JSON_PATH,
    pages_json_path: Path = PAGES_JSON_PATH,
    chunks_json_path: Path = CHUNKS_JSON_PATH,
    expected_model_id: str = MODEL_ID,
) -> tuple[faiss.Index, list[dict[str, Any]], dict[str, Any]]:
    """
    Load the saved FAISS index, chunk metadata, and manifest from disk.
    Rejects missing, stale, or incompatible index/metadata combinations with instructions to rebuild.
    """
    missing = [
        str(p)
        for p in (faiss_index_path, metadata_json_path, manifest_json_path)
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Saved vector index files are missing: {missing}\n"
            f"Please build the index first by running:\n  {REBUILD_CMD}"
        )

    try:
        with manifest_json_path.open("r", encoding="utf-8") as f_man:
            manifest = json.load(f_man)
        with metadata_json_path.open("r", encoding="utf-8") as f_meta:
            metadata = json.load(f_meta)
        index = faiss.read_index(str(faiss_index_path))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read index or metadata files: {exc}\n"
            f"Please rebuild the index by running:\n  {REBUILD_CMD}"
        ) from exc

    # 1. Verify model compatibility
    manifest_model = manifest.get("model_id")
    if manifest_model != expected_model_id:
        raise RuntimeError(
            f"Incompatible embedding model in manifest: found '{manifest_model}', "
            f"expected '{expected_model_id}'.\n"
            f"Please rebuild the index by running:\n  {REBUILD_CMD}"
        )

    # 2. Verify counts and dimensions match
    expected_num_chunks = manifest.get("num_chunks")
    expected_dim = manifest.get("embedding_dimension")
    if not isinstance(metadata, list) or index.ntotal != len(metadata) or len(metadata) != expected_num_chunks:
        raise RuntimeError(
            f"Index/metadata count mismatch: FAISS index has {index.ntotal} vectors, "
            f"chunks_metadata.json has {len(metadata) if isinstance(metadata, list) else 'invalid'} entries, "
            f"manifest expects {expected_num_chunks}.\n"
            f"Please rebuild the index by running:\n  {REBUILD_CMD}"
        )

    if index.d != expected_dim:
        raise RuntimeError(
            f"Index dimension mismatch: FAISS index dimension is {index.d}, "
            f"manifest expects {expected_dim}.\n"
            f"Please rebuild the index by running:\n  {REBUILD_CMD}"
        )

    # 3. Verify artifact integrity fingerprints
    artifacts_fp = manifest.get("artifacts_fingerprint", {})
    actual_meta_sha = compute_file_sha256(metadata_json_path)
    actual_faiss_sha = compute_file_sha256(faiss_index_path)
    if (
        artifacts_fp.get("chunks_metadata_sha256") != actual_meta_sha
        or artifacts_fp.get("faiss_index_sha256") != actual_faiss_sha
    ):
        raise RuntimeError(
            "Index or metadata file fingerprint does not match manifest (files were modified out-of-band).\n"
            f"Please rebuild the index by running:\n  {REBUILD_CMD}"
        )

    # 4. Verify index is not stale relative to pages.json or chunks.json
    input_fp = manifest.get("input_fingerprint", {})
    if pages_json_path.exists():
        current_pages_sha = compute_file_sha256(pages_json_path)
        if input_fp.get("pages_json_sha256") != current_pages_sha:
            raise RuntimeError(
                "Stale index detected: `data/processed/pages.json` has changed since the index was built.\n"
                f"Please rebuild the index by running:\n  {REBUILD_CMD}"
            )

    if chunks_json_path.exists():
        current_chunks_sha = compute_file_sha256(chunks_json_path)
        if input_fp.get("chunks_json_sha256") != current_chunks_sha:
            raise RuntimeError(
                "Stale index detected: `data/processed/chunks.json` has changed since the index was built.\n"
                f"Please rebuild the index by running:\n  {REBUILD_CMD}"
            )

    return index, metadata, manifest


def main() -> int:
    try:
        manifest = build_and_save_index()
        print("\nIndex Build Summary")
        print("-" * 45)
        print(f"Project: {manifest['project']}")
        print(f"Model ID: {manifest['model_id']} (device: {manifest['device']})")
        print(f"Embedding dimension: {manifest['embedding_dimension']}")
        print(f"Model max sequence length: {manifest['model_max_seq_length']} tokens")
        print(f"Longest chunk content tokens: {manifest['longest_content_tokens']} tokens")
        print(
            f"Longest full embedding input (with prefix & special tokens): "
            f"{manifest['longest_embedding_input_tokens']} tokens"
        )
        print(f"Indexed pages: {manifest['indexed_pages_count']} | Skipped pages: {manifest['skipped_pages_count']}")
        print(f"Total indexed chunks: {manifest['num_chunks']}")
        print(f"Saved FAISS index: {FAISS_INDEX_PATH}")
        print(f"Saved chunk metadata: {METADATA_JSON_PATH}")
        print(f"Saved build manifest: {MANIFEST_JSON_PATH}")
        return 0
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

