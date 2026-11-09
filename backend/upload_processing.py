"""
Generic (non-handbook) PDF processing for chat-scoped uploads.

Deliberately independent from the Tulane handbook cleanup in `preprocess_chunk.py`: no page
exclusions, no section boundaries, no heading regexes. Reuses only the tokenizer helper and the
embedding-input formatter so uploaded chunks and queries live in the same MiniLM vector space.

Everything here is synchronous and CPU-only; callers run it in a bounded worker thread.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from pypdf import PdfReader

try:
    from backend.preprocess_chunk import (
        MODEL_ID,
        TARGET_CONTENT_TOKENS,
        count_tokens,
        format_embedding_input,
    )
except ImportError:  # pragma: no cover - script-style imports
    from preprocess_chunk import (  # type: ignore
        MODEL_ID,
        TARGET_CONTENT_TOKENS,
        count_tokens,
        format_embedding_input,
    )

UPLOAD_SECTION_LABEL = "Uploaded PDF"
EMBED_BATCH_SIZE = 8
MIN_TEXT_CHARS_TOTAL = 200  # below this the PDF is treated as image-only / scanned
MIN_TEXT_PAGE_FRACTION = 0.2  # ...or if fewer than 20% of pages carry any real text
PAGE_TEXT_MIN_CHARS = 25
SAMPLE_SEGMENT_CHARS = 2000
MAX_HEADINGS_IN_SAMPLE = 20


class PdfValidationError(Exception):
    """A user-facing validation failure with a stable machine-readable `code`."""

    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


@dataclass
class UploadLimits:
    max_bytes: int = 10 * 1024 * 1024
    max_pages: int = 50
    max_chars: int = 300_000
    max_chunks: int = 400


def sanitize_display_filename(raw: str | None) -> str:
    """Display-only name: basename, no control characters, bounded length. Never used for paths."""
    name = (raw or "").replace("\\", "/").split("/")[-1]
    name = re.sub(r"[\x00-\x1f\x7f<>\"|?*]", "", name).strip().strip(".")
    name = re.sub(r"\s+", " ", name)
    if len(name) > 120:
        stem, dot, ext = name.rpartition(".")
        name = (stem[: 115 - len(ext)] + "." + ext) if dot and len(ext) <= 8 else name[:120]
    return name or "document.pdf"


def check_pdf_signature(head: bytes) -> None:
    """The `%PDF-` marker must appear in the first KiB. MIME type and extension are only hints."""
    if not head:
        raise PdfValidationError("empty_file", "The uploaded file is empty.")
    if b"%PDF-" not in head[:1024]:
        raise PdfValidationError(
            "not_a_pdf", "This file does not look like a PDF (missing the PDF signature)."
        )


def extract_upload_pages(pdf_path: Path, limits: UploadLimits) -> list[dict[str, Any]]:
    """Open and extract per-page text (1-based page numbers). Raises PdfValidationError."""
    if pdf_path.stat().st_size == 0:
        raise PdfValidationError("empty_file", "The uploaded file is empty.")
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:  # noqa: BLE001 - pypdf raises many types
        raise PdfValidationError(
            "malformed_pdf", "This PDF is damaged or could not be opened by the parser."
        ) from exc

    if reader.is_encrypted:
        raise PdfValidationError(
            "encrypted_pdf",
            "This PDF is password-protected or encrypted. Remove the protection and upload it again.",
        )
    try:
        total_pages = len(reader.pages)
    except Exception as exc:  # noqa: BLE001
        raise PdfValidationError("malformed_pdf", "This PDF's page tree could not be read.") from exc
    if total_pages == 0:
        raise PdfValidationError("malformed_pdf", "This PDF has no pages.")
    if total_pages > limits.max_pages:
        raise PdfValidationError(
            "too_many_pages",
            f"This PDF has {total_pages} pages; the limit is {limits.max_pages}. "
            "Upload a shorter excerpt.",
        )

    pages: list[dict[str, Any]] = []
    total_chars = 0
    pages_with_text = 0
    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            raise PdfValidationError(
                "malformed_pdf", f"Text could not be read from page {idx}."
            ) from exc
        text = text.replace("\x00", "")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        total_chars += len(text)
        if total_chars > limits.max_chars:
            raise PdfValidationError(
                "too_much_text",
                f"This PDF contains more than {limits.max_chars:,} characters of text. "
                "Upload a shorter excerpt.",
            )
        if len(re.sub(r"\s", "", text)) >= PAGE_TEXT_MIN_CHARS:
            pages_with_text += 1
        pages.append({"page": idx, "text": text})

    non_ws = sum(len(re.sub(r"\s", "", p["text"])) for p in pages)
    if non_ws < MIN_TEXT_CHARS_TOTAL or pages_with_text / total_pages < MIN_TEXT_PAGE_FRACTION:
        raise PdfValidationError(
            "no_extractable_text",
            "This PDF has little or no selectable text (it may be a scan or images). "
            "Upload a searchable PDF; OCR is not supported.",
        )
    return pages


_HEADING_RE = re.compile(r"^(?:[A-Z][A-Z0-9 ,;:&/()'\-]{3,80}|\d+(?:\.\d+)*\.?\s+[A-Z][^\n]{2,80})$")


def build_classification_sample(pages: list[dict[str, Any]]) -> str:
    """Bounded beginning / middle / end sample plus detected headings, with page markers."""
    texts = [(p["page"], p["text"]) for p in pages if p["text"].strip()]
    joined = "\n\n".join(f"[page {n}]\n{t}" for n, t in texts)
    total = len(joined)
    seg = SAMPLE_SEGMENT_CHARS
    if total <= seg * 3:
        body = joined
    else:
        mid_start = max(0, total // 2 - seg // 2)
        body = (
            f"--- BEGINNING ---\n{joined[:seg]}\n\n"
            f"--- MIDDLE ---\n{joined[mid_start:mid_start + seg]}\n\n"
            f"--- END ---\n{joined[-seg:]}"
        )
    headings: list[str] = []
    for _, text in texts:
        for line in text.splitlines():
            line = line.strip()
            if _HEADING_RE.match(line) and line not in headings:
                headings.append(line)
                if len(headings) >= MAX_HEADINGS_IN_SAMPLE:
                    break
        if len(headings) >= MAX_HEADINGS_IN_SAMPLE:
            break
    heading_block = "\n".join(f"- {h}" for h in headings) or "(none detected)"
    return f"Detected headings:\n{heading_block}\n\nText sample:\n{body}"


def _split_units(text: str) -> list[str]:
    paras = [re.sub(r"\s*\n\s*", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    paras = [p for p in paras if p]
    if len(paras) <= 1 and "\n" in text:  # PDFs often lack blank lines: fall back to lines
        paras = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return paras


def _split_oversized(unit: str, tokenizer: Any, limit: int) -> list[str]:
    if count_tokens(tokenizer, unit) <= limit:
        return [unit]
    out: list[str] = []
    current: list[str] = []
    for word in unit.split():
        trial = " ".join(current + [word])
        if current and count_tokens(tokenizer, trial) > limit:
            out.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        out.append(" ".join(current))
    return out


def chunk_upload_pages(
    pages: list[dict[str, Any]],
    tokenizer: Any,
    *,
    document_id: str,
    display_filename: str,
    limits: UploadLimits,
    target_tokens: int = TARGET_CONTENT_TOKENS,
) -> list[dict[str, Any]]:
    """Token-aware paragraph packing that preserves 1-based PDF page numbers."""
    units: list[tuple[str, int, int]] = []  # (text, page, tokens)
    for p in pages:
        for u in _split_units(p["text"]):
            for piece in _split_oversized(u, tokenizer, target_tokens):
                units.append((piece, p["page"], count_tokens(tokenizer, piece)))

    chunks: list[dict[str, Any]] = []
    cur_text: list[str] = []
    cur_pages: list[int] = []
    cur_tokens = 0

    def flush() -> None:
        nonlocal cur_text, cur_pages, cur_tokens
        if not cur_text:
            return
        idx = len(chunks)
        chunks.append(
            {
                "chunk_id": f"up_{document_id[:12]}_{idx:04d}",
                "document_id": document_id,
                "document_title": display_filename,
                "document_section": UPLOAD_SECTION_LABEL,
                "pdf_pages": sorted(set(cur_pages)),
                "heading": None,
                "token_count": cur_tokens,
                "text": "\n".join(cur_text),
            }
        )
        cur_text, cur_pages, cur_tokens = [], [], 0

    for text, page, toks in units:
        if cur_text and cur_tokens + toks > target_tokens:
            flush()
            if len(chunks) > limits.max_chunks:
                break
        cur_text.append(text)
        cur_pages.append(page)
        cur_tokens += toks
    flush()

    if len(chunks) > limits.max_chunks:
        raise PdfValidationError(
            "too_many_chunks",
            f"This PDF produces more than {limits.max_chunks} searchable passages. "
            "Upload a shorter excerpt.",
        )
    if not chunks:
        raise PdfValidationError(
            "no_extractable_text", "No searchable text could be extracted from this PDF."
        )
    return chunks


def embed_chunks(chunks: list[dict[str, Any]], model: Any) -> np.ndarray:
    """Normalized float32 embeddings with the shared MiniLM model, in small batches."""
    inputs = [
        format_embedding_input(c["document_title"], c["document_section"], c["heading"], c["text"])
        for c in chunks
    ]
    vectors = model.encode(
        inputs,
        batch_size=EMBED_BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def publish_artifacts(
    root: Path,
    document_id: str,
    chunks: list[dict[str, Any]],
    embeddings: np.ndarray,
    meta: dict[str, Any],
) -> Path:
    """Write to `<root>/<id>.staging`, validate by reading back, then atomically rename to `<root>/<id>`."""
    staging = root / f"{document_id}.staging"
    final = root / document_id
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        dim = int(embeddings.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        if index.ntotal != len(chunks):
            raise RuntimeError("index/chunk count mismatch before write")
        faiss.write_index(index, str(staging / "index.faiss"))
        (staging / "chunks.json").write_text(
            json.dumps(chunks, ensure_ascii=False), encoding="utf-8"
        )
        (staging / "meta.json").write_text(
            json.dumps({**meta, "model_id": MODEL_ID, "embedding_dimension": dim, "num_chunks": len(chunks)}),
            encoding="utf-8",
        )
        # Validate what was actually written
        load_published(staging, expected_dim=dim)
        if final.exists():
            shutil.rmtree(final)
        os.replace(staging, final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final


def load_published(directory: Path, expected_dim: int | None = None) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    """Load a server-generated upload index. Validates dimension and metadata counts."""
    index = faiss.read_index(str(directory / "index.faiss"))
    chunks = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    if not isinstance(chunks, list) or index.ntotal != len(chunks) or index.ntotal != meta.get("num_chunks"):
        raise RuntimeError("Uploaded index and chunk metadata counts do not match.")
    dim = expected_dim if expected_dim is not None else meta.get("embedding_dimension")
    if index.d != dim or meta.get("model_id") != MODEL_ID:
        raise RuntimeError("Uploaded index dimension/model does not match the active embedding model.")
    return index, chunks, meta

