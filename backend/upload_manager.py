"""
Chat-scoped PDF upload store, bounded background processing and scoped retrieval cache.

- SQLite table `uploaded_documents` (same database file as conversations) holds status and ownership.
  A partial UNIQUE index guarantees at most one *active* (processing/ready) document per conversation,
  even under concurrent uploads.
- Processing runs in a small thread pool (never on the event loop) with a bounded queue.
- Artifacts (FAISS index + chunk metadata) are server-generated, staged, validated, then published.
- Retrieval only ever sees the index of the document that belongs to the requested conversation.
"""

from __future__ import annotations

import datetime
import logging
import shutil
import sqlite3
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

try:
    from backend.retriever import HandbookRetriever
    from backend.upload_processing import (
        PdfValidationError,
        UploadLimits,
        build_classification_sample,
        chunk_upload_pages,
        embed_chunks,
        extract_upload_pages,
        load_published,
        publish_artifacts,
    )
    from backend.upload_relevance import RelevanceUnavailable, classify_relevance
except ImportError:  # pragma: no cover
    from retriever import HandbookRetriever  # type: ignore
    from upload_processing import (  # type: ignore
        PdfValidationError,
        UploadLimits,
        build_classification_sample,
        chunk_upload_pages,
        embed_chunks,
        extract_upload_pages,
        load_published,
        publish_artifacts,
    )
    from upload_relevance import RelevanceUnavailable, classify_relevance  # type: ignore

log = logging.getLogger("uploads")

ACTIVE_STATUSES = ("processing", "ready")
_COLUMNS = (
    "document_id, conversation_id, display_filename, status, code, message, "
    "page_count, chunk_count, created_at, updated_at"
)


class ActiveDocumentExists(Exception):
    """The conversation already has a processing or ready document."""


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


class UploadStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS uploaded_documents (
                    document_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    display_filename TEXT NOT NULL,
                    status TEXT NOT NULL,
                    code TEXT,
                    message TEXT,
                    page_count INTEGER,
                    chunk_count INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_document_per_conversation
                ON uploaded_documents(conversation_id) WHERE status IN ('processing', 'ready');
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def create_processing(self, conversation_id: str, display_filename: str) -> dict[str, Any]:
        """Atomically claim the conversation's single active slot. Clears stale failed/rejected rows."""
        doc_id = uuid.uuid4().hex
        now = _now()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "DELETE FROM uploaded_documents WHERE conversation_id = ? AND status NOT IN ('processing','ready')",
                (conversation_id,),
            )
            try:
                conn.execute(
                    f"INSERT INTO uploaded_documents ({_COLUMNS}) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (doc_id, conversation_id, display_filename, "processing", None, None, None, None, now, now),
                )
            except sqlite3.IntegrityError as exc:
                conn.execute("ROLLBACK")
                raise ActiveDocumentExists() from exc
            conn.execute("COMMIT")
        finally:
            conn.close()
        return self.get_by_id(doc_id) or {}

    def get_by_id(self, document_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM uploaded_documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_for_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM uploaded_documents WHERE conversation_id = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (conversation_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_ready(self, conversation_id: str) -> dict[str, Any] | None:
        doc = self.get_for_conversation(conversation_id)
        return doc if doc and doc["status"] == "ready" else None

    def finish(
        self,
        document_id: str,
        status: str,
        code: str | None = None,
        message: str | None = None,
        page_count: int | None = None,
        chunk_count: int | None = None,
    ) -> bool:
        """Transition processing -> final status. False if the row was removed/changed meanwhile."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE uploaded_documents SET status=?, code=?, message=?, page_count=?, chunk_count=?, "
                "updated_at=? WHERE document_id=? AND status='processing'",
                (status, code, message, page_count, chunk_count, _now(), document_id),
            )
            conn.commit()
            return cur.rowcount == 1

    def mark_failed(self, document_id: str, code: str, message: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE uploaded_documents SET status='failed', code=?, message=?, updated_at=? "
                "WHERE document_id=? AND status='ready'",
                (code, message, _now(), document_id),
            )
            conn.commit()

    def delete_for_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        doc = self.get_for_conversation(conversation_id)
        if doc is None:
            return None
        with self._connect() as conn:
            conn.execute("DELETE FROM uploaded_documents WHERE document_id = ?", (doc["document_id"],))
            conn.commit()
        return doc

    def recover_interrupted(self) -> list[str]:
        """After a restart nothing can still be processing: mark abandoned jobs failed."""
        with self._connect() as conn:
            ids = [r["document_id"] for r in conn.execute(
                "SELECT document_id FROM uploaded_documents WHERE status='processing'"
            ).fetchall()]
            conn.execute(
                "UPDATE uploaded_documents SET status='failed', code='interrupted', message=?, updated_at=? "
                "WHERE status='processing'",
                ("Processing was interrupted by a server restart. Please upload the PDF again.", _now()),
            )
            conn.commit()
        return ids

    def ready_ids(self) -> set[str]:
        with self._connect() as conn:
            return {r["document_id"] for r in conn.execute(
                "SELECT document_id FROM uploaded_documents WHERE status='ready'"
            ).fetchall()}


class UploadManager:
    def __init__(
        self,
        store: UploadStore,
        root: Path,
        limits: UploadLimits | None = None,
        max_workers: int = 1,
        max_queue: int = 2,
        cache_size: int = 8,
    ) -> None:
        self.store = store
        self.root = Path(root)
        self.limits = limits or UploadLimits()
        self.tmp_dir = self.root / "tmp"
        self.root.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.max_slots = max_workers + max_queue
        self._slots = 0
        self._slot_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="pdf-upload")
        self._cache: OrderedDict[tuple[str, str], HandbookRetriever] = OrderedDict()
        self._cache_lock = threading.Lock()
        self.cache_size = max(1, cache_size)
        self._recover_on_startup()

    # -- startup -------------------------------------------------------------------------------
    def _recover_on_startup(self) -> None:
        self.store.recover_interrupted()
        for p in self.tmp_dir.glob("*"):
            p.unlink(missing_ok=True)
        keep = self.store.ready_ids()
        for child in self.root.iterdir():
            if child.name == "tmp" or not child.is_dir():
                continue
            if child.name.endswith(".staging") or child.name not in keep:
                shutil.rmtree(child, ignore_errors=True)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    # -- slots / submission --------------------------------------------------------------------
    def try_reserve_slot(self) -> bool:
        with self._slot_lock:
            if self._slots >= self.max_slots:
                return False
            self._slots += 1
            return True

    def _release_slot(self) -> None:
        with self._slot_lock:
            self._slots = max(0, self._slots - 1)

    def tmp_path_for(self, document_id: str) -> Path:
        return self.tmp_dir / f"{document_id}.pdf"

    def submit(
        self,
        doc: dict[str, Any],
        model: Any,
        llm_client: Any,
        on_done: Callable[[], None] | None = None,
    ) -> None:
        """Queue processing. The caller must hold a slot from `try_reserve_slot`."""
        self._executor.submit(self._run_job, doc, model, llm_client, on_done)

    # -- the job -------------------------------------------------------------------------------
    def _run_job(self, doc: dict[str, Any], model: Any, llm_client: Any, on_done: Callable[[], None] | None) -> None:
        doc_id = doc["document_id"]
        tmp = self.tmp_path_for(doc_id)
        published: Path | None = None
        try:
            pages = extract_upload_pages(tmp, self.limits)
            if self.store.get_by_id(doc_id) is None:
                return  # removed while processing
            decision = classify_relevance(llm_client, build_classification_sample(pages), doc["display_filename"])
            if decision.decision == "reject":
                self.store.finish(doc_id, "rejected", "not_faculty_related", decision.reason.strip(), len(pages), 0)
                return
            if decision.decision == "needs_clarification":
                self.store.finish(
                    doc_id, "needs_clarification", "needs_clarification",
                    f"{decision.reason.strip()} Upload a faculty-specific excerpt or a clearer document.",
                    len(pages), 0,
                )
                return
            chunks = chunk_upload_pages(
                pages, model.tokenizer, document_id=doc_id,
                display_filename=doc["display_filename"], limits=self.limits,
            )
            embeddings = embed_chunks(chunks, model)
            if self.store.get_by_id(doc_id) is None:
                return
            published = publish_artifacts(
                self.root, doc_id, chunks, embeddings,
                {"document_id": doc_id, "display_filename": doc["display_filename"], "page_count": len(pages)},
            )
            if not self.store.finish(doc_id, "ready", None, decision.reason.strip(), len(pages), len(chunks)):
                shutil.rmtree(published, ignore_errors=True)  # removed during processing
                published = None
        except PdfValidationError as exc:
            self.store.finish(doc_id, "failed", exc.code, exc.message, None, None)
        except Exception as exc:  # noqa: BLE001
            log.error("upload processing failed: %s", type(exc).__name__)  # never log contents
            if published is not None:
                shutil.rmtree(published, ignore_errors=True)
            self.store.finish(
                doc_id, "failed", "processing_error",
                "The PDF could not be processed. Please try again or upload a different file.", None, None,
            )
        finally:
            tmp.unlink(missing_ok=True)
            shutil.rmtree(self.root / f"{doc_id}.staging", ignore_errors=True)
            self._release_slot()
            if on_done:
                on_done()

    # -- retrieval -----------------------------------------------------------------------------
    def get_retriever(self, conversation_id: str, model: Any) -> HandbookRetriever | None:
        """Retriever for THIS conversation's ready document only (ownership verified in SQL)."""
        doc = self.store.get_ready(conversation_id)
        if doc is None:
            return None
        key = (conversation_id, doc["document_id"])
        with self._cache_lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                return hit
        try:
            index, chunks, meta = load_published(self.root / doc["document_id"])
        except Exception as exc:  # noqa: BLE001
            log.error("uploaded index unavailable: %s", type(exc).__name__)
            self.store.mark_failed(
                doc["document_id"], "artifacts_missing",
                "The stored index for this PDF is missing or invalid. Remove it and upload again.",
            )
            return None
        retriever = HandbookRetriever(model=model, index=index, metadata=chunks, manifest=meta)
        with self._cache_lock:
            self._cache[key] = retriever
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return retriever

    # -- removal -------------------------------------------------------------------------------
    def remove(self, conversation_id: str) -> dict[str, Any] | None:
        """Stop future retrieval and delete artifacts. Safe during processing (the job self-cleans)."""
        doc = self.store.delete_for_conversation(conversation_id)
        with self._cache_lock:
            for key in [k for k in self._cache if k[0] == conversation_id]:
                self._cache.pop(key, None)
        if doc:
            shutil.rmtree(self.root / doc["document_id"], ignore_errors=True)
            shutil.rmtree(self.root / f"{doc['document_id']}.staging", ignore_errors=True)
            self.tmp_path_for(doc["document_id"]).unlink(missing_ok=True)
        return doc

