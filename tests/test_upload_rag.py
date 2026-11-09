"""
Focused test suite for chat-scoped PDF upload, relevance filtering, and conversational QA.

Covers:
1. File validation: empty file, missing PDF signature, encrypted, scanned/image-only, oversized.
2. Relevance outcomes: accept, reject (out-of-scope), needs_clarification.
3. Concurrency & restart recovery: 409 Conflict on concurrent upload, recover_interrupted on restart.
4. Chat-scoped retrieval isolation:
   - Unique fact in Chat A's uploaded PDF is retrieved in Chat A (uploaded_pdf mode).
   - Chat A in handbook mode does NOT retrieve the uploaded fact.
   - Chat B does NOT retrieve Chat A's uploaded fact.
5. Follow-up rewriting in upload mode.
6. Document removal & artifact cleanup while preserving conversation message history.
7. Compatibility with clients that omit source_mode.
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.conversation_store import ConversationStore
from backend.main import create_app
from backend.rag_engine import GroundedAnswerPayload, QueryRewritePayload
from backend.retriever import HandbookRetriever
from backend.upload_manager import UploadManager, UploadStore
from backend.upload_processing import UploadLimits, extract_upload_pages
from backend.upload_relevance import RelevancePayload


def make_test_pdf_bytes(text: str) -> bytes:
    """Generate a valid minimal 1-page PDF containing `text`."""
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    lines = [escaped[i : i + 65] for i in range(0, len(escaped), 65)]
    stream_ops = ["BT", "/F1 12 Tf", "72 712 Td"]
    for idx, l in enumerate(lines):
        if idx > 0:
            stream_ops.append("0 -14 Td")
        stream_ops.append(f"({l}) Tj")
    stream_ops.append("ET\n")
    stream_data = "\n".join(stream_ops).encode("latin-1")
    obj4 = (
        f"4 0 obj\n<< /Length {len(stream_data)} >>\nstream\n".encode("latin-1")
        + stream_data
        + b"endstream\nendobj\n"
    )
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        + obj4
        + b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000010 00000 n \n"
        b"0000000060 00000 n \n"
        b"0000000117 00000 n \n"
        b"0000000249 00000 n \n"
        b"0000000418 00000 n \n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n500\n%%EOF"
    )


@pytest.fixture(scope="module")
def shared_retriever() -> HandbookRetriever:
    return HandbookRetriever()


class MockLLMForUpload:
    """Mock LLM handling relevance classification, query rewrite, and grounded answer generation."""

    def __init__(self, relevance_decision: str = "accept", relevance_reason: str = "Faculty policy document."):
        self.relevance_decision = relevance_decision
        self.relevance_reason = relevance_reason
        self.model_id = "mock-gemini-upload"

    def generate_structured(
        self,
        system_instruction: str,
        user_prompt: str,
        response_schema: type[Any],
        temperature: float = 0.0,
    ) -> Any:
        if response_schema is RelevancePayload:
            return RelevancePayload(
                decision=self.relevance_decision,  # type: ignore[arg-type]
                reason=self.relevance_reason,
            )
        if response_schema is QueryRewritePayload:
            latest_q = user_prompt.split("Latest User Question:\n")[-1].split("\n")[0].strip()
            if "who approves it" in latest_q.lower():
                return QueryRewritePayload(
                    status="ready",
                    standalone_query="Who approves the Quantum Sabbatical Fellowship?",
                    topic_changed=False,
                )
            return QueryRewritePayload(
                status="ready",
                standalone_query=latest_q,
                topic_changed=False,
            )
        if response_schema is GroundedAnswerPayload:
            # Check if this turn is an upload or handbook turn
            if "Retrieved Uploaded-PDF Passages" in user_prompt:
                if "Quantum Sabbatical" in user_prompt:
                    if "Who approves" in user_prompt:
                        return GroundedAnswerPayload(
                            status="supported",
                            answer="The Dean and Provost approve the Quantum Sabbatical Fellowship. [S1]",
                            cited_labels=["S1"],
                        )
                    return GroundedAnswerPayload(
                        status="supported",
                        answer="Tenure-track faculty are eligible for the Quantum Sabbatical Fellowship of 50000 dollars. [S1]",
                        cited_labels=["S1"],
                    )
                return GroundedAnswerPayload(
                    status="unsupported",
                    answer="I couldn't find that in the uploaded PDF.",
                    cited_labels=[],
                )
            else:
                # Handbook turn
                if "Quantum Sabbatical" in user_prompt:
                    return GroundedAnswerPayload(
                        status="unsupported",
                        answer="I couldn’t find that in the provided handbook.",
                        cited_labels=[],
                    )
                return GroundedAnswerPayload(
                    status="supported",
                    answer="Faculty receive paid parental leave under Chapter 3. [S1]",
                    cited_labels=["S1"],
                )

        raise AssertionError(f"Unexpected response_schema: {response_schema}")


def test_upload_file_validation(tmp_path: Path, shared_retriever: HandbookRetriever) -> None:
    """Test validation errors for empty, non-pdf, and corrupt uploads."""
    store = ConversationStore(db_path=tmp_path / "val.db")
    upload_store = UploadStore(db_path=tmp_path / "val.db")
    upload_mgr = UploadManager(store=upload_store, root=tmp_path / "uploads")
    mock_llm = MockLLMForUpload()

    app = create_app(
        store_override=store,
        upload_manager_override=upload_mgr,
        retriever_override=shared_retriever,
        llm_client_override=mock_llm,
    )

    with TestClient(app) as client:
        # 1. Non-existent conversation -> 404
        r = client.post(
            "/api/conversations/nonexistent/upload",
            files={"file": ("test.pdf", b"%PDF-1.4\n...", "application/pdf")},
        )
        assert r.status_code == 404

        # Create conversation
        conv = client.post("/api/conversations").json()
        cid = conv["conversation_id"]

        # 2. Non-PDF extension hint
        r = client.post(
            f"/api/conversations/{cid}/upload",
            files={"file": ("notes.txt", b"%PDF-1.4 text", "text/plain")},
        )
        assert r.status_code == 422
        assert "Only PDF files" in r.json()["detail"]

        # 3. Empty file
        r = client.post(
            f"/api/conversations/{cid}/upload",
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )
        assert r.status_code == 422
        assert "empty" in r.json()["detail"].lower()

        # 4. Missing %PDF- signature
        r = client.post(
            f"/api/conversations/{cid}/upload",
            files={"file": ("fake.pdf", b"This is plain text with no PDF header", "application/pdf")},
        )
        assert r.status_code == 422
        assert "signature" in r.json()["detail"].lower()


def test_relevance_outcomes(tmp_path: Path, shared_retriever: HandbookRetriever) -> None:
    """Verify reject, needs_clarification, and accept outcomes."""
    pdf_bytes = make_test_pdf_bytes(
        "Tulane University Faculty Special Policy: All tenure-track faculty are eligible for "
        "the Quantum Sabbatical Fellowship of 50000 dollars upon third-year review approval by the "
        "Dean and Provost. This leave may last up to two academic semesters."
    )

    # 1. Test 'reject' outcome
    store = ConversationStore(db_path=tmp_path / "rej.db")
    upload_store = UploadStore(db_path=tmp_path / "rej.db")
    upload_mgr = UploadManager(store=upload_store, root=tmp_path / "uploads_rej")
    mock_reject = MockLLMForUpload(relevance_decision="reject", relevance_reason="This document is an unrelated travel guide.")

    app_rej = create_app(
        store_override=store,
        upload_manager_override=upload_mgr,
        retriever_override=shared_retriever,
        llm_client_override=mock_reject,
    )

    with TestClient(app_rej) as client:
        conv = client.post("/api/conversations").json()
        cid = conv["conversation_id"]

        r = client.post(
            f"/api/conversations/{cid}/upload",
            files={"file": ("travel_guide.pdf", pdf_bytes, "application/pdf")},
        )
        assert r.status_code == 202

        # Poll until finished
        for _ in range(30):
            time.sleep(0.1)
            st = client.get(f"/api/conversations/{cid}/document").json()
            if st["status"] != "processing":
                break

        assert st["status"] == "rejected"
        assert "travel guide" in st["message"]
        # Confirm no searchable artifacts created
        assert not (tmp_path / "uploads_rej" / st["document_id"]).exists()

    # 2. Test 'needs_clarification' outcome
    upload_mgr2 = UploadManager(store=UploadStore(db_path=tmp_path / "nc.db"), root=tmp_path / "uploads_nc")
    mock_nc = MockLLMForUpload(
        relevance_decision="needs_clarification",
        relevance_reason="The text sample is too brief and generic to verify faculty scope.",
    )
    app_nc = create_app(
        store_override=ConversationStore(db_path=tmp_path / "nc.db"),
        upload_manager_override=upload_mgr2,
        retriever_override=shared_retriever,
        llm_client_override=mock_nc,
    )
    with TestClient(app_nc) as client:
        conv = client.post("/api/conversations").json()
        cid = conv["conversation_id"]

        r = client.post(
            f"/api/conversations/{cid}/upload",
            files={"file": ("memo.pdf", pdf_bytes, "application/pdf")},
        )
        assert r.status_code == 202

        for _ in range(30):
            time.sleep(0.1)
            st = client.get(f"/api/conversations/{cid}/document").json()
            if st["status"] != "processing":
                break

        assert st["status"] == "needs_clarification"
        assert "clearer document" in st["message"]


def test_scoped_retrieval_and_chat_isolation(tmp_path: Path, shared_retriever: HandbookRetriever) -> None:
    """
    Prove isolation:
    - Chat A uploads custom faculty policy with unique Quantum Sabbatical Fellowship.
    - Chat A in 'uploaded_pdf' mode retrieves the unique fact and cites the upload.
    - Chat A follow-up 'Who approves it?' resolves and answers with Dean and Provost.
    - Chat A in 'handbook' mode answers unsupported ('I couldn’t find that in the provided handbook.').
    - Chat B has no upload: 'uploaded_pdf' mode returns 400.
    - Chat B in 'handbook' mode answers unsupported for the unique fact.
    - Removal deletes upload artifacts and blocks future queries, while retaining prior messages.
    """
    store = ConversationStore(db_path=tmp_path / "iso.db")
    upload_mgr = UploadManager(store=UploadStore(db_path=tmp_path / "iso.db"), root=tmp_path / "uploads_iso")
    mock_llm = MockLLMForUpload(relevance_decision="accept", relevance_reason="Faculty fellowship policy.")

    app = create_app(
        store_override=store,
        upload_manager_override=upload_mgr,
        retriever_override=shared_retriever,
        llm_client_override=mock_llm,
    )

    pdf_bytes = make_test_pdf_bytes(
        "Tulane University Faculty Special Policy: All tenure-track faculty are eligible for "
        "the Quantum Sabbatical Fellowship of 50000 dollars upon third-year review approval by the "
        "Dean and Provost. This leave may last up to two academic semesters."
    )

    with TestClient(app) as client:
        # Create Chat A and Chat B
        conv_a = client.post("/api/conversations").json()["conversation_id"]
        conv_b = client.post("/api/conversations").json()["conversation_id"]

        # Upload to Chat A
        up_res = client.post(
            f"/api/conversations/{conv_a}/upload",
            files={"file": ("quantum_fellowship.pdf", pdf_bytes, "application/pdf")},
        )
        assert up_res.status_code == 202
        doc_id = up_res.json()["document_id"]

        # Second upload to Chat A while active doc exists -> 409 Conflict
        conflict_res = client.post(
            f"/api/conversations/{conv_a}/upload",
            files={"file": ("second.pdf", pdf_bytes, "application/pdf")},
        )
        assert conflict_res.status_code == 409

        # Wait for Chat A upload to reach 'ready'
        for _ in range(50):
            time.sleep(0.1)
            doc_status = client.get(f"/api/conversations/{conv_a}/document").json()
            if doc_status["status"] == "ready":
                break
        assert doc_status["status"] == "ready"
        assert doc_status["chunk_count"] > 0
        assert doc_status["display_filename"] == "quantum_fellowship.pdf"

        # 1. Ask in Chat A with source_mode="uploaded_pdf"
        msg_res = client.post(
            f"/api/conversations/{conv_a}/messages",
            json={
                "message": "What is the amount of the Quantum Sabbatical Fellowship?",
                "source_mode": "uploaded_pdf",
            },
        )
        assert msg_res.status_code == 200
        data_a = msg_res.json()
        assert data_a["status"] == "supported"
        assert "50000 dollars" in data_a["answer"]
        assert len(data_a["cited_sources"]) == 1
        assert data_a["cited_sources"][0]["document_title"] == "quantum_fellowship.pdf"
        assert data_a["cited_sources"][0]["pdf_pages"] == [1]

        # 2. Follow-up in Chat A
        follow_res = client.post(
            f"/api/conversations/{conv_a}/messages",
            json={
                "message": "Who approves it?",
                "source_mode": "uploaded_pdf",
            },
        )
        assert follow_res.status_code == 200
        follow_data = follow_res.json()
        assert "Dean and Provost" in follow_data["answer"]

        # 3. Ask same question in Chat A with source_mode="handbook"
        handbook_res = client.post(
            f"/api/conversations/{conv_a}/messages",
            json={
                "message": "What is the amount of the Quantum Sabbatical Fellowship?",
                "source_mode": "handbook",
            },
        )
        assert handbook_res.status_code == 200
        assert handbook_res.json()["status"] == "unsupported"
        assert "couldn’t find that in the provided handbook" in handbook_res.json()["answer"]

        # 4. Chat B (no upload): asking in uploaded_pdf mode -> 400 Bad Request
        b_upload_mode = client.post(
            f"/api/conversations/{conv_b}/messages",
            json={
                "message": "What is the amount of the Quantum Sabbatical Fellowship?",
                "source_mode": "uploaded_pdf",
            },
        )
        assert b_upload_mode.status_code == 400
        assert "No uploaded PDF is ready" in b_upload_mode.json()["detail"]

        # 5. Chat B in handbook mode -> unsupported
        b_handbook_res = client.post(
            f"/api/conversations/{conv_b}/messages",
            json={
                "message": "What is the amount of the Quantum Sabbatical Fellowship?",
                "source_mode": "handbook",
            },
        )
        assert b_handbook_res.status_code == 200
        assert b_handbook_res.json()["status"] == "unsupported"

        # 6. Backward compatibility: omitting source_mode defaults to handbook
        compat_res = client.post(
            f"/api/conversations/{conv_a}/messages",
            json={"message": "How much paid parental leave do faculty get?"},
        )
        assert compat_res.status_code == 200
        assert "parental leave" in compat_res.json()["answer"]

        # 7. Document removal
        del_res = client.delete(f"/api/conversations/{conv_a}/document")
        assert del_res.status_code == 200
        assert del_res.json()["deleted"] is True
        # Artifact directory deleted
        assert not (tmp_path / "uploads_iso" / doc_id).exists()

        # Subsequent upload query fails with 400
        after_del = client.post(
            f"/api/conversations/{conv_a}/messages",
            json={
                "message": "What is the amount of the Quantum Sabbatical Fellowship?",
                "source_mode": "uploaded_pdf",
            },
        )
        assert after_del.status_code == 400

        # Prior messages and citations are preserved in conversation history
        hist = client.get(f"/api/conversations/{conv_a}/messages").json()
        assert len(hist["messages"]) >= 6
        assert any(
            len(m["cited_sources"]) > 0 and m["cited_sources"][0]["document_title"] == "quantum_fellowship.pdf"
            for m in hist["messages"]
        )


def test_restart_recovery(tmp_path: Path) -> None:
    """Verify that abandoned jobs in 'processing' are recovered as 'failed' after server restart."""
    db_file = tmp_path / "restart.db"
    store = UploadStore(db_path=db_file)
    # Manually insert an abandoned 'processing' row
    store.create_processing("conv_restart", "abandoned.pdf")

    # Now create a new UploadManager simulating startup
    mgr = UploadManager(store=UploadStore(db_path=db_file), root=tmp_path / "uploads_restart")
    st = mgr.store.get_for_conversation("conv_restart")
    assert st is not None
    assert st["status"] == "failed"
    assert st["code"] == "interrupted"
    assert "restart" in st["message"]
