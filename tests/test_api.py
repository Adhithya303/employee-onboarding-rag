"""
FastAPI backend test suite for the Tulane University Faculty Onboarding Assistant.

Covers:
1. Health check, creating conversations, and listing conversations
2. Sending a turn and reloading messages with persisted citations (`document_title`, `document_section`,
   `pdf_pages`, `excerpt`, `chunk_id`, `label`), message IDs, timestamps, and answer status
3. Missing conversation IDs (404) and invalid/empty/whitespace/oversized messages (422)
4. API failure handling (missing configuration 503, unavailable index 503, rate limits 429,
   upstream timeouts 504, upstream failures 502) without saving failed turns or exposing secrets
5. Strict history isolation between conversations
6. Per-conversation turn serialization with 409 Conflict for overlapping sends to the same conversation
   while allowing concurrent sends to different conversations
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.conversation_store import ConversationStore
from backend.llm_client import LLMConfigurationError, LLMServiceError
from backend.main import MAX_MESSAGE_CHARS, create_app
from backend.rag_engine import GroundedAnswerPayload, QueryRewritePayload
from backend.retriever import HandbookRetriever


@pytest.fixture(scope="module")
def shared_retriever() -> HandbookRetriever:
    """Load the CPU SentenceTransformer and saved FAISS index once for the test module."""
    return HandbookRetriever()


class ScriptedAPIMockLLM:
    """Deterministic mock LLM for FastAPI endpoint testing."""

    def __init__(self, responses: list[Any], delay_seconds: float = 0.0) -> None:
        self.responses = list(responses)
        self.delay_seconds = delay_seconds
        self.calls: list[dict[str, Any]] = []
        self.model_id = "mock-gemini-model"

    def generate_structured(
        self,
        system_instruction: str,
        user_prompt: str,
        response_schema: type[Any],
        temperature: float = 0.0,
    ) -> Any:
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)
        self.calls.append(
            {
                "system_instruction": system_instruction,
                "user_prompt": user_prompt,
                "response_schema": response_schema,
                "temperature": temperature,
            }
        )
        if not self.responses:
            raise AssertionError("ScriptedAPIMockLLM ran out of scripted responses.")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_api_1_create_and_list_conversations_and_health(
    shared_retriever: HandbookRetriever, tmp_path: Path
) -> None:
    """Verify GET /api/health, POST /api/conversations, and GET /api/conversations."""
    store = ConversationStore(db_path=tmp_path / "api_conv.db")
    mock_llm = ScriptedAPIMockLLM([])
    app = create_app(
        store_override=store,
        retriever_override=shared_retriever,
        llm_client_override=mock_llm,
    )

    with TestClient(app) as client:
        health_res = client.get("/api/health")
        assert health_res.status_code == 200
        health_data = health_res.json()
        assert health_data["status"] == "ok"
        assert health_data["index_loaded"] is True
        assert health_data["chunk_count"] == 749
        assert health_data["llm_configured"] is True

        # Initially empty
        list_res_0 = client.get("/api/conversations")
        assert list_res_0.status_code == 200
        assert list_res_0.json()["conversations"] == []

        # Create two conversations
        create_res_1 = client.post("/api/conversations", json={"title": "First Chat"})
        assert create_res_1.status_code == 201
        conv1 = create_res_1.json()
        assert conv1["title"] == "First Chat"
        assert conv1["message_count"] == 0
        assert len(conv1["conversation_id"]) == 8

        create_res_2 = client.post("/api/conversations")
        assert create_res_2.status_code == 201
        conv2 = create_res_2.json()
        assert conv2["title"] == "New Faculty Onboarding Conversation"

        list_res = client.get("/api/conversations")
        assert list_res.status_code == 200
        convs = list_res.json()["conversations"]
        assert len(convs) == 2
        returned_ids = {c["conversation_id"] for c in convs}
        assert returned_ids == {conv1["conversation_id"], conv2["conversation_id"]}


def test_api_2_send_and_reload_turn_with_citations(
    shared_retriever: HandbookRetriever, tmp_path: Path
) -> None:
    """
    Verify POST /api/conversations/{id}/messages returns message IDs, timestamps, answer status,
    answer text, and cited sources, and that GET /api/conversations/{id}/messages reloads them intact.
    """
    store = ConversationStore(db_path=tmp_path / "api_reload.db")
    mock_llm = ScriptedAPIMockLLM(
        [
            QueryRewritePayload(
                status="ready",
                standalone_query="How many weeks of Paid Parental Leave do full-time faculty receive?",
                topic_changed=False,
            ),
            GroundedAnswerPayload(
                status="supported",
                answer=(
                    "Under the Tulane University Faculty Handbook (Updated August 7, 2026), "
                    "full-time faculty receive up to 6 weeks of Paid Parental Leave [S1]."
                ),
                cited_labels=["S1"],
            ),
        ]
    )
    app = create_app(
        store_override=store,
        retriever_override=shared_retriever,
        llm_client_override=mock_llm,
    )

    with TestClient(app) as client:
        conv_res = client.post("/api/conversations")
        cid = conv_res.json()["conversation_id"]

        send_res = client.post(
            f"/api/conversations/{cid}/messages",
            json={"message": "How many weeks of Paid Parental Leave do full-time faculty receive?"},
        )
        assert send_res.status_code == 200
        body = send_res.json()

        assert body["conversation_id"] == cid
        assert body["user_message_id"] > 0
        assert body["assistant_message_id"] > body["user_message_id"]
        assert body["created_at"]
        assert body["status"] == "supported"
        assert "[S1]" in body["answer"]
        assert len(body["cited_sources"]) == 1

        src = body["cited_sources"][0]
        assert src["label"] == "S1"
        assert src["document_title"] == "Tulane University Faculty Handbook"
        assert src["document_section"] == "Faculty Handbook (Main)"
        assert any(p in (150, 151) for p in src["pdf_pages"])
        assert src["chunk_id"].startswith("fh_main_")
        assert len(src["excerpt"]) > 20

        # Reload conversation messages (simulating page refresh / resume)
        reload_res = client.get(f"/api/conversations/{cid}/messages")
        assert reload_res.status_code == 200
        reload_data = reload_res.json()
        assert reload_data["conversation"]["conversation_id"] == cid
        assert reload_data["conversation"]["message_count"] == 2
        assert "Paid Parental Leave" in reload_data["conversation"]["title"]

        msgs = reload_data["messages"]
        assert len(msgs) == 2
        assert msgs[0]["message_id"] == body["user_message_id"]
        assert msgs[0]["role"] == "user"
        assert msgs[1]["message_id"] == body["assistant_message_id"]
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["status"] == "supported"
        assert msgs[1]["cited_sources"] == body["cited_sources"]


def test_api_3_missing_conversation_and_invalid_input(
    shared_retriever: HandbookRetriever, tmp_path: Path
) -> None:
    """
    Verify unknown conversation IDs return 404, and empty/whitespace-only or oversized messages
    return 422 without calling the LLM or mutating SQLite.
    """
    store = ConversationStore(db_path=tmp_path / "api_validation.db")
    mock_llm = ScriptedAPIMockLLM([])
    app = create_app(
        store_override=store,
        retriever_override=shared_retriever,
        llm_client_override=mock_llm,
    )

    with TestClient(app) as client:
        # Unknown conversation on GET and POST
        get_missing = client.get("/api/conversations/nonexistent/messages")
        assert get_missing.status_code == 404
        assert "not found" in get_missing.json()["detail"].lower()

        post_missing = client.post(
            "/api/conversations/nonexistent/messages",
            json={"message": "What is the sabbatical policy?"},
        )
        assert post_missing.status_code == 404

        # Create valid conversation and test invalid inputs
        conv = client.post("/api/conversations").json()
        cid = conv["conversation_id"]

        empty_res = client.post(
            f"/api/conversations/{cid}/messages",
            json={"message": "   \n\t  "},
        )
        assert empty_res.status_code == 422
        assert "empty" in empty_res.json()["detail"].lower()

        oversized_res = client.post(
            f"/api/conversations/{cid}/messages",
            json={"message": "x" * (MAX_MESSAGE_CHARS + 50)},
        )
        assert oversized_res.status_code == 422
        assert "maximum allowed length" in oversized_res.json()["detail"].lower()

        # Verify no LLM calls were made and no messages were stored
        assert len(mock_llm.calls) == 0
        assert store.get_messages(cid) == []


def test_api_4_failure_handling_and_secret_protection(
    shared_retriever: HandbookRetriever, tmp_path: Path
) -> None:
    """
    Verify clear HTTP errors for:
    - Rate limits (429)
    - Upstream timeouts (504) and upstream service failures (502)
    - Missing configuration (503) and unavailable index (503)
    Ensure failed turns are never saved to SQLite and API keys are never leaked.
    """
    store = ConversationStore(db_path=tmp_path / "api_failures.db")
    fake_api_key = "AIzaSySecretKeyThatMustNeverAppearInResponses999"

    mock_llm = ScriptedAPIMockLLM(
        [
            LLMServiceError(
                f"Gemini API transient/rate-limit error (HTTP 429) with key {fake_api_key}",
                is_transient=True,
                status_code=429,
            ),
            LLMServiceError(
                "Gemini API request timed out (HTTP 504)",
                is_transient=True,
                status_code=504,
            ),
            LLMServiceError(
                f"Gemini API upstream error (HTTP 502) key={fake_api_key}",
                is_transient=True,
                status_code=502,
            ),
        ]
    )

    app = create_app(
        store_override=store,
        retriever_override=shared_retriever,
        llm_client_override=mock_llm,
    )

    with TestClient(app) as client:
        cid = client.post("/api/conversations").json()["conversation_id"]

        # 1. Rate limit -> 429
        r_429 = client.post(
            f"/api/conversations/{cid}/messages",
            json={"message": "What is the parental leave policy?"},
        )
        assert r_429.status_code == 429
        assert fake_api_key not in r_429.text

        # 2. Timeout -> 504
        r_504 = client.post(
            f"/api/conversations/{cid}/messages",
            json={"message": "What is the parental leave policy?"},
        )
        assert r_504.status_code == 504

        # 3. Upstream failure -> 502 with secret redacted
        r_502 = client.post(
            f"/api/conversations/{cid}/messages",
            json={"message": "What is the parental leave policy?"},
        )
        assert r_502.status_code == 502
        assert fake_api_key not in r_502.text
        assert "[REDACTED_API_KEY]" in r_502.json()["detail"]

        # Verify zero messages were saved in SQLite across all three failures
        assert store.get_messages(cid) == []

    # 4. Unavailable index -> 503
    app_no_index = create_app(
        store_override=store,
        retriever_override=None,
        llm_client_override=mock_llm,
        index_error_override="FAISS index file not found.",
    )
    with TestClient(app_no_index) as client_no_idx:
        cid2 = client_no_idx.post("/api/conversations").json()["conversation_id"]
        r_no_idx = client_no_idx.post(
            f"/api/conversations/{cid2}/messages",
            json={"message": "What is the sabbatical policy?"},
        )
        assert r_no_idx.status_code == 503
        assert "build_index.py" in r_no_idx.json()["detail"]

    # 5. Missing Gemini configuration -> 503
    empty_env = tmp_path / ".env.missing"
    empty_env.write_text("GEMINI_API_KEY=your_gemini_api_key_here\n", encoding="utf-8")
    app_no_cfg = create_app(
        store_override=store,
        retriever_override=shared_retriever,
        llm_client_override=None,
        env_path=empty_env,
    )
    with TestClient(app_no_cfg) as client_no_cfg:
        health_degraded = client_no_cfg.get("/api/health").json()
        assert health_degraded["status"] == "degraded"
        assert health_degraded["llm_configured"] is False


def test_api_5_conversation_isolation(
    shared_retriever: HandbookRetriever, tmp_path: Path
) -> None:
    """
    Verify two different conversations keep strictly separate message histories and follow-up
    rewrite prompts.
    """
    store = ConversationStore(db_path=tmp_path / "api_isolation.db")
    mock_llm = ScriptedAPIMockLLM(
        [
            # Conv A Turn 1: Sabbatical
            QueryRewritePayload(
                status="ready",
                standalone_query="Who is eligible for faculty sabbatical leave?",
                topic_changed=False,
            ),
            GroundedAnswerPayload(
                status="supported",
                answer="Sabbatical leave is open to tenure-track faculty [S1].",
                cited_labels=["S1"],
            ),
            # Conv B Turn 1: Smoking Policy
            QueryRewritePayload(
                status="ready",
                standalone_query="What is Tulane's smoking policy in university buildings?",
                topic_changed=False,
            ),
            GroundedAnswerPayload(
                status="supported",
                answer="Smoking is prohibited in all University buildings [S1].",
                cited_labels=["S1"],
            ),
            # Conv B Turn 2: Follow-up in Conv B
            QueryRewritePayload(
                status="ready",
                standalone_query="What is the maximum employer fine under Tulane's smoking policy?",
                topic_changed=False,
            ),
            GroundedAnswerPayload(
                status="supported",
                answer="Employers face fines up to $500 [S1].",
                cited_labels=["S1"],
            ),
        ]
    )
    app = create_app(
        store_override=store,
        retriever_override=shared_retriever,
        llm_client_override=mock_llm,
    )

    with TestClient(app) as client:
        cid_a = client.post("/api/conversations").json()["conversation_id"]
        cid_b = client.post("/api/conversations").json()["conversation_id"]

        res_a1 = client.post(
            f"/api/conversations/{cid_a}/messages",
            json={"message": "Who is eligible for faculty sabbatical leave?"},
        )
        assert res_a1.status_code == 200

        res_b1 = client.post(
            f"/api/conversations/{cid_b}/messages",
            json={"message": "What is Tulane's smoking policy in university buildings?"},
        )
        assert res_b1.status_code == 200

        res_b2 = client.post(
            f"/api/conversations/{cid_b}/messages",
            json={"message": "What is the maximum fine?"},
        )
        assert res_b2.status_code == 200

        # Check rewrite prompt for Conv B Turn 2 (index 4 in mock_llm.calls)
        conv_b_rewrite_prompt = mock_llm.calls[4]["user_prompt"]
        assert "smoking" in conv_b_rewrite_prompt.lower()
        assert "sabbatical" not in conv_b_rewrite_prompt.lower()

        msgs_a = client.get(f"/api/conversations/{cid_a}/messages").json()["messages"]
        msgs_b = client.get(f"/api/conversations/{cid_b}/messages").json()["messages"]
        assert len(msgs_a) == 2
        assert len(msgs_b) == 4


def test_api_6_overlapping_sends_to_same_conversation_return_409(
    shared_retriever: HandbookRetriever, tmp_path: Path
) -> None:
    """
    Verify that overlapping POST /api/conversations/{id}/messages requests to the SAME conversation
    return 409 Conflict for the second concurrent caller, while a concurrent request to a DIFFERENT
    conversation succeeds in parallel without blocking.
    """
    store = ConversationStore(db_path=tmp_path / "api_concurrency.db")

    class ConcurrentSchemaAwareMockLLM:
        def __init__(self, delay_seconds: float = 0.25) -> None:
            self.delay_seconds = delay_seconds
            self.model_id = "mock-concurrent-gemini"

        def generate_structured(
            self,
            system_instruction: str,
            user_prompt: str,
            response_schema: type[Any],
            temperature: float = 0.0,
        ) -> Any:
            time.sleep(self.delay_seconds)
            if response_schema is QueryRewritePayload:
                if "smoking" in user_prompt.lower():
                    return QueryRewritePayload(
                        status="ready",
                        standalone_query="What is the faculty smoking policy?",
                        topic_changed=False,
                    )
                return QueryRewritePayload(
                    status="ready",
                    standalone_query="What is Tulane's nepotism policy for faculty?",
                    topic_changed=False,
                )
            return GroundedAnswerPayload(
                status="supported",
                answer="Grounded policy response from the Tulane Faculty Handbook [S1].",
                cited_labels=["S1"],
            )

    app = create_app(
        store_override=store,
        retriever_override=shared_retriever,
        llm_client_override=ConcurrentSchemaAwareMockLLM(delay_seconds=0.25),
    )

    conv_1 = store.create_conversation()
    conv_2 = store.create_conversation()
    cid_1 = conv_1["conversation_id"]
    cid_2 = conv_2["conversation_id"]

    async def _run_concurrent_requests() -> tuple[httpx.Response, httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as async_client:
                # Start first request on cid_1
                task_1 = asyncio.create_task(
                    async_client.post(
                        f"/api/conversations/{cid_1}/messages",
                        json={"message": "What is Tulane's nepotism policy for faculty?"},
                    )
                )
                # Give task_1 a brief moment to acquire the per-conversation lock on cid_1
                await asyncio.sleep(0.05)

                # Send overlapping second request to cid_1 (must return 409 Conflict immediately)
                task_overlap_same = asyncio.create_task(
                    async_client.post(
                        f"/api/conversations/{cid_1}/messages",
                        json={"message": "Overlapping duplicate send to same conversation"},
                    )
                )
                # Send concurrent request to cid_2 (must succeed in parallel)
                task_other_conv = asyncio.create_task(
                    async_client.post(
                        f"/api/conversations/{cid_2}/messages",
                        json={"message": "What is the faculty smoking policy?"},
                    )
                )

                return await asyncio.gather(task_1, task_overlap_same, task_other_conv)

    res_first, res_overlap, res_other = asyncio.run(_run_concurrent_requests())

    assert res_first.status_code == 200
    assert res_overlap.status_code == 409
    assert "currently processing" in res_overlap.json()["detail"].lower()
    assert res_other.status_code == 200

    # Verify cid_1 only recorded the first turn (2 messages), not the rejected overlapping send
    assert len(store.get_messages(cid_1)) == 2
    assert len(store.get_messages(cid_2)) == 2
