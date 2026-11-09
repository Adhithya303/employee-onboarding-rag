"""
Focused validation test suite for the Tulane University Faculty Onboarding Assistant
conversational RAG engine (using a mocked LLM + real CPU FAISS retrieval and SQLite persistence).

Covers all required scenarios:
1. Follow-up question rewriting
2. Ambiguous follow-up clarification
3. Explicit topic changes
4. Unsupported questions ("I couldn’t find that in the provided handbook.")
5. Partial answers
6. Unknown citation labels (retry-once recovery & rejection on persistent bad labels)
7. Conversation isolation and SQLite persistence (including /new, /list, /resume, /history)
8. API failure handling (bounded transient retries, no endless auth retries, no failed turn saved,
   secret redaction, missing .env instructions)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from google.genai import errors as genai_errors

from backend.conversation_store import ConversationStore
from backend.llm_client import (
    GeminiSettings,
    GeminiStructuredClient,
    LLMConfigurationError,
    LLMServiceError,
    _redact_secrets,
    load_gemini_settings,
)
from backend.rag_engine import (
    UNSUPPORTED_ANSWER_TEXT,
    CitationValidationError,
    ConversationalRAGEngine,
    GroundedAnswerPayload,
    QueryRewritePayload,
)
from backend.retriever import HandbookRetriever


@pytest.fixture(scope="module")
def shared_retriever() -> HandbookRetriever:
    """Load the real CPU SentenceTransformer and saved FAISS index once for the test module."""
    return HandbookRetriever()


class ScriptedMockLLM:
    """
    Deterministic mock implementing `StructuredLLMProtocol` that records prompts
    and returns scripted Pydantic payloads or raises configured exceptions.
    """

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate_structured(
        self,
        system_instruction: str,
        user_prompt: str,
        response_schema: type[Any],
        temperature: float = 0.0,
    ) -> Any:
        self.calls.append(
            {
                "system_instruction": system_instruction,
                "user_prompt": user_prompt,
                "response_schema": response_schema,
                "temperature": temperature,
            }
        )
        if not self.responses:
            raise AssertionError("ScriptedMockLLM ran out of scripted responses.")
        next_item = self.responses.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        return next_item


def test_1_followup_question_rewriting(shared_retriever: HandbookRetriever, tmp_path: Path) -> None:
    """
    After discussing faculty Paid Parental Leave, a follow-up like 'Who approves it?'
    must include prior turn history in the rewrite prompt, rewrite to a standalone
    parental-leave approval query, retrieve page 151 chunks, and return grounded citations.
    """
    store = ConversationStore(db_path=tmp_path / "conv.db")
    conv = store.create_conversation()
    cid = conv["conversation_id"]

    mock_llm = ScriptedMockLLM(
        [
            # Turn 1: initial question about Paid Parental Leave
            QueryRewritePayload(
                status="ready",
                standalone_query="How many weeks of Paid Parental Leave do full-time faculty receive?",
                topic_changed=False,
            ),
            GroundedAnswerPayload(
                status="supported",
                answer=(
                    "Under the Tulane University Faculty Handbook (Updated August 7, 2026), "
                    "full-time faculty receive up to 6 weeks of continuous Paid Parental Leave "
                    "at 100% of compensation within a rolling backward 12-month period [S2]."
                ),
                cited_labels=["S2"],
            ),
            # Turn 2: follow-up "Who approves it?"
            QueryRewritePayload(
                status="ready",
                standalone_query="Who approves faculty Paid Parental Leave requests at Tulane University?",
                topic_changed=False,
            ),
            GroundedAnswerPayload(
                status="supported",
                answer=(
                    "For Tulane faculty, Paid Parental Leave requests must be made to the leave "
                    "administrator (at least 30 days in advance when foreseeable), and university "
                    "approvals are made by the dean and the Senior Vice President for Academic "
                    "Affairs and Provost [S1]."
                ),
                cited_labels=["S1"],
            ),
        ]
    )

    engine = ConversationalRAGEngine(retriever=shared_retriever, llm_client=mock_llm, store=store)

    res1 = engine.ask(cid, "How many weeks of Paid Parental Leave do full-time faculty receive?")
    assert res1.status == "supported"
    assert any(150 in s.pdf_pages or 151 in s.pdf_pages for s in res1.cited_sources)

    res2 = engine.ask(cid, "Who approves it?")
    assert res2.status == "supported"
    assert res2.rewritten_query == "Who approves faculty Paid Parental Leave requests at Tulane University?"
    # Verify that Turn 1 history was passed into Turn 2's rewrite call
    rewrite_call_2_prompt = mock_llm.calls[2]["user_prompt"]
    assert "Paid Parental Leave" in rewrite_call_2_prompt
    assert "Who approves it?" in rewrite_call_2_prompt
    assert len(res2.cited_sources) == 1
    assert 151 in res2.cited_sources[0].pdf_pages


def test_2_ambiguous_followup_clarification(
    shared_retriever: HandbookRetriever, tmp_path: Path
) -> None:
    """
    When a user asks an ambiguous question (e.g., 'How long does it last?' after a turn that
    mentioned both Paid Parental Leave and Short-Term Disability Leave, or a pronoun with no
    referent), the engine must ask a clarification question without guessing or citing sources.
    """
    store = ConversationStore(db_path=tmp_path / "conv.db")
    conv = store.create_conversation()
    cid = conv["conversation_id"]

    mock_llm = ScriptedMockLLM(
        [
            QueryRewritePayload(
                status="clarify",
                clarification_question=(
                    "Are you asking about the duration of Paid Parental Leave or Short-Term "
                    "Disability Leave for Tulane faculty?"
                ),
            )
        ]
    )

    engine = ConversationalRAGEngine(retriever=shared_retriever, llm_client=mock_llm, store=store)
    res = engine.ask(cid, "How long does it last?")

    assert res.status == "clarify"
    assert "Paid Parental Leave or Short-Term Disability Leave" in res.answer
    assert res.rewritten_query is None
    assert res.cited_sources == []
    assert res.retrieved_chunks == []
    # Only the rewriter should have been called; answer generation should not run on clarify
    assert len(mock_llm.calls) == 1


def test_3_explicit_topic_changes(shared_retriever: HandbookRetriever, tmp_path: Path) -> None:
    """
    When the user switches from a leave topic to a Research Misconduct Policy question,
    `topic_changed` is preserved and retrieval pulls from `Appended Research Misconduct Policy`.
    """
    store = ConversationStore(db_path=tmp_path / "conv.db")
    conv = store.create_conversation()
    cid = conv["conversation_id"]

    # Seed prior turn on parental leave
    store.save_completed_turn(
        conversation_id=cid,
        user_question="How much paid parental leave do faculty get?",
        assistant_answer="Full-time faculty receive 6 weeks of Paid Parental Leave [S1].",
        rewritten_query="How much paid parental leave do faculty get?",
        status="supported",
        cited_sources=[],
    )

    mock_llm = ScriptedMockLLM(
        [
            QueryRewritePayload(
                status="ready",
                standalone_query=(
                    "Within how many days must a research misconduct investigation be completed "
                    "and what is the standard of proof?"
                ),
                topic_changed=True,
            ),
            GroundedAnswerPayload(
                status="supported",
                answer=(
                    "Under the appended Tulane University Research Misconduct Policy, an "
                    "Investigation must be completed within 120 days of initiation (or 180 days "
                    "for NSF-sponsored research), and the standard of proof is a Preponderance "
                    "of the Evidence [S1]."
                ),
                cited_labels=["S1"],
            ),
        ]
    )

    engine = ConversationalRAGEngine(retriever=shared_retriever, llm_client=mock_llm, store=store)
    res = engine.ask(
        cid,
        "Switching topics: within how many days must a research misconduct investigation be completed and what is the standard of proof?",
    )

    assert res.topic_changed is True
    assert res.status == "supported"
    assert res.cited_sources[0].document_section == "Appended Research Misconduct Policy"
    assert any(p >= 192 for p in res.cited_sources[0].pdf_pages)


def test_4_unsupported_questions(shared_retriever: HandbookRetriever, tmp_path: Path) -> None:
    """
    When retrieved passages lack the answer, the engine must return the exact string:
    'I couldn’t find that in the provided handbook.' with no citations.
    """
    store = ConversationStore(db_path=tmp_path / "conv.db")
    conv = store.create_conversation()
    cid = conv["conversation_id"]

    mock_llm = ScriptedMockLLM(
        [
            QueryRewritePayload(
                status="ready",
                standalone_query="What is the campus shuttle bus schedule to the airport?",
                topic_changed=False,
            ),
            GroundedAnswerPayload(
                status="unsupported",
                answer=UNSUPPORTED_ANSWER_TEXT,
                cited_labels=[],
            ),
        ]
    )

    engine = ConversationalRAGEngine(retriever=shared_retriever, llm_client=mock_llm, store=store)
    res = engine.ask(cid, "What is the campus shuttle bus schedule to the airport?")

    assert res.status == "unsupported"
    assert res.answer == "I couldn’t find that in the provided handbook."
    assert res.cited_sources == []


def test_5_partial_answers(shared_retriever: HandbookRetriever, tmp_path: Path) -> None:
    """
    When only part of a multi-part question is supported by the handbook, the engine answers
    the supported part with citations and explicitly identifies what is missing.
    """
    store = ConversationStore(db_path=tmp_path / "conv.db")
    conv = store.create_conversation()
    cid = conv["conversation_id"]

    mock_llm = ScriptedMockLLM(
        [
            QueryRewritePayload(
                status="ready",
                standalone_query="What is the faculty smoking policy and what brand of air purifiers are installed in classrooms?",
                topic_changed=False,
            ),
            GroundedAnswerPayload(
                status="partial",
                answer=(
                    "In the Tulane University Faculty Handbook, smoking is prohibited in all "
                    "University buildings—including work areas, classrooms, lounges, restrooms, "
                    "hallways, and vehicles—and employers face fines up to $500 for non-compliance [S1]."
                ),
                cited_labels=["S1"],
                missing_information=(
                    "The provided handbook does not contain information about classroom air purifier brands."
                ),
            ),
        ]
    )

    engine = ConversationalRAGEngine(retriever=shared_retriever, llm_client=mock_llm, store=store)
    res = engine.ask(
        cid,
        "What is the faculty smoking policy and what brand of air purifiers are installed in classrooms?",
    )

    assert res.status == "partial"
    assert "[S1]" in res.answer
    assert "air purifier" in res.answer.lower()
    assert len(res.cited_sources) == 1
    assert 61 in res.cited_sources[0].pdf_pages


def test_6_unknown_citation_labels_retry_and_rejection(
    shared_retriever: HandbookRetriever, tmp_path: Path
) -> None:
    """
    Part A: If the model returns an unknown citation label (e.g., [S99]) on attempt 1,
            the engine rejects it and retries once; when attempt 2 returns valid [S1], it succeeds.
    Part B: If the model returns unknown citation labels on both attempts, `CitationValidationError`
            is raised and the failed turn is NOT saved to SQLite.
    """
    store = ConversationStore(db_path=tmp_path / "conv.db")
    conv = store.create_conversation()
    cid = conv["conversation_id"]

    # Part A: Recover on retry
    mock_llm_recover = ScriptedMockLLM(
        [
            QueryRewritePayload(
                status="ready",
                standalone_query="What is Tulane's nepotism policy for faculty?",
                topic_changed=False,
            ),
            # Attempt 1: bogus label [S99]
            GroundedAnswerPayload(
                status="supported",
                answer="Tulane prohibits an employee from supervising a relative [S99].",
                cited_labels=["S99"],
            ),
            # Attempt 2 (retry): valid label [S1]
            GroundedAnswerPayload(
                status="supported",
                answer=(
                    "Under the Tulane University Faculty Handbook Nepotism Policy, it is against "
                    "University policy for an employee to supervise a relative or work in positions "
                    "with an audit or control function over a relative without prior written approval [S1]."
                ),
                cited_labels=["S1"],
            ),
        ]
    )

    engine_recover = ConversationalRAGEngine(
        retriever=shared_retriever, llm_client=mock_llm_recover, store=store
    )
    res = engine_recover.ask(cid, "What is Tulane's nepotism policy for faculty?")
    assert res.status == "supported"
    assert len(res.cited_sources) == 1
    assert res.cited_sources[0].label == "S1"
    # 1 rewrite call + 2 answer generation calls = 3 total LLM calls
    assert len(mock_llm_recover.calls) == 3
    assert "CORRECTION REQUIRED" in mock_llm_recover.calls[2]["user_prompt"]

    # Part B: Persistent invalid labels must raise CitationValidationError and NOT save to SQLite
    conv_fail = store.create_conversation()
    cid_fail = conv_fail["conversation_id"]
    mock_llm_fail = ScriptedMockLLM(
        [
            QueryRewritePayload(
                status="ready",
                standalone_query="What is Tulane's nepotism policy for faculty?",
                topic_changed=False,
            ),
            GroundedAnswerPayload(
                status="supported",
                answer="Nepotism policy claim [S42].",
                cited_labels=["S42"],
            ),
            GroundedAnswerPayload(
                status="supported",
                answer="Still citing non-existent label [S99].",
                cited_labels=["S99"],
            ),
        ]
    )
    engine_fail = ConversationalRAGEngine(
        retriever=shared_retriever, llm_client=mock_llm_fail, store=store
    )
    with pytest.raises(CitationValidationError):
        engine_fail.ask(cid_fail, "What is Tulane's nepotism policy for faculty?")

    # Verify failed turn was NOT saved in SQLite
    assert store.get_messages(cid_fail) == []


def test_7_conversation_isolation_and_persistence(
    shared_retriever: HandbookRetriever, tmp_path: Path
) -> None:
    """
    Verify that separate conversations stored in SQLite are strictly isolated during follow-up
    rewriting and can be listed and resumed across store instances.
    """
    db_file = tmp_path / "isolated.db"
    store1 = ConversationStore(db_path=db_file)
    conv_a = store1.create_conversation()
    conv_b = store1.create_conversation()

    store1.save_completed_turn(
        conversation_id=conv_a["conversation_id"],
        user_question="Tell me about faculty sabbatical leave.",
        assistant_answer="Sabbatical leave is a privilege of tenure-track faculty [S1].",
        rewritten_query="Tell me about faculty sabbatical leave.",
        status="supported",
        cited_sources=[
            {
                "label": "S1",
                "chunk_id": "fh_main_0069_1abfabd425",
                "document_title": "Tulane University Faculty Handbook",
                "document_section": "Faculty Handbook (Main)",
                "pdf_pages": [25],
                "excerpt": "Sabbatical leaves are considered to be a privilege of tenure track faculty...",
            }
        ],
    )

    store1.save_completed_turn(
        conversation_id=conv_b["conversation_id"],
        user_question="Tell me about the Smoking Policy.",
        assistant_answer="Smoking is prohibited in all University buildings [S1].",
        rewritten_query="Tell me about the Smoking Policy.",
        status="supported",
        cited_sources=[],
    )

    # Re-open a fresh ConversationStore instance on the same SQLite file to verify persistence
    store2 = ConversationStore(db_path=db_file)
    listed = store2.list_conversations()
    assert len(listed) == 2

    msgs_a = store2.get_messages(conv_a["conversation_id"])
    msgs_b = store2.get_messages(conv_b["conversation_id"])
    assert len(msgs_a) == 2
    assert len(msgs_b) == 2
    assert "sabbatical" in msgs_a[0]["content"].lower()
    assert "smoking" in msgs_b[0]["content"].lower()

    # Run a follow-up in conv_b and verify conv_a's sabbatical history is NEVER leaked into conv_b
    mock_llm = ScriptedMockLLM(
        [
            QueryRewritePayload(
                status="ready",
                standalone_query="What is the maximum fine for violating the Tulane smoking policy?",
                topic_changed=False,
            ),
            GroundedAnswerPayload(
                status="supported",
                answer="The maximum fine for an employer is $500 [S1].",
                cited_labels=["S1"],
            ),
        ]
    )
    engine = ConversationalRAGEngine(retriever=shared_retriever, llm_client=mock_llm, store=store2)
    engine.ask(conv_b["conversation_id"], "What is the maximum fine?")

    rewrite_prompt_b = mock_llm.calls[0]["user_prompt"]
    assert "Smoking Policy" in rewrite_prompt_b
    assert "sabbatical" not in rewrite_prompt_b.lower()


def test_8_api_failure_handling_and_key_protection(
    shared_retriever: HandbookRetriever, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Verify:
    - Missing `.env` / placeholder keys raise `LLMConfigurationError` with setup instructions.
    - Authentication errors (401/403) are NOT retried endlessly (1 attempt only).
    - Transient rate-limit errors (429/503) ARE retried up to `max_retries`.
    - API failures do not save partial/failed turns in SQLite.
    - API keys are redacted from error strings and settings repr.
    """
    # 1. Missing / placeholder configuration check
    empty_env = tmp_path / ".env.empty"
    empty_env.write_text(
        "GEMINI_API_KEY=your_gemini_api_key_here\nGEMINI_MODEL=your_available_gemini_model_id_here\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    with pytest.raises(LLMConfigurationError) as cfg_exc_info:
        load_gemini_settings(env_path=empty_env)
    assert "GEMINI_API_KEY" in str(cfg_exc_info.value)
    assert "search_demo.py" in str(cfg_exc_info.value)

    # 2. Secret redaction check
    fake_secret = "AIzaSyFakeSecretKeyForUnitTesting123456"
    settings = GeminiSettings(api_key=fake_secret, model_id="test-model", timeout_seconds=10.0, max_retries=2)
    assert fake_secret not in repr(settings)
    assert fake_secret not in _redact_secrets(f"Error with key {fake_secret}", api_key=fake_secret)

    # 3. Verify non-retryable auth error (401) stops on first attempt, whereas transient (503) retries
    sleep_calls: list[float] = []
    client = GeminiStructuredClient(settings=settings, sleep_fn=lambda s: sleep_calls.append(s))

    auth_attempts = 0

    def fake_auth_fail(*args: Any, **kwargs: Any) -> Any:
        nonlocal auth_attempts
        auth_attempts += 1
        raise genai_errors.APIError(401, {"error": {"message": f"Invalid key {fake_secret}"}})

    monkeypatch.setattr(client._client.models, "generate_content", fake_auth_fail)
    with pytest.raises(LLMServiceError) as auth_err:
        client.generate_structured("sys", "user", QueryRewritePayload)
    assert auth_attempts == 1  # Must NOT retry 401 auth failure
    assert sleep_calls == []
    assert fake_secret not in str(auth_err.value)

    # 4. Verify transient 503 error retries boundedly (`1 + max_retries = 3` attempts)
    transient_attempts = 0

    def fake_transient_fail(*args: Any, **kwargs: Any) -> Any:
        nonlocal transient_attempts
        transient_attempts += 1
        raise genai_errors.APIError(503, {"error": {"message": "Service temporarily unavailable"}})

    monkeypatch.setattr(client._client.models, "generate_content", fake_transient_fail)
    with pytest.raises(LLMServiceError) as trans_err:
        client.generate_structured("sys", "user", QueryRewritePayload)
    assert trans_err.value.is_transient is True
    assert transient_attempts == 3  # 1 initial + 2 retries
    assert len(sleep_calls) == 2

    # 5. Verify failed API call during `engine.ask` does NOT save a turn in SQLite
    store = ConversationStore(db_path=tmp_path / "fail.db")
    conv = store.create_conversation()
    engine = ConversationalRAGEngine(retriever=shared_retriever, llm_client=client, store=store)
    with pytest.raises(LLMServiceError):
        engine.ask(conv["conversation_id"], "What is the faculty parental leave policy?")
    assert store.get_messages(conv["conversation_id"]) == []

