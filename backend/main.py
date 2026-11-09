"""
FastAPI backend for the Tulane University Faculty Onboarding Assistant.

Security & Deployment Scope Note:
- This application is designed as a local, single-user onboarding assistant demo.
- Conversation IDs are short convenience identifiers stored in local SQLite (`data/conversations/chat_history.db`)
  and are NOT an authentication or access-control mechanism.
- All document embeddings and FAISS vector search run locally on CPU (`device="cpu"`).
- Hosted LLM calls use the official Google Gen AI SDK (`google-genai`) with credentials kept strictly
  on the backend in `.env` (`GEMINI_API_KEY`, `GEMINI_MODEL`). Secrets and raw stack traces are never
  exposed in API responses.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Literal

import anyio
from fastapi import FastAPI, HTTPException, Path as PathParam, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

try:
    from backend.conversation_store import DEFAULT_DB_PATH, ConversationStore
    from backend.llm_client import (
         DEFAULT_ENV_PATH,
        GeminiStructuredClient,
        LLMConfigurationError,
        LLMServiceError,
        _redact_secrets,
    )
    from backend.rag_engine import (
        CitationValidationError,
        ConversationalRAGEngine,
        StructuredLLMProtocol,
    )
    from backend.retriever import HandbookRetriever
except ImportError:
    from conversation_store import DEFAULT_DB_PATH, ConversationStore  # type: ignore
    from llm_client import (  # type: ignore
        DEFAULT_ENV_PATH,
        GeminiStructuredClient,
        LLMConfigurationError,
        LLMServiceError,
        _redact_secrets,
    )
    from rag_engine import (  # type: ignore
        CitationValidationError,
        ConversationalRAGEngine,
        StructuredLLMProtocol,
    )
    from retriever import HandbookRetriever  # type: ignore

MAX_MESSAGE_CHARS = 2000
DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173,"
    "http://127.0.0.1:5173,"
    "http://localhost:4173,"
    "http://127.0.0.1:4173"
)


# ---------------------------------------------------------------------------
# Typed Request & Response Schemas
# ---------------------------------------------------------------------------


class CitedSourceSchema(BaseModel):
    """Citation metadata returned with assistant answers and persisted in SQLite."""

    label: str = Field(description="Inline evidence label, e.g. 'S1'.")
    chunk_id: str = Field(description="Deterministic handbook chunk identifier.")
    document_title: str = Field(description="Title of the source document.")
    document_section: str = Field(
        description="Handbook section ('Faculty Handbook (Main)' or 'Appended Research Misconduct Policy')."
    )
    pdf_pages: list[int] = Field(description="1-based PDF page numbers for this passage.")
    excerpt: str = Field(description="Short supporting excerpt from the chunk.")


class MessageSchema(BaseModel):
    """Single user or assistant message in a conversation."""

    message_id: int
    conversation_id: str
    role: Literal["user", "assistant"]
    content: str
    status: Literal["supported", "partial", "unsupported", "clarify"] | str | None = None
    cited_sources: list[CitedSourceSchema] = Field(default_factory=list)
    created_at: str


class ConversationSummarySchema(BaseModel):
    """Metadata summary for a conversation in the sidebar list."""

    conversation_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


class ConversationsListResponse(BaseModel):
    """Response schema for GET /api/conversations."""

    conversations: list[ConversationSummarySchema]


class CreateConversationRequest(BaseModel):
    """Optional request body for POST /api/conversations."""

    title: str | None = Field(default=None, max_length=120)


class ConversationMessagesResponse(BaseModel):
    """Response schema for GET /api/conversations/{id}/messages."""

    conversation: ConversationSummarySchema
    messages: list[MessageSchema]


class SendMessageRequest(BaseModel):
    """Request schema for POST /api/conversations/{id}/messages."""

    message: str = Field(
        description="The user's handbook question or follow-up.",
    )

    @field_validator("message")
    @classmethod
    def validate_message_content(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("Message must be a string.")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Message cannot be empty or whitespace only.")
        if len(cleaned) > MAX_MESSAGE_CHARS:
            raise ValueError(
                f"Message exceeds maximum allowed length of {MAX_MESSAGE_CHARS} characters "
                f"(received {len(cleaned)} characters)."
            )
        return cleaned


class SendMessageResponse(BaseModel):
    """Response schema for POST /api/conversations/{id}/messages."""

    conversation_id: str
    conversation_title: str
    user_message_id: int
    assistant_message_id: int
    created_at: str
    status: Literal["supported", "partial", "unsupported", "clarify"]
    answer: str
    cited_sources: list[CitedSourceSchema] = Field(default_factory=list)
    user_message: MessageSchema
    assistant_message: MessageSchema


class HealthResponse(BaseModel):
    """Response schema for GET /api/health."""

    status: Literal["ok", "degraded"]
    index_loaded: bool
    chunk_count: int
    llm_configured: bool
    model_id: str | None = None
    detail: str | None = None


# ---------------------------------------------------------------------------
# Application State Container
# ---------------------------------------------------------------------------


class AppResources:
    """Holds singleton resources initialized once during FastAPI lifespan."""

    def __init__(
        self,
        store: ConversationStore,
        retriever: HandbookRetriever | None = None,
        llm_client: StructuredLLMProtocol | None = None,
        index_error: str | None = None,
        llm_error: str | None = None,
    ) -> None:
        self.store = store
        self.retriever = retriever
        self.llm_client = llm_client
        self.index_error = index_error
        self.llm_error = llm_error
        self.conversation_locks: dict[str, asyncio.Lock] = {}

    def get_conversation_lock(self, conversation_id: str) -> asyncio.Lock:
        lock = self.conversation_locks.get(conversation_id)
        if lock is None:
            lock = asyncio.Lock()
            self.conversation_locks[conversation_id] = lock
        return lock


def _parse_cors_origins() -> list[str]:
    raw = os.environ.get("CORS_ORIGINS", DEFAULT_CORS_ORIGINS)
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    return origins or ["http://localhost:5173", "http://127.0.0.1:5173"]


def _format_stored_message(msg: dict[str, Any]) -> MessageSchema:
    raw_sources = msg.get("cited_sources") or []
    sources = [
        CitedSourceSchema(
            label=str(s.get("label", "")),
            chunk_id=str(s.get("chunk_id", "")),
            document_title=str(s.get("document_title", "Tulane University Faculty Handbook")),
            document_section=str(s.get("document_section", "")),
            pdf_pages=[int(p) for p in (s.get("pdf_pages") or [])],
            excerpt=str(s.get("excerpt", "")),
        )
        for s in raw_sources
    ]
    return MessageSchema(
        message_id=int(msg["message_id"]),
        conversation_id=str(msg["conversation_id"]),
        role=msg["role"],
        content=str(msg["content"]),
        status=msg.get("status"),
        cited_sources=sources,
        created_at=str(msg["created_at"]),
    )


def _map_llm_service_error_to_http(exc: LLMServiceError) -> HTTPException:
    safe_detail = _redact_secrets(str(exc))
    code = exc.status_code
    lower = safe_detail.lower()

    if code == 429 or "rate limit" in lower or "429" in lower or "quota" in lower:
        return HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                "The Gemini API rate limit or quota was reached. "
                "Please wait a moment and try sending your question again."
            ),
        )
    if code in (408, 504) or "timed out" in lower or "timeout" in lower:
        return HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=(
                "The request to the Gemini API timed out before completing. "
                "Please try sending your question again."
            ),
        )
    if code in (401, 403):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_Unavailable
            if hasattr(status, "HTTP_503_SERVICE_Unavailable")
            else status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Gemini API authentication failed. Please verify GEMINI_API_KEY in the backend .env file."
            ),
        )
    return HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail=f"Upstream language model request failed: {safe_detail}",
    )


def create_app(
    *,
    db_path: Path = DEFAULT_DB_PATH,
    env_path: Path = DEFAULT_ENV_PATH,
    retriever_override: HandbookRetriever | None = None,
    llm_client_override: StructuredLLMProtocol | None = None,
    store_override: ConversationStore | None = None,
    index_error_override: str | None = None,
    llm_error_override: str | None = None,
) -> FastAPI:
    """
    Create and configure the FastAPI application.
    Supports dependency overrides for deterministic unit/integration testing without reloading models.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        store = store_override or await anyio.to_thread.run_sync(
            lambda: ConversationStore(db_path=db_path)
        )

        retriever: HandbookRetriever | None = retriever_override
        index_error: str | None = index_error_override
        if retriever is None and index_error is None:
            try:
                retriever = await anyio.to_thread.run_sync(HandbookRetriever)
            except Exception as exc:
                index_error = _redact_secrets(str(exc))

        llm_client: StructuredLLMProtocol | None = llm_client_override
        llm_error: str | None = llm_error_override
        if llm_client is None and llm_error is None:
            try:
                llm_client = await anyio.to_thread.run_sync(
                    lambda: GeminiStructuredClient(env_path=env_path)
                )
            except LLMConfigurationError as cfg_exc:
                llm_error = _redact_secrets(str(cfg_exc))
            except Exception as exc:
                llm_error = _redact_secrets(str(exc))

        app.state.resources = AppResources(
            store=store,
            retriever=retriever,
            llm_client=llm_client,
            index_error=index_error,
            llm_error=llm_error,
        )
        yield

    app = FastAPI(
        title="Tulane University Faculty Onboarding Assistant API",
        description=(
            "Local single-user conversational RAG API grounded in the supplied Tulane University "
            "Faculty Handbook (Updated August 7, 2026) and appended Research Misconduct Policy."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_parse_cors_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Accept"],
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        messages: list[str] = []
        for err in exc.errors():
            msg = str(err.get("msg", "Invalid request input."))
            if msg.startswith("Value error, "):
                msg = msg[len("Value error, ") :]
            messages.append(msg)
        combined = " ".join(messages) if messages else "Invalid request payload."
        return JSONResponse(
            status_code=getattr(
                status,
                "HTTP_422_UNPROCESSABLE_CONTENT",
                status.HTTP_422_UNPROCESSABLE_ENTITY,
            ),
            content={"detail": combined},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        # Never expose raw tracebacks or secrets in API responses
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "An unexpected internal server error occurred while processing the request."
            },
        )

    def _get_resources(request: Request) -> AppResources:
        return request.app.state.resources

    @app.get("/api/health", response_model=HealthResponse)
    async def get_health(request: Request) -> HealthResponse:
        res = _get_resources(request)
        index_loaded = res.retriever is not None and res.index_error is None
        chunk_count = int(res.retriever.index.ntotal) if index_loaded and res.retriever else 0
        llm_configured = res.llm_client is not None and res.llm_error is None
        model_id = getattr(res.llm_client, "model_id", None) if llm_configured else None

        issues: list[str] = []
        if not index_loaded:
            issues.append(
                f"FAISS index unavailable: {res.index_error or 'Run backend/build_index.py first.'}"
            )
        if not llm_configured:
            issues.append(
                f"Gemini configuration unavailable: {res.llm_error or 'Check GEMINI_API_KEY and GEMINI_MODEL in .env.'}"
            )

        return HealthResponse(
            status="ok" if (index_loaded and llm_configured) else "degraded",
            index_loaded=index_loaded,
            chunk_count=chunk_count,
            llm_configured=llm_configured,
            model_id=model_id,
            detail=" | ".join(issues) if issues else None,
        )

    @app.post(
        "/api/conversations",
        response_model=ConversationSummarySchema,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_conversation(
        request: Request,
        payload: CreateConversationRequest | None = None,
    ) -> ConversationSummarySchema:
        res = _get_resources(request)
        title = payload.title if payload is not None else None
        created = await anyio.to_thread.run_sync(
            lambda: res.store.create_conversation(title=title)
        )
        return ConversationSummarySchema(**created)

    @app.get("/api/conversations", response_model=ConversationsListResponse)
    async def list_conversations(
        request: Request,
        limit: int = Query(default=50, ge=1, le=200),
    ) -> ConversationsListResponse:
        res = _get_resources(request)
        rows = await anyio.to_thread.run_sync(
            lambda: res.store.list_conversations(limit=limit)
        )
        return ConversationsListResponse(
            conversations=[ConversationSummarySchema(**r) for r in rows]
        )

    @app.get(
        "/api/conversations/{conversation_id}/messages",
        response_model=ConversationMessagesResponse,
    )
    async def get_conversation_messages(
        request: Request,
        conversation_id: str = PathParam(..., min_length=1, max_length=64),
    ) -> ConversationMessagesResponse:
        res = _get_resources(request)
        cid = conversation_id.strip()

        def _load_conv_and_messages() -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
            conv = res.store.get_conversation(cid)
            if conv is None:
                return None, []
            msgs = res.store.get_messages(cid)
            return conv, msgs

        conv, raw_messages = await anyio.to_thread.run_sync(_load_conv_and_messages)
        if conv is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation '{cid}' was not found.",
            )

        return ConversationMessagesResponse(
            conversation=ConversationSummarySchema(**conv),
            messages=[_format_stored_message(m) for m in raw_messages],
        )

    @app.post(
        "/api/conversations/{conversation_id}/messages",
        response_model=SendMessageResponse,
    )
    async def send_conversation_message(
        request: Request,
        payload: SendMessageRequest,
        conversation_id: str = PathParam(..., min_length=1, max_length=64),
    ) -> SendMessageResponse:
        res = _get_resources(request)
        cid = conversation_id.strip()

        # 1. Verify the conversation exists before checking service readiness or acquiring lock
        conv = await anyio.to_thread.run_sync(lambda: res.store.get_conversation(cid))
        if conv is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation '{cid}' was not found.",
            )

        # 2. Verify FAISS index and Gemini LLM configuration are available
        if res.retriever is None or res.index_error is not None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "The handbook vector index is unavailable or out of date. "
                    "Please run `.\\.venv\\Scripts\\python backend\\build_index.py` from the project root."
                ),
            )

        # If LLM wasn't configured at startup, try loading once more in case the user updated .env
        llm_client = res.llm_client
        if llm_client is None:
            try:
                llm_client = await anyio.to_thread.run_sync(
                    lambda: GeminiStructuredClient(env_path=env_path)
                )
                res.llm_client = llm_client
                res.llm_error = None
            except LLMConfigurationError as cfg_exc:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=_redact_secrets(str(cfg_exc)),
                ) from cfg_exc
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Gemini configuration error: {_redact_secrets(str(exc))}",
                ) from exc

        # 3. Serialize turns within the same conversation; reject overlapping sends with 409 Conflict
        conv_lock = res.get_conversation_lock(cid)
        if conv_lock.locked():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "This conversation is currently processing a previous message. "
                    "Please wait for the current response to finish before sending another."
                ),
            )

        async with conv_lock:
            def _execute_turn() -> tuple[Any, dict[str, Any] | None]:
                engine = ConversationalRAGEngine(
                    retriever=res.retriever,  # type: ignore[arg-type]
                    llm_client=llm_client,
                    store=res.store,
                )
                turn_result = engine.ask(
                    conversation_id=cid,
                    user_question=payload.message,
                )
                updated_conv = res.store.get_conversation(cid)
                return turn_result, updated_conv

            try:
                turn_result, updated_conv = await anyio.to_thread.run_sync(_execute_turn)
            except KeyError as key_exc:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Conversation '{cid}' was not found.",
                ) from key_exc
            except LLMConfigurationError as cfg_exc:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=_redact_secrets(str(cfg_exc)),
                ) from cfg_exc
            except LLMServiceError as svc_exc:
                raise _map_llm_service_error_to_http(svc_exc) from svc_exc
            except CitationValidationError as cit_exc:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=(
                        "The language model returned an improperly cited response after retry. "
                        "Please try submitting your question again."
                    ),
                ) from cit_exc

        created_at = turn_result.created_at or (
            updated_conv["updated_at"] if updated_conv else ""
        )
        user_msg_id = int(turn_result.user_message_id or 0)
        assistant_msg_id = int(turn_result.assistant_message_id or 0)
        conv_title = (
            str(updated_conv["title"])
            if updated_conv is not None
            else str(conv["title"])
        )

        cited_source_schemas = [
            CitedSourceSchema(
                label=s.label,
                chunk_id=s.chunk_id,
                document_title=s.document_title,
                document_section=s.document_section,
                pdf_pages=list(s.pdf_pages),
                excerpt=s.excerpt,
            )
            for s in turn_result.cited_sources
        ]

        user_message_schema = MessageSchema(
            message_id=user_msg_id,
            conversation_id=cid,
            role="user",
            content=payload.message,
            status=turn_result.status,
            cited_sources=[],
            created_at=created_at,
        )
        assistant_message_schema = MessageSchema(
            message_id=assistant_msg_id,
            conversation_id=cid,
            role="assistant",
            content=turn_result.answer,
            status=turn_result.status,
            cited_sources=cited_source_schemas,
            created_at=created_at,
        )

        return SendMessageResponse(
            conversation_id=cid,
            conversation_title=conv_title,
            user_message_id=user_msg_id,
            assistant_message_id=assistant_msg_id,
            created_at=created_at,
            status=turn_result.status,
            answer=turn_result.answer,
            cited_sources=cited_source_schemas,
            user_message=user_message_schema,
            assistant_message=assistant_message_schema,
        )

    return app


app = create_app()
