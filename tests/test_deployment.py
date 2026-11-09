"""Deployment-oriented tests: readiness probe, optional access token, CORS, configurable DB path."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.conversation_store import ConversationStore
from backend.main import create_app


class _StubLLM:
    model_id = "stub"


class _StubRetriever:
    class index:  # noqa: N801 - mimics faiss index attribute
        ntotal = 3


def _app(tmp_path: Path, **kw):
    return create_app(store_override=ConversationStore(db_path=tmp_path / "t.db"), **kw)


def test_ready_503_when_index_missing(tmp_path: Path) -> None:
    app = _app(tmp_path, index_error_override="missing", llm_client_override=_StubLLM())
    with TestClient(app) as c:
        assert c.get("/api/health").status_code == 200  # diagnostic only
        r = c.get("/api/ready")
        assert r.status_code == 503 and r.json()["ready"] is False


def test_ready_503_when_llm_missing(tmp_path: Path) -> None:
    app = _app(tmp_path, retriever_override=_StubRetriever(), llm_error_override="no key")
    with TestClient(app) as c:
        assert c.get("/api/ready").status_code == 503


def test_ready_200_when_ready(tmp_path: Path) -> None:
    app = _app(tmp_path, retriever_override=_StubRetriever(), llm_client_override=_StubLLM())
    with TestClient(app) as c:
        r = c.get("/api/ready")
        assert r.status_code == 200 and r.json()["chunk_count"] == 3


def test_access_token_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACCESS_TOKEN", "s3cret")
    monkeypatch.setenv("CORS_ORIGINS", "https://localhost")
    app = _app(tmp_path, retriever_override=_StubRetriever(), llm_client_override=_StubLLM())
    with TestClient(app) as c:
        assert c.get("/api/ready").status_code == 200  # probes stay public
        assert c.get("/api/health").status_code == 200
        r = c.get("/api/conversations", headers={"Origin": "https://localhost"})
        assert r.status_code == 401
        assert r.headers.get("access-control-allow-origin") == "https://localhost"  # CORS still outermost
        assert c.get("/api/conversations", headers={"X-Access-Token": "wrong"}).status_code == 401
        assert c.post("/api/conversations", json={}).status_code == 401
        assert c.get("/api/conversations", headers={"X-Access-Token": "s3cret"}).status_code == 200
        pre = c.options(
            "/api/conversations",
            headers={
                "Origin": "https://localhost",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "x-access-token",
            },
        )
        assert pre.status_code == 200
        pre_del = c.options(
            "/api/conversations/conv123/document",
            headers={
                "Origin": "https://localhost",
                "Access-Control-Request-Method": "DELETE",
                "Access-Control-Request-Headers": "x-access-token",
            },
        )
        assert pre_del.status_code == 200


def test_no_token_means_open_local_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ACCESS_TOKEN", raising=False)
    app = _app(tmp_path, retriever_override=_StubRetriever(), llm_client_override=_StubLLM())
    with TestClient(app) as c:
        assert c.get("/api/conversations").status_code == 200


def test_db_path_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import backend.conversation_store as cs

    target = tmp_path / "disk" / "chat.db"
    monkeypatch.setenv("CONVERSATION_DB_PATH", str(target))
    try:
        importlib.reload(cs)
        assert cs.DEFAULT_DB_PATH == target
    finally:
        monkeypatch.delenv("CONVERSATION_DB_PATH")
        importlib.reload(cs)
    assert cs.DEFAULT_DB_PATH.name == "chat_history.db"


def test_unwritable_db_path_falls_back_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.conversation_store import ConversationStore

    # Simulates Render deployment where /var/data was configured in env but disk was not attached
    monkeypatch.setenv("CONVERSATION_DB_PATH", "Z:/nonexistent_drive_root/var/data/chat_history.db")
    store = ConversationStore()
    assert store.db_path.exists() or store.db_path.parent.exists()
    assert "Z:" not in str(store.db_path)


