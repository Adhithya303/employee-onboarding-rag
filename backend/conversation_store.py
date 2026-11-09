"""
SQLite conversation and message store for the Tulane University Faculty Onboarding Assistant.

Provides isolated conversation persistence, bounded recent message retrieval for follow-up
rewriting, and atomic turn recording so failed LLM calls are never saved as assistant answers.
"""

from __future__ import annotations

import datetime
import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_DB_PATH = PROJECT_ROOT / "data" / "conversations" / "chat_history.db"
# Local default is unchanged. Deployments (e.g. Render persistent disk) can override it with
# the CONVERSATION_DB_PATH environment variable, e.g. /var/data/chat_history.db.
DEFAULT_DB_PATH = (
    Path(os.environ["CONVERSATION_DB_PATH"]) if os.environ.get("CONVERSATION_DB_PATH") else _LOCAL_DB_PATH
)


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


class ConversationStore:
    """SQLite-backed store for conversations and turns."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA busy_timeout = 15000;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    rewritten_query TEXT,
                    status TEXT,
                    cited_sources_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_conv_id
                ON messages(conversation_id, message_id);
                """
            )

    def create_conversation(self, title: str | None = None) -> dict[str, Any]:
        """Start and persist a new conversation with a short unique ID."""
        conv_id = uuid.uuid4().hex[:8]
        now = _utc_now_iso()
        resolved_title = (title or "New Faculty Onboarding Conversation").strip()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations (conversation_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (conv_id, resolved_title, now, now),
            )
            conn.commit()

        return {
            "conversation_id": conv_id,
            "title": resolved_title,
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
        }

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        """Fetch a conversation by ID, returning None if not found."""
        cid = conversation_id.strip()
        if not cid:
            return None
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    c.conversation_id,
                    c.title,
                    c.created_at,
                    c.updated_at,
                    COUNT(m.message_id) AS message_count
                FROM conversations c
                LEFT JOIN messages m ON c.conversation_id = m.conversation_id
                WHERE c.conversation_id = ?
                GROUP BY c.conversation_id
                """,
                (cid,),
            ).fetchone()

        if row is None:
            return None
        return dict(row)

    def list_conversations(self, limit: int = 50) -> list[dict[str, Any]]:
        """List saved conversations ordered by most recently updated."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    c.conversation_id,
                    c.title,
                    c.created_at,
                    c.updated_at,
                    COUNT(m.message_id) AS message_count
                FROM conversations c
                LEFT JOIN messages m ON c.conversation_id = m.conversation_id
                GROUP BY c.conversation_id
                ORDER BY c.updated_at DESC, c.created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_messages(
        self,
        conversation_id: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return messages for `conversation_id` in chronological order.
        If `limit` is specified, returns the most recent `limit` messages in chronological order.
        """
        cid = conversation_id.strip()
        with self._connect() as conn:
            if limit is not None and limit > 0:
                rows = conn.execute(
                    """
                    SELECT * FROM (
                        SELECT
                            message_id,
                            conversation_id,
                            role,
                            content,
                            rewritten_query,
                            status,
                            cited_sources_json,
                            created_at
                        FROM messages
                        WHERE conversation_id = ?
                        ORDER BY message_id DESC
                        LIMIT ?
                    )
                    ORDER BY message_id ASC
                    """,
                    (cid, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT
                        message_id,
                        conversation_id,
                        role,
                        content,
                        rewritten_query,
                        status,
                        cited_sources_json,
                        created_at
                    FROM messages
                    WHERE conversation_id = ?
                    ORDER BY message_id ASC
                    """,
                    (cid,),
                ).fetchall()

        results: list[dict[str, Any]] = []
        for r in rows:
            item = dict(r)
            raw_sources = item.pop("cited_sources_json", None)
            item["cited_sources"] = json.loads(raw_sources) if raw_sources else []
            results.append(item)
        return results

    def save_completed_turn(
        self,
        conversation_id: str,
        user_question: str,
        assistant_answer: str,
        rewritten_query: str | None = None,
        status: str = "answered",
        cited_sources: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Atomically save a completed user + assistant turn.
        Only called after the LLM response is generated and validated so that failed API calls
        are never recorded as assistant answers.
        Returns metadata for the newly created user and assistant message records.
        """
        cid = conversation_id.strip()
        now = _utc_now_iso()
        sources_list = list(cited_sources or [])
        sources_json = json.dumps(sources_list, ensure_ascii=False)

        with self._connect() as conn:
            conv = conn.execute(
                "SELECT title FROM conversations WHERE conversation_id = ?",
                (cid,),
            ).fetchone()
            if conv is None:
                raise KeyError(f"Conversation '{cid}' does not exist.")

            # Auto-title the conversation from the first user question
            current_title = conv["title"]
            existing_count = conn.execute(
                "SELECT COUNT(*) AS cnt FROM messages WHERE conversation_id = ?",
                (cid,),
            ).fetchone()["cnt"]

            new_title = current_title
            if existing_count == 0 and current_title == "New Faculty Onboarding Conversation":
                compact_q = " ".join(user_question.strip().split())
                new_title = compact_q[:60] + ("..." if len(compact_q) > 60 else "")

            user_cursor = conn.execute(
                """
                INSERT INTO messages (
                    conversation_id, role, content, rewritten_query, status, cited_sources_json, created_at
                )
                VALUES (?, 'user', ?, ?, ?, '[]', ?)
                """,
                (cid, user_question.strip(), rewritten_query, status, now),
            )
            user_message_id = int(user_cursor.lastrowid or 0)

            assistant_cursor = conn.execute(
                """
                INSERT INTO messages (
                    conversation_id, role, content, rewritten_query, status, cited_sources_json, created_at
                )
                VALUES (?, 'assistant', ?, ?, ?, ?, ?)
                """,
                (cid, assistant_answer.strip(), rewritten_query, status, sources_json, now),
            )
            assistant_message_id = int(assistant_cursor.lastrowid or 0)

            conn.execute(
                """
                UPDATE conversations
                SET title = ?, updated_at = ?
                WHERE conversation_id = ?
                """,
                (new_title, now, cid),
            )
            conn.commit()

        return {
            "conversation_id": cid,
            "title": new_title,
            "user_message_id": user_message_id,
            "assistant_message_id": assistant_message_id,
            "created_at": now,
        }

