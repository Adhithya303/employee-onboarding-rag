"""
Stage 3B: Interactive terminal conversational RAG application for the
Tulane University Faculty Onboarding Assistant.

Commands supported in interactive mode:
  /new                      Start a new conversation
  /list                     List saved conversations
  /resume <conversation_id> Resume an existing conversation from SQLite
  /history                  Display messages in the active conversation
  /exit                     Exit the application

Notes:
- Loads the CPU SentenceTransformer model and saved FAISS index once per session.
- Reads `GEMINI_API_KEY` and `GEMINI_MODEL` from `.env` in the project root.
- Never exposes API keys or internal chain-of-thought reasoning.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

try:
    from backend.conversation_store import DEFAULT_DB_PATH, ConversationStore
    from backend.llm_client import (
        DEFAULT_ENV_PATH,
        GeminiStructuredClient,
        LLMConfigurationError,
        LLMServiceError,
    )
    from backend.rag_engine import CitationValidationError, ConversationalRAGEngine, RAGTurnResult
    from backend.retriever import HandbookRetriever
except ImportError:
    from conversation_store import DEFAULT_DB_PATH, ConversationStore  # type: ignore
    from llm_client import (  # type: ignore
        DEFAULT_ENV_PATH,
        GeminiStructuredClient,
        LLMConfigurationError,
        LLMServiceError,
    )
    from rag_engine import (  # type: ignore
        CitationValidationError,
        ConversationalRAGEngine,
        RAGTurnResult,
    )
    from retriever import HandbookRetriever  # type: ignore


def print_turn_result(result: RAGTurnResult, debug: bool = False) -> None:
    """Format and print the assistant answer, cited sources, and optional debug info."""
    print("\n" + "=" * 88)
    if debug:
        print("[DEBUG]")
        print(f"  Conversation ID   : {result.conversation_id}")
        print(f"  Rewritten Query   : {result.rewritten_query or '(Clarification requested — no retrieval)'}")
        print(f"  Topic Changed     : {result.topic_changed}")
        if result.retrieved_chunks:
            chunk_summary = ", ".join(
                f"[{c['evidence_label']}] {c['chunk_id']} (p.{','.join(map(str, c['pdf_pages']))})"
                for c in result.retrieved_chunks
            )
            print(f"  Retrieved Chunks  : {chunk_summary}")
        else:
            print("  Retrieved Chunks  : None")
        print("-" * 88)

    print(f"Assistant ({result.status.upper()}):\n")
    print(result.answer)

    if result.cited_sources:
        print("\nCited Sources (from provided Tulane University Faculty Handbook):")
        for src in result.cited_sources:
            pages_str = ", ".join(str(p) for p in src.pdf_pages)
            print(
                f"  [{src.label}] {src.document_title} — {src.document_section} | "
                f"PDF Page(s): {pages_str} (Chunk: {src.chunk_id})"
            )
            print(f"       Excerpt: \"{src.excerpt}\"")
    elif result.status in ("unsupported", "clarify"):
        print("\nCited Sources: None")
    print("=" * 88)


def print_conversations_list(conversations: list[dict[str, Any]], active_id: str) -> None:
    """Print saved conversations from SQLite."""
    print("\nSaved Conversations:")
    print("-" * 88)
    if not conversations:
        print("  (No saved conversations found.)")
    else:
        for conv in conversations:
            marker = "*" if conv["conversation_id"] == active_id else " "
            print(
                f" {marker} ID: {conv['conversation_id']} | "
                f"Messages: {conv['message_count']:2d} | "
                f"Updated: {conv['updated_at']} | "
                f"Title: {conv['title']}"
            )
    print("-" * 88)


def print_conversation_history(conv: dict[str, Any], messages: list[dict[str, Any]]) -> None:
    """Print all turns in the active conversation."""
    print("\n" + "=" * 88)
    print(f"Conversation History — ID: {conv['conversation_id']} | Title: {conv['title']}")
    print("-" * 88)
    if not messages:
        print("  (No messages in this conversation yet.)")
    else:
        for msg in messages:
            role_header = "You" if msg["role"] == "user" else "Assistant"
            print(f"[{msg['created_at']}] {role_header}:")
            for line in msg["content"].splitlines():
                print(f"  {line}")
            if msg["role"] == "assistant" and msg.get("cited_sources"):
                cite_summary = ", ".join(
                    f"[{s['label']}] p.{','.join(map(str, s['pdf_pages']))}"
                    for s in msg["cited_sources"]
                )
                print(f"  Sources: {cite_summary}")
            print("-" * 44)
    print("=" * 88)


def handle_slash_command(
    cmd_line: str,
    store: ConversationStore,
    active_conv: dict[str, Any],
) -> tuple[bool, dict[str, Any], bool]:
    """
    Handle `/new`, `/list`, `/resume <id>`, `/history`, `/exit`.
    Returns `(was_command, updated_active_conv, should_exit)`.
    """
    stripped = cmd_line.strip()
    if not stripped.startswith("/"):
        return False, active_conv, False

    parts = stripped.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd == "/exit":
        print("Exiting Tulane Faculty Onboarding Assistant.")
        return True, active_conv, True

    if cmd == "/new":
        new_conv = store.create_conversation()
        print(f"\nStarted new conversation: {new_conv['conversation_id']}")
        return True, new_conv, False

    if cmd == "/list":
        convs = store.list_conversations()
        print_conversations_list(convs, active_id=active_conv["conversation_id"])
        return True, active_conv, False

    if cmd == "/resume":
        if not arg:
            print("\nUsage: /resume <conversation_id>")
            return True, active_conv, False
        found = store.get_conversation(arg)
        if found is None:
            print(f"\nConversation '{arg}' was not found. Use /list to view available IDs.")
            return True, active_conv, False
        print(
            f"\nResumed conversation: {found['conversation_id']} "
            f"('{found['title']}', {found['message_count']} messages)"
        )
        return True, found, False

    if cmd == "/history":
        refreshed = store.get_conversation(active_conv["conversation_id"]) or active_conv
        msgs = store.get_messages(active_conv["conversation_id"])
        print_conversation_history(refreshed, msgs)
        return True, refreshed, False

    print(
        f"\nUnknown command '{cmd}'. Available commands: "
        "/new, /list, /resume <conversation_id>, /history, /exit"
    )
    return True, active_conv, False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tulane University Faculty Onboarding Assistant — Conversational RAG (CPU + Gemini)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show rewritten standalone query and retrieved chunk IDs for each turn.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to SQLite conversation database.",
    )
    parser.add_argument(
        "--env-path",
        type=Path,
        default=DEFAULT_ENV_PATH,
        help="Path to .env configuration file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional conversation_id to resume at startup.",
    )
    parser.add_argument(
        "--question",
        "-q",
        action="append",
        help="Optional question(s) to run non-interactively before exiting.",
    )
    args = parser.parse_args()

    # 1. Validate Gemini API key and model configuration upfront
    try:
        llm_client = GeminiStructuredClient(env_path=args.env_path)
    except LLMConfigurationError as cfg_err:
        print(f"\nCONFIGURATION ERROR:\n{cfg_err}", file=sys.stderr)
        return 1

    # 2. Load saved FAISS index & CPU SentenceTransformer model once per session
    try:
        print("Loading saved FAISS index and CPU embedding model (once per session)...")
        retriever = HandbookRetriever()
    except Exception as idx_err:
        print(f"\nINDEX ERROR:\n{idx_err}", file=sys.stderr)
        return 1

    store = ConversationStore(db_path=args.db_path)
    engine = ConversationalRAGEngine(
        retriever=retriever,
        llm_client=llm_client,
        store=store,
    )

    if args.resume:
        resumed = store.get_conversation(args.resume)
        if resumed is None:
            print(f"\nERROR: Conversation '{args.resume}' not found in {args.db_path}.", file=sys.stderr)
            return 1
        active_conv = resumed
        print(f"Resumed conversation: {active_conv['conversation_id']} ('{active_conv['title']}')")
    else:
        active_conv = store.create_conversation()
        print(f"Started new conversation: {active_conv['conversation_id']}")

    print(
        f"Configured Gemini Model: {llm_client.model_id} | "
        f"Indexed Chunks: {retriever.index.ntotal} (CPU)"
    )
    print(
        "Scope: Tulane University Faculty Handbook (Updated Aug 7, 2026) & Appended Research Misconduct Policy.\n"
        "Commands: /new | /list | /resume <conversation_id> | /history | /exit"
    )

    # Non-interactive batch mode if --question flags were supplied
    if args.question:
        for q in args.question:
            if not q.strip():
                continue
            was_cmd, active_conv, should_exit = handle_slash_command(q, store, active_conv)
            if should_exit:
                return 0
            if was_cmd:
                continue
            try:
                turn_res = engine.ask(active_conv["conversation_id"], q)
                print_turn_result(turn_res, debug=args.debug)
            except (LLMServiceError, CitationValidationError) as err:
                print(
                    f"\n[SERVICE / VALIDATION ERROR — Turn was NOT saved as an assistant answer]\n  {err}",
                    file=sys.stderr,
                )
                return 1
        return 0

    # Interactive terminal loop
    while True:
        try:
            prompt_str = f"\n[{active_conv['conversation_id']}] Ask a faculty policy question (or /exit): "
            user_input = input(prompt_str).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Tulane Faculty Onboarding Assistant.")
            break

        if not user_input:
            continue

        was_cmd, active_conv, should_exit = handle_slash_command(user_input, store, active_conv)
        if should_exit:
            break
        if was_cmd:
            continue

        try:
            turn_res = engine.ask(active_conv["conversation_id"], user_input)
            print_turn_result(turn_res, debug=args.debug)
        except LLMServiceError as api_err:
            print(
                f"\n[API ERROR — No assistant response was saved for this turn]\n"
                f"  {api_err}\n"
                "  (Note: This is an API/service failure, NOT a missing handbook policy.)"
            )
        except CitationValidationError as val_err:
            print(
                f"\n[VALIDATION ERROR — No assistant response was saved for this turn]\n"
                f"  {val_err}"
            )
        except Exception as exc:
            print(f"\n[UNEXPECTED ERROR — Turn not saved]\n  {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

