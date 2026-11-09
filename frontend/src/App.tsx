import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  ApiRequestError,
  createConversation,
  fetchConversationMessages,
  fetchConversations,
  fetchHealth,
  sendConversationMessage,
} from "./api";
import type {
  AnswerStatus,
  ChatMessage,
  CitedSource,
  ConversationSummary,
  HealthResponse,
} from "./types";

const ACTIVE_CONVERSATION_STORAGE_KEY =
  "tulane_faculty_onboarding_active_conversation_id";

interface ExampleQuestion {
  category: string;
  question: string;
  hint: string;
}

const EXAMPLE_QUESTIONS: ExampleQuestion[] = [
  {
    category: "Faculty Leave & Approvals",
    question:
      "How many weeks of Paid Parental Leave do full-time faculty receive, and who approves requests?",
    hint: "Tests grounded retrieval and follow-up context on Faculty Handbook pp. 150–151",
  },
  {
    category: "Sabbatical Leave",
    question:
      "Who is eligible for faculty sabbatical leave and what are the compensation and return obligations?",
    hint: "Covers tenure-track eligibility and service requirements on pp. 25, 159–160",
  },
  {
    category: "Research Misconduct Policy",
    question:
      "Within how many days must a research misconduct investigation be completed and what is the standard of proof?",
    hint: "Queries the Appended Research Misconduct Policy section (pp. 192–238)",
  },
  {
    category: "Workplace Policies",
    question:
      "What are the restrictions in the Nepotism Policy for faculty supervising or auditing a relative?",
    hint: "Retrieves conflict-of-interest and prior written approval rules (pp. 58–59)",
  },
];

function formatPagesLabel(pages: number[]): string {
  if (!pages || pages.length === 0) {
    return "Page N/A";
  }
  if (pages.length === 1) {
    return `PDF Page ${pages[0]}`;
  }
  return `PDF Pages ${pages.join(", ")}`;
}

function formatTimestamp(isoString: string): string {
  if (!isoString) return "";
  const date = new Date(isoString);
  if (Number.isNaN(date.getTime())) {
    return isoString;
  }
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function getStatusBadgeInfo(status?: AnswerStatus | string | null): {
  label: string;
  className: string;
  description?: string;
} | null {
  switch (status) {
    case "supported":
      return {
        label: "Grounded in Provided Handbook",
        className: "status-badge status-supported",
      };
    case "partial":
      return {
        label: "Partially Supported by Handbook",
        className: "status-badge status-partial",
        description:
          "Only part of this question is addressed in the supplied handbook passages.",
      };
    case "clarify":
      return {
        label: "Clarification Requested",
        className: "status-badge status-clarify",
        description:
          "Your question could refer to more than one faculty policy or topic. Reply below to clarify.",
      };
    case "unsupported":
      return {
        label: "Not Found in Provided Handbook",
        className: "status-badge status-unsupported",
        description:
          "The retrieved handbook passages do not contain information to answer this question.",
      };
    default:
      return null;
  }
}

interface SourceListProps {
  messageId: number;
  sources: CitedSource[];
}

const ExpandableSourceList: React.FC<SourceListProps> = ({
  messageId,
  sources,
}) => {
  const [sectionOpen, setSectionOpen] = useState<boolean>(true);
  const [expandedCards, setExpandedCards] = useState<Record<string, boolean>>(
    () => {
      const initial: Record<string, boolean> = {};
      sources.forEach((src, index) => {
        initial[`${messageId}-${src.label}-${index}`] = index === 0;
      });
      return initial;
    }
  );

  if (!sources || sources.length === 0) {
    return null;
  }

  const toggleCard = (key: string) => {
    setExpandedCards((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
  };

  return (
    <div className="sources-container">
      <div className="sources-header-row">
        <button
          type="button"
          className="sources-toggle-btn"
          onClick={() => setSectionOpen((prev) => !prev)}
          aria-expanded={sectionOpen}
        >
          <span className="sources-toggle-icon">{sectionOpen ? "▼" : "▶"}</span>
          <span>
            Cited Handbook Passages ({sources.length}{" "}
            {sources.length === 1 ? "source" : "sources"})
          </span>
        </button>
        <span className="sources-verification-hint">
          Verify policy claims against the PDF pages and excerpts below.
        </span>
      </div>

      {sectionOpen && (
        <div className="sources-cards-list">
          {sources.map((src, idx) => {
            const cardKey = `${messageId}-${src.label}-${idx}`;
            const isExpanded = Boolean(expandedCards[cardKey]);
            return (
              <div
                key={cardKey}
                className={`source-card ${isExpanded ? "expanded" : "collapsed"}`}
              >
                <button
                  type="button"
                  className="source-card-header"
                  onClick={() => toggleCard(cardKey)}
                  aria-expanded={isExpanded}
                >
                  <div className="source-card-title-group">
                    <span className="source-label-pill">[{src.label}]</span>
                    <span className="source-section-tag">
                      {src.document_section}
                    </span>
                    <span className="source-pages-tag">
                      {formatPagesLabel(src.pdf_pages)}
                    </span>
                  </div>
                  <div className="source-card-meta-right">
                    <code className="source-chunk-id">{src.chunk_id}</code>
                    <span className="source-expand-label">
                      {isExpanded ? "Hide excerpt" : "Show excerpt"}
                    </span>
                  </div>
                </button>

                {isExpanded && (
                  <div className="source-card-body">
                    <div className="source-card-doc-meta">
                      <span>
                        <strong>Document:</strong> {src.document_title}
                      </span>
                      <span>
                        <strong>Section:</strong> {src.document_section}
                      </span>
                      <span>
                        <strong>{formatPagesLabel(src.pdf_pages)}</strong>
                      </span>
                      <span>
                        <strong>Chunk ID:</strong> <code>{src.chunk_id}</code>
                      </span>
                    </div>
                    <blockquote className="source-excerpt">
                      “{src.excerpt}”
                    </blockquote>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export const App: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<
    string | null
  >(() => {
    try {
      return localStorage.getItem(ACTIVE_CONVERSATION_STORAGE_KEY);
    } catch {
      return null;
    }
  });
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draftInput, setDraftInput] = useState<string>("");
  const [isInitializing, setIsInitializing] = useState<boolean>(true);
  const [isLoadingMessages, setIsLoadingMessages] = useState<boolean>(false);
  const [isCreatingChat, setIsCreatingChat] = useState<boolean>(false);
  const [inFlightByConversation, setInFlightByConversation] = useState<
    Record<string, boolean>
  >({});
  const [pendingUserQuestionByConv, setPendingUserQuestionByConv] = useState<
    Record<string, string>
  >({});
  const [errorBanner, setErrorBanner] = useState<string | null>(null);

  // Refs to guard against stale responses when the user switches conversations mid-flight
  const activeConversationIdRef = useRef<string | null>(activeConversationId);
  const loadMessagesSeqRef = useRef<number>(0);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const updateActiveConversationId = useCallback((nextId: string | null) => {
    activeConversationIdRef.current = nextId;
    setActiveConversationId(nextId);
    try {
      if (nextId) {
        localStorage.setItem(ACTIVE_CONVERSATION_STORAGE_KEY, nextId);
      } else {
        localStorage.removeItem(ACTIVE_CONVERSATION_STORAGE_KEY);
      }
    } catch {
      // Ignore localStorage quota/privacy errors
    }
  }, []);

  const scrollToBottom = useCallback(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, pendingUserQuestionByConv, activeConversationId, scrollToBottom]);

  const loadConversationHistory = useCallback(
    async (
      conversationId: string,
      options?: { fallbackOnNotFound?: boolean }
    ): Promise<boolean> => {
      const requestSeq = ++loadMessagesSeqRef.current;
      setIsLoadingMessages(true);
      setErrorBanner(null);

      try {
        const data = await fetchConversationMessages(conversationId);
        // Guard against race condition if the user switched to another chat while loading
        if (
          activeConversationIdRef.current !== conversationId ||
          loadMessagesSeqRef.current !== requestSeq
        ) {
          return true;
        }
        setMessages(data.messages);
        setConversations((prev) => {
          const exists = prev.some(
            (c) => c.conversation_id === data.conversation.conversation_id
          );
          if (!exists) {
            return [data.conversation, ...prev];
          }
          return prev.map((c) =>
            c.conversation_id === data.conversation.conversation_id
              ? data.conversation
              : c
          );
        });
        return true;
      } catch (err) {
        if (
          activeConversationIdRef.current !== conversationId ||
          loadMessagesSeqRef.current !== requestSeq
        ) {
          return false;
        }
        if (err instanceof ApiRequestError && err.status === 404) {
          if (options?.fallbackOnNotFound) {
            updateActiveConversationId(null);
            setMessages([]);
            return false;
          }
          updateActiveConversationId(null);
          setMessages([]);
          setErrorBanner(
            `Saved conversation "${conversationId}" was not found and has been cleared.`
          );
          return false;
        }
        const message =
          err instanceof Error
            ? err.message
            : "Could not load conversation messages from the backend.";
        setErrorBanner(message);
        return false;
      } finally {
        if (
          activeConversationIdRef.current === conversationId &&
          loadMessagesSeqRef.current === requestSeq
        ) {
          setIsLoadingMessages(false);
        }
      }
    },
    [updateActiveConversationId]
  );

  // Initial load: check backend health, load conversation list, and restore stored conversation ID
  useEffect(() => {
    let cancelled = false;

    async function initializeApp() {
      setIsInitializing(true);
      try {
        const [healthRes, convList] = await Promise.all([
          fetchHealth().catch(() => null),
          fetchConversations(),
        ]);
        if (cancelled) return;

        if (healthRes) {
          setHealth(healthRes);
        }
        setConversations(convList);

        const storedId = activeConversationIdRef.current;
        if (storedId) {
          const restoredOk = await loadConversationHistory(storedId, {
            fallbackOnNotFound: true,
          });
          if (cancelled) return;
          if (restoredOk) {
            return;
          }
        }

        // If no stored conversation (or stored ID was stale/missing), select the most recent one if available
        if (convList.length > 0) {
          const firstId = convList[0].conversation_id;
          updateActiveConversationId(firstId);
          await loadConversationHistory(firstId, { fallbackOnNotFound: true });
        } else {
          updateActiveConversationId(null);
          setMessages([]);
        }
      } catch (err) {
        if (cancelled) return;
        const msg =
          err instanceof Error
            ? err.message
            : "Unable to connect to the FastAPI backend. Make sure uvicorn is running.";
        setErrorBanner(msg);
      } finally {
        if (!cancelled) {
          setIsInitializing(false);
        }
      }
    }

    void initializeApp();
    return () => {
      cancelled = true;
    };
  }, [loadConversationHistory, updateActiveConversationId]);

  const handleSelectConversation = async (conversationId: string) => {
    if (conversationId === activeConversationIdRef.current) {
      return;
    }
    setErrorBanner(null);
    setMessages([]);
    updateActiveConversationId(conversationId);
    await loadConversationHistory(conversationId);
  };

  const handleStartNewChat = async () => {
    if (isCreatingChat) return;
    setIsCreatingChat(true);
    setErrorBanner(null);

    try {
      const created = await createConversation();
      setConversations((prev) => [
        created,
        ...prev.filter((c) => c.conversation_id !== created.conversation_id),
      ]);
      loadMessagesSeqRef.current += 1;
      setIsLoadingMessages(false);
      setMessages([]);
      updateActiveConversationId(created.conversation_id);
      textareaRef.current?.focus();
    } catch (err) {
      const msg =
        err instanceof Error
          ? err.message
          : "Failed to create a new conversation.";
      setErrorBanner(msg);
    } finally {
      setIsCreatingChat(false);
    }
  };

  const submitQuestion = async (rawQuestion: string) => {
    const trimmed = rawQuestion.trim();
    if (!trimmed) {
      setErrorBanner("Please enter a question before sending.");
      return;
    }

    const currentConvId = activeConversationIdRef.current;
    if (currentConvId && inFlightByConversation[currentConvId]) {
      return;
    }

    setErrorBanner(null);
    let targetConvId = currentConvId;

    // Clear composer optimistically while preserving `trimmed` to restore if the request fails
    setDraftInput("");

    try {
      if (!targetConvId) {
        const created = await createConversation();
        targetConvId = created.conversation_id;
        setConversations((prev) => [
          created,
          ...prev.filter((c) => c.conversation_id !== created.conversation_id),
        ]);
        updateActiveConversationId(targetConvId);
        setMessages([]);
      }

      const lockedConvId = targetConvId;
      setInFlightByConversation((prev) => ({ ...prev, [lockedConvId]: true }));
      setPendingUserQuestionByConv((prev) => ({
        ...prev,
        [lockedConvId]: trimmed,
      }));

      try {
        const response = await sendConversationMessage(lockedConvId, trimmed);

        // Refresh sidebar list so updated title, message_count, and ordering are reflected
        const updatedList = await fetchConversations().catch(() => null);
        if (updatedList) {
          setConversations(updatedList);
        } else {
          setConversations((prev) =>
            prev.map((c) =>
              c.conversation_id === lockedConvId
                ? {
                    ...c,
                    title: response.conversation_title || c.title,
                    updated_at: response.created_at || c.updated_at,
                    message_count: c.message_count + 2,
                  }
                : c
            )
          );
        }

        // CRITICAL: Only append messages to the main view if the user is STILL viewing this conversation!
        if (activeConversationIdRef.current === lockedConvId) {
          setMessages((prev) => [
            ...prev,
            response.user_message,
            response.assistant_message,
          ]);
        }
      } catch (sendErr) {
        // Restore the user's draft input so nothing is lost and no duplicate message is created
        if (activeConversationIdRef.current === lockedConvId) {
          setDraftInput((currentDraft) =>
            currentDraft.trim() ? currentDraft : trimmed
          );
          const errMsg =
            sendErr instanceof Error
              ? sendErr.message
              : "Failed to send message to the onboarding assistant.";
          setErrorBanner(errMsg);
        }
      } finally {
        setInFlightByConversation((prev) => {
          const next = { ...prev };
          delete next[lockedConvId];
          return next;
        });
        setPendingUserQuestionByConv((prev) => {
          const next = { ...prev };
          delete next[lockedConvId];
          return next;
        });
      }
    } catch (createErr) {
      setDraftInput(trimmed);
      const errMsg =
        createErr instanceof Error
          ? createErr.message
          : "Could not initialize a conversation with the backend.";
      setErrorBanner(errMsg);
    }
  };

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    void submitQuestion(draftInput);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const activeId = activeConversationIdRef.current;
      const isBusy = Boolean(activeId && inFlightByConversation[activeId]);
      if (!isBusy && !isCreatingChat && draftInput.trim()) {
        void submitQuestion(draftInput);
      }
    }
  };

  const isActiveConvBusy = Boolean(
    activeConversationId && inFlightByConversation[activeConversationId]
  );
  const pendingQuestionForActiveConv = activeConversationId
    ? pendingUserQuestionByConv[activeConversationId]
    : undefined;

  const activeConversationSummary = conversations.find(
    (c) => c.conversation_id === activeConversationId
  );

  return (
    <div className="app-shell">
      {/* Sidebar */}
      <aside className="sidebar" aria-label="Saved Conversations">
        <div className="sidebar-brand">
          <div className="brand-emblem">TU</div>
          <div className="brand-text">
            <span className="brand-uni">Tulane University</span>
            <span className="brand-scope">Faculty Handbook RAG</span>
          </div>
        </div>

        <button
          type="button"
          className="new-chat-btn"
          onClick={() => void handleStartNewChat()}
          disabled={isCreatingChat}
        >
          <span>+ New Chat</span>
        </button>

        <div className="sidebar-section-label">
          Saved Conversations ({conversations.length})
        </div>

        <div className="conversation-list">
          {conversations.length === 0 ? (
            <div className="sidebar-empty-note">
              No saved chats yet. Ask a handbook question or click{" "}
              <strong>+ New Chat</strong> to begin.
            </div>
          ) : (
            conversations.map((conv) => {
              const isSelected = conv.conversation_id === activeConversationId;
              const isConvBusy = Boolean(
                inFlightByConversation[conv.conversation_id]
              );
              return (
                <button
                  key={conv.conversation_id}
                  type="button"
                  className={`conversation-item ${isSelected ? "active" : ""}`}
                  onClick={() =>
                    void handleSelectConversation(conv.conversation_id)
                  }
                >
                  <div className="conversation-item-title-row">
                    <span className="conversation-item-title">
                      {conv.title}
                    </span>
                    {isConvBusy && (
                      <span
                        className="conv-busy-dot"
                        title="Processing a response..."
                      />
                    )}
                  </div>
                  <div className="conversation-item-meta">
                    <span>{conv.message_count} msgs</span>
                    <span>•</span>
                    <span>{formatTimestamp(conv.updated_at)}</span>
                  </div>
                </button>
              );
            })
          )}
        </div>

        <div className="sidebar-footer">
          <div className="health-indicator">
            <span
              className={`health-dot ${
                health?.status === "ok"
                  ? "ok"
                  : health?.status === "degraded"
                    ? "degraded"
                    : "unknown"
              }`}
            />
            <span>
              {health
                ? health.status === "ok"
                  ? `Index ready (${health.chunk_count} chunks)`
                  : "Backend degraded (check .env / index)"
                : "Checking backend status..."}
            </span>
          </div>
          <div className="sidebar-disclaimer-mini">
            Local single-user demo • CPU FAISS retrieval
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="main-panel">
        {/* Header */}
        <header className="main-header">
          <div className="header-titles">
            <h1 className="app-title">Faculty Onboarding Assistant</h1>
            <p className="app-subtitle">
              Source: Supplied{" "}
              <strong>
                Tulane University Faculty Handbook (Updated August 7, 2026)
              </strong>{" "}
              &amp; appended{" "}
              <strong>Research Misconduct Policy (March 7, 2022)</strong>.
              Answers reflect only this provided faculty handbook (not general
              staff policy), and its current live policy status is unverified.
            </p>
          </div>
          {activeConversationSummary && (
            <div className="active-conv-pill" title="Active Conversation ID">
              Chat ID: <code>{activeConversationSummary.conversation_id}</code>
            </div>
          )}
        </header>

        {/* Degraded Health Warning Banner */}
        {health && health.status === "degraded" && health.detail && (
          <div className="warning-banner" role="status">
            <strong>Backend Configuration Notice:</strong> {health.detail}
          </div>
        )}

        {/* Error Banner */}
        {errorBanner && (
          <div className="error-banner" role="alert">
            <div className="error-banner-text">
              <strong>Request Error:</strong> {errorBanner}
            </div>
            <button
              type="button"
              className="error-dismiss-btn"
              onClick={() => setErrorBanner(null)}
              aria-label="Dismiss error message"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Messages / Welcome Area */}
        <main className="messages-viewport">
          {isInitializing || isLoadingMessages ? (
            <div className="loading-state-panel">
              <div className="spinner" />
              <p>Loading conversation history...</p>
            </div>
          ) : messages.length === 0 && !pendingQuestionForActiveConv ? (
            <div className="welcome-panel">
              <div className="welcome-card">
                <div className="welcome-badge">
                  Tulane University Faculty Scope Only
                </div>
                <h2>Welcome to the Faculty Onboarding Assistant</h2>
                <p>
                  Ask questions about faculty appointments, tenure and promotion
                  criteria, sabbatical and parental leave, grievance procedures,
                  faculty governance, or the appended Research Misconduct
                  Policy. Every policy answer cites specific handbook passages
                  (<code>[S1]</code>, <code>[S2]</code>) with PDF page numbers
                  and excerpts.
                </p>

                <div className="example-questions-grid">
                  {EXAMPLE_QUESTIONS.map((item, idx) => (
                    <button
                      key={idx}
                      type="button"
                      className="example-question-card"
                      disabled={isActiveConvBusy}
                      onClick={() => void submitQuestion(item.question)}
                    >
                      <span className="example-category">{item.category}</span>
                      <span className="example-question-text">
                        “{item.question}”
                      </span>
                      <span className="example-hint">{item.hint}</span>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="messages-list">
              {messages.map((msg) => {
                const isUser = msg.role === "user";
                const badgeInfo = !isUser
                  ? getStatusBadgeInfo(msg.status)
                  : null;

                return (
                  <article
                    key={msg.message_id}
                    className={`message-row ${
                      isUser ? "message-user" : "message-assistant"
                    } ${msg.status ? `message-status-${msg.status}` : ""}`}
                  >
                    <div className="message-bubble">
                      <div className="message-meta-header">
                        <span className="message-role-label">
                          {isUser
                            ? "You"
                            : "Faculty Onboarding Assistant (Handbook RAG)"}
                        </span>
                        <div className="message-meta-right">
                          {badgeInfo && (
                            <span className={badgeInfo.className}>
                              {badgeInfo.label}
                            </span>
                          )}
                          {msg.created_at && (
                            <span className="message-timestamp">
                              {formatTimestamp(msg.created_at)}
                            </span>
                          )}
                        </div>
                      </div>

                      {badgeInfo?.description && (
                        <div
                          className={`status-callout status-callout-${msg.status}`}
                        >
                          {badgeInfo.description}
                        </div>
                      )}

                      {/* Safe text rendering via React text nodes (no raw HTML injection) */}
                      <div className="message-content-text">
                        {msg.content.split("\n").map((line, idx) => (
                          <p
                            key={idx}
                            className={
                              line.trim() === "" ? "empty-line" : "text-line"
                            }
                          >
                            {line}
                          </p>
                        ))}
                      </div>

                      {!isUser &&
                        msg.cited_sources &&
                        msg.cited_sources.length > 0 && (
                          <ExpandableSourceList
                            messageId={msg.message_id}
                            sources={msg.cited_sources}
                          />
                        )}
                    </div>
                  </article>
                );
              })}

              {/* Pending turn display while a request is in flight for THIS conversation */}
              {pendingQuestionForActiveConv && (
                <>
                  <article className="message-row message-user message-pending">
                    <div className="message-bubble">
                      <div className="message-meta-header">
                        <span className="message-role-label">You</span>
                        <span className="message-timestamp">Sending...</span>
                      </div>
                      <div className="message-content-text">
                        {pendingQuestionForActiveConv
                          .split("\n")
                          .map((line, idx) => (
                            <p key={idx} className="text-line">
                              {line}
                            </p>
                          ))}
                      </div>
                    </div>
                  </article>

                  <article
                    className="message-row message-assistant message-loading"
                    aria-live="polite"
                  >
                    <div className="message-bubble">
                      <div className="loading-inline">
                        <div className="spinner-small" />
                        <span>
                          Searching the Tulane Faculty Handbook and generating a
                          grounded response...
                        </span>
                      </div>
                    </div>
                  </article>
                </>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </main>

        {/* Composer */}
        <footer className="composer-footer">
          <form className="composer-form" onSubmit={handleFormSubmit}>
            <div className="composer-input-wrapper">
              <textarea
                ref={textareaRef}
                className="composer-textarea"
                rows={2}
                maxLength={2000}
                placeholder={
                  isActiveConvBusy
                    ? "Waiting for the current response to complete..."
                    : "Ask a question about the Tulane University Faculty Handbook... (Press Enter to send, Shift+Enter for a newline)"
                }
                value={draftInput}
                onChange={(e) => setDraftInput(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={isActiveConvBusy || isCreatingChat}
                aria-label="Ask a question about the Tulane University Faculty Handbook"
              />
              <button
                type="submit"
                className="composer-send-btn"
                disabled={
                  isActiveConvBusy || isCreatingChat || !draftInput.trim()
                }
              >
                {isActiveConvBusy ? "Sending..." : "Send"}
              </button>
            </div>
            <div className="composer-helper-row">
              <span>
                Press <kbd>Enter</kbd> to send • <kbd>Shift</kbd> +{" "}
                <kbd>Enter</kbd> for a new line
              </span>
              <span>{draftInput.trim().length} / 2000 chars</span>
            </div>
          </form>
        </footer>
      </div>
    </div>
  );
};

export default App;

