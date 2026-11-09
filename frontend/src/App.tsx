import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  AnswerStatus,
  ChatMessage,
  CitedSource,
  ConversationSummary,
  DocumentStatusResponse,
  HealthResponse,
  SourceMode,
} from "./types";
import {
  ACCESS_TOKEN_REQUIRED_EVENT,
  ApiRequestError,
  createConversation,
  deleteDocument,
  fetchConversationMessages,
  fetchConversations,
  fetchDocumentStatus,
  fetchHealth,
  getAccessToken,
  sendConversationMessage,
  setAccessToken,
  uploadDocument,
} from "./api";

const ACTIVE_CONVERSATION_STORAGE_KEY = "tulane_faculty_onboarding_active_conv_id";

interface ExampleQuestionItem {
  category: string;
  question: string;
  hint: string;
}

const EXAMPLE_QUESTIONS: ExampleQuestionItem[] = [
  {
    category: "Faculty Leave",
    question: "How much paid parental leave do faculty get?",
    hint: "Chapter 3: Leave policies for full-time regular faculty.",
  },
  {
    category: "Approval Procedures",
    question: "Who approves faculty sabbatical leave and when must it be requested?",
    hint: "Department Chair, Dean, and Provost approval timelines.",
  },
  {
    category: "Research Misconduct",
    question: "What are the formal stages of a research misconduct inquiry and investigation?",
    hint: "Appended Research Misconduct Policy (PDF pages 192–238).",
  },
  {
    category: "Tenure & Promotion",
    question: "What are the criteria and timeline for tenure review?",
    hint: "Chapter 4: Promotion and tenure review procedures.",
  },
];

function formatTimestamp(isoString: string): string {
  try {
    const d = new Date(isoString);
    if (isNaN(d.getTime())) return isoString;
    return d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  } catch {
    return isoString;
  }
}

function formatPagesLabel(pages: number[]): string {
  if (!pages || pages.length === 0) return "PDF Page: unrecorded";
  if (pages.length === 1) return `PDF Page ${pages[0]}`;
  return `PDF Pages ${pages.join(", ")}`;
}

function getStatusBadgeInfo(status?: AnswerStatus | string | null): {
  label: string;
  className: string;
  description?: string;
} | null {
  if (!status) return null;
  switch (status) {
    case "supported":
      return {
        label: "Supported by Handbook",
        className: "status-badge status-supported",
      };
    case "partial":
      return {
        label: "Partially Supported",
        className: "status-badge status-partial",
        description: "Part of this response is supported; see the missing information note below.",
      };
    case "clarify":
      return {
        label: "Clarification Needed",
        className: "status-badge status-clarify",
        description: "Your reference was ambiguous. Please clarify which specific policy you mean.",
      };
    case "unsupported":
      return {
        label: "Not in Document",
        className: "status-badge status-unsupported",
      };
    default:
      return null;
  }
}

const ExpandableSourceList: React.FC<{
  messageId: number;
  sources: CitedSource[];
}> = ({ messageId, sources }) => {
  const [sectionOpen, setSectionOpen] = useState<boolean>(true);
  const [expandedCards, setExpandedCards] = useState<Record<string, boolean>>({});

  const toggleCard = (key: string) => {
    setExpandedCards((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
  };

  if (!sources || sources.length === 0) return null;

  return (
    <div className="sources-container">
      <div className="sources-header-bar">
        <button
          type="button"
          className="sources-toggle-btn"
          onClick={() => setSectionOpen((prev) => !prev)}
          aria-expanded={sectionOpen}
        >
          <span className="sources-toggle-icon">{sectionOpen ? "▼" : "▶"}</span>
          <span>
            Cited Passages ({sources.length} {sources.length === 1 ? "source" : "sources"})
          </span>
        </button>
        <span className="sources-verification-hint">
          Verify claims against the PDF pages and excerpts below.
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
                    <span className="source-section-tag">{src.document_section}</span>
                    <span className="source-pages-tag">{formatPagesLabel(src.pdf_pages)}</span>
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
                    <blockquote className="source-excerpt">“{src.excerpt}”</blockquote>
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
  const [activeConversationId, setActiveConversationId] = useState<string | null>(() => {
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
  const [inFlightByConversation, setInFlightByConversation] = useState<Record<string, boolean>>({});
  const [pendingUserQuestionByConv, setPendingUserQuestionByConv] = useState<Record<string, string>>({});
  const [errorBanner, setErrorBanner] = useState<string | null>(null);

  // Chat-scoped upload and source mode states
  const [documentByConv, setDocumentByConv] = useState<Record<string, DocumentStatusResponse>>({});
  const [sourceModeByConv, setSourceModeByConv] = useState<Record<string, SourceMode>>({});
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [showRemoveModal, setShowRemoveModal] = useState<boolean>(false);

  // Hosted Access Token Modal state
  const [showTokenModal, setShowTokenModal] = useState<boolean>(false);
  const [tokenInput, setTokenInput] = useState<string>(() => getAccessToken());

  // Refs to guard against stale responses when switching conversations mid-flight
  const activeConversationIdRef = useRef<string | null>(activeConversationId);
  const loadMessagesSeqRef = useRef<number>(0);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const activeDoc: DocumentStatusResponse | undefined = activeConversationId
    ? documentByConv[activeConversationId]
    : undefined;
  const isDocReady = activeDoc?.status === "ready";
  const activeSourceMode: SourceMode =
    (activeConversationId && sourceModeByConv[activeConversationId]) || "handbook";

  const setActiveSourceMode = (mode: SourceMode) => {
    if (activeConversationId) {
      setSourceModeByConv((prev) => ({ ...prev, [activeConversationId]: mode }));
    }
  };

  const updateActiveConversationId = useCallback((nextId: string | null) => {
    activeConversationIdRef.current = nextId;
    setActiveConversationId(nextId);
    setSelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    try {
      if (nextId) {
        localStorage.setItem(ACTIVE_CONVERSATION_STORAGE_KEY, nextId);
      } else {
        localStorage.removeItem(ACTIVE_CONVERSATION_STORAGE_KEY);
      }
    } catch {
      // Ignore storage errors
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

  // Listen for 401 token requirement events
  useEffect(() => {
    const onTokenRequired = () => setShowTokenModal(true);
    window.addEventListener(ACCESS_TOKEN_REQUIRED_EVENT, onTokenRequired);
    return () => window.removeEventListener(ACCESS_TOKEN_REQUIRED_EVENT, onTokenRequired);
  }, []);

  // Poll processing documents across any conversation
  useEffect(() => {
    const processingIds = Object.entries(documentByConv)
      .filter(([_, doc]) => doc.status === "processing")
      .map(([id]) => id);

    if (processingIds.length === 0) return;

    const timer = setInterval(() => {
      for (const cid of processingIds) {
        fetchDocumentStatus(cid)
          .then((doc) => {
            setDocumentByConv((prev) => ({ ...prev, [cid]: doc }));
          })
          .catch(() => {});
      }
    }, 2000);

    return () => clearInterval(timer);
  }, [documentByConv]);

  const loadConversationHistory = useCallback(
    async (
      conversationId: string,
      options?: { fallbackOnNotFound?: boolean }
    ): Promise<boolean> => {
      const requestSeq = ++loadMessagesSeqRef.current;
      setIsLoadingMessages(true);
      setErrorBanner(null);

      // Concurrently fetch document status for this conversation
      void fetchDocumentStatus(conversationId)
        .then((doc) => {
          setDocumentByConv((prev) => ({ ...prev, [conversationId]: doc }));
        })
        .catch(() => {});

      try {
        const data = await fetchConversationMessages(conversationId);
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

  // Initial App Mount
  useEffect(() => {
    let isMounted = true;

    async function initializeApp() {
      setIsInitializing(true);
      try {
        const healthRes = await fetchHealth().catch(() => null);
        if (isMounted) setHealth(healthRes);

        const convs = await fetchConversations().catch(() => []);
        if (!isMounted) return;
        setConversations(convs);

        const savedId = activeConversationIdRef.current;
        if (savedId) {
          await loadConversationHistory(savedId, { fallbackOnNotFound: true });
        } else if (convs.length > 0) {
          updateActiveConversationId(convs[0].conversation_id);
          await loadConversationHistory(convs[0].conversation_id);
        }
      } catch (err) {
        if (!isMounted) return;
        const msg =
          err instanceof Error
            ? err.message
            : "Failed to initialize the onboarding assistant.";
        setErrorBanner(msg);
      } finally {
        if (isMounted) setIsInitializing(false);
      }
    }

    void initializeApp();

    return () => {
      isMounted = false;
    };
  }, [loadConversationHistory, updateActiveConversationId]);

  const handleSelectConversation = async (conversationId: string) => {
    if (activeConversationId === conversationId) return;
    setErrorBanner(null);
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
        err instanceof Error ? err.message : "Failed to create a new conversation.";
      setErrorBanner(msg);
    } finally {
      setIsCreatingChat(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    setSelectedFile(file);
  };

  const handleUploadSubmit = async () => {
    if (!selectedFile || !activeConversationId || isUploading) return;
    setIsUploading(true);
    setErrorBanner(null);
    try {
      const doc = await uploadDocument(activeConversationId, selectedFile);
      setDocumentByConv((prev) => ({
        ...prev,
        [activeConversationId]: doc,
      }));
      setSelectedFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to upload PDF.";
      setErrorBanner(msg);
    } finally {
      setIsUploading(false);
    }
  };

  const handleConfirmRemoveDocument = async () => {
    if (!activeConversationId) return;
    setErrorBanner(null);
    try {
      await deleteDocument(activeConversationId);
      setDocumentByConv((prev) => ({
        ...prev,
        [activeConversationId]: {
          conversation_id: activeConversationId,
          status: "none",
        },
      }));
      setActiveSourceMode("handbook");
      setShowRemoveModal(false);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to remove PDF.";
      setErrorBanner(msg);
    }
  };

  const handleSaveToken = () => {
    const cleaned = tokenInput.trim();
    setAccessToken(cleaned);
    setShowTokenModal(false);
    setErrorBanner(null);
    // Refresh conversations with the new token
    void fetchConversations()
      .then((convs) => setConversations(convs))
      .catch(() => {});
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
      const lockedMode: SourceMode =
        (lockedConvId && sourceModeByConv[lockedConvId]) || "handbook";

      if (lockedMode === "uploaded_pdf") {
        const doc = documentByConv[lockedConvId];
        if (!doc || doc.status !== "ready") {
          setDraftInput(trimmed);
          setErrorBanner(
            "No uploaded PDF is ready for this conversation. Upload a faculty-related PDF first or switch to Handbook mode."
          );
          return;
        }
      }

      setInFlightByConversation((prev) => ({ ...prev, [lockedConvId]: true }));
      setPendingUserQuestionByConv((prev) => ({
        ...prev,
        [lockedConvId]: trimmed,
      }));

      try {
        const response = await sendConversationMessage(
          lockedConvId,
          trimmed,
          lockedMode
        );

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

        if (activeConversationIdRef.current === lockedConvId) {
          setMessages((prev) => [
            ...prev,
            response.user_message,
            response.assistant_message,
          ]);
        }
      } catch (sendErr) {
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
            <span className="brand-scope">Faculty Onboarding Assistant</span>
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
              No saved chats yet. Ask a question or click{" "}
              <strong>+ New Chat</strong> to begin.
            </div>
          ) : (
            conversations.map((conv) => {
              const isSelected = conv.conversation_id === activeConversationId;
              const isConvBusy = Boolean(inFlightByConversation[conv.conversation_id]);
              return (
                <button
                  key={conv.conversation_id}
                  type="button"
                  className={`conversation-item ${isSelected ? "active" : ""}`}
                  onClick={() => void handleSelectConversation(conv.conversation_id)}
                >
                  <div className="conversation-item-title-row">
                    <span className="conversation-item-title">{conv.title}</span>
                    {isConvBusy && (
                      <span className="conv-busy-dot" title="Processing a response..." />
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
          <button
            type="button"
            className="token-config-btn"
            onClick={() => setShowTokenModal(true)}
            title="Configure access token for hosted deployments"
          >
            🔑 Access Token {getAccessToken() ? "(Configured)" : "(None)"}
          </button>
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
              Scope: Supplied{" "}
              <strong>Tulane University Faculty Handbook (Updated August 7, 2026)</strong>{" "}
              &amp; appended <strong>Research Misconduct Policy (March 7, 2022)</strong>, or
              your uploaded faculty PDF.
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
              <strong>Notice:</strong> {errorBanner}
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

        {/* Chat-Scoped PDF Upload Banner */}
        {activeConversationSummary && (
          <div className="chat-doc-banner">
            <div className="chat-doc-panel">
              <div className="chat-doc-header-row">
                <div className="chat-doc-title-group">
                  <span className="chat-doc-icon">📄</span>
                  <span>Chat Document:</span>
                  {(!activeDoc || activeDoc.status === "none") && (
                    <span className="chat-doc-meta-info">No document uploaded yet for this chat.</span>
                  )}
                  {activeDoc?.status === "processing" && (
                    <span className="chat-doc-status-badge processing">
                      <div className="spinner-small" />
                      Validating &amp; Indexing: {activeDoc.display_filename}
                    </span>
                  )}
                  {activeDoc?.status === "ready" && (
                    <span className="chat-doc-status-badge ready">
                      ✓ Ready: {activeDoc.display_filename} ({activeDoc.page_count} pages, {activeDoc.chunk_count} passages)
                    </span>
                  )}
                  {activeDoc?.status === "rejected" && (
                    <span className="chat-doc-status-badge rejected">
                      ✗ Out of Scope: {activeDoc.display_filename}
                    </span>
                  )}
                  {activeDoc?.status === "needs_clarification" && (
                    <span className="chat-doc-status-badge needs_clarification">
                      ⚠ Borderline / Needs Clarification: {activeDoc.display_filename}
                    </span>
                  )}
                  {activeDoc?.status === "failed" && (
                    <span className="chat-doc-status-badge failed">
                      ✗ Processing Failed: {activeDoc.display_filename || "Document"}
                    </span>
                  )}
                </div>

                {activeDoc && activeDoc.status !== "none" && activeDoc.status !== "processing" && (
                  <button
                    type="button"
                    className="chat-doc-btn chat-doc-btn-remove"
                    onClick={() => setShowRemoveModal(true)}
                  >
                    Remove PDF
                  </button>
                )}
              </div>

              {/* Explanatory notes or action messages */}
              {activeDoc && (activeDoc.status === "rejected" || activeDoc.status === "needs_clarification" || activeDoc.status === "failed") && (
                <div className="chat-doc-meta-info">
                  <strong>Reason:</strong> {activeDoc.message || "The document could not be accepted."}
                </div>
              )}

              {/* Upload control when no active doc is ready or processing */}
              {(!activeDoc || activeDoc.status === "none") && (
                <div className="chat-doc-upload-row">
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf,application/pdf"
                    className="chat-doc-file-input"
                    onChange={handleFileChange}
                    disabled={isUploading}
                  />
                  <button
                    type="button"
                    className="chat-doc-btn chat-doc-btn-upload"
                    onClick={() => void handleUploadSubmit()}
                    disabled={isUploading || !selectedFile}
                  >
                    {isUploading ? "Uploading..." : "Upload PDF"}
                  </button>
                  <span className="chat-doc-limits-hint">
                    Max 10 MiB, up to 50 pages. Searchable PDF only (no OCR). Excerpts are sent to Gemini for faculty-relevance validation.
                  </span>
                </div>
              )}
            </div>
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
                <div className="welcome-badge">Faculty Scope Only</div>
                <h2>Welcome to the Faculty Onboarding Assistant</h2>
                <p>
                  Each chat begins in <strong>Handbook mode</strong> grounded in the Tulane University Faculty Handbook.
                  You can also upload a searchable faculty-related PDF to this chat and switch to <strong>Uploaded PDF only</strong> mode.
                  Every answer cites specific passages (<code>[S1]</code>, <code>[S2]</code>) with PDF page numbers and excerpts.
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
                      <span className="example-question-text">“{item.question}”</span>
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
                const badgeInfo = !isUser ? getStatusBadgeInfo(msg.status) : null;

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
                          {isUser ? "You" : "Faculty Onboarding Assistant"}
                        </span>
                        <div className="message-meta-right">
                          {badgeInfo && (
                            <span className={badgeInfo.className}>{badgeInfo.label}</span>
                          )}
                          {msg.created_at && (
                            <span className="message-timestamp">
                              {formatTimestamp(msg.created_at)}
                            </span>
                          )}
                        </div>
                      </div>

                      {badgeInfo?.description && (
                        <div className={`status-callout status-callout-${msg.status}`}>
                          {badgeInfo.description}
                        </div>
                      )}

                      <div className="message-content-text">
                        {msg.content.split("\n").map((line, idx) => (
                          <p
                            key={idx}
                            className={line.trim() === "" ? "empty-line" : "text-line"}
                          >
                            {line}
                          </p>
                        ))}
                      </div>

                      {!isUser && msg.cited_sources && msg.cited_sources.length > 0 && (
                        <ExpandableSourceList
                          messageId={msg.message_id}
                          sources={msg.cited_sources}
                        />
                      )}
                    </div>
                  </article>
                );
              })}

              {pendingQuestionForActiveConv && (
                <>
                  <article className="message-row message-user message-pending">
                    <div className="message-bubble">
                      <div className="message-meta-header">
                        <span className="message-role-label">You</span>
                        <span className="message-timestamp">Sending...</span>
                      </div>
                      <div className="message-content-text">
                        {pendingQuestionForActiveConv.split("\n").map((line, idx) => (
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
                          {activeSourceMode === "uploaded_pdf"
                            ? `Searching uploaded document (${activeDoc?.display_filename || "PDF"})...`
                            : "Searching the Tulane Faculty Handbook and generating response..."}
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

        {/* Source Mode Selector Bar */}
        <div className="source-mode-bar">
          <div className="source-mode-buttons">
            <span className="source-mode-label">Search Mode:</span>
            <button
              type="button"
              className={`source-mode-btn ${activeSourceMode === "handbook" ? "active" : ""}`}
              onClick={() => setActiveSourceMode("handbook")}
            >
              📖 Tulane Faculty Handbook
            </button>
            <button
              type="button"
              className={`source-mode-btn ${activeSourceMode === "uploaded_pdf" ? "active" : ""}`}
              disabled={!isDocReady}
              title={
                !isDocReady
                  ? "Upload and verify a faculty-related PDF above to enable this mode"
                  : `Query exclusively from ${activeDoc?.display_filename}`
              }
              onClick={() => setActiveSourceMode("uploaded_pdf")}
            >
              📄 Uploaded PDF only {activeDoc?.display_filename ? `(${activeDoc.display_filename})` : ""}
            </button>
          </div>
          <div className="source-mode-active-indicator">
            {activeSourceMode === "uploaded_pdf"
              ? `Querying: ${activeDoc?.display_filename || "Uploaded PDF only"}`
              : "Querying: Tulane Faculty Handbook"}
          </div>
        </div>

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
                    : activeSourceMode === "uploaded_pdf"
                      ? `Ask a question about ${activeDoc?.display_filename || "the uploaded PDF"}... (Enter to send, Shift+Enter for newline)`
                      : "Ask a question about the Tulane University Faculty Handbook... (Enter to send, Shift+Enter for newline)"
                }
                value={draftInput}
                onChange={(e) => setDraftInput(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={isActiveConvBusy || isCreatingChat}
                aria-label="Ask a faculty onboarding question"
              />
              <button
                type="submit"
                className="composer-send-btn"
                disabled={isActiveConvBusy || isCreatingChat || !draftInput.trim()}
              >
                {isActiveConvBusy ? "Sending..." : "Send"}
              </button>
            </div>
            <div className="composer-helper-row">
              <span>
                Press <kbd>Enter</kbd> to send • <kbd>Shift</kbd> + <kbd>Enter</kbd> for a new line
              </span>
              <span>{draftInput.trim().length} / 2000 chars</span>
            </div>
          </form>
        </footer>
      </div>

      {/* Remove Document Confirmation Modal */}
      {showRemoveModal && (
        <div className="modal-backdrop" role="dialog" aria-modal="true">
          <div className="modal-dialog">
            <h2 className="modal-title">Remove Uploaded PDF?</h2>
            <p className="modal-body">
              Are you sure you want to remove{" "}
              <strong>{activeDoc?.display_filename || "this PDF"}</strong> from this conversation?
            </p>
            <p className="modal-body">
              <strong>Note on saved history:</strong> Existing chat messages and citations will
              remain preserved in your saved transcript, but future questions will no longer be able
              to retrieve from this document. To upload a replacement PDF, the current document must
              first be removed.
            </p>
            <div className="modal-actions">
              <button
                type="button"
                className="modal-btn-cancel"
                onClick={() => setShowRemoveModal(false)}
              >
                Cancel
              </button>
              <button
                type="button"
                className="modal-btn-danger"
                onClick={() => void handleConfirmRemoveDocument()}
              >
                Confirm Removal
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Access Token Configuration Modal */}
      {showTokenModal && (
        <div className="modal-backdrop" role="dialog" aria-modal="true">
          <div className="modal-dialog">
            <h2 className="modal-title">Hosted Access Token</h2>
            <p className="modal-body">
              If your backend is deployed with an <code>ACCESS_TOKEN</code> gate on Render, enter the token below.
              It is saved securely in your browser's local storage and sent via the <code>X-Access-Token</code> header.
              (Leave empty for local development without authentication.)
            </p>
            <input
              type="text"
              className="modal-input"
              placeholder="Paste X-Access-Token here..."
              value={tokenInput}
              onChange={(e) => setTokenInput(e.target.value)}
            />
            <div className="modal-actions">
              <button
                type="button"
                className="modal-btn-cancel"
                onClick={() => setShowTokenModal(false)}
              >
                Cancel
              </button>
              <button
                type="button"
                className="chat-doc-btn chat-doc-btn-upload"
                onClick={handleSaveToken}
              >
                Save Token
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
