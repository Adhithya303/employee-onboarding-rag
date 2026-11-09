export type AnswerStatus = "supported" | "partial" | "unsupported" | "clarify";
export type SourceMode = "handbook" | "uploaded_pdf";

export type DocumentStatus =
  | "none"
  | "processing"
  | "ready"
  | "rejected"
  | "needs_clarification"
  | "failed";

export interface DocumentStatusResponse {
  document_id?: string | null;
  conversation_id: string;
  display_filename?: string | null;
  status: DocumentStatus;
  code?: string | null;
  message?: string | null;
  page_count?: number | null;
  chunk_count?: number | null;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface DeleteDocumentResponse {
  deleted: boolean;
  conversation_id: string;
  document_id?: string | null;
}

export interface CitedSource {
  label: string;
  chunk_id: string;
  document_title: string;
  document_section: string;
  pdf_pages: number[];
  excerpt: string;
}

export interface ChatMessage {
  message_id: number;
  conversation_id: string;
  role: "user" | "assistant";
  content: string;
  status?: AnswerStatus | string | null;
  cited_sources: CitedSource[];
  created_at: string;
}

export interface ConversationSummary {
  conversation_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface ConversationsListResponse {
  conversations: ConversationSummary[];
}

export interface ConversationMessagesResponse {
  conversation: ConversationSummary;
  messages: ChatMessage[];
}

export interface SendMessageResponse {
  conversation_id: string;
  conversation_title: string;
  user_message_id: number;
  assistant_message_id: number;
  created_at: string;
  status: AnswerStatus;
  answer: string;
  cited_sources: CitedSource[];
  user_message: ChatMessage;
  assistant_message: ChatMessage;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  index_loaded: boolean;
  chunk_count: number;
  llm_configured: boolean;
  model_id?: string | null;
  detail?: string | null;
}

