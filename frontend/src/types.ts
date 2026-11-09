export type AnswerStatus = "supported" | "partial" | "unsupported" | "clarify";

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

