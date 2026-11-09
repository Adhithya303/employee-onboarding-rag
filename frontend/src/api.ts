import type {
  ConversationMessagesResponse,
  ConversationsListResponse,
  ConversationSummary,
  HealthResponse,
  SendMessageResponse,
} from "./types";

const RAW_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
export const API_BASE_URL = RAW_BASE_URL.replace(/\/+$/, "");

export class ApiRequestError extends Error {
  public readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiRequestError";
    this.status = status;
  }
}

async function parseJsonOrThrow<T>(response: Response): Promise<T> {
  let bodyText = "";
  try {
    bodyText = await response.text();
  } catch {
    bodyText = "";
  }

  let parsed: unknown = null;
  if (bodyText) {
    try {
      parsed = JSON.parse(bodyText);
    } catch {
      parsed = null;
    }
  }

  if (!response.ok) {
    let detailMessage = `Request failed with HTTP ${response.status}.`;
    if (
      parsed &&
      typeof parsed === "object" &&
      "detail" in parsed &&
      typeof (parsed as { detail: unknown }).detail === "string"
    ) {
      detailMessage = (parsed as { detail: string }).detail;
    } else if (bodyText.trim()) {
      detailMessage = bodyText.trim();
    }
    throw new ApiRequestError(detailMessage, response.status);
  }

  return parsed as T;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE_URL}/api/health`, {
    method: "GET",
    headers: { Accept: "application/json" },
  });
  return parseJsonOrThrow<HealthResponse>(res);
}

export async function fetchConversations(): Promise<ConversationSummary[]> {
  const res = await fetch(`${API_BASE_URL}/api/conversations`, {
    method: "GET",
    headers: { Accept: "application/json" },
  });
  const data = await parseJsonOrThrow<ConversationsListResponse>(res);
  return data.conversations;
}

export async function createConversation(title?: string): Promise<ConversationSummary> {
  const res = await fetch(`${API_BASE_URL}/api/conversations`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify(title ? { title } : {}),
  });
  return parseJsonOrThrow<ConversationSummary>(res);
}

export async function fetchConversationMessages(
  conversationId: string
): Promise<ConversationMessagesResponse> {
  const res = await fetch(
    `${API_BASE_URL}/api/conversations/${encodeURIComponent(conversationId)}/messages`,
    {
      method: "GET",
      headers: { Accept: "application/json" },
    }
  );
  return parseJsonOrThrow<ConversationMessagesResponse>(res);
}

export async function sendConversationMessage(
  conversationId: string,
  message: string
): Promise<SendMessageResponse> {
  const res = await fetch(
    `${API_BASE_URL}/api/conversations/${encodeURIComponent(conversationId)}/messages`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({ message }),
    }
  );
  return parseJsonOrThrow<SendMessageResponse>(res);
}

