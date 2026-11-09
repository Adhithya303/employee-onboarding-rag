import type {
  ConversationMessagesResponse,
  ConversationsListResponse,
  ConversationSummary,
  DeleteDocumentResponse,
  DocumentStatusResponse,
  HealthResponse,
  SendMessageResponse,
  SourceMode,
} from "./types";

const RAW_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
export const API_BASE_URL = RAW_BASE_URL.replace(/\/+$/, "");

export const ACCESS_TOKEN_STORAGE_KEY = "faculty_onboarding_access_token";
export const ACCESS_TOKEN_REQUIRED_EVENT = "access-token-required";

/** The access token is typed by the user on the device (never bundled into the build). */
export function getAccessToken(): string {
  try {
    return localStorage.getItem(ACCESS_TOKEN_STORAGE_KEY) ?? "";
  } catch {
    return "";
  }
}

export function setAccessToken(token: string): void {
  try {
    if (token) localStorage.setItem(ACCESS_TOKEN_STORAGE_KEY, token);
    else localStorage.removeItem(ACCESS_TOKEN_STORAGE_KEY);
  } catch {
    // Ignore storage errors
  }
}

function authHeaders(base: Record<string, string>): Record<string, string> {
  const token = getAccessToken();
  return token ? { ...base, "X-Access-Token": token } : base;
}

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
    if (response.status === 401 && typeof window !== "undefined") {
      window.dispatchEvent(new Event(ACCESS_TOKEN_REQUIRED_EVENT));
    }
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
    headers: authHeaders({ Accept: "application/json" }),
  });
  const data = await parseJsonOrThrow<ConversationsListResponse>(res);
  return data.conversations;
}

export async function createConversation(title?: string): Promise<ConversationSummary> {
  const res = await fetch(`${API_BASE_URL}/api/conversations`, {
    method: "POST",
    headers: authHeaders({
      "Content-Type": "application/json",
      Accept: "application/json",
    }),
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
      headers: authHeaders({ Accept: "application/json" }),
    }
  );
  return parseJsonOrThrow<ConversationMessagesResponse>(res);
}

export async function sendConversationMessage(
  conversationId: string,
  message: string,
  sourceMode: SourceMode = "handbook"
): Promise<SendMessageResponse> {
  const res = await fetch(
    `${API_BASE_URL}/api/conversations/${encodeURIComponent(conversationId)}/messages`,
    {
      method: "POST",
      headers: authHeaders({
        "Content-Type": "application/json",
        Accept: "application/json",
      }),
      body: JSON.stringify({ message, source_mode: sourceMode }),
    }
  );
  return parseJsonOrThrow<SendMessageResponse>(res);
}

export async function fetchDocumentStatus(
  conversationId: string
): Promise<DocumentStatusResponse> {
  const res = await fetch(
    `${API_BASE_URL}/api/conversations/${encodeURIComponent(conversationId)}/document`,
    {
      method: "GET",
      headers: authHeaders({ Accept: "application/json" }),
    }
  );
  return parseJsonOrThrow<DocumentStatusResponse>(res);
}

export async function uploadDocument(
  conversationId: string,
  file: File
): Promise<DocumentStatusResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(
    `${API_BASE_URL}/api/conversations/${encodeURIComponent(conversationId)}/upload`,
    {
      method: "POST",
      // Do not set Content-Type header so the browser sets the multipart boundary
      headers: authHeaders({ Accept: "application/json" }),
      body: formData,
    }
  );
  return parseJsonOrThrow<DocumentStatusResponse>(res);
}

export async function deleteDocument(
  conversationId: string
): Promise<DeleteDocumentResponse> {
  const res = await fetch(
    `${API_BASE_URL}/api/conversations/${encodeURIComponent(conversationId)}/document`,
    {
      method: "DELETE",
      headers: authHeaders({ Accept: "application/json" }),
    }
  );
  return parseJsonOrThrow<DeleteDocumentResponse>(res);
}

