/**
 * Plain fetch helpers for the sessions API.
 * NOT a React hook — just async functions.
 */

import type { UIMessage } from "ai";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface Session {
  id: string;
  title: string | null;
  created_at: string;
  updated_at: string;
}

// ---------------------------------------------------------------------------
// Session CRUD
// ---------------------------------------------------------------------------

export async function fetchSessions(): Promise<Session[]> {
  const res = await fetch("/api/sessions");
  if (!res.ok) throw new Error("Failed to fetch sessions");
  return res.json();
}

export async function createSessionOnServer(id: string): Promise<Session> {
  const res = await fetch("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id }),
  });
  if (!res.ok) throw new Error("Failed to create session");
  return res.json();
}

export async function deleteSessionOnServer(id: string): Promise<void> {
  const res = await fetch(`/api/sessions/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error("Failed to delete session");
}

export async function generateSessionTitle(id: string): Promise<Session> {
  const res = await fetch(`/api/sessions/${id}/title`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to generate title");
  return res.json();
}

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

/**
 * Fetch messages for a session and convert them to the UIMessage shape
 * that `useChat` expects as `initialMessages`.
 */
export async function fetchSessionMessages(
  sessionId: string,
): Promise<UIMessage[]> {
  const res = await fetch(`/api/sessions/${sessionId}`);
  if (res.status === 404) throw new Error("Session not found");
  if (!res.ok) throw new Error("Failed to fetch session");

  const data: {
    messages: {
      id: string;
      role: string;
      metadata?: UIMessage["metadata"];
      parts: UIMessage["parts"];
    }[];
  } = await res.json();

  return data.messages
    .map((m) => ({
      id: m.id,
      role: m.role as UIMessage["role"],
      ...(m.metadata !== undefined ? { metadata: m.metadata } : {}),
      parts: m.parts,
    }))
    .filter((m) => m.parts.length > 0);
}
