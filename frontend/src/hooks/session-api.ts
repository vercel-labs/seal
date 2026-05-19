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

/** Part types the UI knows how to render. */
const KNOWN_TYPES = new Set([
  "text",
  "file",
  "step-start",
  "source-url",
  "source-document",
  "reasoning",
]);

function isKnownPart(p: Record<string, unknown>): boolean {
  const t = p.type as string | undefined;
  if (!t) return false;
  return KNOWN_TYPES.has(t) || t.startsWith("tool-");
}

function normalizePart(
  part: Record<string, unknown>,
): Record<string, unknown> | null {
  if (!isKnownPart(part)) return null;

  if (typeof part.type === "string" && part.type.startsWith("tool-")) {
    if (part.output !== undefined) {
      return {
        ...part,
        state:
          part.state === "output-error" || typeof part.errorText === "string"
            ? "output-error"
            : part.state === "output-denied" ||
                (typeof part.approval === "object" &&
                  part.approval !== null &&
                  (part.approval as { approved?: unknown }).approved === false)
              ? "output-denied"
              : "output-available",
      };
    }

    if (part.state === "call") {
      return {
        ...part,
        state: "input-available",
      };
    }
  }

  return part;
}

/**
 * Fetch messages for a session and convert them to the UIMessage shape
 * that `useChat` expects as `initialMessages`.
 */
export async function fetchSessionMessages(
  sessionId: string,
): Promise<UIMessage[]> {
  const res = await fetch(`/api/sessions/${sessionId}`);
  if (!res.ok) return [];

  const data: {
    messages: {
      id: string;
      role: string;
      parts: Record<string, unknown>[];
      createdAt?: string;
    }[];
  } = await res.json();

  return data.messages
    .map((m) => ({
      id: m.id,
      role: m.role as UIMessage["role"],
      parts: m.parts
        .map(normalizePart)
        .filter(
          (part): part is Record<string, unknown> => part !== null,
        ) as UIMessage["parts"],
      ...(m.createdAt ? { createdAt: new Date(m.createdAt) } : {}),
    }))
    .filter((m) => m.parts.length > 0);
}
