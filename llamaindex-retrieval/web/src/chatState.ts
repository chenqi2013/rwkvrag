import type { AskResponse } from "./types.ts";
import { answerPresentation } from "./answerPresentation.ts";
import { safeHistoryAnswer } from "./answerFormat.ts";
import { historyAnswer } from "./unifiedAnswer.ts";

export type ChatScope = {
  retrieval_mode: "knowledge_base" | "web" | "hybrid" | "auto";
  knowledge_base_id?: string;
  top_k: number;
};
export type ChatTurn = {
  id: string;
  question: string;
  status: "pending" | "received" | "failed" | "cancelled" | "interrupted";
  response?: AskResponse;
  error?: string;
  invalidResponse?: unknown;
  startedAt: number;
};
export type Conversation = {
  version: 1;
  id: string;
  revision: number;
  createdAt: number;
  updatedAt: number;
  scope: ChatScope;
  draft: string;
  turns: ChatTurn[];
};
export type HistoryMessage = { role: "user" | "assistant"; content: string };
export const DEFAULT_SCOPE: ChatScope = { retrieval_mode: "auto", top_k: 1 };
export const MAX_HISTORY = 64;

export function newChatId(random: Pick<Crypto, "getRandomValues"> & Partial<Pick<Crypto, "randomUUID">> = crypto): string {
  // randomUUID is unavailable on non-HTTPS LAN origins; getRandomValues still works.
  return random.randomUUID?.() ?? Array.from(random.getRandomValues(new Uint8Array(16)),
    byte => byte.toString(16).padStart(2, "0")).join("");
}

export function newConversation(id: string, now = Date.now()): Conversation {
  return { version: 1, id, revision: 0, createdAt: now, updatedAt: now,
    scope: { ...DEFAULT_SCOPE }, draft: "", turns: [] };
}

/** This is a transport eligibility check, never a factual correctness judgment. */
export function carriedAnswer(turn: ChatTurn): string | undefined {
  if (turn.status !== "received" || !turn.response) return undefined;
  const display = answerPresentation(turn.response);
  if (!display.spanValid) return undefined;
  const safe = safeHistoryAnswer(display.answerText, turn.response);
  return safe ? historyAnswer(turn.response, safe) : undefined;
}

/** Preserve complete eligible pairs, order and original strings. Never summarize/window. */
export function conversationHistory(conversation: Conversation): HistoryMessage[] {
  return conversation.turns.flatMap(turn => {
    const answer = carriedAnswer(turn);
    return answer === undefined ? [] : [
      { role: "user" as const, content: turn.question },
      { role: "assistant" as const, content: answer },
    ];
  });
}

export function requestIssue(conversation: Conversation): "empty" | "question_limit" | "history_limit" | "message_limit" | undefined {
  if (!conversation.draft.trim()) return "empty";
  if (Array.from(conversation.draft).length > 10_000) return "question_limit";
  const history = conversationHistory(conversation);
  // Same completed-pair limit as Search Lab, with no silent context deletion.
  if (history.length >= MAX_HISTORY) return "history_limit";
  if (history.some(message => Array.from(message.content).length > 100_000)) return "message_limit";
  return undefined;
}

export function chatRequest(conversation: Conversation) {
  const issue = requestIssue(conversation);
  if (issue) throw new Error(issue);
  return { ...conversation.scope, question: conversation.draft, history: conversationHistory(conversation) };
}

export function reviseConversation(conversation: Conversation, changes: Partial<Pick<Conversation, "scope" | "draft" | "turns">>, now = Date.now()): Conversation {
  if (changes.scope && conversation.turns.length) throw new Error("scope_locked");
  return { ...conversation, ...changes, revision: conversation.revision + 1, updatedAt: now };
}

/** A response can settle only its own still-pending turn. Cancelled/deleted turns stay so. */
export function settleTurn(conversation: Conversation, id: string,
  result: Pick<ChatTurn, "status" | "response" | "error" | "invalidResponse">): Conversation {
  if (result.status === "pending" || !conversation.turns.some(t => t.id === id && t.status === "pending")) return conversation;
  return reviseConversation(conversation, {
    turns: conversation.turns.map(t => t.id === id ? { ...t, ...result } : t),
  });
}

export function interruptPending(conversation: Conversation): Conversation {
  return conversation.turns.some(t => t.status === "pending") ? reviseConversation(conversation, {
    turns: conversation.turns.map(t => t.status === "pending" ? { ...t, status: "interrupted" } : t),
  }) : conversation;
}

// Stored data is versioned and checked before rendering. Invalid records remain in
// storage; never silently replace an unreadable conversation with an empty one.
const record = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);
const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
export function isAskResponse(v: unknown): v is AskResponse {
  return record(v) && typeof v.answer === "string" && record(v.generation) && record(v.retrieval)
    && Array.isArray(v.sources) && v.sources.every(s => record(s) && typeof s.id === "string"
      && typeof s.title === "string" && typeof s.snippet === "string" && typeof s.source === "string"
      && typeof s.document_id === "string" && finite(s.score) && record(s.metadata)
      && (s.uri == null || typeof s.uri === "string"));
}
export function isConversation(v: unknown): v is Conversation {
  if (!record(v) || v.version !== 1 || typeof v.id !== "string" || !v.id
    || !Number.isSafeInteger(v.revision) || (v.revision as number) < 0
    || !finite(v.createdAt) || !finite(v.updatedAt) || typeof v.draft !== "string"
    || !record(v.scope) || !["auto", "knowledge_base", "web", "hybrid"].includes(String(v.scope.retrieval_mode))
    || !Number.isInteger(v.scope.top_k) || Number(v.scope.top_k) < 1 || Number(v.scope.top_k) > 100
    || (v.scope.knowledge_base_id !== undefined && typeof v.scope.knowledge_base_id !== "string")
    || !Array.isArray(v.turns)) return false;
  const ids = new Set<string>();
  return v.turns.every(t => {
    if (!record(t) || typeof t.id !== "string" || !t.id || ids.has(t.id)
      || typeof t.question !== "string" || !finite(t.startedAt)
      || !["pending", "received", "failed", "cancelled", "interrupted"].includes(String(t.status))
      || (t.error !== undefined && typeof t.error !== "string")
      || (t.response !== undefined && !isAskResponse(t.response))
      || (t.status === "received" && !isAskResponse(t.response))) return false;
    ids.add(t.id);
    return true;
  });
}

/** Optimistic concurrency: a stale tab must not overwrite/delete a newer record. */
export function storageRevisionMatches(existing: unknown, next: Conversation, deleting = false): boolean {
  if (existing === undefined) return !deleting && next.revision === 0;
  return isConversation(existing) && existing.id === next.id
    && existing.revision === next.revision - (deleting ? 0 : 1);
}
