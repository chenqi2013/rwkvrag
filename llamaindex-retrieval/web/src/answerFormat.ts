import type { AskResponse } from "./types.ts";
import { citationParts, citedSource } from "./citations.ts";

export type FormattedAnswer = {
  text: string;
  blocked: boolean;
  invalidLabels: string[];
  repeated: boolean;
};

const candidateTag = /\[(?:资料|Source)\s*[^\]\r\n]*(?:\]|$)/g;
const validTag = /^\[(?:资料|Source)\s*[1-9]\d*\]$/;

function repeatedLongSpan(text: string): boolean {
  const normalized = text.replace(/\[(?:资料|Source)\s*[^\]\r\n]*\]/g, "")
    .replace(/\s+/g, "").toLowerCase();
  for (const [width, minimum] of [[96, 2], [32, 3]]) {
    const positions = new Map<string, number[]>();
    for (let i = 0; i + width <= normalized.length; i++) {
      const span = normalized.slice(i, i + width);
      const seen = positions.get(span);
      if (!seen) {
        positions.set(span, [i]);
      } else if (i - seen[seen.length - 1] >= width) {
        seen.push(i);
        if (seen.length >= minimum) return true;
      }
    }
  }
  return false;
}

/** A display gate. The response and saved raw answer are never rewritten. */
export function formatAnswer(text: string, response: Pick<AskResponse, "sources" | "generation">): FormattedAnswer {
  const invalid = new Set<string>();
  for (const tag of text.matchAll(candidateTag)) {
    if (!validTag.test(tag[0])) invalid.add(tag[0]);
  }
  for (const part of citationParts(text)) {
    if (part.label !== undefined && !citedSource(response, part.label)) invalid.add(part.text);
  }
  const repeated = repeatedLongSpan(text);
  const blocked = invalid.size > 0 || repeated;
  return { text: blocked ? "" : text, blocked, invalidLabels: [...invalid], repeated };
}

export function safeHistoryAnswer(text: string, response: Pick<AskResponse, "sources" | "generation">): string | undefined {
  const formatted = formatAnswer(text, response);
  return formatted.blocked || !formatted.text.trim() ? undefined : formatted.text;
}
