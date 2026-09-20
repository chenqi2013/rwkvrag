export interface AtomicRequest { object: string; attribute: string; conditions: string }
export interface EvidenceSpan { start: number; end: number; text: string; context: EvidenceSpan[] }
export interface AtomicClaim {
  id: string;
  kind: "reader_supported" | "unconfirmed";
  target: AtomicRequest;
  value_quote: string | null;
  statement_quote: string;
  statement_position: { start: number; end: number; origin?: string; context_index?: number };
  scope_quote: string | null;
  evidence: EvidenceSpan;
  binding: { source_id: string; document_id: string; indexed_text_sha256: string; source_sha256?: string; parsed_snapshot_sha256?: string; index_version?: string };
  value_positions: { start: number; end: number }[];
  semantic_verified: boolean;
}
export interface AtomicRunSummary { id: string; request: AtomicRequest; created_at: string; status: string }
export interface AtomicRun extends AtomicRunSummary {
  knowledge_base_id: string;
  claims: AtomicClaim[];
  sources: { id: string; title: string; snippet: string; metadata: Record<string, unknown> }[];
  calls: Record<string, unknown>[];
  issues: Record<string, unknown>[];
  coverage: Record<string, unknown>;
  freshness?: string;
}

/** Python offsets count Unicode code points; JavaScript string offsets do not. */
export function exactSpan(text: string, start: number, end: number, expected: string) {
  const chars = Array.from(text);
  if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end <= start || end > chars.length) return undefined;
  const quote = chars.slice(start, end).join("");
  if (quote !== expected) return undefined;
  return { before: chars.slice(0, start).join(""), quote, after: chars.slice(end).join("") };
}
