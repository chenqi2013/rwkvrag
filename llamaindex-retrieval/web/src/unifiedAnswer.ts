import type { AskResponse } from "./types.ts";

type RecordValue = Record<string, unknown>;
const object = (value: unknown): value is RecordValue => !!value && typeof value === "object" && !Array.isArray(value);
const texts = (value: unknown): value is string[] => Array.isArray(value) && value.every(x => typeof x === "string" && x.trim().length > 0);

export const isUnifiedSpan = (protocol: unknown) => protocol === "rwkv-unified-v3" || protocol === "rwkv-unified-v4";

/** Display-only JSON projection. Never recover an intermediate or repair raw. */
export function unifiedAnswer(response?: Pick<AskResponse, "answer" | "generation">) {
  if (!response || !isUnifiedSpan(response.generation.protocol)) return undefined;
  const g = response.generation;
  const nodes = Array.isArray(g.nodes) ? g.nodes.filter(object) : [];
  const roots = nodes.filter(n => typeof g.root_node_id === "string" && n.id === g.root_node_id);
  const root = roots.length === 1 ? roots[0] : undefined;
  const identity = !!root && root.raw_text === response.answer && g.raw_model_answer === response.answer
    && typeof root.call_id === "string" && root.call_id.length > 0 && g.answer_call_id === root.call_id;
  let valid = false, answerText = "";
  let missing: string[] = [], referenceIds: string[] = [];
  if (identity && root?.status === "completed") {
    if (root.kind === "clarify" && g.answer_representation === "model_raw_text") {
      valid = true;
      answerText = response.answer;
    } else if (root.kind === "final" && g.root_contract === "SpanAnswer") {
      try {
        const value: unknown = JSON.parse(response.answer);
        if (object(value) && Object.keys(value).sort().join(",") === "answer,missing,span_ids"
          && typeof value.answer === "string" && value.answer.trim().length > 0
          && texts(value.missing) && texts(value.span_ids)
          && JSON.stringify(value.span_ids) === JSON.stringify(root.span_ids)) {
          valid = true;
          answerText = value.answer; // Exact model field; no sentence filtering/rewriting.
          missing = value.missing;
          referenceIds = value.span_ids;
        }
      } catch { /* The unchanged raw remains in the trace view. */ }
    }
  }
  return { valid, answerText, missing, referenceIds, root,
    attempted: !!root && (typeof root.event_index === "number" || typeof root.call_id === "string"),
    // A rejected root may have been generated; a queued/budget-blocked event
    // may not have been. Count explicit completion receipts, not node status.
    called: !!root && (root.kind === "final" || root.kind === "clarify") && Array.isArray(g.model_calls)
      && g.model_calls.some(c => object(c) && typeof root.call_id === "string" && root.call_id.length > 0
        && c.call_id === root.call_id && c.completion_attempted === true) };
}

/** UI follows its existing completed-only history policy, retaining v3/v4 root JSON. */
export function historyAnswer(response: Pick<AskResponse, "answer" | "generation">, displayed: string) {
  const unified = unifiedAnswer(response);
  if (unified) return response.generation.status === "completed" && unified.valid ? response.answer : undefined;
  return response.generation.pipeline === "rwkv"
    ? response.generation.status === "completed" && response.generation.writer_status === "completed" ? displayed : undefined
    : response.generation.answer_strategy === "single_writer_call" ? displayed : undefined;
}
