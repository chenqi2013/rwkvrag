import type { SearchResult, SaveWebSnapshotRequest, WebSnapshot } from "./types.ts";
import { externalSourceUrl } from "./citations.ts";

export function isWebSource(source: SearchResult) {
  return source.source === "web" || source.metadata?.retrieval_origin === "web";
}

export function webSnapshotId(source: SearchResult): string | undefined {
  const id = source.metadata?.web_snapshot_id;
  return isWebSource(source) && typeof id === "string" && id.trim() ? id : undefined;
}

/** Validate the server's saved receipt, never a model answer or displayed snippet. */
export function snapshotSaveIssue(source: SearchResult, snapshot: WebSnapshot): string | undefined {
  const id = webSnapshotId(source);
  if (!id || snapshot.id !== id) return "identity_mismatch";
  if (source.metadata?.content_status === "snippet_only" || snapshot.content_status !== "fetched") return "not_full_text";
  if (snapshot.save_allowed !== true) return "not_allowed";
  if (typeof snapshot.text !== "string" || !snapshot.text.trim()) return "empty_text";
  if (typeof snapshot.sha256 !== "string" || !/^[a-f0-9]{64}$/i.test(snapshot.sha256)) return "invalid_hash";
  if (!externalSourceUrl(snapshot.url)) return "invalid_url";
  return undefined;
}

export function snapshotSaveRequest(
  source: SearchResult, snapshot: WebSnapshot, knowledgeBaseId: string | undefined, confirmed: boolean,
): SaveWebSnapshotRequest {
  if (!confirmed || !knowledgeBaseId?.trim() || snapshotSaveIssue(source, snapshot)) {
    throw new Error("A valid snapshot, target knowledge base and explicit confirmation are required.");
  }
  return { knowledge_base_id: knowledgeBaseId, expected_sha256: snapshot.sha256, confirmed: true };
}
