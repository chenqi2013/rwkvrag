import assert from "node:assert/strict";
import test from "node:test";
import { api } from "../src/api.ts";
import { snapshotSaveIssue, snapshotSaveRequest, webSnapshotId } from "../src/webSnapshots.ts";

const source = {
  id: "web-result", source: "web", snippet: "An answer-sized excerpt, not the saved body",
  metadata: { web_snapshot_id: "snapshot/with space", content_status: "fetched" },
};
const snapshot = {
  id: "snapshot/with space", title: "Source title", url: "https://example.org/docs",
  text: "The server's complete captured text.", sha256: "a".repeat(64),
  content_status: "fetched", retrieved_at: "2026-09-24T00:00:00Z", provider: "tavily", save_allowed: true,
};

test("legacy web evidence and local documents have no save receipt", () => {
  assert.equal(webSnapshotId({ ...source, metadata: {} }), undefined);
  assert.equal(webSnapshotId({ ...source, source: "upload" }), undefined);
  assert.equal(webSnapshotId({ ...source, metadata: { web_snapshot_id: " " } }), undefined);
  assert.equal(webSnapshotId(source), snapshot.id);
});

test("snippets cannot be promoted to full documents by the source card or a receipt", () => {
  assert.equal(snapshotSaveIssue(source, { ...snapshot, content_status: "snippet_only" }), "not_full_text");
  const snippet = { ...source, metadata: { ...source.metadata, content_status: "snippet_only" } };
  assert.equal(snapshotSaveIssue(snippet, snapshot), "not_full_text");
  assert.throws(() => snapshotSaveRequest(snippet, snapshot, "kb-a", true));
});

test("wrong identity, server rejection, empty body, invalid hash and unsafe URL fail closed", () => {
  for (const [patch, issue] of [
    [{ id: "different" }, "identity_mismatch"],
    [{ save_allowed: false }, "not_allowed"],
    [{ text: "  " }, "empty_text"],
    [{ sha256: "wrong-hash" }, "invalid_hash"],
    [{ url: "javascript:alert(1)" }, "invalid_url"],
    [{ url: "https://user:password@example.org" }, "invalid_url"],
  ]) {
    assert.equal(snapshotSaveIssue(source, { ...snapshot, ...patch }), issue);
    assert.throws(() => snapshotSaveRequest(source, { ...snapshot, ...patch }, "kb-a", true));
  }
});

test("saving requires both explicit confirmation and a selected knowledge base", () => {
  assert.throws(() => snapshotSaveRequest(source, snapshot, "kb-a", false));
  assert.throws(() => snapshotSaveRequest(source, snapshot, undefined, true));
  assert.throws(() => snapshotSaveRequest(source, snapshot, " ", true));
});

test("save request contains only target, previewed hash and confirmation; source remains immutable", () => {
  const original = JSON.stringify({ source, snapshot });
  assert.deepEqual(snapshotSaveRequest(source, snapshot, "kb-a", true), {
    knowledge_base_id: "kb-a", expected_sha256: snapshot.sha256, confirmed: true,
  });
  assert.equal(JSON.stringify({ source, snapshot }), original);
});

test("preview uses GET and confirmation posts exact identity/hash without search, answer or upload", async (t) => {
  const calls = [];
  const saved = { file_id: "file-1", job_id: "job-1", status: "pending" };
  t.mock.method(globalThis, "fetch", async (url, options) => {
    calls.push({ url, options });
    return new Response(JSON.stringify(options?.method === "POST" ? saved : snapshot), { status: 200 });
  });
  const preview = await api.webSnapshot(snapshot.id);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "/v1/admin/web-snapshots/snapshot%2Fwith%20space");
  assert.equal(calls[0].options?.method, undefined);
  const result = await api.saveWebSnapshot(snapshot.id, snapshotSaveRequest(source, preview, "kb-a", true));
  assert.deepEqual(result, saved);
  assert.equal(calls.length, 2);
  assert.equal(calls[1].url, "/v1/admin/web-snapshots/snapshot%2Fwith%20space/save");
  assert.equal(calls[1].options.method, "POST");
  assert.deepEqual(JSON.parse(calls[1].options.body), {
    knowledge_base_id: "kb-a", expected_sha256: snapshot.sha256, confirmed: true,
  });
});

test("receipt GET failures cannot become a successful preview", async (t) => {
  t.mock.method(globalThis, "fetch", async () => new Response(JSON.stringify({ detail: "snapshot_not_found" }), { status: 404 }));
  await assert.rejects(api.webSnapshot(snapshot.id), /snapshot_not_found/);
});

test("an existing failed ingestion is preserved rather than reported as a new successful job", async (t) => {
  const result = { file_id: "old-file", job_id: "old-job", status: "failed", existing: true };
  t.mock.method(globalThis, "fetch", async () => new Response(JSON.stringify(result), { status: 200 }));
  assert.deepEqual(await api.saveWebSnapshot(snapshot.id, snapshotSaveRequest(source, snapshot, "kb-a", true)), result);
});
