import assert from "node:assert/strict";
import test from "node:test";
import { citationParts, citedSource, sourceLabels, externalSourceUrl, savedContext } from "../src/citations.ts";

const source = { id: "saved-2", snippet: "original evidence", metadata: { parent_source_id: "parent", snapshot_sha256: "old-sha" } };
const response = { sources: [source], generation: { citation_map: { "7": "saved-2" } }, retrieval: {} };

test("clickable parts preserve original text and never renumber a citation", () => {
  const text = "回答😀[资料 7]。另一项[Source 99]。[1](https://example.org)";
  const parts = citationParts(text);
  assert.equal(parts.map(p => p.text).join(""), text);
  assert.deepEqual(parts.filter(p => p.label !== undefined).map(p => p.label), [7, 99]);
});

test("explicit citation map takes precedence over positional source order", () => {
  assert.equal(citedSource(response, 7), source);
  assert.equal(citedSource(response, 1), undefined);
  assert.deepEqual(sourceLabels(response, source, 0), [7]);
  assert.equal(citedSource({ ...response, generation: {} }, 1), source);
});

test("missing, ambiguous or invalid source bindings never select a different source", () => {
  assert.equal(citedSource({ ...response, sources: [source, source] }, 7), undefined);
  assert.equal(citedSource(response, 0), undefined);
  assert.equal(citedSource(response, 999), undefined);
  assert.equal(citedSource({ ...response, generation: { citation_map: [] } }, 1), undefined);
});

test("only public URL protocols can become links; local files remain saved evidence", () => {
  for (const value of ["file:///private/doc.md", "javascript:alert(1)", "data:text/html,x", "https://user:secret@example.org", "/tmp/a"]) {
    assert.equal(externalSourceUrl(value), undefined);
  }
  assert.equal(externalSourceUrl("https://example.org/a"), "https://example.org/a");
});

test("snapshot lookup uses saved identity and hash rather than a newer page", () => {
  const value = { ...response, retrieval: { web_search: [{ snapshots: [
    { id: "parent", sha256: "new-sha", text: "newer page" },
    { id: "parent", sha256: "old-sha", text: "old saved page" },
  ] }] } };
  const before = JSON.stringify(value);
  assert.equal(savedContext(value, source), "old saved page");
  assert.equal(JSON.stringify(value), before);
});

test("Reader window keeps its full saved webpage binding", () => {
  const windowSource = { ...source, metadata: { ...source.metadata,
    parent_source_id: "window-2", web_snapshot_parent_id: "full-page" } };
  const value = { ...response, retrieval: { web_search: [{ snapshots: [
    { id: "window-2", sha256: "old-sha", text: "wrong parent" },
    { id: "full-page", sha256: "old-sha", text: "complete saved webpage" },
  ] }] } };
  assert.equal(savedContext(value, windowSource), "complete saved webpage");
});

test("original file URL binds the retained revision and page", async () => {
  const { originalSourceUrl } = await import("../src/citations.ts");
  const item = { ...source, metadata: { file_id: "file-1", source_sha256: "a".repeat(64), page: 2 } };
  assert.equal(originalSourceUrl(item), `/v1/admin/files/file-1/source/${"a".repeat(64)}#page=2`);
  assert.equal(originalSourceUrl({ ...item, metadata: { ...item.metadata, source_sha256: "../../etc/passwd" } }), undefined);
  assert.equal(originalSourceUrl(source), undefined);
});
