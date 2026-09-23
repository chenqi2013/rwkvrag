import assert from "node:assert/strict";
import test from "node:test";
import { formatAnswer, safeHistoryAnswer } from "../src/answerFormat.ts";

const response = {
  sources: [{ id: "requests" }, { id: "httpx" }],
  generation: { citation_map: { "1": "requests", "2": "httpx" } },
};

test("valid cited answer stays byte-for-byte visible", () => {
  const raw = "Requests 发出 HTTP 请求[资料 1]；HTTPX 支持异步客户端[资料 2]。";
  assert.deepEqual(formatAnswer(raw, response), {
    text: raw, blocked: false, invalidLabels: [], repeated: false, missingCitations: false,
  });
});

test("fabricated source labels hide the whole answer without altering its raw source", () => {
  const raw = "可靠事实[资料 1]。编造来源的结论[资料 7]。";
  const snapshot = JSON.stringify(response);
  assert.deepEqual(formatAnswer(raw, response), {
    text: "", blocked: true, invalidLabels: ["[资料 7]"], repeated: false, missingCitations: false,
  });
  assert.equal(JSON.stringify(response), snapshot);
  assert.match(raw, /编造来源/);
  assert.equal(safeHistoryAnswer(raw, response), undefined);
});

test("malformed citation and missing mapped source cannot become valid links", () => {
  assert.deepEqual(formatAnswer("未知[资料 N]", response).invalidLabels, ["[资料 N]"]);
  assert.deepEqual(formatAnswer("未知[资料 2]", {
    ...response, generation: { citation_map: { "1": "requests", "2": "missing" } },
  }).invalidLabels, ["[资料 2]"]);
});

test("repeated claim with rotating source numbers is hidden", () => {
  const claim = "说明 Requests 是一个流行的 HTTP 客户端，但没有记载同步异步支持。";
  const raw = `[资料 1] ${claim}\n[资料 2] ${claim}\n[资料 1] ${claim}`;
  const result = formatAnswer(raw, response);
  assert.equal(result.blocked, true);
  assert.equal(result.repeated, true);
  assert.equal(result.text, "");
  assert.equal(safeHistoryAnswer(raw, response), undefined);
});

test("a long paragraph duplicated once is hidden even if every source number exists", () => {
  const paragraph = "Requests 和 HTTPX 的定位需要分别核对各自材料，异步和同步支持也须分开说明。".repeat(2);
  const raw = `${paragraph}[资料 1]\n${paragraph}[资料 2]`;
  const result = formatAnswer(raw, response);
  assert.equal(result.blocked, true);
  assert.equal(result.repeated, true);
});

test("valid answer is carried into conversation history unchanged", () => {
  const raw = "HTTPX 有同步与异步接口[资料 2]。";
  assert.equal(safeHistoryAnswer(raw, response), raw);
});

test("saved sources without an actual citation are explicitly marked, never cited automatically", () => {
  const raw = "HTTPX 支持异步接口。";
  const result = formatAnswer(raw, response);
  assert.equal(result.missingCitations, true);
  assert.equal(result.blocked, false);
  assert.equal(result.text, raw);
  assert.equal(safeHistoryAnswer(raw, response), raw);
  assert.equal(formatAnswer("资料不足，无法判断。", { sources: [], generation: {} }).missingCitations, false);
});
