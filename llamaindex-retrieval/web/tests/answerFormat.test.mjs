import assert from "node:assert/strict";
import test from "node:test";
import { formatAnswer } from "../src/answerFormat.ts";

const response = {
  sources: [{ id: "requests" }, { id: "httpx" }],
  generation: { citation_map: { "1": "requests", "2": "httpx" } },
};

test("valid cited answer stays byte-for-byte visible", () => {
  const raw = "Requests 发出 HTTP 请求[资料 1]；HTTPX 支持异步客户端[资料 2]。";
  assert.deepEqual(formatAnswer(raw, response), {
    text: raw, blocked: false, invalidLabels: [], repeated: false,
  });
});

test("fabricated source labels hide the whole answer without altering its raw source", () => {
  const raw = "可靠事实[资料 1]。编造来源的结论[资料 7]。";
  const snapshot = JSON.stringify(response);
  assert.deepEqual(formatAnswer(raw, response), {
    text: "", blocked: true, invalidLabels: ["[资料 7]"], repeated: false,
  });
  assert.equal(JSON.stringify(response), snapshot);
  assert.match(raw, /编造来源/);
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
});
