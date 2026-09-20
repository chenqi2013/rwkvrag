import assert from "node:assert/strict";
import test from "node:test";
import { answerPresentation, evidenceWarnings } from "../src/answerPresentation.ts";

test("matrix review failures remain visible without editing model output", () => {
  for (const status of ["answer_quality_failed", "answer_review_failed", "matrix_partial_failure"]) {
    const response = { answer: "原始模型答案", generation: { pipeline: "rwkv", status,
      answer_span: [0, 6], model_calls: [{ stage: "writer", status: "completed" }] } };
    const before = JSON.stringify(response);
    const display = answerPresentation(response);
    assert.equal(display.color, "orange");
    assert.equal(display.answerText, response.answer);
    assert.equal(JSON.stringify(response), before);
  }
});

test("partial retrieval is visible and preserves the original answer", () => {
  const response = { answer: "local evidence only", sources: [{}], generation: {
    pipeline: "rwkv", status: "retrieval_partial_failure", writer_status: "completed",
    answer_span: [0, 19], retrieval_failures: [{ provider: "web", error: "TimeoutError" }],
    model_calls: [{ stage: "writer", status: "completed" }],
  } };
  const before = JSON.stringify(response);
  assert.match(answerPresentation(response).label[1], /retrieval requests failed/);
  assert.match(evidenceWarnings(response)[0][1], /remaining sources/);
  assert.equal(JSON.stringify(response), before);
});

test("routing failure offers an explicit scope choice without inventing an answer", () => {
  const response = { answer: "", generation: { pipeline: "rwkv", status: "routing_failed", model_calls: [] } };
  const display = answerPresentation(response);
  assert.match(display.label[1], /choose a search scope/);
  assert.equal(display.answerText, "");
  assert.equal(display.writerCalled, false);
});

test("Shanghai response exposes missing evidence and invalid citation without replacing model output", () => {
  const response = { answer: "上海地铁线路数量：[资料 1] 上海地铁线路总数：[资料 1] 上海地铁线路数：[资料 1]",
    sources: [], generation: { pipeline: "rwkv", status: "completed", writer_status: "completed",
      answer_span: [0, 46], evidence_count: 0,
      citation_audit: { unknown_label_ids: [1], semantic_support_verified: false } } };
  const before = JSON.stringify(response);
  const warnings = evidenceWarnings(response);
  assert.equal(warnings.length, 2);
  assert.match(warnings[0][0], /未返回有效证据/);
  assert.match(warnings[1][0], /引用无对应来源：\[资料 1\]/);
  assert.equal(answerPresentation(response).answerText, response.answer);
  assert.equal(JSON.stringify(response), before);
});

test("evidence notices tolerate legacy responses and do not infer semantic correctness", () => {
  assert.deepEqual(evidenceWarnings(), []);
  assert.deepEqual(evidenceWarnings({ sources: [{}], generation: {} }), []);
  assert.deepEqual(evidenceWarnings({ sources: [{}], generation: {
    citation_audit: { unknown_label_ids: [], semantic_support_verified: false } } }), []);
});

test("malformed labels and fabricated source quotes are visible without rewriting the answer", () => {
  const response = { answer: "原文：虚构[资料 N]", sources: [{}], generation: {
    citation_audit: { invalid_labels: ["[资料 N]"], quote_audit: { failed: 1 } } } };
  const before = JSON.stringify(response);
  const warnings = evidenceWarnings(response);
  assert.equal(warnings.length, 2);
  assert.match(warnings[0][1], /malformed citation/);
  assert.match(warnings[1][1], /do not match/);
  assert.equal(JSON.stringify(response), before);
});

function native(answer, extra = {}) {
  return { answer, generation: { pipeline: "rwkv", status: "completed",
    model_calls: [{ stage: "writer", status: "completed", completion_attempted: true }],
    ...extra } };
}

test("Python code-point bounds preserve emoji, raw text and the original API object", () => {
  const prefix = ">先想😀\n</think>";
  const body = "第一项😀，e\u0301第二项。";
  const raw = prefix + body + "<end>";
  const response = native(raw, { answer_span: [Array.from(prefix).length,
    Array.from(prefix + body).length] });
  const before = JSON.stringify(response);
  const result = answerPresentation(response);
  assert.equal(result.answerText, body);
  assert.equal(result.rawAnswer, raw);
  assert.notEqual(raw.slice(...response.generation.answer_span), body);
  assert.equal(JSON.stringify(response), before);
  assert.equal(result.writerCalled, true);
  assert.equal(result.color, "blue");
  assert.match(result.label[1], /semantic support not verified/);
});

for (const span of [undefined, null, [0], [-1, 2], [3, 1], [0, 999], [true, 2], [0, 2.5], [0, 2, 3]]) {
  test(`invalid or missing answer_span never exposes thinking as an answer: ${JSON.stringify(span)}`, () => {
    const raw = ">thinking only😀</think>";
    const result = answerPresentation(native(raw, { answer_span: span }));
    assert.equal(result.answerText, "");
    assert.equal(result.rawAnswer, raw);
    assert.equal(result.spanValid, false);
  });
}

test("native completed ignores legacy verification flags and does not claim semantic correctness", () => {
  const result = answerPresentation(native("answer", { answer_span: [0, 6],
    grounding_valid: true, answer_support_passed: true, answer_strategy: "single_writer_call" }));
  assert.equal(result.isNative, true);
  assert.equal(result.color, "blue");
  assert.match(result.label[1], /semantic support not verified/);
});

test("resolver partial failure preserves the answer and reports failed evidence reads", () => {
  const result = answerPresentation(native("answer", { status: "resolver_partial_failure",
    writer_status: "completed", answer_span: [0, 6] }));
  assert.equal(result.answerText, "answer");
  assert.equal(result.color, "orange");
  assert.match(result.label[1], /some evidence reads failed/);
});

test("planner fallback remains visible without changing the model answer", () => {
  const response = { ...native("answer", { status: "planner_partial_failure",
    writer_status: "completed", answer_span: [0, 6], planner_fallback: "original_question" }),
    sources: [{}] };
  const before = JSON.stringify(response);
  assert.equal(answerPresentation(response).answerText, "answer");
  assert.equal(answerPresentation(response).color, "orange");
  assert.match(answerPresentation(response).label[1], /planning failed/);
  assert.match(evidenceWarnings(response)[0][1], /original question/);
  response.generation.status = "length";
  assert.match(answerPresentation(response).label[1], /Output limit reached/);
  assert.match(evidenceWarnings(response)[0][1], /original question/);
  response.generation.status = "planner_partial_failure";
  assert.equal(JSON.stringify(response), before);
});

test("length-limited output stays explicitly incomplete even with a valid final span", () => {
  const result = answerPresentation(native("partial", { status: "length", answer_span: [0, 7],
    model_calls: [{ stage: "writer", status: "length", completion_attempted: true }] }));
  assert.equal(result.answerText, "partial");
  assert.match(result.label[1], /Output limit reached; generation incomplete/);
});

test("writer budget rejection distinguishes stage entry from a model generation request", () => {
  const result = answerPresentation(native("", { status: "budget_exceeded", answer_span: null,
    model_calls: [{ stage: "writer", status: "budget_exceeded", completion_attempted: false }] }));
  assert.equal(result.writerAttempted, true);
  assert.equal(result.writerCalled, false);
  assert.match(result.label[1], /Input exceeds configured budget; not generated/);
});

for (const status of ["planner_failed", "retrieval_failed", "invalid_materials"]) {
  test(`${status} does not invent a Writer call or an insufficient-evidence judgment`, () => {
    const result = answerPresentation(native("", { status, model_calls: [{ stage: "planner" }] }));
    assert.equal(result.writerAttempted, false);
    assert.equal(result.writerCalled, false);
    assert.equal(result.color, "red");
    assert.doesNotMatch(result.label[1], /Insufficient evidence/);
  });
}

test("completed with no writer records remains a missing-record diagnostic", () => {
  const result = answerPresentation(native("answer", { model_calls: [], answer_span: [0, 6] }));
  assert.match(result.label[1], /generation record missing/);
});

test("legacy answer text and existing verification/failure labels stay compatible", () => {
  const response = { answer: "原样回答😀", generation: { answer_strategy: "single_writer_call",
    grounding_valid: true, answer_support_passed: true } };
  const result = answerPresentation(response);
  assert.equal(result.isNative, false);
  assert.equal(result.answerText, response.answer);
  assert.equal(result.color, "green");
  assert.equal(result.label[1], "Evidence verified");
  assert.equal(answerPresentation({ answer: "失败原文", generation: {
    answer_strategy: "generation_failed" } }).label[1], "Generation failed");
  assert.equal(answerPresentation({ answer: "旧拒答", generation: {} }).label[1],
    "Insufficient evidence; not generated");
});
