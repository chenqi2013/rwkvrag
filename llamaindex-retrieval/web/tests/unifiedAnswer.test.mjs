import assert from 'node:assert/strict';
import test from 'node:test';
import { unifiedAnswer, historyAnswer } from '../src/unifiedAnswer.ts';
import { answerPresentation, evidenceWarnings } from '../src/answerPresentation.ts';

function fixture(body = '模型原句🙂\n条件不变') {
  const raw = JSON.stringify({answer: body, span_ids: ['E1.S2'], missing: ['模型缺口']}, null, 2);
  return {answer: raw, sources: [{id: 'source1'}], generation: {
    pipeline: 'rwkv', protocol: 'rwkv-unified-v3', status: 'completed',
    raw_model_answer: raw, root_node_id: 'N3', answer_call_id: 'call3', root_contract: 'SpanAnswer',
    answer_representation: 'model_raw_json', execution_scope: {unread_units: 0},
    nodes: [{id: 'N1', kind: 'leaf', raw_text: 'DO_NOT_SHOW_INTERMEDIATE', status: 'completed'},
      {id: 'N3', kind: 'final', status: 'completed', call_id: 'call3', event_index: 3,
        raw_text: raw, span_ids: ['E1.S2']}],
  }};
}

test('projects only the exact root answer field; raw, missing and source choices are unchanged', () => {
  const r = fixture(); const before = JSON.stringify(r);
  const view = unifiedAnswer(r);
  assert.equal(view.answerText, '模型原句🙂\n条件不变');
  assert.deepEqual(view.missing, ['模型缺口']);
  assert.deepEqual(view.referenceIds, ['E1.S2']);
  assert.equal(JSON.stringify(r), before);
  assert.equal(answerPresentation(r).spanValid, true);
  assert.match(answerPresentation(r).label[1], /factual support unverified/);
  assert.equal(historyAnswer(r, view.answerText), r.answer);
});

for (const failure of ['length', 'invalid_contract', 'blocked_dependency', 'cancelled']) {
  test(`failed root ${failure} never projects a leaf or repairs JSON`, () => {
    const r = fixture(); r.generation.nodes[1].status = failure;
    assert.equal(unifiedAnswer(r).answerText, '');
    assert.equal(answerPresentation(r).rawAnswer, r.answer);
    assert.equal(historyAnswer(r, 'DO_NOT_SHOW_INTERMEDIATE'), undefined);
  });
}

for (const [index, mutation] of [
  r => {r.generation.root_node_id = 'N1';},
  r => {r.generation.answer_call_id = 'other';},
  r => {r.generation.raw_model_answer = 'other';},
  r => {r.generation.nodes[1].raw_text = 'other';},
  r => {r.generation.nodes[1].span_ids = ['E9.S9'];},
  r => {r.generation.nodes.push({...r.generation.nodes[1]});},
].entries()) {
  test(`root identity or selection mismatch ${index} is not guessed around`, () => {
    const r = fixture(); mutation(r);
    assert.equal(unifiedAnswer(r).valid, false);
    assert.equal(answerPresentation(r).answerText, '');
  });
}

test('partial root is visible with warnings but not silently carried as a completed turn', () => {
  const r = fixture('不修正任何错误句子'); r.generation.status = 'partial';
  r.generation.execution_scope = {unread_units: 5, failed_nodes: ['N2'], failed_actions: ['A1'], control_failures: ['review_failed']};
  assert.equal(answerPresentation(r).answerText, '不修正任何错误句子');
  assert.equal(answerPresentation(r).color, 'orange');
  assert.equal(historyAnswer(r, '不修正任何错误句子'), undefined);
  const notices = evidenceWarnings(r).map(n => n[1]).join('\n');
  assert.match(notices, /5 acquired material units/);
  assert.match(notices, /failed_nodes/);
  assert.match(notices, /control_failures/);
});

test('clarification root is model raw text, not a fabricated SpanAnswer', () => {
  const r = fixture(); r.answer = '请明确对象？';
  Object.assign(r.generation, {raw_model_answer: r.answer, root_contract: null, answer_representation: 'model_raw_text', status: 'needs_clarification'});
  Object.assign(r.generation.nodes[1], {kind: 'clarify', raw_text: r.answer});
  assert.equal(unifiedAnswer(r).answerText, r.answer);
});

test('schema-like or malformed final output is preserved raw, not projected or repaired', () => {
  for (const value of ['["answer"]', '{"answer":"x"}', '{"answer":"x","span_ids":[],"missing":[],"extra":1}', '```json\n{}\n```']) {
    const r = fixture(); r.answer = r.generation.raw_model_answer = r.generation.nodes[1].raw_text = value;
    assert.equal(unifiedAnswer(r).valid, false);
    assert.equal(answerPresentation(r).rawAnswer, value);
  }
});

test('source attempt counts come from actions, not requested mode or model prose', () => {
  const r = fixture('I searched the web');
  r.retrieval = {requested_mode: 'web', actions: [{channel: 'knowledge_base', status: 'failed'}]};
  r.generation.execution_scope.stop_reason = 'retrieval_budget_exhausted';
  const notices = evidenceWarnings(r).map(x => x[1]).join('\n');
  assert.match(notices, /knowledge base 1, Web 0/);
  assert.match(notices, /retrieval_budget_exhausted/);
});

test('v4 keeps immutable root projection and discloses control attempts, not claimed success', () => {
  const r = fixture(); r.generation.protocol = 'rwkv-unified-v4';
  r.retrieval = {directions: [{status: 'admitted'}, {status: 'rejected'}], actions: []};
  r.generation.execution_scope.plan_revision_calls = 1;
  r.generation.model_calls = [{purpose: 'direction', completion_attempted: true},
    {purpose: 'direction', completion_attempted: false}, {purpose: 'plan_revision', completion_attempted: true}];
  assert.equal(unifiedAnswer(r).valid, true);
  assert.equal(historyAnswer(r, 'projection'), r.answer);
  assert.match(evidenceWarnings(r).map(x => x[1]).join('\n'), /Direction events: 2; recorded completion attempts: direction 1, plan revision 1/);
  r.generation.nodes[1].status = 'invalid_contract';
  assert.equal(unifiedAnswer(r).valid, false);
  assert.equal(historyAnswer(r, 'projection'), undefined);
});

test('unknown future protocol is not guessed into a span projection', () => {
  const r = fixture(); r.generation.protocol = 'rwkv-unified-v99';
  assert.equal(unifiedAnswer(r), undefined);
});

test('legacy history behavior is retained and v2 is not guessed into a v3 projection', () => {
  const r = fixture(); r.generation.protocol = 'rwkv-unified-v2';
  assert.equal(unifiedAnswer(r), undefined);
  assert.equal(historyAnswer(r, 'visible'), undefined);
  r.generation.writer_status = 'completed';
  assert.equal(historyAnswer(r, 'visible'), 'visible');
});
