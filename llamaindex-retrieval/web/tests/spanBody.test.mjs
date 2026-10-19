import assert from 'node:assert/strict';
import test from 'node:test';
import { unifiedAnswer, historyAnswer } from '../src/unifiedAnswer.ts';
import { citationParts } from '../src/citations.ts';
import { formatAnswer, safeHistoryAnswer } from '../src/answerFormat.ts';
import { answerPresentation, evidenceWarnings } from '../src/answerPresentation.ts';

function fixture(protocol, body) {
  const raw = JSON.stringify({answer: body, span_ids: ['E1.S1'], missing: []}, null, 2);
  return {answer: raw, sources: [{id: 'source1', snippet: body}], generation: {
    pipeline: 'rwkv', protocol, status: 'completed', raw_model_answer: raw,
    root_node_id: 'N2', answer_call_id: 'call2', root_contract: 'SpanAnswer',
    answer_representation: 'model_raw_json', citation_map: {'1': 'source1'},
    nodes: [{id: 'N2', kind: 'final', status: 'completed', call_id: 'call2',
      raw_text: raw, span_ids: ['E1.S1']}],
  }};
}

for (const protocol of ['rwkv-unified-v3', 'rwkv-unified-v4']) {
  for (const body of ['[资料 99]', '[资料 x]', '[Source 99]', '[Source x]', '`[资料 99]`',
    '```text\r\n[Source 99]\r\n```', '[1]', '  🙂\r\n保留空白\t ', '原样保留。'.repeat(40)]) {
    test(`${protocol} displays body literally, without legacy links or heuristic suppression: ${JSON.stringify(body).slice(0, 65)}`, () => {
      const r = fixture(protocol, body), before = JSON.stringify(r);
      const p = unifiedAnswer(r), formatted = formatAnswer(p.answerText, r);
      assert.equal(p.valid, true);
      assert.equal(formatted.blocked, false);
      assert.equal(formatted.text, body);
      assert.deepEqual(formatted.invalidLabels, []);
      assert.equal(formatted.missingCitations, false);
      assert.deepEqual(citationParts(formatted.text, r.generation.protocol), [{text: body}]);
      assert.equal(safeHistoryAnswer(p.answerText, r), body);
      assert.equal(historyAnswer(r, body), r.answer);
      assert.equal(JSON.stringify(r), before);
    });
  }
  test(`${protocol} legacy body diagnostics do not turn a literal into a reference warning`, () => {
    const r = fixture(protocol, '[资料 99]');
    r.generation.citation_audit = {unknown_label_ids: [99], invalid_labels: ['[资料 x]'], missing_valid_citation: true};
    const notices = evidenceWarnings(r).map(n => n[1]).join('\n');
    assert.doesNotMatch(notices, /Citations have no matching source|malformed citation labels|no valid source citation/);
    assert.match(notices, /do not verify factual support/);
  });
  test(`${protocol} invalid root still has no projected answer; raw remains inspectable`, () => {
    const r = fixture(protocol, '[资料 99]');
    r.generation.nodes[0].status = 'invalid_contract';
    const view = answerPresentation(r);
    assert.equal(view.answerText, '');
    assert.equal(view.rawAnswer, r.answer);
    assert.equal(unifiedAnswer(r).valid, false);
    assert.equal(historyAnswer(r, '[资料 99]'), undefined);
  });
}

test('span final invocation is taken from its completion receipt, not admission status', () => {
  const r = fixture('rwkv-unified-v4', '[Source 99]');
  r.generation.nodes[0].status = 'invalid_contract';
  r.generation.model_calls = [{call_id: 'call2', purpose: 'final', completion_attempted: true}];
  assert.equal(unifiedAnswer(r).called, true);
  assert.equal(unifiedAnswer(r).valid, false);
  r.generation.model_calls[0].completion_attempted = false;
  assert.equal(unifiedAnswer(r).called, false);
  r.generation.nodes[0].status = 'completed';
  assert.equal(unifiedAnswer(r).called, false);
  r.generation.model_calls = [{call_id: 'other', completion_attempted: true}];
  assert.equal(unifiedAnswer(r).called, false);
});

test('non-span protocols retain the legacy body-label contract and link parsing', () => {
  for (const protocol of [undefined, 'rwkv-unified-v2', 'rwkv-unified-v99']) {
    const r = fixture(protocol, '[Source 99]');
    assert.equal(formatAnswer('[Source 99]', r).blocked, true);
    assert.deepEqual(citationParts('[Source 99]', protocol), [{text: '[Source 99]', label: 99}]);
  }
});
