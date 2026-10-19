import test from 'node:test';
import assert from 'node:assert/strict';
import { newChatId, newConversation, reviseConversation, settleTurn, interruptPending, carriedAnswer,
  conversationHistory, chatRequest, requestIssue, isAskResponse, isConversation, storageRevisionMatches } from '../src/chatState.ts';
import { api } from '../src/api.ts';

const response = (body = '  原文🙂\r\n[Source 99]  ', protocol = 'rwkv-unified-v4') => {
  const raw = JSON.stringify({ answer: body, span_ids: ['E1.S1'], missing: [] }, null, 2);
  return { answer: raw, sources: [{ id: 's', title: 'source', snippet: body, document_id: 'd',
    source: 'provided_material', score: 1, metadata: {}, uri: null }], retrieval: {}, generation: {
    pipeline: 'rwkv', protocol, status: 'completed', raw_model_answer: raw, root_node_id: 'N1',
    answer_call_id: 'c1', root_contract: 'SpanAnswer', answer_representation: 'model_raw_json',
    nodes: [{ id: 'N1', kind: 'final', status: 'completed', call_id: 'c1', raw_text: raw, span_ids: ['E1.S1'] }],
  } };
};
function turn(id = 't1', r = response()) {
  return { id, question: '  问题🙂\r\n逐字  ', status: 'received', startedAt: 12, response: r };
}
const withTurn = t => reviseConversation(newConversation('a'), { turns: [t], draft: '下一问' });

test('IDs remain available on insecure LAN origins without randomUUID', () => {
  assert.equal(newChatId({ getRandomValues: a => { a.fill(15); return a; } }), '0f'.repeat(16));
  assert.equal(newChatId({ randomUUID: () => 'uuid', getRandomValues: () => { throw Error('not needed'); } }), 'uuid');
  assert.notEqual(newChatId(), newChatId());
});

test('invalid response envelope survives persistence as audit-only data', () => {
  const pending = { ...turn(), status: 'pending', response: undefined };
  const raw = { answer: 'unchanged raw', sources: null };
  const c = settleTurn(withTurn(pending), pending.id, { status: 'failed', invalidResponse: raw });
  assert.deepEqual(structuredClone(c).turns[0].invalidResponse, raw);
  assert.equal(isConversation(c), true);
  assert.deepEqual(conversationHistory(c), []);
});

test('new conversation has its own empty history, automatic retrieval and no shared references', () => {
  const a = newConversation('a', 1), b = newConversation('b', 2);
  a.scope.retrieval_mode = 'web'; a.turns.push(turn());
  assert.deepEqual(conversationHistory(b), []);
  assert.deepEqual(b.scope, { retrieval_mode: 'auto', top_k: 1 });
  assert.equal(isConversation(b), true);
});
for (const protocol of ['rwkv-unified-v3', 'rwkv-unified-v4']) {
  test(`${protocol}: exact role order, full root raw and original questions go over the wire`, () => {
    const r = response(undefined, protocol), t = turn('t', r);
    const c = withTurn(t), before = JSON.stringify(c);
    c.draft = '  保持输入\n\t';
    const payload = chatRequest(c);
    assert.deepEqual(payload.history, [{ role: 'user', content: t.question }, { role: 'assistant', content: r.answer }]);
    assert.equal(payload.question, c.draft);
    assert.equal(payload.history[1].content, r.answer);
    c.draft = '下一问'; assert.equal(JSON.stringify(c), before);
  });
}
for (const status of ['pending', 'failed', 'cancelled', 'interrupted']) {
  test(`${status} turn stays visible but is never carried even if a response is attached`, () => {
    const t = { ...turn(), status }, c = withTurn(t);
    assert.equal(carriedAnswer(t), undefined);
    assert.deepEqual(chatRequest(c).history, []);
    assert.equal(c.turns[0].response.answer, t.response.answer);
  });
}
for (const status of ['length', 'no_evidence', 'call_budget_exceeded', 'partial', 'request_timeout', 'invalid_contract']) {
  test(`${status} response is retained without substitution or history admission`, () => {
    const r = response(); r.generation.status = status;
    const c = withTurn(turn('t', r)), before = JSON.stringify(c);
    assert.deepEqual(conversationHistory(c), []);
    assert.equal(JSON.stringify(c), before);
  });
}
for (const mutation of [r => { r.answer = ''; }, r => { r.generation.answer_call_id = 'wrong'; },
  r => { r.generation.nodes[0].status = 'invalid_contract'; }, r => { r.generation.nodes[0].span_ids = []; }]) {
  test('empty/invalid/mismatched roots cannot populate assistant history', () => {
    const r = response(); mutation(r);
    assert.equal(carriedAnswer(turn('t', r)), undefined);
  });
}
test('legacy single pipeline carries exact projected body, not hidden thinking', () => {
  const r = response(); r.answer = 'thinking\n🙂 answer';
  r.generation = { pipeline: 'rwkv', status: 'completed', writer_status: 'completed', answer_span: [9, 17],
    model_calls: [{ stage: 'writer', completion_attempted: true, status: 'completed' }] };
  assert.equal(carriedAnswer(turn('t', r)), '🙂 answer');
});
test('interleaved failed and partial turns do not reorder eligible pairs or introduce fake assistants', () => {
  const first = turn('one'), last = turn('three', response('精确新答案'));
  const c = reviseConversation(newConversation('a'), { turns: [first, { ...turn('two'), status: 'failed' }, last], draft: 'next' });
  assert.deepEqual(chatRequest(c).history, [
    { role: 'user', content: first.question }, { role: 'assistant', content: first.response.answer },
    { role: 'user', content: last.question }, { role: 'assistant', content: last.response.answer },
  ]);
});
test('source permissions are fixed after the first attempted turn', () => {
  assert.throws(() => reviseConversation(withTurn({ ...turn(), status: 'failed' }), { scope: { retrieval_mode: 'web', top_k: 1 } }), /scope_locked/);
  const empty = newConversation('x');
  assert.equal(reviseConversation(empty, { scope: { retrieval_mode: 'web', top_k: 1 } }).scope.retrieval_mode, 'web');
});
test('history cap blocks rather than silently slicing old turns', () => {
  const c = reviseConversation(newConversation('a'), { turns: Array.from({ length: 32 }, (_, i) => turn(String(i))), draft: 'next' });
  assert.equal(conversationHistory(c).length, 64);
  assert.equal(requestIssue(c), 'history_limit');
  assert.throws(() => chatRequest(c), /history_limit/);
  c.turns.pop(); assert.equal(chatRequest(c).history.length, 62);
});
test('input character limit counts Unicode code points and never rewrites the draft', () => {
  const c = newConversation('a'); c.draft = '🙂'.repeat(10000);
  assert.equal(chatRequest(c).question, c.draft);
  c.draft += 'x'; assert.equal(requestIssue(c), 'question_limit');
  c.draft = ' \t\n'; assert.equal(requestIssue(c), 'empty');
});
test('oversized history message is explicitly blocked, not trimmed', () => {
  const c = withTurn({ ...turn(), question: 'q'.repeat(100001) });
  assert.equal(requestIssue(c), 'message_limit');
  assert.equal(c.turns[0].question.length, 100001);
});
test('settlement owns a turn, does not mutate earlier snapshots or affect another conversation', () => {
  const a = withTurn({ ...turn(), status: 'pending', response: undefined });
  const b = newConversation('b');
  const r = response(), finished = settleTurn(a, 't1', { status: 'received', response: r });
  assert.equal(finished.turns[0].response, r);
  assert.equal(a.turns[0].status, 'pending');
  assert.equal(settleTurn(b, 't1', { status: 'received', response: r }), b);
  assert.equal(settleTurn(finished, 't1', { status: 'received', response: response('late') }), finished);
});
test('cancel and interrupted states are terminal for a late response', () => {
  const c = withTurn({ ...turn(), status: 'pending', response: undefined });
  const stopped = settleTurn(c, 't1', { status: 'cancelled' });
  assert.equal(settleTurn(stopped, 't1', { status: 'received', response: response() }), stopped);
  const restored = interruptPending(c);
  assert.equal(restored.turns[0].status, 'interrupted');
  assert.equal(interruptPending(restored), restored);
  assert.equal(settleTurn(restored, 't1', { status: 'received', response: response() }), restored);
});
test('complete response/raw/provenance survives a structured-clone round trip unchanged', () => {
  const c = withTurn(turn()); c.turns[0].response.generation.span_bindings = { 'E1.S1': { selected_original_text: ' \r\nsource🙂 ' } };
  const restored = structuredClone(c);
  assert.equal(isConversation(restored), true);
  assert.deepEqual(restored, c);
  assert.deepEqual(chatRequest(restored), chatRequest(c));
});
for (const mutation of [c => { c.version = 99; }, c => { c.turns.push(c.turns[0]); },
  c => { c.turns[0].response = {}; }, c => { c.scope.top_k = 0; }, c => { c.turns[0].status = 'success'; },
  c => { c.revision = -1; }, c => { c.scope.retrieval_mode = 'private_magic'; }, c => { c.turns = null; }]) {
  test('malformed stored conversation is rejected without modifying the original', () => {
    const c = withTurn(turn()); mutation(c); const raw = JSON.stringify(c);
    assert.equal(isConversation(c), false); assert.equal(JSON.stringify(c), raw);
  });
}
test('invalid API envelopes are rejected, but invalid model raw is still valid audit data', () => {
  const r = response(); r.answer = 'not JSON at all';
  assert.equal(isAskResponse(r), true);
  assert.equal(isAskResponse({ ...r, sources: null }), false);
  assert.equal(isAskResponse({ ...r, generation: [] }), false);
});
test('storage revisions prevent stale overwrite, stale deletion and resurrection after deletion', () => {
  const c = newConversation('a'), next = reviseConversation(c, { draft: 'first' });
  assert.equal(storageRevisionMatches(undefined, c), true);
  assert.equal(storageRevisionMatches(c, next), true);
  assert.equal(storageRevisionMatches(next, next), false);
  assert.equal(storageRevisionMatches(undefined, next), false);
  assert.equal(storageRevisionMatches(next, c, true), false);
  assert.equal(storageRevisionMatches(next, next, true), true);
  assert.equal(storageRevisionMatches({ bad: 'data' }, c), false);
  assert.equal(storageRevisionMatches(newConversation('other'), next), false);
});
test('ask sends exact payload and optional AbortSignal; existing callers still work', async () => {
  const original = globalThis.fetch, calls = [];
  globalThis.fetch = async (url, options) => { calls.push({ url, options }); return new Response(JSON.stringify(response()), { status: 200 }); };
  try {
    const controller = new AbortController(), payload = chatRequest(withTurn(turn()));
    await api.ask(payload, controller.signal); await api.ask(payload);
    assert.equal(calls[0].url, '/v1/ask'); assert.equal(calls[0].options.method, 'POST');
    assert.equal(calls[0].options.signal, controller.signal);
    assert.deepEqual(JSON.parse(calls[0].options.body), payload);
    assert.equal(calls[1].options.signal, undefined);
  } finally { globalThis.fetch = original; }
});
