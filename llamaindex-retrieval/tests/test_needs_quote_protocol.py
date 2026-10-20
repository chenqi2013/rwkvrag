import json

import pytest

from llamaindex_retrieval import interactive_needs as flow
from llamaindex_retrieval.native_rwkv import NativeRWKVResult
from test_interactive_needs import Model, reply, run, source


QUOTE = '答案1明确写在这一句原文之中。'


@pytest.mark.parametrize('quote,ids,status,valid', [
    (QUOTE, ['S1'], 'found', True),
    ('字' * 10, ['S3'], 'found', True),
    ('字' * 300, ['S3'], 'found', True),
    ('字' * 9, ['S3'], 'found', False),
    ('字' * 301, ['S3'], 'found', False),
    ('另一句完全不存在的原文句子。', ['S1'], 'found', False),
    (QUOTE, ['S1', 'S2'], 'found', False),
    (QUOTE, [], 'found', False),
    (QUOTE, ['unknown'], 'found', False),
    ('', [], 'not_found', True),
    (QUOTE, [], 'not_found', False),
    ('', ['S1'], 'not_found', False),
])
def test_quote_protocol_grounding_rules(quote, ids, status, valid):
    spans = [{'id': 'S1', 'text': QUOTE}, {'id': 'S2', 'text': QUOTE}, {'id': 'S3', 'text': '字' * 301}]
    value = {'status': status, 'quote': quote, 'answer': '原样答案', 'span_ids': ids}
    assert (flow.validate_answer(value, spans, 'quote_v1') is None) is valid


class QuoteModel(Model):
    def __init__(self, value=None, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    async def complete(self, messages, **kwargs):
        instruction, raw = messages[0]['content'].split('\n', 1)
        if instruction == flow.QUOTE_ANSWER:
            data = json.loads(raw)
            self.calls.append((instruction, data, kwargs))
            return reply(self.value or {'status': 'found', 'quote': QUOTE,
                                       'answer': '  模型的原样答案\n保留条件  ',
                                       'span_ids': [data['spans'][0]['id']]})
        return await super().complete(messages, **kwargs)


async def test_valid_quote_uses_runtime_factory_and_keeps_need_raw():
    model = QuoteModel()
    result, _ = await run(model, {'q': [source(0, QUOTE)]}, answer_protocol='quote_v1', final_mode='assemble',
                          supplementary=False)
    node = next(n for n in result['graph']['nodes'] if n['kind'] == 'answer')
    assert node['answer_status'] == 'found' and json.loads(node['raw_text'])['quote'] == QUOTE
    assert result['answer']['answer'] == '  模型的原样答案\n保留条件   [S1]'
    gen = result['answer']['generation']
    assert gen['final_mode'] == 'assemble' and gen['answer_protocol'] == 'quote_v1'
    assert gen['raw_model_answer'] is None and gen['model_final_call'] is False
    assert not any(i in {flow.COMPOSE, flow.CHECKED_COMPOSE} for i, _, _ in model.calls)
    call = next(k for i, _, k in model.calls if i == flow.QUOTE_ANSWER)
    assert call['structured_schema']['properties'] == flow.answer_schema(['S1'], 'quote_v1')


async def test_invalid_quote_counts_as_unresolved_and_retains_raw():
    value = {'status': 'found', 'quote': '伪造出来的句子并没有在原文出现。', 'answer': '无据事实', 'span_ids': ['S1']}
    model = QuoteModel(value)
    result, _ = await run(model, {'q': [source(0, QUOTE)]}, answer_protocol='quote_v1', supplementary=False)
    node = next(n for n in result['graph']['nodes'] if n['kind'] == 'answer')
    assert node['status'] == 'invalid_output' and node['answer_status'] == 'failed'
    assert json.loads(node['raw_text']) == value and node['parsed'] is None
    gen = result['answer']['generation']
    assert gen['missing_need_ids'] == ['I1'] and gen['status'] == 'no_evidence'
    assert not any(i == flow.COMPOSE for i, _, _ in model.calls)


async def test_assemble_keeps_found_text_and_renders_each_missing_need():
    model = Model(needs=['第一项', '第二项'], answer_status={'I2': 'not_found'})
    result, _ = await run(model, {'q': [source(0, '答案1。'), source(1, '答案2。')]},
                          final_mode='assemble', supplementary=False)
    assert result['answer']['answer'] == '短答案 [S1]\n未找到：第二项'
    assert result['answer']['generation']['need_answers'] == {'I1': 'found', 'I2': 'not_found'}
    assert not any(i == flow.COMPOSE for i, _, _ in model.calls)


async def test_assemble_all_missing_has_no_model_final_or_invented_answer():
    result, _ = await run(Model(needs=['缺失项']), {'q': []}, final_mode='assemble', supplementary=False)
    assert result['answer']['answer'] == '未找到：缺失项'
    gen = result['answer']['generation']
    assert gen['status'] == 'no_evidence' and gen['raw_model_answer'] is None


async def test_empty_failed_composition_still_records_model_call():
    class FailedFinal(Model):
        async def complete(self, messages, **kwargs):
            if messages[0]['content'].startswith(flow.COMPOSE):
                return NativeRWKVResult('length', '', 'length', {})
            return await super().complete(messages, **kwargs)

    result, _ = await run(FailedFinal(), {'q': [source(0, '答案1。')]}, supplementary=False)
    gen = result['answer']['generation']
    assert gen['model_final_call'] is True and gen['status'] == 'length'
    assert gen['raw_model_answer'] == ''


class CheckedModel(Model):
    def __init__(self, sentences):
        super().__init__()
        self.sentences = sentences

    async def complete(self, messages, **kwargs):
        instruction, raw = messages[0]['content'].split('\n', 1)
        if instruction == flow.CHECKED_COMPOSE:
            self.calls.append((instruction, json.loads(raw), kwargs))
            self.raw_final = json.dumps({'sentences': self.sentences}, ensure_ascii=False)
            return reply({'sentences': self.sentences})
        return await super().complete(messages, **kwargs)


async def test_checked_compose_retains_raw_and_exposes_a_verbatim_display_projection():
    rows = [{'text': '原样的一句话。', 'span_ids': ['S1']}]
    model = CheckedModel(rows)
    result, _ = await run(model, {'q': [source(0, '答案1。')]}, final_mode='checked_compose', supplementary=False)
    gen = result['answer']['generation']
    assert gen['status'] == 'completed' and gen['final_mode'] == 'checked_compose'
    assert json.loads(gen['raw_model_answer']) == {'sentences': rows}
    assert gen['model_parsed_output'] == {'sentences': rows}
    assert gen['parsed_answer'] == {'answer': '原样的一句话。 [S1]', 'source_ids': ['S1']}
    assert result['answer']['answer'] == gen['raw_model_answer']


@pytest.mark.parametrize('rows', [
    [{'text': '无引用。', 'span_ids': []}],
    [{'text': '未知引用。', 'span_ids': ['S99']}],
    [{'text': '重复引用。', 'span_ids': ['S1', 'S1']}],
    [{'text': '缺字段。'}],
    [{'text': '多字段。', 'span_ids': ['S1'], 'extra': True}],
    [{'text': 1, 'span_ids': ['S1']}],
    [{'text': ' ', 'span_ids': ['S1']}],
    [],
])
async def test_checked_compose_violations_are_invalid_without_editing_raw(rows):
    model = CheckedModel(rows)
    result, _ = await run(model, {'q': [source(0, '答案1。')]}, final_mode='checked_compose', supplementary=False)
    gen = result['answer']['generation']
    assert gen['status'] == 'invalid_output' and gen['parsed_answer'] is None
    assert json.loads(gen['raw_model_answer']) == {'sentences': rows}
    assert result['answer']['answer'] == gen['raw_model_answer']


@pytest.mark.parametrize('options', [{'answer_protocol': 'unknown'}, {'final_mode': 'unknown'}])
async def test_unknown_protocol_options_fail_before_any_model_call(options):
    model = Model()
    with pytest.raises(ValueError):
        await run(model, {}, **options)
    assert not model.calls
