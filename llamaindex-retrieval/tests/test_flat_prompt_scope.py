"""Ordinary evidence answers must not inherit instructions for assumed arithmetic."""
from llamaindex_retrieval import interactive_flat as flat
from test_interactive_flat import Model, run, web
from llamaindex_retrieval.native_rwkv import NativeRWKVResult


async def test_direct_source_answer_does_not_receive_calculation_preface():
    model = Model()
    await run(model, [web()], review=False, supplement=False, final_response_style='concise')
    assert '按题设' not in model.final_prompt
    assert flat.CALC_FINAL not in model.final_prompt


async def test_invalid_calculation_does_not_receive_assumption_preface():
    model = Model()
    model.calculation = True
    model.calcs = [{'label': 'unsupported', 'expression': '999999 * 999998'}]
    result = await run(model, [web()], review=False, supplement=False)
    assert result['calculations'] and 'error' in result['calculations'][0]
    assert flat.CALC_FINAL not in model.final_prompt


async def test_actual_calculation_gets_conditional_source_attribution():
    model = Model()
    model.calculation = True
    model.calcs = [{'label': 'total', 'expression': '3 * 2'}]
    result = await run(model, [web()], question='按每组3人、2组计算总人数。',
                       review=False, supplement=False)
    assert result['calculations'][0]['result'] == 6
    assert flat.CALC_FINAL in model.final_prompt
    assert '从资料直接查得的事实按来源陈述' in model.final_prompt


async def test_explicit_web_page_reads_tail_with_upload_count_budget():
    target = '末页给出完整发布日期为十月二十七日。'
    body = '\n\n'.join(f'第{i}节：' + '这是连续资料中的背景句。' * 30 for i in range(9)) + '\n\n' + target
    source = web(text=body)
    # The provider may report a canonical URL different from the request URL.
    source.uri = 'https://x.example/canonical'
    events = []
    model = Model(answer_quotes=[target], detail_quotes=[])

    async def search(query):
        raise AssertionError('An explicit page must be fetched directly')

    async def fetch(urls):
        assert urls == ['https://x.example/redirect']
        return [source]

    async def files(entities, queries, publish):
        return {}

    result = await flat.answer(model, '末页的发布日期是什么？', channels=['web'],
        search=search, select_files=files, progress=lambda _: None,
        requested_urls=['https://x.example/redirect'], fetch_urls=fetch,
        window_chars=400, window_overlap=40, max_windows_per_source=4, kb_max_windows=40,
        review=False, supplement=False, observe=events.append)
    ready = next(e for e in events if e['kind'] == 'source_ready')
    assert ready['requested'] is True
    assert len(ready['windows']) > 4
    assert ready['windows'][-1][1] == len(body)
    assert target in model.final_prompt
    assert not any(e.get('status') == 'windows_limited' for e in result['errors'])


async def test_search_results_keep_their_separate_window_budget():
    body = '\n\n'.join(f'第{i}节：' + '这是连续资料中的背景句。' * 30 for i in range(9))
    events = []
    result = await run(Model(), [web(text=body)], window_chars=400, window_overlap=40,
                       max_windows_per_source=4, kb_max_windows=40, observe=events.append,
                       review=False, supplement=False)
    ready = next(e for e in events if e['kind'] == 'source_ready')
    assert ready['requested'] is False
    assert len(ready['windows']) == 4
    assert any(e.get('status') == 'windows_limited' for e in result['errors'])


async def test_explicit_follow_up_upgrades_reused_page_without_duplicate_reads():
    target = '末页给出完整发布日期为十月二十七日。'
    body = '\n\n'.join(f'第{i}节：' + '连续资料背景句。' * 30 for i in range(9)) + '\n\n' + target
    source = web(text=body)
    source.uri = 'https://x.example/canonical'
    model = Model(relation='追问', answer_quotes=[target], detail_quotes=[],
                  needs=[{'text': '发布日期', 'user_words': '发布日期'}])
    events = []

    async def fetch(urls):
        assert urls == ['https://x.example/redirect']
        return [source, source]  # Repeated fetch results must not repeat windows.

    async def search(query):
        raise AssertionError('The explicit page is fetched directly')

    async def files(entities, queries, publish):
        return {}

    result = await flat.answer(model, '它的发布日期是什么？', channels=['web'],
        search=search, select_files=files, progress=lambda _: None,
        history=[{'question': '这个页面是什么？', 'entities': ['网页']}], previous_sources=[source],
        requested_urls=['https://x.example/redirect'], fetch_urls=fetch,
        window_chars=400, window_overlap=40, max_windows_per_source=4, kb_max_windows=40,
        review=False, supplement=False, observe=events.append)
    ready = [e for e in events if e['kind'] == 'source_ready']
    assert len(ready) == 2 and ready[0]['requested'] is False and ready[1]['requested'] is True
    assert len(ready[0]['windows']) == 4 and ready[1]['windows'][-1][1] == len(body)
    assert result['telemetry']['read_calls'] == len(ready[1]['windows'])
    assert result['telemetry']['source_count'] == 1 and target in model.final_prompt


async def test_concise_runtime_uses_one_format_contract():
    from llamaindex_retrieval.flat_prompts import final_instruction

    model = Model()
    await run(model, [web()], final_response_style='concise', review=False, supplement=False)
    prompt, options = next((p, k) for p, k in model.prompts if p.endswith(flat.FINAL_CONCISE))
    assert prompt.endswith(final_instruction(concise=True))
    assert 'points补充' not in prompt and '写3到8条' not in prompt
    assert '值得补充的方面' not in prompt
    assert options['structured_schema']['properties']['unknown']['type'] == 'string'
    assert options['structured_schema']['properties']['points']['maxItems'] == 0


async def test_reader_reserves_more_output_for_many_needs():
    model = Model(needs=[{'text': f'要求{i}', 'user_words': f'要求{i}'} for i in range(1, 9)])
    question = '、'.join(f'要求{i}' for i in range(1, 9))
    await run(model, [web()], question=question, review=False, supplement=False)
    _, options = next((p, k) for p, k in model.prompts if p.endswith(flat.READ))
    assert options['max_tokens'] > 1400


async def test_length_retry_increases_budget_and_keeps_failed_raw():
    class LengthOnce(Model):
        async def complete(self, messages, **kwargs):
            if messages[0]['content'].endswith(flat.READ) and not hasattr(self, 'failed_limit'):
                self.failed_limit = kwargs['max_tokens']
                return NativeRWKVResult('length', '{"items":[', 'length', {})
            return await super().complete(messages, **kwargs)

    model = LengthOnce()
    result = await run(model, [web()], review=False, supplement=False)
    _, options = next((p, k) for p, k in model.prompts if p.endswith(flat.READ))
    assert options['max_tokens'] > model.failed_limit
    reads = [node for node in result['graph']['nodes'] if node['stage'] == 'read']
    assert reads[0]['raw_text'] == '{"items":['
    assert reads[1]['retry_of'] == reads[0]['raw_sha256']
