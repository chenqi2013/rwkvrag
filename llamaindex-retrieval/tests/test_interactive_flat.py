"""Flat quote pipeline: structural binding, grouping, history and review with a fake model."""
import json

from llamaindex_retrieval import interactive_flat as flat
from llamaindex_retrieval.native_rwkv import NativeRWKVResult
from llamaindex_retrieval.schemas import SourceItem

AS_OF = '2026-10-22T09:00:00+08:00'
PAGE = '# 标题\n\n![](a.gif)\n\n**全新3.1版本「归港」现已推出！** 详见[公告](https://x.example/a)。\n\n3.1版本于10月1日上线，新增两个地图。'
QUESTION = '雾岭最新版本主题是什么'


class Model:
    def __init__(self, answer_quotes=None, detail_quotes=None, final='最新版本主题是「归港」。[1]',
                 needs=None, relation='新问题', issues=()):
        self.prompts = []
        self.answer_quotes = answer_quotes if answer_quotes is not None else ['全新3.1版本「归港」现已推出！']
        self.detail_quotes = detail_quotes if detail_quotes is not None else ['3.1版本于10月1日上线']
        self.final, self.relation, self.issues = final, relation, list(issues)
        self.points = []
        self.names = ['雾岭']
        self.metrics = []
        self.assumptions, self.calculation, self.calcs = [], False, []
        self.needs = needs or [{'text': '最新版本主题', 'user_words': '最新版本主题'}]

    async def complete(self, messages, **kwargs):
        prompt = messages[0]['content']
        self.prompts.append((prompt, kwargs))
        if prompt.rstrip().endswith(flat.PLAN):
            value = {'relation': self.relation, 'question': '雾岭最新版本主题', 'needs': self.needs,
                     'aspects': ['上线时间'], 'queries': ['雾岭 最新版本'], 'entities': ['雾岭'], 'metrics': self.metrics, 'names': self.names,
                     'assumptions': self.assumptions, 'calculation': self.calculation}
        elif prompt.rstrip().endswith(flat.READ):
            value = {'items': [{'source': 1, 'need': 1, 'answer_quotes': self.answer_quotes,
                                'detail_quotes': self.detail_quotes}]}
        elif prompt.rstrip().endswith(flat.FINAL):
            value = {'answer': self.final, 'points': self.points, 'unknown': ''}
            self.final_prompt = prompt
        elif prompt.rstrip().endswith(flat.CALC):
            value = {'calculations': self.calcs}
        else:
            assert prompt.rstrip().endswith(flat.REVIEW)
            value = {'issues': self.issues}
        return NativeRWKVResult('completed', json.dumps(value, ensure_ascii=False), 'stop', {})


def web(identity='W1', text=PAGE, **metadata):
    return SourceItem(id=identity, document_id=identity, title='雾岭官网', source='web', uri='https://x.example/',
                      score=0, snippet=text, metadata={'content_status': 'fetched',
                      'retrieved_at': '2026-10-21T23:30:00Z', **metadata})


async def run(model, sources, question=QUESTION, **options):
    async def search(query):
        return sources

    async def select_files(entities, queries, publish):
        return {'mode': 'none', 'selected_file_ids': []}
    return await flat.answer(model, question, channels=['web'], search=search, select_files=select_files,
                             progress=lambda text: None, as_of=AS_OF, snippet_batch_wait=0.01, **options)


def test_clean_markup_drops_images_urls_and_emphasis():
    assert flat.clean_markup(PAGE).startswith('# 标题\n\n全新3.1版本「归港」现已推出！ 详见公告。')


def test_bind_quote_exact_whitespace_and_punctuation_insensitive():
    text = '甲乙\n丙丁 戊己，庚辛壬癸。'
    assert flat.bind_quote(text, '「甲乙」') == (0, 2, 'exact')
    assert flat.bind_quote(text, '乙丙丁戊') == (1, 7, 'whitespace_insensitive')
    assert flat.bind_quote(text, '丙丁戊己庚辛') == (3, 11, 'punctuation_insensitive')
    assert flat.bind_quote(text, '不存在的句子') is None


def test_stitched_quote_binds_only_its_verbatim_clauses():
    text = '第一句话很长很长很长。中间无关。第二句话也很长很长很长。'
    assert flat.bind_quote(text, '第一句话很长很长很长，第二句话也很长很长很长') is None
    hits = flat.bind_fragments(text, '第一句话很长很长很长，第二句话也很长很长很长')
    assert [text[a:b] for a, b, _ in hits] == ['第一句话很长很长很长', '第二句话也很长很长很长']


async def test_answer_and_detail_quotes_bind_and_review_runs_without_changing_answer():
    model = Model()
    result = await run(model, [web()])
    spans = {s['text']: s for s in result['graph']['spans']}
    view = result['graph']['reading_views']['W1']['body']
    for text, span in spans.items():
        assert view[span['span']['start']:span['span']['end']] == text
    assert spans['全新3.1版本「归港」现已推出！']['kinds'] == ['answer']
    assert spans['3.1版本于10月1日上线']['kinds'] == ['detail']
    assert result['answer']['generation']['parsed_answer'] == {'answer': '最新版本主题是「归港」。[1]', 'source_ids': ['1']}
    assert result['response_type'] == 'answered' and result['review']['issues'] == []
    assert result['telemetry']['total_calls'] == 4
    read_prompt, final_prompt = model.prompts[1][0], model.prompts[2][0]
    assert read_prompt.index('正文开始') < read_prompt.index('问题：') < read_prompt.index(flat.READ)
    assert '值得补充的方面：上线时间' in read_prompt and 'sha256' not in read_prompt
    assert '今天是2026年10月22日（星期四，北京时间09:00），昨天是10月21日（星期三），明天是10月23日（星期五），后天是10月24日（星期六）。' in final_prompt and '抓取于提问当天' in final_prompt


async def test_numeric_lookup_does_not_add_arithmetic_against_the_model_plan():
    model = Model()
    model.metrics = ['版本号']
    result = await run(model, [web()], review=False)
    assert result['metrics'] == ['版本号']
    assert result['display_text'] == model.final
    assert result['calculations'] == []
    assert not any(prompt.rstrip().endswith(flat.CALC) for prompt, _ in model.prompts)
    assert result['telemetry']['total_calls'] == 3


async def test_model_requested_arithmetic_still_runs_with_numeric_metrics():
    model = Model()
    model.metrics = ['版本号']
    model.calculation = True
    model.calcs = [{'label': '相加', 'expression': '1+2'}]
    result = await run(model, [web()], review=False)
    assert sum(prompt.rstrip().endswith(flat.CALC) for prompt, _ in model.prompts) == 1
    assert result['calculations'][0]['result'] == 3
    assert result['display_text'] == model.final


async def test_empty_read_items_are_a_completed_read_without_selected_evidence():
    class EmptyReader(Model):
        async def complete(self, messages, **kwargs):
            if messages[0]['content'].rstrip().endswith(flat.READ):
                self.prompts.append((messages[0]['content'], kwargs))
                return NativeRWKVResult('completed', '{"items":[]}', 'stop', {})
            return await super().complete(messages, **kwargs)

    result = await run(EmptyReader(), [web()], review=False, supplement=False)
    reads = [node for node in result['graph']['nodes'] if node['stage'] == 'read']
    assert len(reads) == 1 and reads[0]['status'] == 'completed'
    assert reads[0]['raw_text'] == '{"items":[]}' and reads[0]['parsed'] == {'items': []}
    assert result['graph']['spans'] == []
    assert result['not_found_needs'] == ['最新版本主题']
    assert result['telemetry']['total_calls'] == 2


async def test_whole_windows_share_reads_only_after_a_generation_free_budget_check():
    class BudgetedModel(Model):
        async def check_budget(self, messages, **kwargs):
            return NativeRWKVResult('completed', None, None,
                                    {'budget_only': True, 'completion_attempted': False})

    text = '\n'.join(f'雾岭的第{i}段资料保留完整句子与限制条件。' for i in range(50))
    result = await run(BudgetedModel(), [web(text=text)], window_chars=400,
                       window_overlap=50, max_windows_per_source=8, review=False, supplement=False)
    reads = [n for n in result['graph']['nodes'] if n['stage'] == 'read']
    windows = sorted(w for n in reads for w in n['windows'])
    assert len(windows) > len(reads) and max(len(n['windows']) for n in reads) == 2
    assert windows[0][0] == 0 and windows[-1][1] == len(text)
    assert all(a[1] >= b[0] for a, b in zip(windows, windows[1:]))
    assert all(check['trace']['completion_attempted'] is False
               for check in result['graph']['read_batch_checks'])


async def test_oversized_read_batch_splits_without_losing_any_window():
    class TooSmallModel(Model):
        async def check_budget(self, messages, **kwargs):
            return NativeRWKVResult('budget_exceeded', None, None, {'completion_attempted': False})

    text = '\n'.join(f'雾岭的第{i}段资料保留完整句子与限制条件。' for i in range(50))
    result = await run(TooSmallModel(), [web(text=text)], window_chars=400,
                       window_overlap=50, max_windows_per_source=8, review=False, supplement=False)
    reads = [n for n in result['graph']['nodes'] if n['stage'] == 'read']
    windows = sorted(w for n in reads for w in n['windows'])
    assert len(reads) == len(windows) and all(len(n['windows']) == 1 for n in reads)
    assert windows[0][0] == 0 and windows[-1][1] == len(text)
    assert all(a[1] >= b[0] for a, b in zip(windows, windows[1:]))
    assert any(check['status'] == 'budget_exceeded' for check in result['graph']['read_batch_checks'])


async def read_overlapping_windows(text, quote):
    class BudgetedModel(Model):
        async def check_budget(self, messages, **kwargs):
            return NativeRWKVResult('completed', None, None,
                                    {'budget_only': True, 'completion_attempted': False})

    model = BudgetedModel(answer_quotes=[quote], detail_quotes=[])
    pipeline = flat.Flat(model, '设备使用限制是什么', channels=['web'], search=None,
                         select_files=None, progress=lambda text: None, as_of=AS_OF)
    pipeline.needs = ['设备使用限制']
    source = web(text=text)
    pipeline.views[source.id] = {'body': text, 'snippet': '',
                                 'body_sha256': flat.digest(text), 'snippet_sha256': flat.digest('')}
    await pipeline.read([(source, (0, 400), False), (source, (350, 750), False)])
    return pipeline


async def test_whole_quote_in_second_window_precedes_partial_match_in_hinted_window():
    quote = '本设备可以连续运行很长时间，但必须始终连接原厂适配器且不得在密闭空间中使用。'
    pipeline = await read_overlapping_windows('甲' * 370 + quote + '乙' * 400, quote)
    assert [row['text'] for row in pipeline.quotes.values()] == [quote]
    assert all(row['match'] == 'exact' for row in pipeline.quotes.values())
    assert pipeline.unbound == []


async def test_stitched_quote_keeps_conditions_from_every_offered_window_once():
    first, second = '设备持续运行时必须使用原厂适配器', '不得放在密闭空间中使用'
    text = '甲' * 360 + first + '。' + '乙' * 150 + second + '。' + '丙' * 250
    pipeline = await read_overlapping_windows(text, first + '。' + second + '。')
    assert {row['text'] for row in pipeline.quotes.values()} == {first, second}
    assert all(row['match'] == 'fragment' for row in pipeline.quotes.values())
    assert len(pipeline.reads[0]['quote_ids']) == 2
    assert pipeline.unbound == []


async def test_review_findings_are_recorded_and_mark_answer_unverified():
    issue = {'kind': flat.ISSUES[0], 'detail': '另一来源说法不同', 'sources': [1], 'evidence': '3.1版本于10月1日上线',
             'answer_quote': '最新版本主题是「归港」'}
    loose = {'kind': flat.ISSUES[3], 'detail': '无依据的指责', 'sources': [1], 'evidence': '原文没有这句话',
             'answer_quote': '归港'}
    contradicted = {'kind': flat.ISSUES[3], 'detail': '3.1不在原文', 'sources': [1], 'evidence': '3.1版本于10月1日上线',
                    'answer_quote': '3.1'}
    model = Model(issues=[issue, loose, contradicted])
    model.points = ['3.1版本于10月1日上线。[1]']
    result = await run(model, [web()])
    assert [i['detail'] for i in result['review']['unbound_issues']] == ['无依据的指责', '3.1不在原文']
    assert result['display_text'].startswith('最新版本主题是「归港」。[1]')
    assert result['review']['issues'][0]['source_ids'] == ['W1']
    assert result['response_type'] == 'unverified' and result['evidence_verification'] == 'review_flagged'


async def test_needs_not_in_user_words_are_dropped_and_detail_only_need_is_missing():
    needs = [{'text': '最新版本主题', 'user_words': '最新版本主题'}, {'text': '历史版本', 'user_words': '历史版本列表'}]
    result = await run(Model(needs=needs), [web()])
    assert result['needs'] == ['最新版本主题'] and result['dropped_needs'][0]['text'] == '历史版本'
    result = await run(Model(answer_quotes=[]), [web()])
    assert result['not_found_needs'] == ['最新版本主题']


async def test_unbound_quotes_never_reach_final_and_no_answer_without_quotes():
    model = Model(answer_quotes=['原文里没有这句话'], detail_quotes=[])
    result = await run(model, [web()])
    assert result['graph']['spans'] == [] and len(result['graph']['unbound_quotes']) == 1
    assert len(model.prompts) == 2 and result['response_type'] == 'not_found'


async def test_snippet_only_sources_share_one_read_and_are_unverified():
    sources = [web(f'W{i}', text=f'摘要{i}', content_status='snippet_only',
                   search_snippet=f'全新3.1版本「归港」现已推出！来源{i}') for i in range(3)]
    model = Model(detail_quotes=[])
    result = await run(model, sources)
    reads = [p for p, _ in model.prompts if p.rstrip().endswith(flat.READ)]
    assert len(reads) == 1 and reads[0].count('正文开始（搜索摘要') == 3
    assert result['graph']['spans'][0]['section'] == 'snippet'
    assert result['response_type'] == 'unverified'


async def test_follow_up_uses_code_kept_history_without_previous_answers():
    history = [{'question': '雾岭是什么游戏', 'resolved_question': '雾岭是什么游戏', 'entities': ['雾岭'],
                'answer_opening': '雾岭是一款开放世界游戏。'}]
    model = Model(relation='追问', needs=[{'text': '最新版本主题', 'user_words': '最新版本主题'}])
    result = await run(model, [], question='它最新版本主题是什么', history=history, previous_sources=[web()])
    plan = model.prompts[0][0]
    assert '第1轮用户原话：雾岭是什么游戏' in plan and '第1轮回答开头（只用于判断指代的对象）：雾岭是一款开放世界游戏。' in plan
    assert result['reused_source_ids'] == ['W1'] and result['relation'] == '追问'
    assert result['answer']['generation']['citation_bindings'] == {'1': ['W1']}


async def test_new_topic_does_not_reuse_previous_sources():
    result = await run(Model(), [web('W2')], previous_sources=[web()],
                       history=[{'question': '别的问题', 'entities': []}])
    assert result['reused_source_ids'] == []


async def test_invalid_citations_are_recorded_not_removed():
    result = await run(Model(final='主题是「归港」。[1][7]'), [web()])
    assert result['answer']['generation']['parsed_answer']['answer'] == '主题是「归港」。[1][7]'
    assert result['answer']['generation']['invalid_citations'] == ['7']


async def test_changed_title_for_same_source_is_recorded_not_fatal():
    result = await run(Model(), [web(), web().model_copy(update={'title': '另一个标题'})])
    assert result['response_type'] == 'answered'
    assert any(e['status'] == 'duplicate_source_title_changed' for e in result['errors'])


def test_title_candidates_read_small_sets_and_match_names_in_large_ones():
    small = [{'id': f'f{i}', 'title': f'doc{i}.md'} for i in range(3)]
    assert flat.title_candidates(small, [], ['x']) == (['f0', 'f1', 'f2'], 'all_files')
    large = [{'id': f'f{i}', 'title': f'无关{i}.md'} for i in range(40)] + [
        {'id': 'hit', 'title': '雾岭.md'}, {'id': 'num', 'title': '28.md'}, {'id': 'hd', 'title': 'HD 42824.md'}]
    assert flat.title_candidates(large, ['雾岭'], ['雾岭 最新版本']) == (['hit'], 'title_match')
    assert flat.title_candidates(large, ['HD 42824'], ['HD 42824 是什么']) == (['hd'], 'title_match')


def test_answer_opening_keeps_first_sentence_without_citations():
    from llamaindex_retrieval.interactive_app import answer_opening
    assert answer_opening('Go语言最新版本是1.27.0[2]。它发布于8月。') == 'Go语言最新版本是1.27.0。'
    assert answer_opening(None) is None


async def test_structured_points_are_joined_in_order_without_rewording():
    model = Model()
    model.points = ['3.1版本于10月1日上线。[1]']
    result = await run(model, [web()])
    assert result['display_text'] == '最新版本主题是「归港」。[1]\n· 3.1版本于10月1日上线。[1]'
    assert json.loads(result['answer']['generation']['raw_model_answer'])['points'] == model.points


def test_repeated_points_detection():
    assert flat.repeated_points({'answer': '西安今天小雨。[1]', 'points': ['西安今天小雨。[2]']})
    assert flat.repeated_points({'answer': 'A', 'points': ['10月23日（周五）天气预报为晴，最高气温19℃，最低气温11℃。',
                                                         '10月23日天气预报为晴，最高气温19℃，最低气温11℃。[2]']})
    assert not flat.repeated_points({'answer': '主题是归港。', 'points': ['3.1于10月1日上线。', '新增两个地图。']})


async def test_repeated_answer_gets_one_new_call_and_original_is_kept():
    model = Model()
    model.points = ['最新版本主题是「归港」。[1]']
    result = await run(model, [web()])
    finals = [n for n in result['graph']['nodes'] if n['stage'] == 'final']
    assert len(finals) == 2 and finals[1]['retry_of'] == finals[0]['raw_sha256']
    penalties = [k for p, k in model.prompts if p.rstrip().endswith(flat.FINAL)]
    assert penalties[1]['frequency_penalty'] == 1.0 and penalties[1]['presence_penalty'] == 0.6


async def test_optional_repetition_diagnostic_keeps_first_answer_without_regeneration():
    model = Model()
    model.points = ['最新版本主题是「归港」。[1]']
    result = await run(model, [web()], retry_repeated_points=False)
    finals = [n for n in result['graph']['nodes'] if n['stage'] == 'final']
    assert len(finals) == 1
    assert result['display_text'] == '最新版本主题是「归港」。[1]\n· 最新版本主题是「归港」。[1]'
    assert result['answer']['generation']['raw_model_answer'] == finals[0]['raw_text']
    assert result['repetition_checks'] == [{'raw_sha256': finals[0]['raw_sha256'],
        'repeated_points': True, 'regeneration_requested': False, 'raw_text_modified': False}]
    assert result['answer']['generation']['answer_modified'] is False


async def test_no_direct_quote_for_any_need_skips_answer():
    model = Model(answer_quotes=[], final='未知。')
    result = await run(model, [web()], supplement=False)
    assert not any(p.rstrip().endswith(flat.FINAL) for p, _ in model.prompts)
    assert result['response_type'] == 'not_found' and '其他对象' in result['display_text']


def test_name_present_whole_or_long_part_only():
    assert flat.name_present('雾岭', '雾岭官网')
    assert not flat.name_present('星河电视S9', '三星 S9 85英寸 4K电视')
    assert flat.name_present('荣耀MagicBook Pro 16 2026', 'MagicBook Pro 16 搭载酷睿Ultra')
    assert not flat.name_present('日元', '港元兑人民币')


async def test_object_named_nowhere_in_material_gets_no_answer():
    source = web(text='三星S9电视屏幕尺寸为85英寸。').model_copy(update={'title': '三星电视参数'})
    model = Model(answer_quotes=['三星S9电视屏幕尺寸为85英寸'], detail_quotes=[])
    model.names = ['星河电视S9']
    result = await run(model, [source], question='星河电视S9屏幕多大', supplement=False)
    assert result['name_absent_quotes'] == ['星河电视S9'] and result['response_type'] == 'not_found'
    assert not any(p.rstrip().endswith(flat.FINAL) for p, _ in model.prompts)


async def test_one_present_name_or_date_words_do_not_block_answer():
    model = Model()
    model.names = ['雾岭', '最近一周']
    result = await run(model, [web()], question='最近一周雾岭版本主题')
    assert result['names'] == ['雾岭'] and result['response_type'] == 'answered'


async def test_generic_terms_are_not_name_checked():
    source = web(text='新赛季于10月21日开打。').model_copy(update={'title': '赛程'})
    model = Model(answer_quotes=['新赛季于10月21日开打'], detail_quotes=[])
    model.names = []
    result = await run(model, [source], question='新赛季哪天开打', supplement=False)
    assert result['name_absent_quotes'] == [] and result['not_found_needs'] == []
    assert result['response_type'] != 'not_found'


def test_range_notes_use_code_calendar():
    today = flat.local_time(AS_OF, flat.ZoneInfo('Asia/Shanghai'))
    assert flat.range_notes('每年4月1日至10月31日为旺季', today) == ['4月1日至10月31日：今天在此期间']
    assert flat.range_notes('11月1日至次年3月31日为淡季', today) == ['11月1日至次年3月31日：今天不在此期间']


async def test_numbers_absent_from_quotes_are_listed_without_changing_answer():
    model = Model(final='最新版本主题是「归港」。[1]')
    model.points = ['3.1版本于2025年10月1日上线。[1]']
    result = await run(model, [web()])
    assert result['unsupported_numbers'] == [{'text': '3.1版本于2025年10月1日上线。[1]', 'numbers': ['2025']}]
    assert '2025年' in result['display_text'] and result['response_type'] == 'unverified'


async def test_comparison_lists_which_sources_name_each_object():
    model = Model()
    model.names = ['雾岭', '星落']
    await run(model, [web()], question='雾岭和星落哪个版本新')
    assert '所问对象与写明它的来源（代码按名称统计）：\n雾岭：来源[1]\n星落：摘抄中没有写明它的原文' in model.final_prompt


def test_chinese_numerals_and_safe_arithmetic():
    assert flat.cn_to_int('五千四百六十四') == 5464 and flat.cn_to_int('三万') == 30000
    assert '5464' in flat.numeric_view('長五千四百六十四公里')
    assert flat.evaluate('190*2+95') == 475 and flat.evaluate('1318/350') == 1318 / 350
    for bad in ('__import__("os")', '2**10', 'a+1'):
        try:
            flat.evaluate(bad)
            raise AssertionError(bad)
        except (ValueError, SyntaxError):
            pass


async def test_calculation_uses_only_quoted_numbers_and_code_result():
    page = '成人票190元，儿童半价95元。'
    model = Model(answer_quotes=['成人票190元，儿童半价95元'], detail_quotes=[], final='共475元。[1]')
    model.calculation = True
    model.calcs = [{'label': '两大一小', 'expression': '190*2+95'}, {'label': '编造', 'expression': '200*3'},
                   {'label': '原样', 'expression': '190'}]
    result = await run(model, [web(text=page)], question='雾岭两个大人一个小孩门票多少')
    rows = {c['label']: c for c in result['calculations']}
    assert rows['两大一小']['result'] == 475 and 'error' in rows['编造'] and rows['原样']['error'] == 'no operation'
    assert '两大一小：190*2+95 = 475' in model.final_prompt and '200*3' not in model.final_prompt


def test_calendar_line_week_and_last_friday():
    pipeline = flat.Flat(Model(), 'q', channels=['web'], search=None, select_files=None,
                         progress=lambda t: None, as_of='2026-10-23T14:00:00+08:00')
    line = pipeline.calendar_line()
    assert '本周是10月19日至10月25日' in line and '上周五是10月16日' in line and '最近一周是10月17日至10月23日' in line


async def test_follow_up_queries_get_previous_object_when_dropped():
    history = [{'question': '沐桥AI眼镜多少钱', 'entities': ['沐桥AI眼镜']}]
    model = Model(relation='追问')
    result = await run(model, [web()], question='它支持翻译吗', history=history)
    assert result['queries'] == ['沐桥AI眼镜 雾岭 最新版本']


async def test_quote_attributed_to_wrong_batched_source_still_binds():
    sources = [web(f'W{i}', text=f'摘要{i}', content_status='snippet_only', search_snippet=f'雾岭第{i}条消息内容很长') for i in range(3)]
    model = Model(answer_quotes=['雾岭第2条消息内容很长'], detail_quotes=[])
    result = await run(model, sources)
    assert [q['source_id'] for q in result['graph']['spans']] == ['W2'] and result['graph']['unbound_quotes'] == []


def test_partial_name_keeps_head_or_model_code():
    assert not flat.name_present('栖桐冻干猫粮', 'Go Raw冻干猫粮召回')
    assert flat.name_present('荣耀MagicBook Pro 16 2026', 'MagicBook Pro 16 搭载酷睿Ultra')
    assert flat.name_present('西安秦始皇帝陵博物院', '秦始皇帝陵博物院开放时间')


async def test_generic_names_dropped_and_follow_up_inherits_names():
    model = Model()
    model.names = ['去年同期', '最新版本主题']
    result = await run(model, [web()], question='去年同期最新版本主题')
    assert result['names'] == []
    model = Model(relation='追问')
    model.names = []
    history = [{'question': '朴茗扫地机是谁做的', 'entities': ['朴茗扫地机'], 'names': ['朴茗扫地机']}]
    result = await run(model, [web()], question='它的吸力多少', history=history)
    assert result['names'] == ['朴茗扫地机'] and result['response_type'] == 'not_found'


async def test_partly_absent_comparison_lists_absent_object():
    model = Model()
    model.names = ['雾岭', '翊岑E7']
    await run(model, [web()], question='雾岭和翊岑E7哪个新')
    assert '都没有出现，关于它们只能写未知，不能写出任何数值或特点：翊岑E7' in model.final_prompt


def test_final_quotes_rotate_across_sources():
    rows = [{'id': f'{s}{j}', 'source_id': s, 'kinds': ['answer'], 'needs': [1], 'text': f'{s}-内容{j}',
             'section': 'body', 'span': {'start': j}} for s in ('A', 'B') for j in range(3)]
    pipeline = flat.Flat(Model(), 'q', channels=['web'], search=None, select_files=None, progress=lambda t: None,
                         as_of=AS_OF, per_need_limit=2)
    pipeline.needs, pipeline.quotes = ['x'], {r['id']: r for r in rows}
    pipeline.sources = [web('A'), web('B')]
    _, _, ids = pipeline.final_materials()
    assert ids == ['A0', 'B0']


async def test_live_observations_preserve_model_inputs_and_answers_with_mutating_or_broken_observer():
    from copy import deepcopy
    plain_model, observed_model = Model(), Model()
    baseline = await run(plain_model, [web()], review=False, supplement=False)
    events = []

    def observe(event):
        events.append(deepcopy(event))
        if event['kind'] == 'source_ready':
            event['reading_view']['body'] = 'must not enter the model'
            event['source']['snippet'] = 'must not replace the original'
        if event['kind'] == 'binding_finished':
            for span in event['spans']:
                span['text'] = 'must not replace quotes'
        if event['kind'] == 'plan_ready':
            event['needs'].clear()
            raise RuntimeError('observer unavailable')

    actual = await run(observed_model, [web()], review=False, supplement=False, observe=observe)
    assert observed_model.prompts == plain_model.prompts
    assert actual['display_text'] == baseline['display_text']
    assert actual['graph']['spans'] == baseline['graph']['spans']
    assert actual['graph']['reading_views'] == baseline['graph']['reading_views']
    assert len(actual['observation_errors']) == 1
    assert [e['sequence'] for e in events] == list(range(1, len(events) + 1))
    starts = {e['operation']: e for e in events if e['kind'] == 'call_started'}
    ends = {e['operation']: e for e in events if e['kind'] == 'call_finished'}
    assert starts.keys() == ends.keys()
    assert {e['stage'] for e in starts.values()} == {'plan', 'read', 'final'}
    assert all(ends[k]['elapsed_ms'] >= v['elapsed_ms'] for k, v in starts.items())
    source_event = next(e for e in events if e['kind'] == 'source_ready')
    assert source_event['reading_view']['body'] == baseline['graph']['reading_views']['W1']['body']
    assert any(e['kind'] == 'binding_finished' and e['bound_quotes'] == 2 for e in events)


async def test_live_observations_report_cancelled_acquisition_as_cancelled():
    import asyncio
    events = []
    started = asyncio.Event()

    async def search(query):
        started.set()
        await asyncio.Event().wait()

    pipeline = flat.Flat(Model(), QUESTION, channels=['web'], search=search,
                         select_files=None, progress=lambda _: None, observe=events.append)
    task = asyncio.create_task(pipeline.acquire(['query'], 'initial'))
    await started.wait()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    finish = [e for e in events if e['kind'] == 'acquisition_finished']
    assert len(finish) == 1 and finish[0]['status'] == 'cancelled'
