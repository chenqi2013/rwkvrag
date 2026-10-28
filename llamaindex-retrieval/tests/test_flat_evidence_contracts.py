"""Regression cases for complete JSON frames, verbatim evidence and explicit URLs."""
from copy import deepcopy
import re
import pytest

from llamaindex_retrieval import interactive_flat as flat
from llamaindex_retrieval.interactive_app import question_urls
from llamaindex_retrieval.native_rwkv import NativeRWKVResult
from llamaindex_retrieval.structured_native import json_schema
from test_interactive_flat import Model, run, web


@pytest.mark.parametrize('name,text', [('Artemis II', 'Artemis III'),
                                      ('Artemis III', 'Artemis II'),
                                      ('Blue Moon', 'the Moon')])
def test_literal_name_diagnostics_do_not_equate_distinct_names(name, text):
    assert not flat.name_present(name, text)


def test_literal_name_diagnostic_keeps_word_boundaries_in_english_sentences():
    assert flat.name_present('Artemis II', 'The Artemis II mission will fly.')
    assert flat.name_present('Blue Moon', 'The Blue Moon lander is described here.')
    assert flat.name_present('Ubuntu', 'İ Ubuntu')
    assert flat.name_present('R', 'İ R')
    text = 'İ A complete statement.'
    start, end, _ = flat.bind_quote(text, 'a complete statement.')
    assert text[start:end] == 'A complete statement.'


async def test_schema_invalid_read_payload_is_audited_and_retried():
    class WrongItems(Model):
        broken = False

        async def complete(self, messages, **kwargs):
            if messages[0]['content'].endswith(flat.READ) and not self.broken:
                self.broken = True
                return NativeRWKVResult('completed', '{"items":"wrong-type"}', 'stop', {})
            return await super().complete(messages, **kwargs)

    result = await run(WrongItems(), [web()], review=False, supplement=False)
    reads = [n for n in result['graph']['nodes'] if n['stage'] == 'read']
    assert len(reads) == 2
    assert reads[0]['status'] == 'invalid_output' and reads[0]['parsed'] is None
    assert reads[0]['raw_text'] == '{"items":"wrong-type"}'
    assert reads[1]['quote_ids']


@pytest.mark.parametrize('source,metadata', [('web', {'published_date': '2026-09-01'}),
                                            ('upload', {'published_at': '2026-09-01'})])
async def test_date_only_publication_is_kept_for_web_and_upload(source, metadata):
    material = web(**metadata)
    material.source = source
    model = Model()
    await run(model, [material], review=False, supplement=False)
    assert '发布：2026-09-01（原文日期，未提供时刻）' in model.final_prompt


def test_schema_marker_guard_preserves_patterns_literals_and_caller_data():
    literal = {'type': 'string', 'pattern': 'literal data ✿'}
    schema = {'type': 'object', 'properties': {
        'point': {'type': 'string', 'pattern': flat.POINT_PATTERN},
        'optional': {'type': ['string', 'null']},
        'literal': {'const': literal},
        'choice': {'type': 'string', 'enum': ['✿', 'ordinary']}}}
    original = deepcopy(schema)
    guarded = json_schema(schema)
    assert schema == original
    point = guarded['properties']['point']
    assert point['allOf'] == [{'pattern': flat.POINT_PATTERN}]
    assert re.fullmatch(point['pattern'], '有出处的句子。[1]')
    assert not re.fullmatch(point['pattern'], '未完✿')
    assert guarded['properties']['optional']['pattern'] == point['pattern']
    assert guarded['properties']['literal']['const'] == literal
    assert guarded['properties']['choice']['enum'] == ['✿', 'ordinary']
    assert json_schema(schema, allow_delimiter=True) == original


async def test_complete_long_quote_and_distinct_dates_reach_writer():
    text = 'A complete statement ' + 'with a necessary condition ' * 15 + 'and its conclusion.'
    model = Model(answer_quotes=[text], detail_quotes=[])
    await run(model, [web(text=text, published_date='2026-09-01T00:00:00Z')], review=False)
    read = next(kwargs for prompt, kwargs in model.prompts if prompt.endswith(flat.READ))
    item = read['structured_schema']['properties']['items']['items']
    assert item['properties']['answer_quotes']['items']['maxLength'] >= len(text) > 360
    assert text in model.final_prompt
    assert '发布：2026-09-01' in model.final_prompt and '抓取：2026-10-22' in model.final_prompt


async def test_bad_read_frame_is_retained_and_retried_once_without_partial_evidence():
    class BrokenFirst(Model):
        broken = 0

        async def complete(self, messages, **kwargs):
            if messages[0]['content'].endswith(flat.READ) and not self.broken:
                self.broken += 1
                return NativeRWKVResult('completed', '{"items":[', 'stop', {'call_id': 'failed'})
            return await super().complete(messages, **kwargs)

    result = await run(BrokenFirst(), [web()], review=False, supplement=False)
    reads = [n for n in result['graph']['nodes'] if n['stage'] == 'read']
    assert len(reads) == 2
    assert reads[0]['raw_text'] == '{"items":[' and reads[0]['parsed'] is None
    assert reads[0]['status'] == 'invalid_output' and reads[0]['transport_status'] == 'completed'
    assert reads[0]['quote_ids'] == []
    assert reads[1]['retry_of'] == reads[0]['raw_sha256'] and reads[1]['quote_ids']


async def test_english_evidence_is_not_vetoed_by_chinese_object_name():
    model = Model(answer_quotes=['Generative AI requires risk management.'], detail_quotes=[],
                  final='生成式AI需要风险管理。[1]')
    model.names = ['生成式AI']
    result = await run(model, [web(text='Generative AI requires risk management.')],
                       question='生成式AI的要求是什么', review=False, supplement=False)
    assert result['name_absent_quotes'] == ['生成式AI']
    assert result['display_text'] == model.final
    assert result['answer']['generation']['semantic_support_verified'] is False


def test_literal_urls_keep_order_deduplicate_and_strip_question_punctuation():
    assert question_urls('比较 https://github.com/a/one，和 https://github.com/b/two。'
                         '另见(https://github.com/a/one)。') == [
        'https://github.com/a/one', 'https://github.com/b/two']
    url = 'https://example.org/wiki/Example_(category)'
    assert question_urls(f'比较({url})。') == [url]
    assert question_urls(url) == [url]
    assert question_urls('HTTPS://example.org/Doc') == ['HTTPS://example.org/Doc']


async def test_explicit_urls_are_acquired_instead_of_searching_an_altered_query():
    requested, seen = ['https://github.com/a/one', 'https://github.com/b/two'], []

    async def fetch(urls):
        seen.append(urls)
        return [web()]

    result = await run(Model(), [], requested_urls=requested, fetch_urls=fetch,
                       supplement=False, review=False)
    assert seen == [requested]
    assert len(result['acquisition']) == 1
    assert result['acquisition'][0]['action'] == 'extract'
    assert result['acquisition'][0]['requested_urls'] == requested
    assert result['graph']['spans']
