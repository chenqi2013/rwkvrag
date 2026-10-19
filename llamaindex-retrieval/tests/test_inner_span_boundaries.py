"""Selectable boundaries, not semantic fixes or model-output repair."""
import json

from llama_index.core import Document
import pytest

from llamaindex_retrieval.unified_contracts import digest
from llamaindex_retrieval.unified_span_reading import SpanCatalog, SpanReading, span_prompt
from llamaindex_retrieval.verbatim_chunking import packed_verbatim_nodes, verbatim_nodes
from test_unified_span_reading import material


def check(text, window=64, overlap=8, **extra):
    original = material(text, **extra)
    before = json.dumps(original)
    catalog = SpanCatalog.build(original, window=window, overlap=overlap)
    projected = catalog.model_material()
    spans = projected['body_spans']
    covered = set()
    assert json.dumps(original) == before
    for i, s in enumerate(spans):
        a, b = s['span']['start'] - 17, s['span']['end'] - 17
        assert s['id'] == f'E9.S{i+1}' and s['text'].strip()
        assert s['text'] == text[a:b]
        assert s['span']['sha256'] == digest(s['text'])
        covered.update(range(a, b))
        for c in s['context_spans']:
            assert c['text'] == text[c['start'] - 17:c['end'] - 17]
            assert c['sha256'] == digest(c['text'])
    if text.strip():
        assert covered == set(range(len(text)))
        assert 'text' not in projected
        for n in verbatim_nodes(Document(text=text, metadata={'kind': extra.get('kind')}), window, overlap):
            assert any(s['span']['start'] <= 17 + n.start_char_idx and s['span']['end'] >= 17 + n.end_char_idx for s in spans)
    else:
        assert spans == [] and projected['text'] == text
    return catalog, spans


@pytest.mark.parametrize('text', [
    '', ' \t\r\n', '\n' * 500,
    '# H\r\n\r\nUse:\r\n```py\r\nopen_it()\r\n```\r\n\r\n',
    '\n' * 90 + '```py\na()\n```\n' + '\n' * 90,
    '# H\n```py\n' + 'a()\n' * 100 + '```\n\n',
    '# H\n| a | b |\n| --- | --- |\n' + '| x | 0 |\n' * 40 + '\n',
    '# H\n' + '- item\n  text\n' * 40 + '\n# Tail\nend\n',
    '# H\n' + '中文🙂\r\n text ' * 100 + '\n# End\nend\n',
    '# A\n\nfirst\n\n# B\n\nsecond\n',
])
@pytest.mark.parametrize('window,overlap', [(32, 4), (640, 80)])
def test_nonempty_selectors_and_all_original_characters(text, window, overlap):
    check(text, window, overlap)


def test_short_lead_and_code_share_a_selector_without_any_fabrication():
    text = '# Note\n\nEnable:\n```py\nfirst_api()\n```\nRead:\n```py\nsecond_api()\n```\n\n'
    catalog, spans = check(text, 640, 80)
    assert len(spans) == 1 and spans[0]['text'] == text
    value = SpanReading.model_validate({'findings': [{'span_id': 'E9.S1', 'statement': 'Unsupported model claim'}], 'missing': []})
    before = value.model_dump_json()
    binding = value.bind(catalog)[0]
    assert binding['selected_original_text'] == text
    assert binding['semantic_support_verified'] is False
    assert before == value.model_dump_json()


def test_large_code_and_surrounding_whitespace_not_cut_to_soft_budget():
    text = '\r\n```py\n' + 'x()\n' * 100 + '```\n \r\n'
    _, spans = check(text, 32, 4)
    assert len(spans) == 1 and spans[0]['text'] == text
    assert len(spans[0]['text']) > 32  # Hard model budget remains the caller's job.


def test_qa_whole_with_whitespace():
    text = '\n# Q\n' + 'question\n' * 50 + '# A\nanswer\n\n'
    _, spans = check(text, 32, 4, kind='qa-markdown')
    assert len(spans) == 1 and spans[0]['atomic']


def test_whitespace_material_is_not_an_addressable_source():
    catalog, _ = check(' \n\t')
    with pytest.raises(ValueError, match='unknown_span_id'):
        SpanReading.model_validate({'findings': [{'span_id': 'E9.S1', 'statement': 'a'}], 'missing': []}).bind(catalog)
    payload = {'history': [], 'latest_user_message': {'role': 'user', 'content': 'Exact question'}, 'original_material': material(' \n\t')}
    original = json.loads(span_prompt(payload, catalog).split('\n输入数据：\n')[1])['original_material']
    assert original['text'] == ' \n\t' and original['body_spans'] == []


def test_outer_packing_default_and_original_chunker_unchanged():
    d = Document(text='```py\n' + 'x()\n' * 100 + '```\n\n', id_='a')
    old = packed_verbatim_nodes(d, 32, 4)
    attached = packed_verbatim_nodes(d, 32, 4, absorb_whitespace=True)
    assert len(old) == 2 and len(attached) == 1
    assert old[-1].text == '\n' and attached[0].text == d.text
    assert all(n.metadata['packing_policy'] == 'adjacent_verbatim_v1' for n in old)
    assert len(attached[0].metadata['packed_components']) == 2
    assert verbatim_nodes(d, 32, 4)[0].text == old[0].text
