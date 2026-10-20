import json

import pytest

from llamaindex_retrieval import kb_navigation as nav
from test_interactive_needs import card, reply


class PickAll:
    def __init__(self, invalid_first=False):
        self.calls = []
        self.invalid_first = invalid_first

    async def complete(self, messages, **kwargs):
        instruction, data = messages[0]['content'].split('\n', 1)
        data = json.loads(data)
        self.calls.append((instruction, data, kwargs))
        limit = kwargs['structured_schema']['properties']['ids']['maxItems']
        ids = [d['id'] for d in data['documents']][:limit]
        if self.invalid_first and len(self.calls) == 1:
            ids = ['unknown']
        return reply({'ids': ids})


async def test_titles_then_cards_5000_files_has_20_title_4_card_and_1_final_calls():
    cards = [card(i, '历史', summary='卡片内容') for i in range(5000)]
    model = PickAll()
    selected, receipt = await nav.choose(model, '问题', cards, mode='titles_then_cards')
    assert len(selected) == 4 and len(receipt['decisions']) == 25
    titles = [(i, d, k) for i, d, k in model.calls if i.startswith(nav.TITLE_PICK[:20])]
    assert len(titles) == 20 and all(len(d['documents']) == 250 for _, d, _ in titles)
    assert all(set(item) == {'id', 'title'} for _, d, _ in titles for item in d['documents'])
    assert all(k['structured_schema']['properties']['ids']['maxItems'] == 5 for _, _, k in titles)
    cards_calls = [(i, d, k) for i, d, k in model.calls if not i.startswith(nav.TITLE_PICK[:20])]
    assert len(cards_calls) == 5 and all(len(d['documents']) <= 25 for _, d, _ in cards_calls)
    assert all('summary' in item for _, d, _ in cards_calls for item in d['documents'])
    offered = {c['id'] for c in cards}
    assert set(selected) <= offered and receipt['title_survivors'] == 100


async def test_title_only_control_uses_100_and_three_picks_until_final():
    model = PickAll()
    selected, receipt = await nav.choose(model, '问题', [card(i, '其他') for i in range(5000)], mode='titles_only')
    assert len(receipt['decisions']) == 53 and len(selected) == 4
    assert all(len(d['documents']) <= 100 for _, d, _ in model.calls)
    assert all(set(item) == {'id', 'title'} for _, d, _ in model.calls for item in d['documents'])
    assert all(k['structured_schema']['properties']['ids']['maxItems'] == 3 for _, _, k in model.calls[:-1])
    assert model.calls[-1][2]['structured_schema']['properties']['ids']['maxItems'] == 4


async def test_failed_title_selector_is_recorded_and_cannot_invent_file_ids():
    model = PickAll(invalid_first=True)
    cards = [card(i, '其他') for i in range(500)]
    selected, receipt = await nav.choose(model, '问题', cards, mode='titles_then_cards')
    assert receipt['decisions'][0]['parsed'] is None
    assert json.loads(receipt['decisions'][0]['raw_text']) == {'ids': ['unknown']}
    assert selected and set(selected) <= {c['id'] for c in cards[250:]}


@pytest.mark.parametrize('mode', ['titles_only', 'titles_then_cards'])
async def test_empty_catalog_makes_no_calls(mode):
    model = PickAll()
    selected, _ = await nav.choose(model, '问题', [], mode=mode)
    assert not selected and not model.calls
