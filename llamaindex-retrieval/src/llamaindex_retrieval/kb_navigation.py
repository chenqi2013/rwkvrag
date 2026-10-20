"""Model-written navigation cards and hierarchical knowledge-base file selection.

Cards are built once per file from its title and opening text, independent of any
question. At query time the model scores categories, then cards in small batches;
code only batches, ranks by the model's scores and caps how many files are read.
Cards are navigation hints, never evidence. No lexical or vector scoring.
"""
import json

from . import interactive_sources as structure
from .interactive_sources import gather

CATEGORIES = ['人物', '历史', '地理与地点', '政治与法律', '军事', '经济与企业', '社会与文化', '宗教与哲学',
              '语言与文学', '艺术与娱乐', '体育', '教育', '自然科学', '天文与地球科学', '生物',
              '医学与健康', '工程与技术', '计算机与互联网', '交通运输', '其他']
CARD = ('为知识库中的一份文档写导航卡片，供以后判断该文档可能回答哪些问题。summary用一句话写清文档讲的对象和主要内容'
        '（不超过60字），entities列出文档中最重要的人名、地名、机构、作品或术语（最多6个），category选最贴切的一类。'
        '只依据给出的标题和开头，不补充常识。')
CATEGORY_PICK = ('根据问题判断答案最可能在哪些类别的文档中，按可能性从高到低选出最多3个类别。')
CARD_PICK = ('以下是知识库文档的导航卡片（模型写的摘要，不是事实依据）。选出最可能包含问题答案的文档，'
             '最多选2份，按可能性从高到低排列；都不相关时返回空数组。')
FINAL_PICK = ('以下导航卡片已从各批中初选出来。选出最可能包含问题答案的文档，按可能性从高到低排列，'
              '最多选{n}份；都不相关时返回空数组。')
TITLE_PICK = ('以下是知识库的文档标题，只用于选择文件，不能作为回答的事实依据。'
              '选出最可能包含问题答案的文档ID，最多{n}份，按可能性从高到低排列；都不相关时返回空数组。')
OPENING_CHARS = 1200
CARD_BATCH = 25        # ~2k input tokens per selection call.
MAX_CATEGORIES = 3
MAX_FILES = 4          # Files read per knowledge-base query.
TITLE_BATCH = 250      # 5000 titles: 20 calls + at most 4 card calls + 1 final.
TITLE_CONTROL_BATCH = 100


async def call(model, instruction, data, properties, *, limit, evidence=()):
    result = await model.complete([{'role': 'user', 'content': instruction + '\n' +
        json.dumps(data, ensure_ascii=False, separators=(',', ':'))}],
        structured_schema=structure.obj(properties), max_tokens=limit,
        assistant_prefill='<think></think', temperature=0, seed=11, stage='reader',
        evidence_ids=list(evidence))
    event = {'status': result.status, 'raw_text': result.raw_text,
             'raw_sha256': structure.digest(result.raw_text or ''), 'parsed': None, 'trace': result.trace}
    if result.status == 'completed':
        try:
            value = json.loads(result.raw_text, object_pairs_hook=structure.unique_object)
            if not isinstance(value, dict) or set(value) != set(properties):
                raise ValueError('invalid fields')
            event['parsed'] = value
        except (ValueError, TypeError) as error:
            event['validation_error'] = str(error)
    return event


async def build_card(model, file_id, title, text):
    event = await call(model, CARD, {'title': title, 'opening': text[:OPENING_CHARS]}, {
        'summary': {'type': 'string', 'maxLength': 120},
        'entities': {'type': 'array', 'maxItems': 6, 'items': {'type': 'string', 'minLength': 1, 'maxLength': 40}},
        'category': {'type': 'string', 'enum': CATEGORIES}}, limit=192, evidence=[file_id])
    card = {'id': file_id, 'title': title, 'status': event['status'], 'raw_text': event['raw_text'],
            'opening_sha256': structure.digest(text[:OPENING_CHARS])}
    parsed = event['parsed']
    if parsed is not None and parsed['category'] in CATEGORIES and isinstance(parsed['summary'], str):
        card.update(summary=parsed['summary'], entities=[e for e in parsed['entities'] if isinstance(e, str)],
                    category=parsed['category'])
    else:
        # A failed card keeps the file reachable by title under 其他; recorded, not hidden.
        card.update(summary='', entities=[], category='其他', card_failed=True)
    return card


def picks(event, ids, limit):
    """Ordered unique picks from a selection call; anything else makes the call invalid."""
    if event['parsed'] is None:
        return []
    chosen = event['parsed']['ids']
    if (not isinstance(chosen, list) or any(i not in ids for i in chosen)
            or len(chosen) != len(set(chosen)) or len(chosen) > limit):
        event.update(parsed=None, validation_error='invalid picks')
        return []
    return chosen


def pick_schema(ids, limit):
    return {'ids': {'type': 'array', 'maxItems': limit, 'items': {'type': 'string', 'enum': list(ids)}}}


async def select(model, instruction, query, group, limit):
    """Cards are shown under short local IDs (D1..Dn) so outputs stay short; mapped back here."""
    local = {f'D{i+1}': c for i, c in enumerate(group)}
    event = await call(model, instruction, {'question': query, 'documents': [
        {'id': k, 'title': c['title'], 'summary': c['summary'], 'entities': c['entities']} for k, c in local.items()]},
        pick_schema(local, limit), limit=48, evidence=[c['id'] for c in group])
    return event, [local[k]['id'] for k in picks(event, list(local), limit)]


async def title_round(model, query, cards, batch_size, batch_picks, max_files, *, initial=False):
    groups = [cards[i:i+batch_size] for i in range(0, len(cards), batch_size)]
    final = len(groups) == 1 and not initial
    limit = max_files if final else batch_picks

    async def one(group):
        local = {f'D{i+1}': c for i, c in enumerate(group)}
        event = await call(model, TITLE_PICK.format(n=limit), {'question': query, 'documents': [
            {'id': k, 'title': c['title']} for k, c in local.items()]},
            pick_schema(local, limit), limit=max(48, 12*limit), evidence=[c['id'] for c in group])
        return event, [local[k] for k in picks(event, list(local), limit)]

    winners, decisions = [], []
    for event, chosen in await gather([one(g) for g in groups]):
        decisions.append({'stage': 'final' if final else 'titles', **event})
        winners.extend(chosen)
    return winners, decisions, final


async def choose(model, query, cards, *, max_files=MAX_FILES, use_categories=False, mode='cards'):
    """Categories, then a two-round tournament over card batches; returns (file IDs, receipt)."""
    if mode not in {'cards', 'titles_only', 'titles_then_cards'}:
        raise ValueError('invalid navigation mode')
    if mode != 'cards' and (use_categories or not 1 <= max_files <= MAX_FILES):
        raise ValueError('title navigation requires categories off and at most four final files')
    receipt = {'query': query, 'catalog_size': len(cards), 'decisions': []}
    if mode == 'titles_only':
        current = cards
        while current:
            current, decisions, final = await title_round(model, query, current, TITLE_CONTROL_BATCH, 3, max_files)
            receipt['decisions'].extend(decisions)
            if final:
                break
        ids = [c['id'] for c in current]
        receipt.update(mode=mode, title_batch=TITLE_CONTROL_BATCH, title_picks=3, selected_file_ids=ids)
        return ids, receipt
    if mode == 'titles_then_cards':
        cards, decisions, _ = await title_round(model, query, cards, TITLE_BATCH, 5, max_files, initial=True)
        receipt['decisions'].extend(decisions)
        receipt.update(mode=mode, title_batch=TITLE_BATCH, title_picks=5, title_survivors=len(cards))
    present = [c for c in CATEGORIES if any(card['category'] == c for card in cards)]
    candidates = cards
    # Category narrowing is optional: on the FineWiki dev set it dropped 6/16 gold files whose
    # card category the model did not pick (nav-dev-20261019 v1 -> v2).
    if use_categories and len(cards) > CARD_BATCH * 4:
        event = await call(model, CATEGORY_PICK, {'question': query, 'categories': present},
                           pick_schema(present, MAX_CATEGORIES), limit=64)
        receipt['decisions'].append({'stage': 'category', **event})
        # A failed or empty category call searches every category rather than none; recorded above.
        chosen = picks(event, present, MAX_CATEGORIES) or present
        receipt['categories'] = chosen
        candidates = [c for c in cards if c['category'] in chosen]
    by_id = {c['id']: c for c in candidates}
    round_ = candidates
    while True:
        batches = [round_[i:i+CARD_BATCH] for i in range(0, len(round_), CARD_BATCH)]
        final = len(batches) == 1
        instruction = FINAL_PICK.format(n=max_files) if final else CARD_PICK
        results = await gather([select(model, instruction, query, g, max_files if final else 2) for g in batches])
        winners = []
        for event, chosen in results:
            receipt['decisions'].append({'stage': 'final' if final else 'cards', **event})
            winners += chosen
        if final or not winners:
            break
        round_ = [by_id[i] for i in winners]
    selected = winners[:max_files]
    receipt['selected_file_ids'] = selected
    return selected, receipt
