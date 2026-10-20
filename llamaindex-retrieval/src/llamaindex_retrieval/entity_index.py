"""Model-authored entity navigation; code performs normalized equality only."""
from functools import lru_cache
from hashlib import sha256
from importlib.resources import files
import json
from pathlib import Path
import unicodedata

from opencc import OpenCC

from .interactive_spans import generate, strings

BUILD = ('为这份文档生成供精确查找使用的实体索引。entities最多12个文档主要对象的规范名称或术语，'
         'aliases最多12个原文支持的别称、缩写或其他文字写法。只依据标题、开头和给出的章节标题，'
         '不要写泛泛的类别，不补充常识，不回答任何问题。')
SELECT = ('以下文档由实体精确查找得到。标题和导航卡片仅供定位，不能作为回答依据。'
          '选择最可能包含完整问题答案的至多3个文档ID，按可能性排序；都不合适则返回空数组。')
NORMALIZATION = 'nfkc-casefold-opencc-t2s-codepoint-space-punctuation-v1'
_OPENCC = OpenCC('t2s')


@lru_cache(maxsize=16384)
def _character(char):
    # One codepoint per conversion: no multi-character phrase can match.
    return _OPENCC.convert(char)


def normalize(value):
    if not isinstance(value, str):
        raise ValueError('entity must be a string')
    text = unicodedata.normalize('NFKC', value).casefold()
    text = ''.join(_character(c) for c in text)
    return ''.join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith('P'))


def normalization_manifest():
    dictionary = files('opencc').joinpath('dictionary/TSCharacters.txt').read_bytes()
    return {'version': NORMALIZATION, 'character_dictionary_sha256': sha256(dictionary).hexdigest(),
            'unicode_version': unicodedata.unidata_version, 'conversion_unit': 'one Unicode codepoint'}


def entity_schema(limit):
    return {name: {**strings(), 'maxItems': limit,
                   'items': {'type': 'string', 'minLength': 1, 'maxLength': 96}}
            for name in ('entities', 'aliases')}


def valid_terms(value, limit):
    return (isinstance(value, dict) and set(value) == {'entities', 'aliases'}
            and all(isinstance(value[k], list) and len(value[k]) <= limit
                    and all(isinstance(t, str) and t.strip() and len(t) <= 96 for t in value[k])
                    for k in ('entities', 'aliases')))


def build_input(title, opening, headings):
    return {'title': title, 'opening': opening, 'headings': headings}


async def build_record(model, file_id, title, opening, headings=()):
    """One generation per file; budget probes remove whole trailing headings."""
    headings = list(headings)
    properties = entity_schema(12)
    while headings and hasattr(model, 'check_budget'):
        check = await generate(model, BUILD, build_input(title, opening, headings), properties,
                               limit=384, stage='index', evidence=[file_id], budget_only=True)
        if check['status'] == 'completed':
            break
        if check['status'] != 'budget_exceeded':
            return {'id': file_id, 'title': title, 'status': check['status'], 'parsed': None,
                    'raw_text': None, 'budget_failure': check}
        headings.pop()
    data = build_input(title, opening, headings)
    event = await generate(model, BUILD, data, properties, limit=384, stage='index', evidence=[file_id])
    if event['parsed'] is not None and not valid_terms(event['parsed'], 12):
        event.update(parsed=None, status='invalid_output', validation_error='invalid entity fields')
    return {'id': file_id, 'title': title, 'input': data, **event}


class EntityIndex:
    def __init__(self, records, cards=()):
        self.records, self.order, self.exact = {}, [], {}
        self.cards = {c['id']: c for c in cards}
        for row in records:
            fid = row['id']
            if not isinstance(fid, str) or not fid or fid in self.records:
                raise ValueError('duplicate or invalid entity file ID')
            self.records[fid] = row
            self.order.append(fid)
            value = row.get('parsed')
            if row.get('status') != 'completed' or not valid_terms(value, 12):
                continue
            for term in value['entities'] + value['aliases']:
                normalized = normalize(term)
                if normalized and fid not in self.exact.setdefault(normalized, []):
                    self.exact[normalized].append(fid)

    @classmethod
    def from_paths(cls, paths, cards=()):
        records = [json.loads(line) for path in paths
                   for line in Path(path).read_text().splitlines() if line.strip()]
        return cls(records, cards)

    def lookup(self, entities, aliases=(), *, allowed_ids=None):
        allowed = set(self.order if allowed_ids is None else allowed_ids)
        terms = list(dict.fromkeys(normalize(t) for t in [*entities, *aliases] if normalize(t)))
        matched = {t: [fid for fid in self.exact.get(t, ()) if fid in allowed] for t in terms}
        union = {fid for ids in matched.values() for fid in ids}
        # Corpus order, never number of hits or term overlap.
        candidates = [fid for fid in self.order if fid in union]
        return candidates, {'normalization': NORMALIZATION, 'normalized_terms': terms,
                            'exact_matches': matched, 'candidate_file_ids': candidates}

    def selection_input(self, question, ids, *, summary_chars=120, entities=3):
        rows = []
        for i, fid in enumerate(ids, 1):
            row = self.records[fid]
            card = self.cards.get(fid, {})
            rows.append({'id': f'D{i}', 'title': row['title'],
                         'summary': card.get('summary', '')[:summary_chars],
                         'entities': (row.get('parsed') or {}).get('entities', [])[:entities]})
        return {'question': question, 'documents': rows}
