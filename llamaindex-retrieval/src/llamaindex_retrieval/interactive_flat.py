"""Flat quote pipeline: plan once, read sources once, answer once, review once.

The model decides the turn relation, needs, queries, which sentences answer a
need or add detail, the final answer and the review findings. Code only
fetches, cleans markup, windows text, binds quotes and need wording to exact
character offsets, groups quotes and formats times. No model-written summary
is passed between calls or between conversation turns.
"""
import ast
import asyncio
from copy import deepcopy
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
import html
import json
import math
import operator
from pathlib import PurePath
import re
from time import monotonic
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from llama_index.core import Document
from jsonschema import Draft202012Validator, ValidationError

from .flat_prompts import (
    CALC, ISSUES, PLAN, READ, READ_ASSESSMENT, REVIEW,
    CALC_FINAL as CALC_FINAL,
    FINAL as FINAL,
    FINAL_CONCISE as FINAL_CONCISE,
    final_instruction,
)
from .interactive_sources import add_sources, digest, gather, obj, unique_object
from .verbatim_chunking import packed_verbatim_nodes

PROTOCOL = 'interactive-flat-quotes-v9'
CLEANING = 'markdown-links-images-v1'
WEEKDAYS = '一二三四五六日'
RELATIONS = ['新问题', '追问', '更正']
IMAGE = re.compile(r'!\[[^\]]*\]\([^)]*\)')
LINK = re.compile(r'\[([^\]]*)\]\((?:[^()]|\([^)]*\))*\)')
JUNK_LINE = re.compile(r'^\s*(?:javascript:\S*|https?://\S+|\[\s*\]|[-*•|]\s*)\s*$')
EMPHASIS = re.compile(r'\*\*|__|(?<!\w)\*(?=\S)|(?<=\S)\*(?!\w)')
POINT_PATTERN = r'^[^\n]{4,300}(\[[0-9]{1,2}\])+[。]?$'
POWERS = {'100', '1000', '10000', '100000000'}
BENIGN = {'duplicate_source_title_changed', 'windows_limited', 'quotes_limited'}
CITATION = re.compile(r'\[(\d{1,3})\]')
CODE_TEXT = re.compile(r'(?<!`)(`+)(?!`)[\s\S]*?(?<!`)\1(?!`)')
CLAUSE = re.compile(r'[，。；;！？!?]')
QUOTE_MARKS = '「」『』“”"\'‘’《》 '


def clean_markup(text):
    """Deterministic reading view for web markdown: drop images/URLs/emphasis, keep link text."""
    text = IMAGE.sub('', text)
    text = LINK.sub(lambda m: m.group(1), text)
    text = html.unescape(EMPHASIS.sub('', text))
    lines = [line.rstrip() for line in text.split('\n') if not JUNK_LINE.match(line)]
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(lines)).strip()


def local_time(value, zone):
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        return None
    if moment.utcoffset() is None:
        return None
    return moment.astimezone(zone)


def relative_day(moment, today):
    days = (today.date() - moment.date()).days
    return {0: '提问当天', 1: '提问前1天'}.get(days, f'提问前{days}天' if days > 0 else f'提问后{-days}天')


def _project(text, keep):
    projected = [(i, folded) for i, char in enumerate(text) if keep(char)
                 for folded in char.lower()]
    return [i for i, _ in projected], ''.join(char for _, char in projected)


def bind_quote(text, quote):
    """Exact, whitespace- or punctuation-insensitive substring match; returns (start, end, mode)."""
    for candidate in dict.fromkeys([quote, quote.strip(QUOTE_MARKS)]):
        if candidate and (start := text.find(candidate)) >= 0:
            return start, start + len(candidate), 'exact'
    for mode, keep, minimum in (('whitespace_insensitive', lambda c: not c.isspace(), 4),
                                ('punctuation_insensitive', str.isalnum, 6)):
        _, target = _project(quote, keep)
        if len(target) < minimum:
            continue
        positions, compact = _project(text, keep)
        start = compact.find(target)
        if start >= 0:
            return positions[start], positions[start + len(target) - 1] + 1, mode
    return None


def bind_fragments(text, quote, minimum=8):
    """A stitched quote binds clause by clause; only verbatim clauses survive."""
    found = []
    for part in CLAUSE.split(quote):
        part = part.strip(QUOTE_MARKS)
        if len(part) >= minimum and (hit := bind_quote(text, part)):
            found.append((hit[0], hit[1], 'fragment'))
    return found


def _normal(text):
    return re.sub(r'[\W_]+', '', text).lower()


def name_present(entity, text):
    """Literal retrieval/diagnostic match only; never an object-identity decision."""
    name = _normal(entity)
    positions, compact = _project(text, str.isalnum)
    if not name:
        return True
    def ascii_word(char):
        return char.isascii() and char.isalnum()
    return any(not (ascii_word(name[0]) and positions[hit.start()]
                    and ascii_word(text[positions[hit.start()] - 1]))
               and not (ascii_word(name[-1]) and positions[hit.end() - 1] + 1 < len(text)
                        and ascii_word(text[positions[hit.end() - 1] + 1]))
               for hit in re.finditer(re.escape(name), compact.lower()))


CN_DIGITS = {'零': 0, '〇': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
CN_UNITS = {'十': 10, '百': 100, '千': 1000, '万': 10000, '亿': 100000000}
CN_NUMBER = re.compile('[零〇一二两三四五六七八九十百千万亿]{2,}')


def cn_to_int(text):
    total, section, digit = 0, 0, 0
    for char in text:
        if char in CN_DIGITS:
            digit = CN_DIGITS[char]
        elif char in ('万', '亿'):
            section = (section + digit) * CN_UNITS[char]
            total, section, digit = total + section, 0, 0
        else:
            section += (digit or 1) * CN_UNITS[char]
            digit = 0
    return total + section + digit


def numeric_view(text):
    """Digits as written plus Chinese numerals converted, thousands separators removed."""
    text = re.sub(r'(?<![\d.])\d{1,3}(?:[,，]\d{3})+(?!\d)',
                  lambda m: m.group().replace(',', '').replace('，', ''), text)
    return text + ' ' + ' '.join(str(cn_to_int(m)) for m in CN_NUMBER.findall(text))


def numeric_values(text):
    """Compare complete numeric literals, including equivalent scientific notation."""
    values = set()
    for n in re.findall(r'(?<![\d.])(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?![\d.])', numeric_view(text)):
        try:
            values.add(decimal_literal(n))
        except ValueError:
            # An unusable number in a document must not abort all calculations.
            continue
    return values


def decimal_literal(text):
    """Bind the original token, before Python rounds a floating-point constant."""
    if len(text) > 128 or not re.fullmatch(r'(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', text):
        raise ValueError('unsupported numeric literal')
    try:
        value = Decimal(text)
    except InvalidOperation as error:
        raise ValueError('numeric literal out of range') from error
    if not value.is_finite() or abs(value.as_tuple().exponent) > 10000:
        raise ValueError('numeric literal out of range')
    return value


def evaluate(expression):
    """Bounded numeric AST; the model alone chooses the formula and operands."""
    if len(expression) > 512:
        raise ValueError('expression too long')
    tree = ast.parse(expression.replace('×', '*').replace('÷', '/'), mode='eval')
    if sum(1 for _ in ast.walk(tree)) > 128:
        raise ValueError('expression too complex')
    operations = {ast.Add: operator.add, ast.Sub: operator.sub,
                  ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow}
    functions = {'sqrt': math.sqrt, 'log': math.log, 'ln': math.log,
                 'log10': math.log10, 'exp': math.exp, 'abs': abs}
    def bounded(value):
        if type(value) is int:
            if value.bit_length() > 4096:
                raise ValueError('integer result too large')
        elif type(value) is not float or not math.isfinite(value):
            raise ValueError('result is not a finite real number')
        return value
    def walk(node, depth=0):
        if depth > 32:
            raise ValueError('expression too deep')
        def child(node):
            return walk(node, depth + 1)
        if isinstance(node, ast.Expression):
            return child(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return bounded(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            a, b = child(node.left), child(node.right)
            if isinstance(node.op, ast.Pow) and abs(b) > 1024:
                raise ValueError('exponent out of range')
            return bounded(operations[type(node.op)](a, b))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return bounded(-child(node.operand) if isinstance(node.op, ast.USub) else child(node.operand))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in functions and len(node.args) == 1 and not node.keywords):
            return bounded(functions[node.func.id](child(node.args[0])))
        raise ValueError('unsupported expression')
    return walk(tree)


def citation_numbers(text):
    """Inline/fenced code brackets denote code, not evidence references."""
    return CITATION.findall(CODE_TEXT.sub('', text))


RANGE = re.compile(r'(\d{1,2})月(\d{1,2})日\s*(?:至|到|-|—|~|～)\s*(次年)?(\d{1,2})月(\d{1,2})日')


def range_notes(text, today):
    """Code-side calendar arithmetic: whether today falls inside each month-day range in a quote."""
    notes = []
    for m in RANGE.finditer(text):
        a, b = (int(m.group(1)), int(m.group(2))), (int(m.group(4)), int(m.group(5)))
        now = (today.month, today.day)
        inside = a <= now <= b if a <= b and not m.group(3) else (now >= a or now <= b)
        notes.append(f'{m.group(0)}：今天{"在" if inside else "不在"}此期间')
    return notes


def _key(text):
    return re.sub(r'[\W\d_]+', '', CITATION.sub('', text)).lower()


def repeated_points(value):
    """True when a point repeats the answer or another point (same letters, ignoring numbers/punctuation marks)."""
    keys = [_key(value['answer'])] + [_key(p) for p in value.get('points') or [] if isinstance(p, str)]
    keys = [k for k in keys if k]
    return any(a == b or (len(a) >= 8 and a in b) or (len(b) >= 8 and b in a)
               or (min(len(a), len(b)) >= 4 and SequenceMatcher(None, a, b).ratio() >= 0.85)
               for i, a in enumerate(keys) for b in keys[i + 1:])


def title_candidates(catalog, entities, queries, *, read_all_below=12, limit=6):
    """Lexical title match only: small upload sets are read whole, large ones by name match."""
    if len(catalog) <= read_all_below:
        return [row['id'] for row in catalog], 'all_files'
    terms = [re.sub(r'\s+', '', t).lower() for t in [*entities, *queries] if t.strip()]
    scored = []
    for row in catalog:
        stem = re.sub(r'\s+', '', PurePath(row['title']).stem).lower()
        # Bare numbers or very short ASCII stems match inside unrelated names.
        if len(stem) < 2 or not re.search(r'[^\W\d_]', stem) or (stem.isascii() and len(stem) < 4):
            continue
        # A title matches when it names an entity or appears inside a query.
        score = max((len(stem) for t in terms if stem in t or (len(t) >= 2 and t == stem)), default=0)
        if score:
            scored.append((score, row['id']))
    scored.sort(key=lambda x: -x[0])
    return [identity for _, identity in scored[:limit]], 'title_match'


class Flat:
    def __init__(self, model, question, *, channels, search, select_files, progress, history=(),
                 previous_sources=(), as_of=None, timezone_name='Asia/Shanghai', window_chars=2400,
                 window_overlap=200, max_windows_per_source=4, snippet_batch=4, snippet_batch_wait=1.5,
                 deadline_seconds=180, acquisition_timeout=70, final_reserve_seconds=50,
                 final_quote_limit=40, per_need_limit=12, final_frequency_penalty=0.3,
                 supplement=True, review=True, kb_window_chars=6000, kb_max_windows=40,
                 read_window_batch=2, retry_repeated_points=True, observe=None,
                 requested_urls=(), fetch_urls=None, read_require_assessment=False,
                 final_response_style='standard'):
        if (not set(channels) <= {'knowledge_base', 'web'} or not 400 <= window_chars <= 24000
                or not 0 <= window_overlap < window_chars // 2 or max_windows_per_source < 1
                or not 1 <= snippet_batch <= 8 or not 0 < final_reserve_seconds < deadline_seconds
                or type(read_window_batch) is not int or not 1 <= read_window_batch <= 2
                or type(retry_repeated_points) is not bool or type(read_require_assessment) is not bool
                or final_response_style not in ('standard', 'concise')):
            raise ValueError('invalid flat pipeline options')
        self.model, self.original_question, self.question = model, question, question
        self.channels, self.search, self.select_files = list(dict.fromkeys(channels)), search, select_files
        self.requested_urls, self.fetch_urls = list(requested_urls), fetch_urls
        self.progress, self.history = progress, list(history)[-3:]
        self.previous_sources = list(previous_sources)
        self.zone = ZoneInfo(timezone_name)
        self.as_of = as_of or datetime.now(self.zone).isoformat(timespec='seconds')
        self.today = local_time(self.as_of, self.zone)
        if self.today is None:
            raise ValueError('as_of requires an explicit timezone')
        self.window_chars, self.window_overlap = window_chars, window_overlap
        # Uploaded documents are long and chosen by the user, so they are read in larger, more windows.
        self.kb_window_chars, self.kb_max_windows = kb_window_chars, kb_max_windows
        self.max_windows, self.snippet_batch, self.snippet_batch_wait = max_windows_per_source, snippet_batch, snippet_batch_wait
        self.read_window_batch = read_window_batch
        self.read_require_assessment = read_require_assessment
        self.final_response_style = final_response_style
        self.retry_repeated_points = retry_repeated_points
        self.repetition_checks = []
        self.read_batch_checks = []
        self.final_quote_limit, self.per_need_limit = final_quote_limit, per_need_limit
        self.final_frequency_penalty, self.supplement, self.review_enabled = final_frequency_penalty, supplement, review
        self.started = monotonic()
        self.deadline = self.started + deadline_seconds
        self.reading_deadline = self.deadline - final_reserve_seconds
        self.acquisition_timeout = acquisition_timeout
        self.relation, self.needs, self.aspects, self.queries, self.entities = '新问题', [], [], [], []
        self.metrics, self.names, self.name_absent = [], [], []
        self.assumptions, self.calculation, self.calc_event, self.calculations = [], False, None, []
        self.need_bindings, self.dropped_needs = [], []
        self.sources, self.views, self.errors, self.acquisition = [], {}, [], []
        self.source_window_limits = {}
        self.reads, self.quotes, self.unbound = [], {}, []
        self.pending_snippets, self.flush_task = [], None
        self.tasks = set()
        self.plan_event = self.final_event = self.review_event = None
        self.final_attempts = []
        self.reused_source_ids = []
        self.timeline = {}
        self.observe = observe
        self.observation_sequence = 0
        self.call_sequence = 0
        self.observation_errors = []

    def emit(self, kind, **fields):
        """Expose execution facts without changing prompts, selection or answers."""
        if self.observe is None:
            return
        self.observation_sequence += 1
        try:
            self.observe(deepcopy({'sequence': self.observation_sequence, 'kind': kind,
                                  'elapsed_ms': (monotonic() - self.started) * 1000, **fields}))
        except Exception as error:
            self.observation_errors.append({'sequence': self.observation_sequence,
                                            'error_type': type(error).__name__})

    # ---------- prompt fragments (plain text, material first, question last) ----------
    def today_line(self):
        # Relative days are arithmetic, so code states them instead of leaving them to the model.
        days = '，'.join(f'{name}是{(self.today + timedelta(days=d)):%m月%d日}（星期{WEEKDAYS[(self.today + timedelta(days=d)).weekday()]}）'
                        for name, d in (('昨天', -1), ('明天', 1), ('后天', 2)))
        return f'今天是{self.today:%Y年%m月%d日}（星期{WEEKDAYS[self.today.weekday()]}，北京时间{self.today:%H:%M}），{days}。'

    def calendar_line(self, years=True):
        """Week and month ranges computed by code; the model only picks which one the question means."""
        d = self.today.date()
        monday = d - timedelta(days=d.weekday())
        last_monday = monday - timedelta(days=7)
        first = d.replace(day=1)
        last_month_end = first - timedelta(days=1)

        def fmt(x):
            return f'{x.month}月{x.day}日'
        last_week = '、'.join(f'上周{WEEKDAYS[i]}是{fmt(last_monday + timedelta(days=i))}' for i in range(7))
        return (f'日期参照：本周是{fmt(monday)}至{fmt(monday + timedelta(days=6))}，{last_week}；'
                f'最近一周是{fmt(d - timedelta(days=6))}至{fmt(d)}；本月是{d.year}年{d.month}月；'
                f'上个月是{last_month_end.year}年{last_month_end.month}月' +
                (f'；今年是{d.year}年，去年是{d.year - 1}年。' if years else '。'))

    def ask_block(self, aspects=True):
        needs = '\n'.join(f'{i}. {need}' for i, need in enumerate(self.needs, 1))
        # Object names come from the model's own plan; they make same-name checks explicit.
        objects = f"\n所问对象：{'、'.join(self.entities)}（原文说的是名称不同的其他对象时不能使用）" if self.entities else ''
        metrics = f"\n所问指标：{'、'.join(self.metrics)}（原文是其他指标的数值时不能当作所问指标）" if self.metrics else ''
        extra = f"\n值得补充的方面：{'、'.join(self.aspects)}" if aspects and self.aspects else ''
        assumed = f"\n本次理解的假设（回答开头要写明）：{'；'.join(self.assumptions)}" if self.assumptions else ''
        original = f'用户本轮原话：{self.original_question}\n' if self.original_question != self.question else ''
        return f'{self.today_line()}\n{self.calendar_line()}\n{original}问题：{self.question}{objects}{metrics}{assumed}\n信息项：\n{needs}{extra}'

    def header(self, source):
        meta = source.metadata
        place = '网页 ' + (urlsplit(source.uri or '').hostname or '') if source.source == 'web' else '上传资料'
        parts = [source.title[:120], place]
        published_value = meta.get('published_date') or meta.get('published_at') or ''
        published = local_time(published_value, self.zone)
        publication = f'{published:%Y-%m-%d}（{relative_day(published, self.today)}）' if published else '未知'
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', published_value):
            try:
                publication = date.fromisoformat(published_value).isoformat() + '（原文日期，未提供时刻）'
            except ValueError:
                pass
        if source.source == 'web' or published_value:
            parts.append('发布：' + publication)
        if source.source == 'web':
            fetched = local_time(meta.get('retrieved_at') or '', self.zone)
            if fetched:
                parts.append(f'抓取：{fetched:%Y-%m-%d %H:%M}（{relative_day(fetched, self.today)}）')
            if meta.get('content_status') == 'snippet_only':
                parts.append('正文未能获取，仅有搜索摘要')
        return '｜'.join(parts)

    def short_header(self, source):
        # Publication and retrieval are different observations; preserve both for the Writer.
        return self.header(source) + (f'｜URL：{source.uri}' if source.uri else '')

    async def call(self, stage, prompt, properties, limit, evidence=(), frequency_penalty=0.0, presence_penalty=0.0):
        self.call_sequence += 1
        operation = f'call-{self.call_sequence}'
        self.emit('call_started', operation=operation, stage=stage, evidence_ids=list(evidence))
        try:
            result = await self.model.complete([{'role': 'user', 'content': prompt}],
                structured_schema=obj(properties), max_tokens=limit,
                assistant_prefill='<think></think', temperature=0, seed=11, frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stage='writer' if stage in {'final', 'review'} else 'reader', evidence_ids=list(evidence))
        except BaseException as error:
            self.emit('call_finished', operation=operation, stage=stage, evidence_ids=list(evidence),
                      status='cancelled' if isinstance(error, asyncio.CancelledError) else 'failed',
                      error_type=type(error).__name__)
            raise
        event = {'stage': stage, 'status': result.status, 'raw_text': result.raw_text,
                 'transport_status': result.status,
                 'raw_sha256': digest(result.raw_text or ''), 'parsed': None, 'trace': result.trace,
                 'finished_ms': (monotonic() - self.started) * 1000}
        if result.status == 'completed':
            try:
                value = json.loads(result.raw_text, object_pairs_hook=unique_object)
                if not isinstance(value, dict) or set(value) != set(properties):
                    raise ValueError('invalid fields')
                Draft202012Validator(obj(properties)).validate(value)
                event['parsed'] = value
            except (ValueError, TypeError, ValidationError) as error:
                event['validation_error'] = str(error)
                event['status'] = 'invalid_output'
        self.emit('call_finished', operation=operation, stage=stage, evidence_ids=list(evidence),
                  status=event['status'], valid_output=event['parsed'] is not None)
        return event

    # ---------- planning ----------
    def history_block(self):
        if not self.history:
            return ''
        lines = ['前几轮对话（只用于理解指代，不是事实依据）：']
        for number, turn in enumerate(self.history, 1):
            objects = '、'.join(turn.get('entities') or []) or '无'
            lines.append(f"第{number}轮用户原话：{turn['question']}｜整理为：{turn.get('resolved_question') or turn['question']}｜所问对象：{objects}")
            if turn.get('answer_opening'):
                # Only the first sentence: it names what "它/这个" refers to; it is not evidence.
                lines.append(f"第{number}轮回答开头（只用于判断指代的对象）：{turn['answer_opening']}")
        return '\n'.join(lines) + '\n\n'

    async def planning(self):
        self.progress('理解问题，规划检索')
        prompt = f'{self.history_block()}{self.today_line()}\n{self.calendar_line()}\n用户本轮原话：{self.original_question}\n\n{PLAN}'
        text = {'type': 'string', 'minLength': 1, 'maxLength': 120}
        strings = {'type': 'array', 'items': text}
        need = obj({'text': text, 'user_words': text})
        self.plan_event = await self.call('plan', prompt, {
            'relation': {'type': 'string', 'enum': RELATIONS},
            'question': {'type': 'string', 'minLength': 1, 'maxLength': 1600},
            'needs': {'type': 'array', 'items': need, 'minItems': 1, 'maxItems': 8},
            'aspects': {**strings, 'maxItems': 4},
            'queries': {**strings, 'minItems': 1, 'maxItems': 2},
            'entities': {**strings, 'maxItems': 8}, 'metrics': {**strings, 'maxItems': 4},
            'names': {**strings, 'maxItems': 6}, 'assumptions': {**strings, 'maxItems': 3},
            'calculation': {'type': 'boolean'}}, 900)
        value = self.plan_event['parsed']
        if value is None or any(not isinstance(value[k], list) for k in ('needs', 'aspects', 'queries', 'entities', 'metrics', 'names', 'assumptions')):
            self.plan_event['parsed'] = None
            self.errors.append({'stage': 'plan', 'status': self.plan_event['status']})
            return False
        self.relation, self.question = value['relation'], value['question']
        for row in value['needs']:
            words = row.get('user_words') or '' if isinstance(row, dict) else ''
            hit = bind_quote(self.original_question, words) if words.strip() else None
            if hit and isinstance(row.get('text'), str) and row['text'].strip():
                self.need_bindings.append({'text': row['text'].strip(), 'user_words': words,
                                           'span': {'start': hit[0], 'end': hit[1], 'match': hit[2]}})
            else:
                # Needs whose wording is not in the user's message are recorded, not searched or answered.
                self.dropped_needs.append(row)
        if not self.need_bindings:
            # Structural fallback: the user's own message is the single need.
            self.need_bindings.append({'text': self.original_question.strip(), 'user_words': self.original_question,
                                       'span': {'start': 0, 'end': len(self.original_question), 'match': 'whole_message'}})
        self.needs = list(dict.fromkeys(b['text'] for b in self.need_bindings))[:8]
        self.aspects = list(dict.fromkeys(a.strip() for a in value['aspects'] if a.strip()))[:4]
        self.queries = list(dict.fromkeys(q.strip() for q in value['queries'] if q.strip()))[:2]
        self.entities = list(dict.fromkeys(e.strip() for e in value['entities'] if e.strip()))
        self.metrics = list(dict.fromkeys(m.strip() for m in value['metrics'] if m.strip()))
        # Only names that occur in the user's turn or the history are literal-checked.
        context = self.original_question + ''.join(t.get('question', '') + ''.join(t.get('entities') or []) for t in self.history)
        self.names = [n.strip() for n in value['names'] if n.strip() and (name_present(n, context) or n.strip() in self.entities)]
        # Names written in the code's own date lines are date references, not objects.
        dates = _normal(self.today_line() + self.calendar_line())
        relative = ('今天', '昨天', '明天', '后天', '本周', '这周', '上周', '最近', '本月', '上个月', '今年', '去年')
        needs, metrics = [_normal(x) for x in self.needs], [_normal(x) for x in self.metrics]
        # Date words come from the code's own date lines; a "name" equal to a need or overlapping a
        # metric is the asked quantity itself, not an object to look up.
        self.names = [n for n in self.names if _normal(n) not in dates and not _normal(n).startswith(relative)
                      and _normal(n) not in needs
                      and not any(_normal(n) in m or m in _normal(n) for m in metrics if m)]
        if not self.names and self.relation != '新问题' and self.history:
            self.names = list(self.history[-1].get('names') or [])
        self.assumptions = [a.strip() for a in value['assumptions'] if a.strip()]
        # The model distinguishes looking up a quantity from doing arithmetic.
        self.calculation = value['calculation'] is True
        if self.relation != '新问题' and self.history:
            # Follow-up queries must carry the referenced object; add it when the plan dropped it.
            previous = [e for e in self.history[-1].get('entities') or [] if e.strip()]
            if previous and not any(name_present(e, q) for e in previous for q in self.queries):
                self.queries = [f'{previous[0]} {q}' for q in self.queries]
        if not self.queries:
            self.errors.append({'stage': 'plan', 'status': 'empty_plan'})
            return False
        self.timeline['plan_ms'] = (monotonic() - self.started) * 1000
        self.emit('plan_ready', question=self.question, needs=list(self.needs), queries=list(self.queries),
                  entities=list(self.entities), channels=list(self.channels))
        return True

    # ---------- acquisition and reading ----------
    async def publish(self, source, *, requested=False):
        count = len(self.sources)
        previous_limit = self.source_window_limits.get(source.id)
        try:
            add_sources(self.sources, [source])
        except ValueError:
            # Same URL and text returned under another title: keep the first observation.
            self.errors.append({'stage': 'acquire', 'status': 'duplicate_source_title_changed', 'source_id': source.id})
            return
        if len(self.sources) == count:
            if (not requested or source.source != 'web' or previous_limit is None
                    or previous_limit >= self.kb_max_windows):
                return
            # A follow-up may already have scheduled the old cited page under
            # the search limit. An explicit fetch can upgrade that same source,
            # including canonical redirects, without reading its prefix twice.
            source = next(s for s in self.sources if s.id == source.id)
        body = source.snippet if source.metadata.get('content_status') != 'snippet_only' else ''
        view = clean_markup(body) if source.source == 'web' else body
        snippet = source.metadata.get('search_snippet') if source.source == 'web' else None
        snippet = clean_markup(snippet) if isinstance(snippet, str) and snippet.strip() else ''
        if snippet and snippet in view:
            snippet = ''
        if source.metadata.get('content_status') == 'snippet_only' and not snippet:
            snippet = clean_markup(source.snippet)
        self.views[source.id] = {'body': view, 'snippet': snippet, 'cleaning': CLEANING if source.source == 'web' else 'none',
                                 'body_sha256': digest(view), 'snippet_sha256': digest(snippet)}
        windows = []
        upload = source.source == 'knowledge_base'
        size, limit = (self.kb_window_chars, self.kb_max_windows) if upload else (self.window_chars, self.max_windows)
        # A page explicitly supplied by the user is selected reading material,
        # including a fetch that resolves to a different canonical URL. Preserve
        # the web window size; only its count limit follows uploaded documents.
        if source.source == 'web' and requested:
            limit = self.kb_max_windows
        if view.strip():
            nodes = list(packed_verbatim_nodes(Document(text=view, id_=source.id), size,
                                               self.window_overlap, absorb_whitespace=True))
            windows = [(n.metadata['source_span']['start'], n.metadata['source_span']['end']) for n in nodes]
        if len(windows) > limit:
            self.errors.append({'stage': 'read', 'status': 'windows_limited', 'source_id': source.id,
                                'windows': len(windows), 'read': limit})
            windows = windows[:limit]
        self.source_window_limits[source.id] = limit
        start = previous_limit or 0
        if previous_limit is not None and len(windows) <= start:
            return
        self.emit('source_ready', source=source.model_dump(mode='json'), requested=requested,
                  reading_view=dict(self.views[source.id]), windows=[list(w) for w in windows])
        if not windows:
            if snippet:
                # Short snippet-only sources share one reading call.
                self.pending_snippets.append((source, None, True))
                if len(self.pending_snippets) >= self.snippet_batch:
                    self.flush_snippets()
                elif self.flush_task is None:
                    self.flush_task = self.spawn(self.delayed_flush())
            return
        # Pack complete neighbouring windows only when the real model can
        # preflight the combined prompt and output budget without generation.
        batch_size = self.read_window_batch if callable(getattr(self.model, 'check_budget', None)) else 1
        segments = [(source, window, index == 0 and bool(snippet))
                    for index, window in enumerate(windows) if index >= start]
        for index in range(0, len(segments), batch_size):
            self.spawn(self.read(segments[index:index + batch_size]))
        self.progress(f'取材与阅读 · {len(self.sources)} 份资料 · 已读 {len(self.reads)}')

    def spawn(self, job):
        task = asyncio.create_task(job)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    def flush_snippets(self):
        batch, self.pending_snippets = self.pending_snippets, []
        for i in range(0, len(batch), self.snippet_batch):
            self.spawn(self.read(batch[i:i + self.snippet_batch]))

    async def delayed_flush(self):
        try:
            await asyncio.sleep(self.snippet_batch_wait)
        finally:
            self.flush_task = None
        self.flush_snippets()

    async def read(self, segments):
        parts = []
        for number, (source, window, include_snippet) in enumerate(segments, 1):
            view = self.views[source.id]
            lines = [f'【资料{number}】{self.header(source)}']
            if window is not None:
                lines.append('正文开始\n' + view['body'][window[0]:window[1]] + '\n正文结束')
            if include_snippet:
                lines.append('正文开始（搜索摘要，网页正文未核实）\n' + view['snippet'] + '\n正文结束')
            parts.append('\n'.join(lines))
        instruction = READ_ASSESSMENT if self.read_require_assessment else READ
        prompt = '\n\n'.join(parts) + f'\n\n{self.ask_block()}\n\n{instruction}'
        def quotes(count, size):
            return {'type': 'array', 'maxItems': count, 'items': {'type': 'string', 'minLength': 2, 'maxLength': size}}
        # A complete sentence can be longer than 240/360 characters. Bound the field
        # by material actually supplied, not a cap that forces cuts inside a quote.
        quote_size = max(len(self.views[s.id]['body'][w[0]:w[1]]) if w else 0
                         for s, w, _ in segments)
        quote_size = max(quote_size, *(len(self.views[s.id]['snippet']) if include else 0
                                      for s, _, include in segments), 2)
        item = obj({'source': {'type': 'integer', 'enum': list(range(1, len(segments) + 1))},
                    'need': {'type': 'integer', 'enum': list(range(1, len(self.needs) + 1))},
                    'answer_quotes': quotes(3, quote_size), 'detail_quotes': quotes(5, quote_size)})
        ids = [s.id for s, _, _ in segments]
        properties = {'items': {'type': 'array', 'items': item, 'maxItems': len(self.needs) * len(segments)}}
        if self.read_require_assessment:
            properties['items']['minItems'] = 1
        # Reserve for the actual source/need matrix, instead of giving eight
        # needs the same completion budget as a single lookup. Whole-window
        # preflight below uses this exact reservation before batching.
        output_limit = min(4096, max(1400, 700 + 350 * len(self.needs) * len(segments)))
        if len(segments) > 1 and any(window is not None for _, window, _ in segments):
            check = {'source_ids': ids, 'windows': [window for _, window, _ in segments],
                     'prompt_sha256': digest(prompt), 'completion_attempted': False}
            self.read_batch_checks.append(check)
            try:
                async with asyncio.timeout_at(self.reading_deadline):
                    budget = await self.model.check_budget([{'role': 'user', 'content': prompt}],
                        structured_schema=obj(properties), max_tokens=output_limit,
                        assistant_prefill='<think></think', temperature=0, seed=11,
                        stage='reader', evidence_ids=ids)
                check.update(status=budget.status, trace=budget.trace)
                fits = (budget.status == 'completed' and budget.trace.get('budget_only') is True
                        and budget.trace.get('completion_attempted') is False)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                check.update(status='failed', error_type=type(error).__name__)
                fits = False
            if not fits:
                # Keep every whole window; an oversized batch is split rather
                # than truncated or sent as a generation that cannot fit.
                for segment in segments:
                    self.spawn(self.read([segment]))
                return
        try:
            async with asyncio.timeout_at(self.reading_deadline):
                event = await self.call('read', prompt, properties, output_limit, evidence=ids)
        except TimeoutError:
            self.errors.append({'stage': 'read', 'status': 'deadline', 'source_ids': ids})
            return
        event.update(source_ids=ids, windows=[w for _, w, _ in segments], quote_ids=[])
        self.reads.append(event)
        if event['parsed'] is None:
            self.errors.append({'stage': 'read', 'status': event['status'], 'source_ids': ids})
            if event['status'] not in {'invalid_output', 'length'}:
                return
            # Regenerate independently; never repair or partially admit the failed raw text.
            retry_prompt = ('格式要求：重新完整输出紧凑JSON，必须闭合所有引号和括号。'
                            '只摘直接相关的原文，不重复引文。\n\n' + prompt)
            try:
                async with asyncio.timeout_at(self.reading_deadline):
                    retry_limit = min(4096, output_limit * 2) if event['status'] == 'length' else output_limit
                    retry = await self.call('read', retry_prompt, properties, retry_limit, evidence=ids)
            except TimeoutError:
                self.errors.append({'stage': 'read', 'status': 'deadline', 'source_ids': ids})
                return
            retry.update(source_ids=ids, windows=[w for _, w, _ in segments], quote_ids=[],
                         retry_of=event['raw_sha256'])
            self.reads.append(retry)
            if retry['parsed'] is None:
                self.errors.append({'stage': 'read', 'status': retry['status'], 'source_ids': ids})
                return
            event = retry
        self.emit('binding_started', source_ids=ids)
        for row in event['parsed']['items'] if isinstance(event['parsed']['items'], list) else []:
            if (not isinstance(row, dict) or row.get('need') not in range(1, len(self.needs) + 1)
                    or row.get('source') not in range(1, len(segments) + 1)):
                continue
            for kind in ('answer', 'detail'):
                for quote in row.get(kind + '_quotes') or []:
                    if isinstance(quote, str) and quote.strip():
                        # The quote may come from another source shown in the same call.
                        order = [row['source'] - 1] + [i for i in range(len(segments)) if i != row['source'] - 1]
                        bound = any(self.bind(event, segments[i], row['need'], kind, quote,
                                              record=False, match_mode='whole') for i in order)
                        if not bound:
                            # A stitched quote can contain clauses from several
                            # offered windows. Collect every matched clause.
                            for i in order:
                                bound = self.bind(event, segments[i], row['need'], kind, quote,
                                                  record=False, match_mode='fragments') or bound
                        if not bound:
                            self.unbound.append({'source_id': segments[row['source'] - 1][0].id, 'need': row['need'],
                                                 'kind': kind, 'quote': quote, 'read_sha256': event['raw_sha256']})
        self.progress(f'取材与阅读 · {len(self.sources)} 份资料 · 已读 {len(self.reads)}')
        self.emit('binding_finished', source_ids=ids, quote_ids=list(event['quote_ids']),
                  bound_quotes=len(self.quotes), unbound_quotes=len(self.unbound),
                  spans=[self.quotes[q] for q in event['quote_ids']])

    def bind(self, event, segment, need, kind, quote, record=True, match_mode='all'):
        source, window, include_snippet = segment
        view = self.views[source.id]
        candidates = []
        if window is not None:
            candidates.append(('body', view['body'], window[0], view['body'][window[0]:window[1]]))
        if include_snippet:
            candidates.append(('snippet', view['snippet'], 0, view['snippet']))
        def whole(text, quote):
            hit = bind_quote(text, quote)
            return [hit] if hit else []
        matchers = {'all': (whole, bind_fragments), 'whole': (whole,), 'fragments': (bind_fragments,)}
        found = False
        for matcher in matchers[match_mode]:
            for section, full, offset, text in candidates:
                hits = matcher(text, quote)
                for start, end, mode in hits:
                    start, end = offset + start, offset + end
                    identity = 'Q' + digest(f'{source.id}\0{section}\0{start}:{end}')[:16]
                    row = self.quotes.setdefault(identity, {'id': identity, 'source_id': source.id, 'section': section,
                        'text': full[start:end], 'needs': [], 'kinds': [], 'match': mode,
                        'span': {'start': start, 'end': end, 'unit': 'unicode_code_points', 'view': section,
                                 'view_sha256': view[section + '_sha256'], 'sha256': digest(full[start:end])}})
                    if need not in row['needs']:
                        row['needs'].append(need)
                    if kind not in row['kinds']:
                        row['kinds'].append(kind)
                    if identity not in event['quote_ids']:
                        event['quote_ids'].append(identity)
                if hits:
                    found = True
                    if matcher is whole:
                        return True
        if found:
            return True
        if record:
            self.unbound.append({'source_id': source.id, 'need': need, 'kind': kind, 'quote': quote,
                                 'read_sha256': event['raw_sha256']})
        return False

    async def acquire(self, queries, round_name):
        async def one(query, urls=None):
            receipt = {'round': round_name, 'channel': 'web', 'query': query, 'status': 'pending'}
            if urls is not None:
                receipt.update(requested_urls=urls, action='extract')
            self.acquisition.append(receipt)
            self.emit('acquisition_started', **receipt)
            try:
                async with asyncio.timeout(self.acquisition_timeout):
                    sources = await self.fetch_urls(urls) if urls is not None else await self.search(query)
                    for source in sources:
                        await self.publish(source, requested=urls is not None)
                receipt['status'] = 'completed'
            except asyncio.CancelledError:
                receipt['status'] = 'cancelled'
                raise
            except Exception as error:
                receipt.update(status='timeout' if isinstance(error, TimeoutError) else 'failed',
                               error=getattr(error, 'safe_code', type(error).__name__))
                self.errors.append({'stage': 'acquire', **receipt})
            finally:
                self.emit('acquisition_finished', **receipt)

        async def files():
            receipt = {'round': round_name, 'channel': 'knowledge_base', 'status': 'pending'}
            self.acquisition.append(receipt)
            self.emit('acquisition_started', **receipt)
            try:
                async with asyncio.timeout(self.acquisition_timeout):
                    chosen = await self.select_files(self.entities, self.queries, self.publish)
                receipt.update(status='completed', **chosen)
            except asyncio.CancelledError:
                receipt['status'] = 'cancelled'
                raise
            except Exception as error:
                receipt.update(status='failed', error=type(error).__name__)
                self.errors.append({'stage': 'acquire', **receipt})
            finally:
                self.emit('acquisition_finished', **receipt)
        jobs = []
        if 'web' in self.channels:
            if round_name == 'initial' and self.requested_urls and self.fetch_urls is not None:
                jobs.append(one('', self.requested_urls))
            else:
                jobs.extend(one(q) for q in queries)
        if 'knowledge_base' in self.channels and round_name == 'initial':
            jobs.append(files())
        await gather(jobs)

    async def drain(self):
        while self.tasks or self.pending_snippets:
            if self.flush_task is not None:
                self.flush_task.cancel()
                self.flush_task = None
            self.flush_snippets()
            await asyncio.gather(*list(self.tasks), return_exceptions=True)
            # Completed tasks can spawn smaller reads before their discard
            # callbacks run. Yield even when gather completes synchronously.
            await asyncio.sleep(0)

    def missing_needs(self):
        covered = {n for q in self.quotes.values() if 'answer' in q['kinds'] for n in q['needs']}
        return [i for i in range(1, len(self.needs) + 1) if i not in covered]

    # ---------- answering ----------
    def final_materials(self):
        """Group bound quotes by source; answers before details; cap by count, never truncate a quote."""
        chosen, per_need, seen = [], {}, set()
        ranked = sorted(self.quotes.values(), key=lambda q: ('answer' not in q['kinds'], min(q['needs']),
                                                             q['source_id'], q['span']['start']))
        # Within each (kind, need) group take quotes round-robin across sources, so one long
        # document cannot fill every slot before another document is represented.
        order, groups = [], {}
        for quote in ranked:
            groups.setdefault(('answer' not in quote['kinds'], min(quote['needs'])), {}).setdefault(
                quote['source_id'], []).append(quote)
        for key in sorted(groups):
            lanes = list(groups[key].values())
            for i in range(max(map(len, lanes))):
                order.extend(lane[i] for lane in lanes if i < len(lane))
        for quote in order:
            need = min(quote['needs'])
            key = re.sub(r'\W+', '', quote['text'])
            if key in seen or any(key in other for other in seen):
                continue
            seen.add(key)
            if len(chosen) >= self.final_quote_limit or per_need.get(need, 0) >= self.per_need_limit:
                continue
            per_need[need] = per_need.get(need, 0) + 1
            chosen.append(quote)
        if len(chosen) < len(seen):
            self.errors.append({'stage': 'final', 'status': 'quotes_limited',
                                'offered': len(chosen), 'bound': len(self.quotes)})
        by_source = {}
        for quote in chosen:
            by_source.setdefault(quote['source_id'], []).append(quote)
        sources = {s.id: s for s in self.sources}
        blocks, numbers = [], {}
        for number, (source_id, rows) in enumerate(by_source.items(), 1):
            numbers[str(number)] = source_id
            lines = [f'[{number}] ' + self.short_header(sources[source_id])]
            for quote in sorted(rows, key=lambda q: (q['section'], q['span']['start'])):
                label = '（搜索摘要，正文未核实）' if quote['section'] == 'snippet' else ''
                notes = range_notes(quote['text'], self.today)
                note = f"〔{'；'.join(notes)}〕" if notes else ''
                lines.append(f'「{quote["text"].strip()}」{label}{note}')
            blocks.append('\n'.join(lines))
        return blocks, numbers, [q['id'] for q in chosen]

    async def calculate(self):
        """Model writes expressions; code checks literals against evidence/question and computes."""
        self.progress('依据原文与题设数字计算')
        row = obj({'label': {'type': 'string', 'minLength': 1, 'maxLength': 80},
                   'expression': {'type': 'string', 'minLength': 1, 'maxLength': 512}})
        self.calc_event = await self.call('calc', f'{self.materials}\n\n{self.ask_block(aspects=False)}\n\n{CALC}',
            {'calculations': {'type': 'array', 'items': row, 'maxItems': 8}}, 800)
        allowed = numeric_values(self.materials + '\n' + self.original_question + '\n' + self.question)
        lines = []
        for item in (self.calc_event['parsed'] or {}).get('calculations') or []:
            expression = (item.get('expression') or '').strip() if isinstance(item, dict) else ''
            entry = {'label': item.get('label') if isinstance(item, dict) else None, 'expression': expression}
            try:
                normalized = expression.replace('×', '*').replace('÷', '/')
                tree = ast.parse(normalized, mode='eval')
                literals = [decimal_literal(ast.get_source_segment(normalized, node)) for node in ast.walk(tree)
                            if isinstance(node, ast.Constant) and type(node.value) in (int, float)]
                if not re.search(r'[-+*/×÷]|\b(?:sqrt|log|ln|log10|exp|abs)\s*\(', expression.lstrip('-')):
                    raise ValueError('no operation')
                # Small integers are counts from the question (两个、三人); other numbers must be quoted.
                if not literals or any(n not in allowed
                                       and not (n == n.to_integral_value() and (n <= 12 or n in {Decimal(p) for p in POWERS}))
                                       for n in literals):
                    raise ValueError('number not in quoted material or question')
                value = evaluate(expression)
                entry['result'] = value
                # Default :g keeps six significant digits and changes large byte counts.
                # Give the Writer the same result retained in the calculation record.
                lines.append(f"{entry['label']}：{expression} = {entry['result']}")
            except (ValueError, SyntaxError, ZeroDivisionError, TypeError, OverflowError) as error:
                entry['error'] = str(error)
            self.calculations.append(entry)
        if lines:
            # Code guarantees the arithmetic, not that the expression fits the question.
            self.materials += ('\n\n代码按模型列出的算式计算的结果（算术正确；算式是否适用于问题，'
                               '依据上面原文判断，不适用就不要使用）：\n' + '\n'.join(lines))

    async def answer(self):
        blocks, numbers, ids = self.final_materials()
        self.emit('evidence_ready', source_ids=list(numbers.values()), quote_ids=ids,
                  citation_bindings={number: [sid] for number, sid in numbers.items()})
        self.materials = '以下是从各资料中原样摘抄的原文：\n\n' + '\n\n'.join(blocks)
        if self.calculation:
            await self.calculate()
        self.progress('依据摘抄原文生成回答')
        missing = self.missing_needs()
        concise = self.final_response_style == 'concise'
        gap = ('\n阅读阶段尚未选出直接回答句的信息项：' + '、'.join(self.needs[i - 1] for i in missing)
               + '。这不是事实缺失的判定。请检查全部所选回答句和细节句是否已经包含该项依据；依据不足才写未知。') if missing else ''
        instruction = final_instruction(concise=concise, has_calculations=(self.calculation
                                        and any('result' in row for row in self.calculations)))
        prompt = f'{self.materials}\n\n{self.ask_block(aspects=not concise)}{gap}\n\n{instruction}'
        schema = {'answer': {'type': 'string', 'minLength': 1, 'maxLength': 4000},
                  'points': {'type': 'array', 'minItems': 0, 'maxItems': 0 if concise else 8,
                             # Each point ends with its source numbers; the numbers are checked in code.
                             'items': {'type': 'string', 'pattern': POINT_PATTERN}},
                  'unknown': {'type': 'string', 'maxLength': 800}}
        self.final_event = await self.call('final', prompt, schema, 2048, evidence=ids,
                                           frequency_penalty=self.final_frequency_penalty)
        self.final_attempts = [self.final_event]
        repeated = bool(self.final_event['parsed'] and repeated_points(self.final_event['parsed']))
        if repeated:
            self.repetition_checks.append({'raw_sha256': self.final_event['raw_sha256'],
                'repeated_points': True, 'regeneration_requested': self.retry_repeated_points,
                'raw_text_modified': False})
        if repeated and self.retry_repeated_points:
            # Code never deletes model sentences; a repeated answer gets one new, separately traced call.
            self.progress('回答要点重复，重新生成一次')
            retry = await self.call('final', prompt, schema, 2048, evidence=ids,
                                    frequency_penalty=1.0, presence_penalty=0.6)
            retry['retry_of'] = self.final_event['raw_sha256']
            self.final_attempts.append(retry)
            if retry['parsed'] and not repeated_points(retry['parsed']):
                self.final_event = retry
        if self.final_event['parsed']:
            value = self.final_event['parsed']
            # Fields are joined in order; the model's words are not changed.
            lines = [value['answer'].strip(), *('· ' + p.strip() for p in value['points'] if isinstance(p, str))]
            if isinstance(value['unknown'], str) and value['unknown'].strip():
                lines.append('未找到依据：' + value['unknown'].strip())
            self.final_event['display_text'] = '\n'.join(lines)
        self.final_event.update(numbers=numbers, quote_ids=ids)

    async def review(self):
        """Independent check; findings are recorded beside the answer and never change it."""
        self.progress('核对回答与原文')
        answer = self.final_event['display_text']
        issue = obj({'kind': {'type': 'string', 'enum': ISSUES},
                     'detail': {'type': 'string', 'minLength': 1, 'maxLength': 200},
                     'sources': {'type': 'array', 'items': {'type': 'integer'}, 'maxItems': 6},
                     'evidence': {'type': 'string', 'maxLength': 300},
                     'answer_quote': {'type': 'string', 'maxLength': 200}})
        self.review_event = await self.call('review',
            f'{self.materials}\n\n{self.ask_block(aspects=False)}\n回答：\n{answer}\n\n{REVIEW}',
            {'issues': {'type': 'array', 'items': issue, 'maxItems': 4}}, 700, evidence=self.final_event['quote_ids'])

    async def run(self):
        if not await self.planning():
            return
        try:
            async with asyncio.timeout_at(self.reading_deadline):
                if self.relation != '新问题' and self.previous_sources:
                    # Follow-ups first re-read the previous turn's cited pages with the new needs.
                    for source in self.previous_sources:
                        self.reused_source_ids.append(source.id)
                        await self.publish(source)
                await self.acquire(self.queries, 'initial')
                self.timeline['acquisition_ms'] = (monotonic() - self.started) * 1000
                await self.drain()
                missing = self.missing_needs()
                # One supplementary web round, queried by the model-written need text.
                if missing and self.supplement and 'web' in self.channels:
                    extra = [self.needs[i - 1] for i in missing if self.needs[i - 1] not in self.queries]
                    if extra:
                        self.progress(f'补查 {len(extra)} 个缺少原文的信息项')
                        await self.acquire(extra, 'supplement')
                        await self.drain()
        except TimeoutError:
            self.errors.append({'stage': 'reading', 'status': 'deadline'})
        finally:
            for task in list(self.tasks):
                task.cancel()
            await asyncio.gather(*list(self.tasks), return_exceptions=True)
        self.timeline['reading_done_ms'] = (monotonic() - self.started) * 1000
        self.emit('reading_finished', source_count=len(self.sources), read_calls=len(self.reads),
                  bound_quotes=len(self.quotes), unbound_quotes=len(self.unbound))
        if not self.quotes and not self.calculation:
            return
        material = ' '.join(s.title + ' ' + v['body'] + ' ' + v['snippet']
                            for s in self.sources if (v := self.views.get(s.id)))
        self.name_absent = [n for n in self.names if not name_present(n, material)]
        # Literal absence is an audit observation, not proof of a different object.
        # The Reader and Writer decide identity from the original evidence.
        # A selected definition/condition can support a derived answer. Direct-quote
        # coverage is diagnostic; the model decides whether the premises suffice.
        try:
            async with asyncio.timeout_at(self.deadline):
                await self.answer()
                self.timeline['answer_ms'] = (monotonic() - self.started) * 1000
                if self.review_enabled and self.final_event['parsed']:
                    await self.review()
        except TimeoutError:
            self.errors.append({'stage': 'review' if self.final_event else 'final', 'status': 'deadline'})

    def unsupported_numbers(self, value):
        """Numbers written in the answer or points that occur nowhere in the offered quotes or the date line."""
        haystack = numeric_view(self.materials + self.today_line() + self.calendar_line(years=False) + self.original_question)
        found = []
        for line in [value.get('answer') or '', *(p for p in value.get('points') or [] if isinstance(p, str))]:
            text = re.sub(r'(?<=\d)[,，](?=\d{3})', '', CITATION.sub('', line))
            missing = [n for n in re.findall(r'\d+(?:\.\d+)?', text) if len(n) >= 2 and n not in haystack]
            if missing:
                found.append({'text': line, 'numbers': list(dict.fromkeys(missing))})
        return found

    def review_result(self):
        event = self.review_event
        if event is None:
            return {'status': 'not_run', 'issues': []}
        numbers = self.final_event['numbers']
        bound, unbound = [], []
        for item in (event['parsed'] or {}).get('issues') or []:
            if not isinstance(item, dict):
                continue
            row = {**item, 'source_ids': [numbers[str(n)] for n in item.get('sources', []) if str(n) in numbers]}
            # A finding counts only when it quotes the answer and the material verbatim.
            evidence = (item.get('evidence') or '').strip()
            quoted = (item.get('answer_quote') or '').strip()
            ok = (len(evidence) >= 4 and bind_quote(self.materials, evidence)
                  and len(quoted) >= 2 and bind_quote(self.final_event['display_text'], quoted))
            if ok and item.get('kind') == ISSUES[3]:
                # "Not in the material": every number in the quoted answer part present there contradicts it.
                numbers_in = re.findall(r'\d+(?:\.\d+)?', quoted)
                ok = not numbers_in or any(n not in self.materials for n in numbers_in)
            (bound if ok else unbound).append(row)
        return {'status': event['status'] if event['parsed'] is not None else 'invalid_output',
                'issues': bound, 'unbound_issues': unbound, 'raw_text': event['raw_text'], 'changes_answer': False}

    def result(self):
        event = self.final_event
        parsed = (event or {}).get('parsed')
        sources = {s.id: s for s in self.sources}
        review = self.review_result()
        # Structural only: with no direct quote for any need, points come from detail quotes
        # that may describe a similarly named object; the UI says so, the answer is unchanged.
        off_object = bool(parsed and parsed.get('points') and len(self.missing_needs()) == len(self.needs))
        unsupported = self.unsupported_numbers(parsed) if parsed else []
        if parsed:
            text = event['display_text']
            numbers = event['numbers']
            cited = list(dict.fromkeys(n for n in citation_numbers(text) if n in numbers))
            invalid = sorted(set(citation_numbers(text)) - set(numbers))
            used = [numbers[n] for n in cited]
            snippet_only = any(q['section'] == 'snippet' and q['source_id'] in used
                               for q in self.quotes.values() if q['id'] in event['quote_ids'])
            kind = ('unverified' if snippet_only or not cited or review['issues'] or off_object or unsupported
                    else ('partial' if self.missing_needs() else 'answered'))
            answer = {'answer': event['raw_text'], 'sources': [sources[i].model_dump(mode='json') for i in used],
                      'generation': {'protocol': PROTOCOL, 'status': event['status'],
                          'parsed_answer': {'answer': text, 'source_ids': cited},
                          'raw_model_answer': event['raw_text'], 'citation_bindings': {n: [numbers[n]] for n in numbers},
                          'invalid_citations': invalid, 'semantic_support_verified': False, 'answer_modified': False}}
            verification = ('review_flagged' if review['issues'] else
                            'includes_search_snippets' if snippet_only else 'original_body')
        else:
            failures = [e for e in self.errors if e.get('status') not in BENIGN]
            kind = 'error' if failures and not self.quotes else ('unverified' if event else 'not_found')
            no_direct = any(e.get('status') == 'no_direct_quote' for e in self.errors)
            text = (event or {}).get('raw_text') or (
                '本次阅读没有选出直接回答所问信息的原文，暂无法依据资料回答。' if no_direct
                else '本次检索未找到能回答问题的原文。' if not failures
                else '本次资料获取或阅读未完成，暂无法依据资料回答。')
            answer = {'answer': text, 'sources': [], 'generation': {'protocol': PROTOCOL,
                      'status': (event or {}).get('status', 'no_evidence'), 'parsed_answer': None,
                      'raw_model_answer': (event or {}).get('raw_text'), 'citation_bindings': {}}}
            verification = 'no_selected_evidence'
        calls = [e for e in [self.plan_event, *self.reads, self.calc_event, *self.final_attempts, self.review_event] if e]
        return {'answer': answer, 'response_type': kind, 'display_text': text, 'as_of': self.as_of,
            'relation': self.relation, 'resolved_question': self.question, 'queries': self.queries,
            'needs': self.needs, 'need_bindings': self.need_bindings, 'dropped_needs': self.dropped_needs,
            'aspects': self.aspects, 'entities': self.entities, 'plan': self.plan_event, 'errors': self.errors,
            'review': review, 'points_without_direct_quote': off_object, 'unsupported_numbers': unsupported,
            'repetition_checks': self.repetition_checks, 'observation_errors': self.observation_errors,
            'name_absent_quotes': self.name_absent, 'metrics': self.metrics, 'names': self.names,
            'assumptions': self.assumptions, 'calculations': self.calculations, 'history_used': self.history, 'reused_source_ids': self.reused_source_ids,
            'acquisition': self.acquisition, 'acquisition_incomplete': any(e['stage'] == 'acquire' and e.get('status') not in BENIGN for e in self.errors),
            'not_found_needs': [self.needs[i - 1] for i in self.missing_needs()],
            'evidence_verification': verification, 'rounds': [],
            'graph': {'nodes': [{k: v for k, v in e.items() if k != 'trace'} for e in calls],
                      'read_batch_checks': self.read_batch_checks,
                      'reading_views': self.views, 'spans': list(self.quotes.values()),
                      'unbound_quotes': self.unbound,
                      'source_snapshots': [s.model_dump(mode='json') for s in self.sources]},
            'telemetry': {'total_ms': (monotonic() - self.started) * 1000, 'total_calls': len(calls),
                'read_calls': len(self.reads), 'source_count': len(self.sources),
                'bound_quotes': len(self.quotes), 'unbound_quotes': len(self.unbound),
                'answer_chars': len(text), **self.timeline}}


async def answer(model, question, **kwargs):
    pipeline = Flat(model, question, **kwargs)
    try:
        async with asyncio.timeout_at(pipeline.deadline):
            await pipeline.run()
    except TimeoutError:
        pipeline.errors.append({'stage': 'request', 'status': 'timeout'})
    return pipeline.result()
