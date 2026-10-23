"""Flat quote pipeline: plan once, read sources once, answer once, review once.

The model decides the turn relation, needs, queries, which sentences answer a
need or add detail, the final answer and the review findings. Code only
fetches, cleans markup, windows text, binds quotes and need wording to exact
character offsets, groups quotes and formats times. No model-written summary
is passed between calls or between conversation turns.
"""
import ast
import asyncio
from datetime import datetime, timedelta
from difflib import SequenceMatcher
import html
import json
from pathlib import PurePath
import re
from time import monotonic
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from llama_index.core import Document

from .interactive_sources import add_sources, digest, gather, obj, unique_object
from .verbatim_chunking import packed_verbatim_nodes

PROTOCOL = 'interactive-flat-quotes-v2'
CLEANING = 'markdown-links-images-v1'
WEEKDAYS = '一二三四五六日'
RELATIONS = ['新问题', '追问', '更正']
ISSUES = ['相反或不一致的原文没有提到', '数值单位或计量基准与原文不一致', '把其他对象的内容当作所问对象', '回答中有原文找不到的内容']
PLAN = ('判断用户这一轮是新问题、追问还是更正，relation写“新问题”“追问”或“更正”：原话中有“它”“这个”“那”“呢”等指代、'
        '或省略了所问对象时是追问；原话自己写明了一个与前几轮无关的对象时是新问题。'
        'question写完整的问题：追问和更正要把指代换成前几轮中的具体对象（例如具体型号或版本号），question中不能再出现“它”“这个”等指代；'
        '新问题不受前几轮影响。needs列出用户实际问到的内容，通常1到3项；用户要求分别说明几个对象或比较几个方面时，'
        '每个对象或方面各列一项，最多4项。每项写text和user_words，'
        'user_words必须从“用户本轮原话”中原样照抄一段文字，表示这一项来自原话的哪里。'
        'aspects写回答时值得补充的相关方面，例如时间、条件、背景、对比，最多4项，可以为空。'
        'queries写1到2条互补的检索句，每条都写出要查的对象；问到今天、最新等时写明具体日期或年份。'
        'entities只写所问的具体对象名称（产品、型号、地点、机构、人物），比较类问题写出每个被比较的对象，追问时写前几轮中被指代的具体对象；'
        'metrics写所问的数值指标，例如中间价、收盘价、续航里程、门票价格，没有就为空。'
        'names只写entities中的专有名称（具体产品型号、公司机构、地名、人名），不写“新赛季”“总冠军”“大模型”这类泛称，没有就为空。'
        'assumptions写你为理解问题所做的假设：名称可能指多种事物时写明按哪一种理解，问题依赖地区或版本而原话没说时写明按哪个理解，没有就为空。'
        'calculation填true表示回答需要用找到的数字做计算（换算、求差、求和、相除等），否则填false。不要回答问题。')
READ = ('只从上面各份资料“正文开始”到“正文结束”之间的文字里摘抄。逐字照抄原文中连续的一段文字，不改写、不拼接、不总结，'
        '不摘抄标题、日期说明或信息项本身。对每份资料的每个信息项：answer_quotes摘抄直接回答该项的句子，最多3句；'
        'detail_quotes摘抄同一对象的相关细节，例如时间、数值和单位、条件、原因、背景、对比或不同说法，最多5段，每段可以是连续的两三句。'
        '只摘抄确实在说所问对象的内容，名称不同的其他对象不要摘抄；没有相关内容就让列表为空。')
FINAL = ('依据上面摘抄的原文回答问题。answer用一两句话直接回答用户所问；points逐条写出原文中与问题相关的其他要点，'
         '例如时间、数值、条件、原因、对比、适用范围和各来源说法的差异，原文充分时写3到6条。每条要点用中文自己的话概括一个与问题直接相关的事实，'
         '与answer和其他要点的内容都不相同，句末标注来源编号；不要照抄整段外文原文，不写与所问对象无关的内容；'
         '问“最新”时以版本号最大或日期最晚的为准；要点用中文写，不照抄外文段落，不写“原文没有提供……”这类说明；'
         '有“本次理解的假设”时，answer开头先写明按哪种理解回答；'
         '比较类问题先分别写出每个对象对应的数值或特点并标注来源，再给出比较结论，结论中的数值必须与这些数值一致；'
         '某个对象没有原文时说明无法比较，不能用别的对象代替；'
         '“不支持”“没有”“未发布”这类否定也必须有原文明确写出，原文没有提到时写未知，不能写成否定；'
         '原文后〔〕中是代码计算的日期判断，以它为准；'
         'unknown写没有任何原文支持的信息项，全部有支持时为空字符串。'
         '每个事实后用[1]、[2]这样的来源编号标注。数值照原文写出单位和计量基准，例如“每100美元”。'
         '几条原文说法不同时，先看是否指同一对象、同一版本、同一时间：不同版本或不同时间的不是冲突，以最新的为准并写出版本或时间；'
         '同一对象同一时间却说法不同时，写出各来源的说法并说明不一致。原文说的对象与所问对象名称不一致时，不能当作所问对象的事实；'
         '原文数值对应的指标与所问指标不同时（例如问中间价、原文是牌价），不能作为answer，可在要点中写明它是哪种指标。'
         '找不到所问对象本身的原文时，answer写未知，points只能写原文中关于所问对象本身的内容，没有就让points为空。'
         '只用摘抄原文中的内容，不补充常识，不重复同一句话，不复制资料标题或“今天是”这类说明。引用标有搜索摘要的原文时注明（据搜索摘要）。')
CALC = ('回答问题需要用上面原文中的数字做计算。为每个需要计算的结果写label和expression：expression只用阿拉伯数字和+ - * / 括号，'
        '数字必须是原文或问题中出现的数字（单位换算系数也必须出现在原文或问题中），形式如“单价*人数+另一单价”，用上面原文中的实际数字代入。'
        '只写确实需要计算的结果，原文已直接给出的数字不用写。不要写计算结果，代码会计算。')
REVIEW = ('对照上面的原文检查这个回答，只找确实存在的以下问题：' + '；'.join(f'{i}. {k}' for i, k in enumerate(ISSUES, 1)) +
          '。每条写kind、detail、相关来源编号sources，在answer_quote中逐字照抄回答里有问题的那一句，并在evidence中逐字照抄能证明这个问题的一句原文。'
          '回答与原文一致的地方不是问题；日期、数值写法不同但意思相同不是问题。大多数回答没有问题，没有问题就让issues为空。不补充常识。')
IMAGE = re.compile(r'!\[[^\]]*\]\([^)]*\)')
LINK = re.compile(r'\[([^\]]*)\]\((?:[^()]|\([^)]*\))*\)')
JUNK_LINE = re.compile(r'^\s*(?:javascript:\S*|https?://\S+|\[\s*\]|[-*•|]\s*)\s*$')
EMPHASIS = re.compile(r'\*\*|__|(?<!\w)\*(?=\S)|(?<=\S)\*(?!\w)')
POINT_PATTERN = r'^[^\[\]\n]{4,300}(\[[0-9]{1,2}\])+[。]?$'
POWERS = {'100', '1000', '10000', '100000000'}
BENIGN = {'duplicate_source_title_changed', 'windows_limited', 'quotes_limited'}
CITATION = re.compile(r'\[(\d{1,3})\]')
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
    positions = [i for i, char in enumerate(text) if keep(char)]
    return positions, ''.join(text[i] for i in positions).lower()


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
    """Literal check that an object name occurs: whole name, or for long names a 60% contiguous part."""
    name, text = _normal(entity), _normal(text)
    if not name:
        return True
    if name in text:
        return True
    size = max(4, -(-len(name) * 3 // 5))
    # Short names (brand + generic word, e.g. "栖桐冻干猫粮") must match whole; long names may drop a
    # qualifier ("西安…"), and a window that is mostly a model code (e.g. "magicbookpro16") also counts.
    return any(name[i:i + size] in text for i in range(len(name) - size + 1)
               if len(name) >= 8 or sum(c.isascii() and c.isalnum() for c in name[i:i + size]) >= 4)


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
    text = re.sub(r'(?<=\d)[,，\s](?=\d{3})', '', text)
    return text + ' ' + ' '.join(str(cn_to_int(m)) for m in CN_NUMBER.findall(text))


def evaluate(expression):
    """Arithmetic on literal numbers only: + - * / and parentheses."""
    tree = ast.parse(expression.replace('×', '*').replace('÷', '/'), mode='eval')
    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub, ast.Mult, ast.Div):
            a, b = walk(node.left), walk(node.right)
            return {ast.Add: a + b, ast.Sub: a - b, ast.Mult: a * b}.get(type(node.op)) if type(node.op) is not ast.Div else a / b
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -walk(node.operand)
        raise ValueError('unsupported expression')
    return walk(tree)


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
                 supplement=True, review=True, kb_window_chars=6000, kb_max_windows=40):
        if (not set(channels) <= {'knowledge_base', 'web'} or not 400 <= window_chars <= 24000
                or not 0 <= window_overlap < window_chars // 2 or max_windows_per_source < 1
                or not 1 <= snippet_batch <= 8 or not 0 < final_reserve_seconds < deadline_seconds):
            raise ValueError('invalid flat pipeline options')
        self.model, self.original_question, self.question = model, question, question
        self.channels, self.search, self.select_files = list(dict.fromkeys(channels)), search, select_files
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
        self.reads, self.quotes, self.unbound = [], {}, []
        self.pending_snippets, self.flush_task = [], None
        self.tasks = set()
        self.plan_event = self.final_event = self.review_event = None
        self.final_attempts = []
        self.reused_source_ids = []
        self.timeline = {}

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
        return f'{self.today_line()}\n{self.calendar_line()}\n问题：{self.question}{objects}{metrics}{assumed}\n信息项：\n{needs}{extra}'

    def header(self, source):
        meta = source.metadata
        place = '网页 ' + (urlsplit(source.uri or '').hostname or '') if source.source == 'web' else '上传资料'
        parts = [source.title[:120], place]
        if source.source == 'web':
            published = local_time(meta.get('published_date') or '', self.zone)
            parts.append('发布：' + (f'{published:%Y-%m-%d}（{relative_day(published, self.today)}）' if published else '未知'))
            fetched = local_time(meta.get('retrieved_at') or '', self.zone)
            if fetched:
                parts.append(f'抓取：{fetched:%Y-%m-%d %H:%M}（{relative_day(fetched, self.today)}）')
            if meta.get('content_status') == 'snippet_only':
                parts.append('正文未能获取，仅有搜索摘要')
        return '｜'.join(parts)

    def short_header(self, source):
        if source.source != 'web':
            return f'{source.title[:80]}（上传资料）'
        fetched = local_time(source.metadata.get('retrieved_at') or '', self.zone)
        when = f'，抓取于{relative_day(fetched, self.today)}' if fetched else ''
        return f"{source.title[:80]}（网页 {urlsplit(source.uri or '').hostname or ''}{when}）"

    async def call(self, stage, prompt, properties, limit, evidence=(), frequency_penalty=0.0, presence_penalty=0.0):
        result = await self.model.complete([{'role': 'user', 'content': prompt}],
            structured_schema=obj(properties), max_tokens=limit,
            assistant_prefill='<think></think', temperature=0, seed=11, frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stage='writer' if stage in {'final', 'review'} else 'reader', evidence_ids=list(evidence))
        event = {'stage': stage, 'status': result.status, 'raw_text': result.raw_text,
                 'raw_sha256': digest(result.raw_text or ''), 'parsed': None, 'trace': result.trace,
                 'finished_ms': (monotonic() - self.started) * 1000}
        if result.status == 'completed':
            try:
                value = json.loads(result.raw_text, object_pairs_hook=unique_object)
                if not isinstance(value, dict) or set(value) != set(properties):
                    raise ValueError('invalid fields')
                event['parsed'] = value
            except (ValueError, TypeError) as error:
                event['validation_error'] = str(error)
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
            'question': {'type': 'string', 'minLength': 1, 'maxLength': 400},
            'needs': {'type': 'array', 'items': need, 'minItems': 1, 'maxItems': 4},
            'aspects': {**strings, 'maxItems': 4},
            'queries': {**strings, 'minItems': 1, 'maxItems': 2},
            'entities': {**strings, 'maxItems': 8}, 'metrics': {**strings, 'maxItems': 4},
            'names': {**strings, 'maxItems': 6}, 'assumptions': {**strings, 'maxItems': 3},
            'calculation': {'type': 'boolean'}}, 600)
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
        self.needs = list(dict.fromkeys(b['text'] for b in self.need_bindings))[:4]
        self.aspects = list(dict.fromkeys(a.strip() for a in value['aspects'] if a.strip()))[:4]
        self.queries = list(dict.fromkeys(q.strip() for q in value['queries'] if q.strip()))[:2]
        self.entities = list(dict.fromkeys(e.strip() for e in value['entities'] if e.strip()))
        self.metrics = list(dict.fromkeys(m.strip() for m in value['metrics'] if m.strip()))
        # Only names that occur in the user's turn or the history are literal-checked.
        context = self.original_question + ''.join(t.get('question', '') + ''.join(t.get('entities') or []) for t in self.history)
        self.names = [n.strip() for n in value['names'] if n.strip() and (name_present(n, context) or n.strip() in self.entities)]
        if not self.names:
            # Structural fallback: entities written literally in the user's message are names to check.
            self.names = [e for e in self.entities if len(_normal(e)) >= 2 and _normal(e) in _normal(self.original_question)]
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
        # Any asked numeric indicator may need arithmetic; the calc call may return nothing.
        self.calculation = value['calculation'] is True or bool(self.metrics)
        if self.relation != '新问题' and self.history:
            # Follow-up queries must carry the referenced object; add it when the plan dropped it.
            previous = [e for e in self.history[-1].get('entities') or [] if e.strip()]
            if previous and not any(name_present(e, q) for e in previous for q in self.queries):
                self.queries = [f'{previous[0]} {q}' for q in self.queries]
        if not self.queries:
            self.errors.append({'stage': 'plan', 'status': 'empty_plan'})
            return False
        self.timeline['plan_ms'] = (monotonic() - self.started) * 1000
        return True

    # ---------- acquisition and reading ----------
    async def publish(self, source):
        count = len(self.sources)
        try:
            add_sources(self.sources, [source])
        except ValueError:
            # Same URL and text returned under another title: keep the first observation.
            self.errors.append({'stage': 'acquire', 'status': 'duplicate_source_title_changed', 'source_id': source.id})
            return
        if len(self.sources) == count:
            return
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
        if view.strip():
            nodes = list(packed_verbatim_nodes(Document(text=view, id_=source.id), size,
                                               self.window_overlap, absorb_whitespace=True))
            windows = [(n.metadata['source_span']['start'], n.metadata['source_span']['end']) for n in nodes]
        if len(windows) > limit:
            self.errors.append({'stage': 'read', 'status': 'windows_limited', 'source_id': source.id,
                                'windows': len(windows), 'read': limit})
            windows = windows[:limit]
        if not windows:
            if snippet:
                # Short snippet-only sources share one reading call.
                self.pending_snippets.append((source, None, True))
                if len(self.pending_snippets) >= self.snippet_batch:
                    self.flush_snippets()
                elif self.flush_task is None:
                    self.flush_task = self.spawn(self.delayed_flush())
            return
        for index, window in enumerate(windows):
            self.spawn(self.read([(source, window, index == 0 and bool(snippet))]))
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
        prompt = '\n\n'.join(parts) + f'\n\n{self.ask_block()}\n\n{READ}'
        def quotes(count, size):
            return {'type': 'array', 'maxItems': count, 'items': {'type': 'string', 'minLength': 2, 'maxLength': size}}
        item = obj({'source': {'type': 'integer', 'enum': list(range(1, len(segments) + 1))},
                    'need': {'type': 'integer', 'enum': list(range(1, len(self.needs) + 1))},
                    'answer_quotes': quotes(3, 240), 'detail_quotes': quotes(5, 360)})
        ids = [s.id for s, _, _ in segments]
        try:
            async with asyncio.timeout_at(self.reading_deadline):
                event = await self.call('read', prompt, {'items': {'type': 'array', 'items': item,
                    'maxItems': len(self.needs) * len(segments)}}, 1400, evidence=ids)
        except TimeoutError:
            self.errors.append({'stage': 'read', 'status': 'deadline', 'source_ids': ids})
            return
        event.update(source_ids=ids, windows=[w for _, w, _ in segments], quote_ids=[])
        self.reads.append(event)
        if event['parsed'] is None:
            self.errors.append({'stage': 'read', 'status': event['status'], 'source_ids': ids})
            return
        for row in event['parsed']['items'] if isinstance(event['parsed']['items'], list) else []:
            if (not isinstance(row, dict) or row.get('need') not in range(1, len(self.needs) + 1)
                    or row.get('source') not in range(1, len(segments) + 1)):
                continue
            for kind in ('answer', 'detail'):
                for quote in row.get(kind + '_quotes') or []:
                    if isinstance(quote, str) and quote.strip():
                        # The quote may come from another source shown in the same call.
                        order = [row['source'] - 1] + [i for i in range(len(segments)) if i != row['source'] - 1]
                        if not any(self.bind(event, segments[i], row['need'], kind, quote, record=False) for i in order):
                            self.unbound.append({'source_id': segments[row['source'] - 1][0].id, 'need': row['need'],
                                                 'kind': kind, 'quote': quote, 'read_sha256': event['raw_sha256']})
        self.progress(f'取材与阅读 · {len(self.sources)} 份资料 · 已读 {len(self.reads)}')

    def bind(self, event, segment, need, kind, quote, record=True):
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
        for matcher in (whole, bind_fragments):
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
                    event['quote_ids'].append(identity)
                if hits:
                    return True
        if record:
            self.unbound.append({'source_id': source.id, 'need': need, 'kind': kind, 'quote': quote,
                                 'read_sha256': event['raw_sha256']})
        return False

    async def acquire(self, queries, round_name):
        async def one(query):
            receipt = {'round': round_name, 'channel': 'web', 'query': query, 'status': 'pending'}
            self.acquisition.append(receipt)
            try:
                async with asyncio.timeout(self.acquisition_timeout):
                    for source in await self.search(query):
                        await self.publish(source)
                receipt['status'] = 'completed'
            except Exception as error:
                receipt.update(status='timeout' if isinstance(error, TimeoutError) else 'failed',
                               error=getattr(error, 'safe_code', type(error).__name__))
                self.errors.append({'stage': 'acquire', **receipt})

        async def files():
            receipt = {'round': round_name, 'channel': 'knowledge_base', 'status': 'pending'}
            self.acquisition.append(receipt)
            try:
                async with asyncio.timeout(self.acquisition_timeout):
                    chosen = await self.select_files(self.entities, self.queries, self.publish)
                receipt.update(status='completed', **chosen)
            except Exception as error:
                receipt.update(status='failed', error=type(error).__name__)
                self.errors.append({'stage': 'acquire', **receipt})
        jobs = [one(q) for q in queries] if 'web' in self.channels else []
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
        """Model writes expressions from quoted numbers; code checks the numbers are quoted and computes."""
        self.progress('依据原文数字计算')
        row = obj({'label': {'type': 'string', 'minLength': 1, 'maxLength': 80},
                   'expression': {'type': 'string', 'minLength': 1, 'maxLength': 120}})
        self.calc_event = await self.call('calc', f'{self.materials}\n\n{self.ask_block(aspects=False)}\n\n{CALC}',
            {'calculations': {'type': 'array', 'items': row, 'maxItems': 4}}, 400)
        allowed = numeric_view(self.materials + self.original_question + self.question)
        lines = []
        for item in (self.calc_event['parsed'] or {}).get('calculations') or []:
            expression = (item.get('expression') or '').strip() if isinstance(item, dict) else ''
            entry = {'label': item.get('label') if isinstance(item, dict) else None, 'expression': expression}
            try:
                literals = re.findall(r'\d+(?:\.\d+)?', expression)
                if not re.search(r'[-+*/×÷]', expression.lstrip('-')):
                    raise ValueError('no operation')
                # Small integers are counts from the question (两个、三人); other numbers must be quoted.
                if not literals or any(n not in allowed and not (n.isdigit() and (int(n) <= 12 or n in POWERS))
                                       for n in literals):
                    raise ValueError('number not in quoted material or question')
                value = evaluate(expression)
                entry['result'] = round(value, 4) if isinstance(value, float) else value
                lines.append(f"{entry['label']}：{expression} = {entry['result']:g}")
            except (ValueError, SyntaxError, ZeroDivisionError, TypeError) as error:
                entry['error'] = str(error)
            self.calculations.append(entry)
        if lines:
            # Code guarantees the arithmetic, not that the expression fits the question.
            self.materials += ('\n\n代码按模型列出的算式计算的结果（算术正确；算式是否适用于问题，'
                               '依据上面原文判断，不适用就不要使用）：\n' + '\n'.join(lines))

    async def answer(self):
        blocks, numbers, ids = self.final_materials()
        self.materials = '以下是从各资料中原样摘抄的原文：\n\n' + '\n\n'.join(blocks)
        if self.calculation:
            await self.calculate()
        if len(self.names) >= 2:
            # Comparison: code lists, per asked name, which numbered sources literally mention it.
            sources = {s.id: s for s in self.sources}
            rows = []
            for name in self.names:
                hits = [n for n, sid in numbers.items() if any(
                    name_present(name, q['text']) or name_present(name, sources[sid].title)
                    for q in self.quotes.values() if q['source_id'] == sid and q['id'] in ids)]
                rows.append(f"{name}：{'来源' + '、'.join(f'[{n}]' for n in hits) if hits else '摘抄中没有写明它的原文'}")
            self.materials += '\n\n所问对象与写明它的来源（代码按名称统计）：\n' + '\n'.join(rows)
        self.progress('依据摘抄原文生成回答')
        # Enough quoted material requires points; thin material may answer in one sentence.
        missing = self.missing_needs()
        absent = getattr(self, 'name_absent', [])
        gone = ('\n以下对象在本次获取的所有资料中都没有出现，关于它们只能写未知，不能写出任何数值或特点：' + '、'.join(absent)) if absent else ''
        # Points are required only when quoted material exists for every need and every asked object.
        least = 2 if len(ids) >= 4 and not missing and not absent else 0
        gap = ('\n以下信息项没有找到直接写明所问对象的原文，只能写未知：' + '、'.join(self.needs[i - 1] for i in missing)) if missing else ''
        prompt = f'{self.materials}\n\n{self.ask_block()}{gap}{gone}\n\n{FINAL}'
        schema = {'answer': {'type': 'string', 'minLength': 1, 'maxLength': 600},
                  'points': {'type': 'array', 'minItems': least, 'maxItems': 6,
                             # Each point ends with its source numbers; the numbers are checked in code.
                             'items': {'type': 'string', 'pattern': POINT_PATTERN}},
                  'unknown': {'type': 'string', 'maxLength': 300}}
        self.final_event = await self.call('final', prompt, schema, 1600, evidence=ids,
                                           frequency_penalty=self.final_frequency_penalty)
        self.final_attempts = [self.final_event]
        if self.final_event['parsed'] and repeated_points(self.final_event['parsed']):
            # Code never deletes model sentences; a repeated answer gets one new, separately traced call.
            self.progress('回答要点重复，重新生成一次')
            retry = await self.call('final', prompt, schema, 1600, evidence=ids,
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
            f'{self.materials}\n\n{self.today_line()}\n问题：{self.question}\n回答：\n{answer}\n\n{REVIEW}',
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
        if not self.quotes:
            return
        material = ' '.join(s.title + ' ' + v['body'] + ' ' + v['snippet']
                            for s in self.sources if (v := self.views.get(s.id)))
        self.name_absent = [n for n in self.names if not name_present(n, material)]
        if self.names and len(self.name_absent) == len(self.names):
            # No fetched page literally names any asked object: the object was not found,
            # so quotes about similarly named objects never become an answer.
            self.errors.append({'stage': 'final', 'status': 'no_direct_quote', 'names_absent': self.name_absent})
            return
        if len(self.missing_needs()) == len(self.needs):
            # No sentence directly answers any need (often a fictional object matched to a similar one):
            # no answer is generated, so detail quotes about other objects never become an answer.
            self.errors.append({'stage': 'final', 'status': 'no_direct_quote'})
            return
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
            cited = list(dict.fromkeys(n for n in CITATION.findall(text) if n in numbers))
            invalid = sorted(set(CITATION.findall(text)) - set(numbers))
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
                '本次检索没有找到直接写明所问对象的原文，无法回答；找到的资料说的是其他对象。' if no_direct
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
            'name_absent_quotes': self.name_absent, 'metrics': self.metrics, 'names': self.names,
            'assumptions': self.assumptions, 'calculations': self.calculations, 'history_used': self.history, 'reused_source_ids': self.reused_source_ids,
            'acquisition': self.acquisition, 'acquisition_incomplete': any(e['stage'] == 'acquire' and e.get('status') not in BENIGN for e in self.errors),
            'not_found_needs': [self.needs[i - 1] for i in self.missing_needs()],
            'evidence_verification': verification, 'rounds': [],
            'graph': {'nodes': [{k: v for k, v in e.items() if k != 'trace'} for e in calls],
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
