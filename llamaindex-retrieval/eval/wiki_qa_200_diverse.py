#!/usr/bin/env python3
"""Run a source-grounded, style-diverse 200-question production evaluation."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from opencc import OpenCC


REFUSALS = ("无法确定", "未检索到", "无法从资料", "无法回答")
T2S = OpenCC("t2s")
CITATION = re.compile(r"\[资料\s*[1-9]\d*\]")
YEAR = re.compile(r"(?<!\d)((?:公元前)?(?:1\d{3}|20\d{2}|\d{3})年(?:\d{1,2}月(?:\d{1,2}日)?)?)")
NUMBER = re.compile(r"(?<!\d)(\d[\d,]*(?:\.\d+)?(?:万|亿|千)?(?:人|平方公里|公里|米|个|座|名|届|次))")
BARE_NUMBER = re.compile(r"(?<![\d资料 ])(\d[\d,]*(?:\.\d+)?(?:万|亿|千)?)(?!\d)")
SENTENCE = re.compile(r"[^。！？!?\n]{10,420}[。！？!?]?")
NOISE = ("Category:", "参考资料", "外部链接", "参见", "延伸阅读", "资料来源")
BAD_TITLE_MARKERS = ("列表", "年新加坡", "年中国", "年香港", "年美国")
BAD_SECTION_MARKERS = ("图辑", "圖輯", "参考", "參考", "注释", "註釋", "外部链接", "延伸阅读")
LOCATION_MARKERS = ("位于", "位於", "坐落于", "坐落於", "地处", "地處")
TIME_ACTIONS = {
    "成立": ("成立", "创立", "創立", "创建", "創建", "建立", "组建", "組建"),
    "出生": ("出生", "生于", "生於"),
    "逝世": ("逝世", "去世", "病逝", "卒于", "卒於"),
    "开通": ("开通", "開通", "启用", "啟用", "通车", "通車"),
    "上映": ("上映", "首映", "发行", "發行"),
}
AGENT_ACTIONS = (
    "创建", "創建", "创立", "創立", "创办", "創辦", "建立", "发明", "發明", "发现", "發現",
    "设计", "設計", "建造", "导演", "導演", "执导", "執導", "创作", "創作", "撰写", "撰寫",
    "提出", "发起", "發起", "领导", "領導", "主持", "组织", "組織", "制作", "製作",
    "开发", "開發",
)
ENTITY_PRONOUNS = ("该型", "該型", "该建筑", "該建築", "该作品", "該作品", "本片", "本书", "本書", "该片", "該片")
AGENT_RELATION = re.compile(
    rf"^.{{0,100}}?(?:是由|由)(?P<agent>[^，,。；;（）()\n]{{2,48}}?)(?:所)?"
    rf"(?P<action>{'|'.join(map(re.escape, AGENT_ACTIONS))})"
    r"(?=于|於|的|，|,|。|；|;|、|和|及|并|並|$)"
)


def request_json(url: str, payload: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def clean(value: str) -> str:
    value = re.sub(r"\[[0-9]{1,4}\]", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def content(source: dict[str, Any]) -> str:
    text = str(source.get("text") or "")
    title = str(source.get("title") or "").strip()
    first, separator, rest = text.partition("\n")
    if separator and (first.strip() == title or first.startswith(f"{title} >")):
        text = rest
    return text.strip()


def sentences(source: dict[str, Any]) -> list[str]:
    return [
        clean(match.group(0))
        for match in SENTENCE.finditer(content(source)[:6000])
        if not any(marker in match.group(0) for marker in NOISE)
    ]


def stable_choice(values: tuple[str, ...], key: str) -> str:
    digest = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16)
    return values[digest % len(values)]


def style_question(question_type: str, title: str, detail: str = "") -> str:
    styles: dict[str, tuple[str, ...]] = {
        "definition": (
            f"{title}是什么？",
            f"请简要介绍一下{title}。",
            f"能通俗说说{title}吗？",
            f"我不太了解{title}，它指的是什么？",
            f"关于{title}，用两三句话说明一下。",
            f"你知道{title}是干什么的吗？",
        ),
        "location": (
            f"{title}位于哪里？",
            f"请问{title}在什么地方？",
            f"{title}具体坐落在哪儿？",
            f"你知道{title}属于哪里吗？",
            f"想去{title}，它的位置在哪里？",
        ),
        "time": (
            f"{title}是哪一年{detail}的？",
            f"请问{title}什么时候{detail}？",
            f"{title}{detail}的具体时间是什么？",
            f"你知道{title}何时{detail}吗？",
            f"说下{title}{detail}的年份。",
        ),
        "agent": (
            f"{title}是谁{detail}的？",
            f"请问谁{detail}了{title}？",
            f"{title}的{detail}者是谁？",
            f"你知道{title}由谁{detail}吗？",
            f"说下{title}背后的{detail}者。",
        ),
        "list": (
            f"{title}的{detail}有哪些？",
            f"请列出{title}中关于{detail}的内容。",
            f"{title}在{detail}方面都包括什么？",
            f"能整理一下{title}的{detail}列表吗？",
            f"我想知道{title}有哪些{detail}。",
        ),
        "cause": (
            f"{title}为什么会走向{detail}？",
            f"导致{title}{detail}的原因有哪些？",
            f"{title}究竟是怎么一步步{detail}的？",
            f"请概括{title}{detail}的主要原因和过程。",
            f"从资料看，{title}{detail}是哪些因素造成的？",
        ),
        "numeric": (
            f"{title}的{detail}是多少？",
            f"请问{title}{detail}有多少？",
            f"你知道{title}的{detail}数据吗？",
            f"说下{title}{detail}的具体数字。",
        ),
    }
    return stable_choice(styles[question_type], f"{question_type}:{title}:{detail}")


def expected_terms(value: str, limit: int = 4) -> list[str]:
    names = re.findall(r"[\u3400-\u9fffA-Za-z·]{2,24}", value)
    blocked = {"中国", "一个", "以及", "其中", "因此", "由于", "这个", "成为", "进行"}
    return list(dict.fromkeys(item for item in names if item not in blocked))[:limit]


def make_definition(source: dict[str, Any]) -> dict[str, Any] | None:
    title = clean(str(source.get("title") or ""))
    title_pattern = re.escape(title)
    relation = re.compile(
        rf"^(?:《)?{title_pattern}(?:》)?[^。；]{{0,24}}?"
        rf"(?:是|為|为|指(?:的是)?|属于|屬於)"
    )
    candidates = [
        sentence
        for sentence in sentences(source)[:8]
        if relation.search(sentence) and not sentence.startswith(("）", ")", "—", "-", "，", ","))
    ]
    if (
        not 2 <= len(title) <= 45
        or any(marker in title for marker in BAD_TITLE_MARKERS)
        or not candidates
    ):
        return None
    reference = candidates[0]
    return make_case(source, "definition", style_question("definition", title), reference)


def make_location(source: dict[str, Any]) -> dict[str, Any] | None:
    title = clean(str(source.get("title") or ""))
    if any(marker in title for marker in BAD_TITLE_MARKERS):
        return None
    for sentence in sentences(source)[:12]:
        if not sentence.startswith((title, f"《{title}》")):
            continue
        marker = next((item for item in LOCATION_MARKERS if item in sentence), None)
        if not marker or sentence.find(marker) > 100:
            continue
        tail = sentence.split(marker, 1)[1].strip(" ：:，,")
        terms = expected_terms(tail, 3)
        if terms:
            return make_case(
                source,
                "location",
                style_question("location", title),
                sentence,
                terms,
            )
    return None


def make_time(source: dict[str, Any]) -> dict[str, Any] | None:
    title = clean(str(source.get("title") or ""))
    if re.search(r"\d{4}年", title) or any(marker in title for marker in BAD_TITLE_MARKERS):
        return None
    for sentence in sentences(source)[:16]:
        subject = subject_prefix(sentence, title)
        if subject is None:
            continue
        for canonical, variants in TIME_ACTIONS.items():
            action_matches = [
                (sentence.find(variant), variant)
                for variant in variants
                if sentence.find(variant) >= subject
            ]
            if not action_matches:
                continue
            action_position, _ = min(action_matches)
            relation = sentence[subject:action_position]
            if (
                action_position - subject > 36
                or any(
                    marker in relation
                    for marker in (
                        "之父", "之母", "父亲", "父親", "母亲", "母親", "弟弟", "兄长", "兄長",
                        "的作者", "的导演", "的導演", "的演员", "的演員", "前身", "公司", "俱乐部",
                        "俱樂部", "大会", "大會", "纪念", "紀念", "一部", "电影", "電影",
                        "舞台剧", "舞台劇", "动画", "動畫", "作品", "歌曲", "品牌的",
                    )
                )
            ):
                continue
            lifespan = re.search(
                rf"{re.escape(title)}\s*[（(]"
                r"(?P<birth>\d{3,4}年(?:\d{1,2}月(?:\d{1,2}日)?)?)[—–-]"
                r"(?P<death>\d{3,4}年(?:\d{1,2}月(?:\d{1,2}日)?)?)[）)]",
                sentence,
            )
            if lifespan and canonical in {"出生", "逝世"}:
                value = lifespan.group("birth" if canonical == "出生" else "death")
                return make_case(
                    source,
                    "time",
                    style_question("time", title, canonical),
                    sentence,
                    [value],
                )
            japanese_year = re.search(
                rf"{re.escape(title)}[^。！？\n]{{0,30}}?"
                r"(?:昭和|大正|明治)\d+年[（(](?P<year>\d{4})[）)]"
                rf"[^。！？\n]{{0,20}}?(?:{'|'.join(map(re.escape, variants))})",
                sentence,
            )
            if japanese_year:
                return make_case(
                    source,
                    "time",
                    style_question("time", title, canonical),
                    sentence,
                    [f"{japanese_year.group('year')}年"],
                )
            dates = list(YEAR.finditer(sentence))
            if dates:
                date = min(dates, key=lambda match: abs(match.start() - action_position))
            else:
                continue
            if abs(date.start() - action_position) <= 45:
                return make_case(
                    source,
                    "time",
                    style_question("time", title, canonical),
                    sentence,
                    [date.group(1)],
                )
    return None


def subject_prefix(sentence: str, title: str) -> int | None:
    """Return where a relation about the exact article subject may begin."""
    prefixes = (title, f"《{title}》")
    prefix = next((item for item in prefixes if sentence.startswith(item)), None)
    if prefix is None:
        return None
    position = len(prefix)
    tail = sentence[position:]
    if not tail:
        return position
    # A continued noun phrase usually names a child entity, such as
    # "Pivotal中国研发中心", rather than the article subject itself.
    if re.match(r"[\u3400-\u9fffA-Za-z0-9]", tail) and not tail.startswith(
        ("是", "为", "為", "于", "於", "在", "由", "生", "卒", "创", "創", "建", "成", "开", "開")
    ):
        return None
    return position


def make_agent(source: dict[str, Any]) -> dict[str, Any] | None:
    title = clean(str(source.get("title") or ""))
    if any(marker in title for marker in BAD_TITLE_MARKERS):
        return None
    for sentence in sentences(source)[:18]:
        subject = subject_prefix(sentence, title)
        if subject is None and not sentence.startswith(ENTITY_PRONOUNS):
            continue
        if subject is not None and sentence[subject:].startswith(("的房屋", "的材料", "的作者", "的导演", "的導演")):
            continue
        if sentence.count("由") + sentence.count("是由") > 2:
            continue
        prefix = sentence[: sentence.find("由") if "由" in sentence else 0]
        if (
            not sentence.startswith(ENTITY_PRONOUNS)
            and any(marker in prefix for marker in ("他", "她", "它", "他们", "他們", "她们", "其作品", "其著作"))
        ):
            continue
        relation = extract_agent_relation(sentence, title)
        if relation:
            agent, action = relation
            return make_case(
                source,
                "agent",
                style_question("agent", title, action),
                sentence,
                [agent],
            )
    return None


def extract_agent_relation(sentence: str, title: str) -> tuple[str, str] | None:
    match = AGENT_RELATION.search(sentence)
    if not match:
        return None
    if re.match(r"的[\u3400-\u9fffA-Za-z]{2,24}", sentence[match.end():]):
        return None
    agent = clean(match.group("agent")).strip("的")
    agent = re.sub(r"(?:于|於|在)?\d{4}年.*$", "", agent).strip()
    if "委托" in agent or "委託" in agent:
        agent = re.split(r"委托|委託", agent)[-1].strip()
    if "——" in agent:
        agent = agent.split("——", 1)[-1].strip()
    agent = re.sub(r"(?:联合|聯合)$", "", agent).strip()
    for role in ("发明家兼工程师", "發明家兼工程師", "建筑师", "建築師", "设计师", "設計師", "拉比"):
        if role in agent:
            agent = agent.split(role, 1)[-1].strip()
    if (
        not 2 <= len(agent) <= 40
        or title in agent
        or agent in {"中国人", "中國人", "建筑", "建築", "韩国", "韓國", "巴伐利亚", "巴伐利亞"}
        or agent.endswith(("国人", "國人", "人士", "团队", "團隊"))
        or any(
            marker in agent
            for marker in (
                "担任", "擔任", "负责", "負責", "编剧", "編劇", "首度", "首次",
                "设计与", "設計與", "设计建造", "設計建造", "作造型", "委托", "委託",
            )
        )
        or any(action in agent for action in AGENT_ACTIONS)
    ):
        return None
    return agent, match.group("action")


def make_list(source: dict[str, Any]) -> dict[str, Any] | None:
    title = clean(str(source.get("title") or ""))
    section = clean(str((source.get("metadata") or {}).get("section") or ""))
    detail = section.split(">")[-1].strip() if section else "主要内容"
    text = content(source)
    items = [
        clean(match.group(1).split("：", 1)[0].split(":", 1)[0])
        for match in re.finditer(r"(?:^|\n)\s*[-*•]\s*([^\n]{2,80})", text)
    ]
    if len(items) < 3:
        list_match = re.search(r"列表[：:]\s*([^\n]{8,1800})", text)
        if list_match:
            items = [clean(item) for item in re.split(r"[、，,；;]", list_match.group(1))]
    items = [item for item in items if 2 <= len(item) <= 40][:8]
    if (
        not title
        or not detail
        or len(items) < 3
        or any(marker in title for marker in BAD_TITLE_MARKERS)
        or any(marker in detail for marker in BAD_SECTION_MARKERS)
    ):
        return None
    return make_case(
        source,
        "list",
        style_question("list", title, detail),
        clean(text[:1800]),
        items[:4],
    )


def make_cause(source: dict[str, Any]) -> dict[str, Any] | None:
    title = clean(str(source.get("title") or ""))
    section = clean(str((source.get("metadata") or {}).get("section") or ""))
    section_leaf = section.split(">")[-1].strip()
    section_leaf = re.sub(r"[（(][^）)]*[）)]\s*$", "", section_leaf).strip()
    event = next(
        (
            marker
            for marker in ("灭亡", "滅亡", "衰落", "失败", "失敗", "解体", "解體")
            if marker in section_leaf
            and re.fullmatch(
                rf"(?:分裂[与與和及]?)?(?:衰落[与與和及]?)?{re.escape(marker)}|{re.escape(marker)}",
                section_leaf,
            )
        ),
        None,
    )
    candidates = sentences(source)[:14]
    if (
        not title
        or any(marker in title for marker in BAD_TITLE_MARKERS)
        or not event
        or title.endswith(event)
        or len(candidates) < 2
    ):
        return None
    reference = "".join(candidates[:4])
    question_title = re.sub(rf"{re.escape(event)}$", "", title).strip() or title
    return make_case(source, "cause", style_question("cause", question_title, event), reference)


def make_numeric(source: dict[str, Any]) -> dict[str, Any] | None:
    title = clean(str(source.get("title") or ""))
    if any(marker in title for marker in BAD_TITLE_MARKERS):
        return None
    labels = ("人口", "面积", "面積", "全长", "全長", "高度", "海拔")
    for sentence in sentences(source)[:14]:
        if (
            not sentence.startswith((title, f"《{title}》"))
            or not (label := next((item for item in labels if item in sentence), None))
        ):
            continue
        label_tail = sentence.split(label, 1)[1]
        values = list(NUMBER.finditer(label_tail))
        value: str | None = None
        if label == "人口":
            population = re.match(
                r"\s*(?:为|為|有|约|約|达|達)?\s*(\d[\d,]*(?:\.\d+)?(?:万|亿|千)?)(?:人|(?=[，,。；;\s]))",
                label_tail,
            )
            if population:
                value = f"{population.group(1)}人"
        elif values:
            value = values[0].group(1)
        if value:
            return make_case(
                source,
                "numeric",
                style_question("numeric", title, label),
                sentence,
                [value],
            )
    return None


def make_case(
    source: dict[str, Any],
    question_type: str,
    question: str,
    reference: str,
    terms: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "question": question,
        "question_type": question_type,
        "style": style_name(question),
        "expected_title": source.get("title"),
        "expected_document_id": source.get("document_id"),
        "expected_terms": terms or expected_terms(reference),
        "reference": clean(reference),
        "section": (source.get("metadata") or {}).get("section"),
    }


def style_name(question: str) -> str:
    if question.startswith(("你知道", "我想", "我不太", "想去")):
        return "conversational"
    if question.startswith(("请", "能")):
        return "instructional"
    if question.startswith(("说下", "关于")):
        return "colloquial"
    return "direct"


def random_sources(opensearch_url: str, index: str, query: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    payload = {
        "size": 1600,
        "query": {
            "function_score": {
                "query": {"bool": {"filter": [{"term": {"source": "finewiki-zh"}}], "must": [query]}},
                "random_score": {"seed": seed, "field": "_seq_no"},
            }
        },
        "_source": ["title", "text", "document_id", "metadata", "content_type", "chunk_order"],
    }
    response = request_json(f"{opensearch_url.rstrip('/')}/{index}/_search", payload)
    return [dict(hit.get("_source") or {}) for hit in response.get("hits", {}).get("hits", [])]


def build_cases(
    opensearch_url: str,
    index: str,
    seed: int,
    excluded_cases: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    specs: list[tuple[str, int, dict[str, Any], Callable[[dict[str, Any]], dict[str, Any] | None]]] = [
        ("definition", 60, {"bool": {"filter": [{"term": {"chunk_order": 0}}, {"term": {"content_type": "prose"}}]}}, make_definition),
        ("location", 30, {"bool": {"filter": [{"term": {"chunk_order": 0}}], "must": [{"match": {"body_tokens": "位于 坐落 地处"}}]}}, make_location),
        ("time", 30, {"bool": {"filter": [{"term": {"chunk_order": 0}}], "must": [{"match": {"body_tokens": "成立 创建 出生 逝世 开通 上映"}}]}}, make_time),
        ("agent", 25, {"bool": {"filter": [{"term": {"chunk_order": 0}}], "must": [{"match": {"body_tokens": "由 创建 设计 发明 发现 建造 导演 创作"}}]}}, make_agent),
        ("list", 30, {"terms": {"content_type": ["list", "table_summary", "table"]}}, make_list),
        ("cause", 15, {"match": {"section_tokens": "灭亡 衰落 失败 解体"}}, make_cause),
        ("numeric", 10, {"bool": {"filter": [{"term": {"chunk_order": 0}}], "must": [{"match": {"body_tokens": "人口 面积 全长 高度 海拔"}}]}}, make_numeric),
    ]
    cases: list[dict[str, Any]] = []
    excluded_cases = excluded_cases or []
    seen_questions = {str(item.get("question") or "") for item in excluded_cases}
    excluded_documents = {
        str(item.get("expected_document_id") or "")
        for item in excluded_cases
        if item.get("expected_document_id")
    }
    seen_documents: Counter[str] = Counter()
    for offset, (question_type, target, query, factory) in enumerate(specs):
        sources = random_sources(opensearch_url, index, query, seed + offset * 997)
        for source in sources:
            case = factory(source)
            if case is None or case["question"] in seen_questions:
                continue
            document_id = str(case.get("expected_document_id") or "")
            if document_id in excluded_documents or seen_documents[document_id] >= 2:
                continue
            cases.append(case)
            seen_questions.add(case["question"])
            seen_documents[document_id] += 1
            if sum(item["question_type"] == question_type for item in cases) >= target:
                break
        actual = sum(item["question_type"] == question_type for item in cases)
        if actual < target:
            raise RuntimeError(f"only generated {actual}/{target} {question_type} cases")
    random.Random(seed).shuffle(cases)
    return cases


def normalized_terms(text: str) -> set[str]:
    text = T2S.convert(re.sub(r"\[资料\s*\d+\]", "", text).lower())
    cjk = "".join(re.findall(r"[\u3400-\u9fff]", text))
    return {cjk[index:index + 2] for index in range(max(0, len(cjk) - 1))}


def score(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    answer = str(response.get("answer") or "")
    sources = list(response.get("sources") or [])
    source_ids = {str(source.get("document_id") or "") for source in sources}
    reasons: list[str] = []
    if str(case["expected_document_id"]) not in source_ids:
        reasons.append("retrieval_miss")
    if not answer or any(marker in answer for marker in REFUSALS):
        reasons.append("answer_refused")
    if answer and not CITATION.search(answer):
        reasons.append("missing_citation")
    normalized_answer = T2S.convert(answer.lower())
    expected = [str(value) for value in case.get("expected_terms") or [] if str(value)]
    matched = [value for value in expected if T2S.convert(value.lower()) in normalized_answer]
    if case["question_type"] in {"time", "agent", "numeric"}:
        required = 1 if expected else 0
    else:
        required = 0
    if required and len(matched) < required:
        reasons.append("expected_fact_missing")
    reference_overlap = len(normalized_terms(case["reference"]) & normalized_terms(answer))
    if case["question_type"] == "location" and reference_overlap < 2:
        reasons.append("low_reference_coverage")
    generation = dict(response.get("generation") or {})
    if generation.get("answer_support_passed") is False:
        reasons.append("answer_support_failed")
    ambiguity = False
    if "expected_fact_missing" in reasons and case["question_type"] == "numeric":
        answer_body = CITATION.sub("", answer)
        answer_numbers = {match.group(1).replace(",", "") for match in BARE_NUMBER.finditer(answer_body)}
        exact_sources = [
            source for source in sources
            if str(source.get("document_id") or "") == str(case["expected_document_id"])
        ]
        evidence_numbers = {
            match.group(1).replace(",", "")
            for source in exact_sources
            for match in BARE_NUMBER.finditer(str(source.get("snippet") or ""))
        }
        if answer_numbers & evidence_numbers:
            reasons.remove("expected_fact_missing")
    category = "passed"
    if "retrieval_miss" in reasons:
        category = "retrieval_failure"
    elif "answer_refused" in reasons:
        category = "generation_or_gate_failure"
    elif ambiguity:
        category = "ambiguous_reference"
    elif reasons:
        category = "answer_quality_failure"
    return {
        **case,
        "answer": answer,
        "sources": [
            {
                "title": source.get("title"),
                "document_id": source.get("document_id"),
                "section": (source.get("metadata") or {}).get("section"),
                "snippet": str(source.get("snippet") or "")[:500],
            }
            for source in sources
        ],
        "retrieval": response.get("retrieval"),
        "generation": generation,
        "matched_expected_terms": matched,
        "failure_reasons": list(dict.fromkeys(reasons)),
        "failure_category": category,
        "passed": not reasons,
    }


def run_case(api_url: str, case: dict[str, Any], top_k: int) -> dict[str, Any]:
    started = time.monotonic()
    error: Exception | None = None
    for attempt in range(3):
        try:
            response = request_json(
                f"{api_url.rstrip('/')}/v1/ask",
                {"question": case["question"], "top_k": top_k},
                timeout=150,
            )
            result = score(case, response)
            break
        except Exception as current_error:
            error = current_error
            if attempt < 2:
                time.sleep(attempt + 1)
    else:
        result = {
            **case,
            "answer": "",
            "sources": [],
            "passed": False,
            "failure_reasons": ["request_error"],
            "failure_category": "request_failure",
            "error": repr(error),
        }
    result["latency_seconds"] = round(time.monotonic() - started, 3)
    return result


def summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed = sum(bool(result.get("passed")) for result in results)
    by_type: dict[str, dict[str, Any]] = {}
    for question_type in sorted({result["question_type"] for result in results}):
        group = [result for result in results if result["question_type"] == question_type]
        group_passed = sum(bool(result.get("passed")) for result in group)
        by_type[question_type] = {
            "passed": group_passed,
            "total": len(group),
            "pass_rate": round(group_passed / len(group), 4),
        }
    latencies = sorted(float(result["latency_seconds"]) for result in results)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 4) if total else 0,
        "by_type": by_type,
        "by_style": dict(Counter(result["style"] for result in results)),
        "failure_categories": dict(Counter(result["failure_category"] for result in results if not result.get("passed"))),
        "failure_reasons": dict(Counter(reason for result in results for reason in result.get("failure_reasons", []))),
        "latency_seconds": {
            "average": round(sum(latencies) / len(latencies), 3) if latencies else 0,
            "p50": latencies[len(latencies) // 2] if latencies else 0,
            "p95": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://127.0.0.1:8090")
    parser.add_argument("--opensearch-url", default="http://127.0.0.1:9200")
    parser.add_argument("--index", default="rwkvrag-knowledge-v1")
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("eval/wiki_qa_200_diverse_results.json"))
    parser.add_argument("--cases-output", type=Path, default=Path("eval/wiki_qa_200_diverse_cases.json"))
    parser.add_argument("--cases-input", type=Path)
    parser.add_argument(
        "--exclude-cases",
        type=Path,
        help="Exclude questions and source documents contained in an existing case file.",
    )
    args = parser.parse_args()
    cases = (
        json.loads(args.cases_input.read_text(encoding="utf-8"))
        if args.cases_input
        else build_cases(
            args.opensearch_url,
            args.index,
            args.seed,
            json.loads(args.exclude_cases.read_text(encoding="utf-8"))
            if args.exclude_cases
            else None,
        )
    )
    args.cases_output.parent.mkdir(parents=True, exist_ok=True)
    args.cases_output.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_case, args.api_url, case, args.top_k) for case in cases]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            results.append(future.result())
            if completed % 20 == 0:
                print(f"completed {completed}/{len(cases)}", flush=True)
    order = {case["question"]: index for index, case in enumerate(cases)}
    results.sort(key=lambda item: order[item["question"]])
    report = {"summary": summary(results), "results": results}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
