"""Freeze independent oracle probes without touching earlier evaluation inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'llamaindex-retrieval/src'))
from llamaindex_retrieval.state_tokens import Vocabulary

SCENARIOS = ROOT / 'llamaindex-retrieval/eval/layered-oracle-20260923-v1/SCENARIOS.json'
VOCAB = ROOT / 'llamaindex-retrieval/statetune/assets/rwkv_vocab_v20230424.txt'
ATTRIBUTES = {'windows', 'license', 'gpu'}
VALUES = {
    'windows': {'native', 'experimental', 'unsupported', 'not_checked'},
    'license': {'MIT', 'Apache-2.0', 'GPL-3.0'},
    'gpu': {'supported', 'unsupported', 'not_checked'},
}
STATUSES = {'current', 'legacy'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def facts_for(scenario):
    facts = []
    for source in scenario['sources']:
        for attr, value, status in source['facts']:
            facts.append({'project': source['project'], 'attribute': attr,
                          'value': value, 'source_id': source['id'], 'status': status})
    return facts


def validate(scenarios):
    if len(scenarios) != 4 or [len(x['projects']) for x in scenarios] != [2, 3, 4, 6]:
        raise ValueError('Expected the frozen 2/3/4/6-project diagnostic curve')
    if len({x['id'] for x in scenarios}) != len(scenarios):
        raise ValueError('Duplicate scenario identity')
    for s in scenarios:
        projects = s['projects']
        if len(set(projects)) != len(projects):
            raise ValueError('Duplicate project name')
        sources = s['sources']
        if len({x['id'] for x in sources}) != len(sources):
            raise ValueError('Duplicate source identity')
        for source in sources:
            if source['project'] not in projects or not source['text'].strip():
                raise ValueError('Unbound or empty source')
            for attr, value, status in source['facts']:
                if attr not in ATTRIBUTES or value not in VALUES[attr] or status not in STATUSES:
                    raise ValueError('Invalid gold tuple')
        decision = s['decision']
        eligible, rejected, unresolved = (set(decision[x]) for x in ('eligible', 'rejected', 'unresolved'))
        if (eligible | rejected | unresolved != set(projects) or eligible & rejected
                or eligible & unresolved or rejected & unresolved
                or not set(decision['recommended']) <= eligible):
            raise ValueError('Decision partition or recommendation invalid')
        current = [f for f in facts_for(s) if f['status'] == 'current']
        for name in projects:
            p = [f for f in current if f['project'] == name]
            windows = {f['value'] for f in p if f['attribute'] == 'windows'}
            licenses = {f['value'] for f in p if f['attribute'] == 'license'}
            if 'GPL-3.0' in licenses:
                expected = 'rejected'
            elif len(windows) > 1 or len(licenses) > 1:
                expected = 'unresolved'
            elif windows in ({'unsupported'}, {'experimental'}):
                expected = 'rejected'
            elif windows == {'native'} and licenses in ({'MIT'}, {'Apache-2.0'}):
                expected = 'eligible'
            else:
                expected = 'unresolved'
            if name not in decision[expected]:
                raise ValueError(f'Gold decision disagrees with explicit hard conditions: {s["id"]}/{name}')


def prompt(body):
    return 'User: ' + body + '\n\nAssistant: <think></think>\n'


def make_cases(scenarios, vocab):
    cases = []
    for s in scenarios:
        source_text = '\n\n'.join(f'[{x["id"]}][项目={x["project"]}] {x["text"]}' for x in s['sources'])
        extraction = (
            '请逐条抽取下面资料明确陈述的 Windows、许可证和 GPU 事实。每条都要绑定项目、属性、值、来源和版本状态；'
            '同一属性的相反记录都保留，旧版记 legacy，当前版记 current；没有写出的属性不要补。'
            '值只能用 windows:native/experimental/unsupported/not_checked；'
            'license:MIT/Apache-2.0/GPL-3.0；gpu:supported/unsupported/not_checked。'
            '只输出一个 JSON 对象，结构为 {"facts":[{"project":"项目名","attribute":"windows|license|gpu",'
            '"value":"枚举值","source_id":"S1","status":"current|legacy"}]}。\n'
            f'项目：{json.dumps(s["projects"], ensure_ascii=False)}\n资料：\n{source_text}'
        )
        gold_facts = facts_for(s)
        comparison = (
            '以下事实表由人工核准，请直接依据它判断，不重新阅读原始文档。'
            '硬条件必须全部满足；有明确不满足则 rejected；缺失、not_checked 或同一当前版本矛盾则 unresolved。'
            'legacy 不能覆盖 current。偏好只在 eligible 之间选择，GPU supported 优先。'
            '只输出一个 JSON 对象，键恰为 eligible、rejected、unresolved、recommended，各值为项目名数组。\n'
            f'项目：{json.dumps(s["projects"], ensure_ascii=False)}\n'
            f'要求：{json.dumps(s["requirements"], ensure_ascii=False)}\n'
            f'已核准事实：{json.dumps(gold_facts, ensure_ascii=False)}'
        )
        selected = [x for x in s['sources'] if any(status == 'current' for _, _, status in x['facts'])]
        evidence = '\n'.join(f'[资料 {i}] 项目={x["project"]}；原文：{x["text"]}' for i, x in enumerate(selected, 1))
        writer = (
            '请给用户一段简洁的多项目比较和选择回答。以下资格与推荐已经核准；不得改变项目归类、补充材料没有的能力或数字。'
            '只引用下方实际提供的逐字材料，用 [资料 N] 标出依据；证据不足的候选明确写尚不能判断。'
            '写完结论即可停止，不要重复。\n'
            f'用户要求：{json.dumps(s["requirements"], ensure_ascii=False)}\n'
            f'已核准结论：{json.dumps(s["decision"], ensure_ascii=False)}\n'
            f'已选逐字资料：\n{evidence}'
        )
        for stage, body, role, limit, expected in (
            ('extract', extraction, 'resolver', 1024, {'facts': gold_facts}),
            ('compare', comparison, 'resolver', 512, s['decision']),
            ('writer', writer, 'writer', 2048, s['decision']),
        ):
            wire = prompt(body)
            ids = vocab.encode(wire)
            if len(ids) + limit > 16384:
                raise ValueError('Full prompt plus output budget exceeds context')
            cases.append({'id': s['id'] + '/' + stage, 'suite': 'layered_oracle_v1',
                          'category': stage, 'state_role': role, 'prompt': wire,
                          'prompt_sha256': sha(wire.encode()), 'input_ids': ids,
                          'max_output_tokens': limit, 'expected': expected,
                          'project_count': len(s['projects']),
                          'source_ids': [x['id'] for x in s['sources']],
                          'writer_source_labels': [f'资料 {i}' for i in range(1, len(selected)+1)] if stage == 'writer' else [],
                          'evaluation_kind': 'new_synthetic_fixed_evidence_oracle_not_retrieval'})
    return cases


def main(out):
    scenarios = json.loads(SCENARIOS.read_text())
    validate(scenarios)
    vocab = Vocabulary(VOCAB)
    cases = make_cases(scenarios, vocab)
    out.mkdir(parents=True, exist_ok=False)
    payload = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in cases).encode()
    (out / 'cases.jsonl').write_bytes(payload)
    pins = {'scenarios_sha256': sha(SCENARIOS.read_bytes()), 'prepare_sha256': sha(Path(__file__).read_bytes()),
            'vocab_sha256': sha(VOCAB.read_bytes()), 'cases_sha256': sha(payload),
            'members': len(cases), 'planned_raw_records': len(cases) * 4,
            'projects_per_scenario': [2, 3, 4, 6], 'live_retrieval': False,
            'training_data_used': False, 'frozen_previous_cases_unchanged': True,
            'reviewer': 'implementer_not_independent'}
    (out / 'PINS.json').write_text(json.dumps(pins, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(pins, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out)
