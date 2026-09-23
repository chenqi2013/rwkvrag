"""Freeze one-hard-condition oracle probes, closer to v10 condition nodes."""
import argparse
import hashlib
import json
from pathlib import Path

from prepare_v1 import SCENARIOS, VOCAB, facts_for, prompt, validate
from llamaindex_retrieval.state_tokens import Vocabulary


def sha(data):
    return hashlib.sha256(data).hexdigest()


def gold_status(facts, attribute):
    current = {fact['value'] for fact in facts
               if fact['attribute'] == attribute and fact['status'] == 'current'}
    if len(current) > 1:
        return 'conflict'
    if not current or current == {'not_checked'}:
        return 'unknown'
    if attribute == 'windows':
        return 'satisfied' if current == {'native'} else 'not_satisfied'
    return 'satisfied' if current <= {'MIT', 'Apache-2.0'} else 'not_satisfied'


def make_cases(scenarios, vocab):
    cases = []
    requirements = {
        'windows': '当前版 Windows 必须原生支持，实验支持不符合',
        'license': '当前版许可证必须为 MIT 或 Apache-2.0',
    }
    for scenario in scenarios:
        for project in scenario['projects']:
            facts = [fact for fact in facts_for(scenario) if fact['project'] == project]
            for attribute, requirement in requirements.items():
                relevant = [fact for fact in facts if fact['attribute'] == attribute]
                body = (
                    '只判断指定项目是否满足以下一个生效硬条件，不做推荐、不考虑偏好或其他项目。'
                    '只看当前版；旧版不覆盖当前版。同一当前版相反记录为 conflict；'
                    '缺失或 not_checked 为 unknown；明确不满足为 not_satisfied；'
                    '有明确证据满足才是 satisfied。'
                    '只输出一个英文单词：satisfied、not_satisfied、unknown 或 conflict。\n'
                    f'项目：{project}\n硬条件：{requirement}\n'
                    f'已核准的本字段事实：{json.dumps(relevant, ensure_ascii=False)}'
                )
                wire = prompt(body)
                ids = vocab.encode(wire)
                if len(ids) + 64 > 16384:
                    raise ValueError('Context overflow')
                cases.append({'id': f'{scenario["id"]}/condition/{project}/{attribute}',
                              'suite': 'layered_oracle_v3_seen_condition',
                              'category': 'condition', 'state_role': 'resolver',
                              'prompt': wire, 'prompt_sha256': sha(wire.encode()),
                              'input_ids': ids, 'max_output_tokens': 64,
                              'expected': gold_status(facts, attribute),
                              'attribute': attribute, 'project_count': len(scenario['projects']),
                              'evaluation_kind': 'seen_synthetic_gold_field_not_retrieval'})
    return cases


def main(out):
    scenarios = json.loads(SCENARIOS.read_text())
    validate(scenarios)
    cases = make_cases(scenarios, Vocabulary(VOCAB))
    if len(cases) != 30:
        raise ValueError('Expected 15 projects x 2 hard conditions')
    out.mkdir(parents=True, exist_ok=False)
    payload = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in cases).encode()
    (out / 'cases.jsonl').write_bytes(payload)
    pins = {'scenarios_sha256': sha(SCENARIOS.read_bytes()),
            'prepare_sha256': sha(Path(__file__).read_bytes()),
            'vocab_sha256': sha(VOCAB.read_bytes()),
            'cases_sha256': sha(payload), 'members': 30,
            'planned_raw_records': 120, 'seen_scenarios': True,
            'gold_fact_intervention': True, 'live_retrieval': False,
            'old_inputs_unchanged': True}
    (out / 'PINS.json').write_text(json.dumps(pins, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(pins, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out)
