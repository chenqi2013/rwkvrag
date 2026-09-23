"""Freeze independent protocol probes on the previously seen V1 oracle scenarios."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'llamaindex-retrieval/src'))
from llamaindex_retrieval.state_tokens import Vocabulary
from llamaindex_retrieval.writer_prompt import writer_prompt_v2
from prepare_v1 import SCENARIOS, VOCAB, facts_for, prompt, validate


def sha(data):
    return hashlib.sha256(data).hexdigest()


def make_cases(scenarios, vocab):
    cases = []

    def add(scenario, stage, name, body, role, limit, expected, labels=None):
        wire = prompt(body)
        ids = vocab.encode(wire)
        if len(ids) + limit > 16384:
            raise ValueError(f'Context overflow: {scenario["id"]}/{stage}/{name}')
        cases.append({
            'id': f'{scenario["id"]}/{stage}/{name}', 'suite': 'layered_oracle_v2_seen_protocol',
            'category': stage, 'state_role': role, 'prompt': wire,
            'prompt_sha256': sha(wire.encode()), 'input_ids': ids,
            'max_output_tokens': limit, 'expected': expected,
            'project_count': len(scenario['projects']),
            'writer_source_labels': labels or [],
            'evaluation_kind': 'seen_synthetic_fixed_evidence_oracle_not_retrieval',
        })

    for scenario in scenarios:
        sources = '\n\n'.join(
            f'[{item["id"]}][项目={item["project"]}] {item["text"]}'
            for item in scenario['sources'])
        extract = (
            '请逐条抽取资料明确陈述的 Windows、许可证和 GPU 事实。'
            '每条绑定项目、属性、值、来源和版本状态；同一当前版的相反记录都保留，'
            '旧版记 legacy，当前版记 current；没有写出的属性不要补。'
            '值只能用 windows:native/experimental/unsupported/not_checked；'
            'license:MIT/Apache-2.0/GPL-3.0；gpu:supported/unsupported/not_checked。'
            '只输出一个 JSON 数组，不要对象外壳或解释；数组中每项恰有 '
            'project、attribute、value、source_id、status 五个字符串字段。\n'
            f'项目：{json.dumps(scenario["projects"], ensure_ascii=False)}\n资料：\n{sources}'
        )
        add(scenario, 'extract_array', 'all', extract, 'resolver', 1024, facts_for(scenario))

        for project in scenario['projects']:
            facts = [row for row in facts_for(scenario) if row['project'] == project]
            gold = next(status for status in ('eligible', 'rejected', 'unresolved')
                        if project in scenario['decision'][status])
            judge = (
                '以下事实已由人工核准。只判断指定项目当前版本是否同时满足全部硬条件。'
                '明确不满足任一硬条件是 rejected；事实缺失、not_checked、或当前版互相矛盾'
                '且没有明确不满足时是 unresolved；全部满足才是 eligible。'
                '旧版不能覆盖当前版。GPU 是偏好，不影响资格。'
                '只输出一个英文单词：eligible、rejected 或 unresolved。\n'
                f'项目：{project}\n'
                f'硬条件：{json.dumps(scenario["requirements"]["hard"], ensure_ascii=False)}\n'
                f'偏好：{json.dumps(scenario["requirements"]["preference"], ensure_ascii=False)}\n'
                f'已核准事实：{json.dumps(facts, ensure_ascii=False)}'
            )
            add(scenario, 'judge_project', project, judge, 'resolver', 64, gold)

        selected = [item for item in scenario['sources']
                    if any(status == 'current' for _, _, status in item['facts'])]
        evidence = [{'label': index, 'source_id': item['id'],
                     'project': item['project'], 'text': item['text']}
                    for index, item in enumerate(selected, 1)]
        decision = scenario['decision']
        compact = {
            'required_objects': scenario['projects'],
            'active_requirements': scenario['requirements'],
            'candidate_eligibility': [
                {'object': name, 'status': next(status for status in
                    ('eligible', 'rejected', 'unresolved') if name in decision[status])}
                for name in scenario['projects']],
            'selection': {'recommended_objects': decision['recommended']},
        }
        task = json.dumps({'latest_question': '比较这些项目是否满足硬条件，并按偏好推荐；'
                            '列出不能判断的项目及原因。',
                           'objects': scenario['projects'],
                           'requirements': scenario['requirements']}, ensure_ascii=False)
        writer = writer_prompt_v2(task, evidence, ['Windows', '许可证', 'GPU']) + (
            '\n以下是已完成的局部判断，不是新的证据。汇总其结论与限制，'
            '不再逐项重做比较或重复字段问题。每个事实仍须由上面的逐字证据支持；'
            '不得把未知写成不支持，不推荐被排除的候选。\n'
            + json.dumps(compact, ensure_ascii=False))
        add(scenario, 'writer_standard', 'all', writer, 'writer', 2048,
            decision, [f'资料 {index}' for index in range(1, len(selected) + 1)])
    return cases


def main(out):
    scenarios = json.loads(SCENARIOS.read_text())
    validate(scenarios)
    cases = make_cases(scenarios, Vocabulary(VOCAB))
    out.mkdir(parents=True, exist_ok=False)
    payload = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in cases).encode()
    (out / 'cases.jsonl').write_bytes(payload)
    pins = {
        'source_scenarios_sha256': sha(SCENARIOS.read_bytes()),
        'prepare_sha256': sha(Path(__file__).read_bytes()),
        'writer_prompt_sha256': sha((ROOT / 'llamaindex-retrieval/src/llamaindex_retrieval/writer_prompt.py').read_bytes()),
        'vocab_sha256': sha(VOCAB.read_bytes()), 'cases_sha256': sha(payload),
        'members': len(cases), 'planned_raw_records': len(cases) * 4,
        'stage_members': {stage: sum(row['category'] == stage for row in cases)
                          for stage in ('extract_array', 'judge_project', 'writer_standard')},
        'seen_scenarios': True, 'live_retrieval': False, 'production_prompt_exact': False,
        'old_inputs_unchanged': True,
    }
    (out / 'PINS.json').write_text(json.dumps(pins, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(pins, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out)
