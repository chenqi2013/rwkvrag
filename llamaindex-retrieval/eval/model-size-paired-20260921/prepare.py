"""Export seen cases without changing their original records or gold."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'llamaindex-retrieval/src'))
from llamaindex_retrieval.writer_prompt import writer_prompt_v2

def sha(b):
    return hashlib.sha256(b).hexdigest()

def main():
    old = ROOT / 'llamaindex-retrieval/eval/g1j72-20260920/cases.jsonl'
    exported = ROOT / 'data/reports/test-questions-and-answers-20260921/全部题目与回答.json'
    cases = []
    for c in map(json.loads, old.read_text().splitlines()):
        cases.append({'id': 'old28/' + c['id'], 'suite': 'old28',
                      'category': c['group'], 'case': c, 'payload': c})
    for r in json.loads(exported.read_text())['historical_160']:
        c = r['case']
        cases.append({'id': 'full160/' + c['id'], 'suite': c['suite'],
                      'category': c.get('category', 'unclassified'), 'case': c,
                      'payload': c['payload']})
    rows = []
    for i, c in enumerate(cases):
        p = c.pop('payload')
        evidence = [{'label': f'资料 {j}', 'text': m['snippet']}
                    for j, m in enumerate(p['materials'], 1)]
        task = json.dumps({'history': p.get('history', []),
                           'latest_question': p['question']}, ensure_ascii=False)
        prompt = 'User: ' + writer_prompt_v2(task, evidence, []) + '\n\nAssistant: <think></think>\n'
        rows.append({**c, 'ordinal': i, 'question': p['question'],
                     'history': p.get('history', []), 'evidence': evidence,
                     'prompt': prompt, 'prompt_sha256': sha(prompt.encode())})
    assert len(rows) == 188 and len({r['id'] for r in rows}) == 188
    with (HERE / 'INPUTS.json').open('x') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    with (HERE / 'SOURCE-BINDING.json').open('x') as f:
        json.dump({str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in
                   [old, exported, ROOT/'llamaindex-retrieval/src/llamaindex_retrieval/writer_prompt.py']}, f, indent=2)

if __name__ == '__main__':
    main()
