import json
from pathlib import Path
import random
from hashlib import sha256

from transformers import AutoTokenizer
from protocol import prompt

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent/'vllm-decoding-20260920'


def main():
    old = json.loads((PRIOR/'inputs.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained('data/services/vllm-decode-20260920/model-metadata')
    inputs = []
    for previous in old:
        case = previous['case']
        variants = {}
        for arm in ('json', 'multiline'):
            body = prompt(case, arm)
            text = tokenizer.apply_chat_template([{'role': 'user', 'content': body}],
                tokenize=False, add_generation_prompt=True, rwkv_generation_prompt='fake_think')
            ids = tokenizer.encode(text, add_special_tokens=False)
            assert tokenizer.decode(ids) == text and text.endswith('<think></think')
            assert len(ids)+32 <= 4096
            variants[arm] = {'body': body, 'prompt': text, 'prompt_token_ids': ids,
                             'prompt_sha256': sha256(text.encode()).hexdigest()}
        assert variants['json'] == {k: previous[k] for k in variants['json']}
        inputs.append({'case': case, 'variants': variants})
    order = list(range(len(inputs)))
    random.Random(20260920).shuffle(order)
    schedule = []
    for round_number in (1, 2):
        for i, index in enumerate(order if round_number == 1 else list(reversed(order))):
            position = i if round_number == 1 else len(order)-1-i
            arms = ('json', 'multiline') if (position+round_number)%2 else ('multiline', 'json')
            schedule.extend({'case_id': inputs[index]['case']['id'], 'arm': arm,
                             'round': round_number, 'seed': 11} for arm in arms)
    for name, value in [('inputs.json', inputs), ('schedule.json', schedule)]:
        path = HERE/name
        with path.open('x') as output:
            output.write(json.dumps(value, ensure_ascii=False, indent=2))
    print(json.dumps({'cases': len(inputs), 'calls': len(schedule), 'previous_json_baseline_exact': True}))


if __name__ == '__main__':
    main()
