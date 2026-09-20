import importlib.util
import json
from pathlib import Path

BASE = Path(__file__).parents[1] / 'vllm-decoding-20260920/protocol.py'
spec = importlib.util.spec_from_file_location('frozen_decoding_protocol', BASE)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
metrics = previous.metrics
parse = previous.parse


def parameters(condition):
    return previous.parameters({'arm': 'top1', 'seed': 11})


def block(name, value):
    return f'[{name} characters={len(value)}]\n{value}\n[/{name}]\n'


def render(data):
    source = json.dumps(data['source'], ensure_ascii=False)
    result = block('source', source) + block('question', data['question'])
    result += f'[contexts count={len(data["contexts"])}]\n'
    for index, context in enumerate(data['contexts']):
        assert set(context) == {'text'}, 'Do not silently discard context metadata'
        result += block(f'context.{index}', context['text'])
    result += '[/contexts]\n' + block('text', data['text'])
    return result


def reconstruct(text):
    """Length-based check preserves embedded delimiters and trailing whitespace."""
    offset = 0

    def read(name):
        nonlocal offset
        prefix = f'[{name} characters='
        assert text.startswith(prefix, offset)
        end = text.index(']\n', offset)
        length = int(text[offset+len(prefix):end])
        start = end + 2
        value = text[start:start+length]
        suffix = f'\n[/{name}]\n'
        assert text.startswith(suffix, start+length)
        offset = start+length+len(suffix)
        return value

    source = json.loads(read('source'))
    question = read('question')
    prefix = '[contexts count='
    assert text.startswith(prefix, offset)
    end = text.index(']\n', offset)
    count = int(text[offset+len(prefix):end])
    offset = end+2
    contexts = [{'text': read(f'context.{i}')} for i in range(count)]
    suffix = '[/contexts]\n'
    assert text.startswith(suffix, offset)
    offset += len(suffix)
    body = read('text')
    assert offset == len(text)
    return {'source': source, 'question': question, 'contexts': contexts, 'text': body}


def prompt(case, arm):
    baseline = previous.prompt(case)
    if arm == 'json':
        return baseline
    if arm != 'multiline':
        raise ValueError('unknown presentation arm')
    instruction, separator, payload = baseline.partition('\n')
    data = json.loads(payload)
    serialized = render(data)
    assert reconstruct(serialized) == data
    return instruction + separator + serialized
