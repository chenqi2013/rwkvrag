from hashlib import sha256
import json
from pathlib import Path
import random

from transformers import AutoTokenizer
from protocol import prompt

HERE=Path(__file__).resolve().parent


def main():
    cases=json.loads((HERE/'cases.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained('data/services/vllm-decode-20260920/model-metadata')
    inputs=[]
    for entry in cases:
        variants={}
        for arm in ('json','multiline'):
            body=prompt(entry['original_case'],arm)
            text=tokenizer.apply_chat_template([{'role':'user','content':body}],tokenize=False,
                add_generation_prompt=True,rwkv_generation_prompt='fake_think')
            ids=tokenizer.encode(text,add_special_tokens=False)
            assert tokenizer.decode(ids)==text and len(ids)+32<=4096
            variants[arm]={'body':body,'prompt':text,'prompt_token_ids':ids,
                'prompt_sha256':sha256(text.encode()).hexdigest()}
        inputs.append({'case':entry['case'],'variants':variants})
    old=json.loads((HERE.parent/'reader-presentation-20260920/inputs.json').read_text())
    old={v['case']['id']:v['variants'] for v in old}
    for v in inputs:
        if v['case']['cohort']=='historical160':
            assert v['variants']==old[v['case']['id'].split(':',1)[1]]
    order=list(range(len(inputs)));random.Random(20260920).shuffle(order)
    schedule=[]
    for round_number in (1,2):
        for i,index in enumerate(order if round_number==1 else list(reversed(order))):
            position=i if round_number==1 else len(order)-1-i
            arms=('json','multiline') if (position+round_number)%2 else ('multiline','json')
            schedule.extend({'case_id':inputs[index]['case']['id'],'arm':arm,'round':round_number,'seed':11} for arm in arms)
    for name,value in [('inputs.json',inputs),('schedule.json',schedule)]:
        with (HERE/name).open('x') as f:f.write(json.dumps(value,ensure_ascii=False,separators=(',',':')))
    print(json.dumps({'rows':len(inputs),'calls':len(schedule),'historical160_inputs_exact':True,
        'max_input_tokens':max(len(v['prompt_token_ids']) for i in inputs for v in i['variants'].values())}))


if __name__=='__main__':main()
