"""Lossless CPU conversion of the existing checkpoint; every tensor rechecked."""
from hashlib import file_digest
import json
from pathlib import Path
import re
import shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SOURCE = Path('/mnt/nas-model/g1j/rwkv7-g1j-2.9b-20260831-ctx16384.pth')
REFERENCE = Path('/mnt/nas-model/g1j/hf/rwkv7-g1j-7.2b-20260831-ctx16384')
TARGET = Path('/home/chase/rwkvrag/data/services/model-size-paired-20260921/model-2.9b')

def sha(path):
    with path.open('rb') as f:
        return file_digest(f, 'sha256').hexdigest()

def mapped(name):
    if name in {'blocks.0.att.v0', 'blocks.0.att.v1', 'blocks.0.att.v2'}:
        return None
    for old, new in [('emb.', 'model.embed_tokens.'), ('head.', 'lm_head.'),
                     ('ln_out.', 'model.norm.'), ('blocks.0.ln0.', 'model.embedding_norm.')]:
        if name.startswith(old):
            return new + name[len(old):]
    name = re.sub(r'^blocks\.(\d+)\.', r'model.layers.\1.', name)
    for old, new in [('.ln1.', '.input_layernorm.'), ('.ln2.', '.post_attention_layernorm.'),
                     ('.att.', '.linear_attn.'), ('.ffn.', '.mlp.'),
                     ('.linear_attn.receptance.', '.linear_attn.r_proj.'),
                     ('.linear_attn.key.', '.linear_attn.k_proj.'),
                     ('.linear_attn.value.', '.linear_attn.v_proj.'),
                     ('.linear_attn.output.', '.linear_attn.o_proj.'),
                     ('.linear_attn.ln_x.', '.linear_attn.g_norm.')]:
        name = name.replace(old, new)
    return name

def main():
    torch.set_num_threads(4)
    TARGET.mkdir(parents=True, exist_ok=False)
    source_sha = sha(SOURCE)
    state = torch.load(SOURCE, map_location='cpu', weights_only=True, mmap=True)
    tensors = {mapped(k): v.squeeze().contiguous() for k, v in state.items() if mapped(k)}
    ref = json.loads((REFERENCE/'model.safetensors.index.json').read_text())['weight_map']
    assert set(tensors) == set(ref) and len(tensors) == 1059
    config = json.loads((REFERENCE/'config.json').read_text())
    config.update(hidden_size=2560, intermediate_size=10240, num_attention_heads=40,
                  decay_low_rank_dim=96, a_low_rank_dim=96, v_low_rank_dim=64,
                  gate_low_rank_dim=320)
    assert state['emb.weight'].shape == (65536,2560)
    assert state['blocks.0.att.w1'].shape == (2560,96)
    assert state['blocks.0.att.g1'].shape == (2560,320)
    (TARGET/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    for name in ['tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
                 'generation_config.json', 'fake_think_generation_config.json', 'tools_generation_config.json']:
        shutil.copyfile(REFERENCE/name, TARGET/name)
    groups, group, size = [], {}, 0
    for k,v in sorted(tensors.items()):
        if size > 1_500_000_000:
            groups.append(group); group={}; size=0
        group[k]=v; size+=v.numel()*v.element_size()
    if group: groups.append(group)
    weight_map={}
    for i,g in enumerate(groups,1):
        name=f'model-{i:05d}-of-{len(groups):05d}.safetensors'
        save_file(g,str(TARGET/name),metadata={'format':'pt'})
        with safe_open(TARGET/name,framework='pt') as f:
            for k,v in g.items():
                actual=f.get_tensor(k)
                assert actual.dtype==v.dtype and actual.shape==v.shape and torch.equal(actual,v),k
                weight_map[k]=name
        print(json.dumps({'shard_verified':name,'tensors':len(g)}),flush=True)
    index={'metadata':{'total_size':sum(t.numel()*t.element_size() for t in tensors.values())},'weight_map':weight_map}
    (TARGET/'model.safetensors.index.json').write_text(json.dumps(index,indent=2)+'\n')
    from transformers import RwkvConfig, RwkvForCausalLM
    with torch.device('meta'):
        model=RwkvForCausalLM(RwkvConfig.from_pretrained(TARGET))
    expected=model.state_dict()
    assert set(expected)==set(tensors)
    assert all(expected[k].shape==v.shape for k,v in tensors.items())
    record={'source':str(SOURCE),'source_sha256':source_sha,'source_tensors':len(state),
            'verified_tensors':len(tensors),'dropped_unused_layer0':['blocks.0.att.v0','blocks.0.att.v1','blocks.0.att.v2'],
            'mapping_script_sha256':sha(Path(__file__)), 'tensor_values_unchanged':True,
            'config_shapes_verified':True,'files':{p.name:sha(p) for p in TARGET.iterdir() if p.is_file()}}
    (TARGET/'CONVERSION.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'verified':len(tensors),'source_sha256':source_sha}),flush=True)

if __name__=='__main__':main()
