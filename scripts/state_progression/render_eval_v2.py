"""Render immutable paired model replies and their input evidence for local review."""
import argparse
import base64
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    run=json.loads((args.eval/'RUN.json').read_text())
    if sha(args.cases)!=run['inputs_sha256']:raise ValueError('Evaluation input hash mismatch')
    cases=[json.loads(line) for line in args.cases.read_text().splitlines()]
    records=[json.loads(p.read_text()) for p in (args.eval/'records').glob('*.json')]
    if len(records)!=len(cases)*4:raise ValueError('Incomplete paired records')
    outputs={}
    for row in records:
        key=(row['id'],row['round'],row['arm'])
        if key in outputs:raise ValueError('Duplicate paired record')
        outputs[key]={k:row.get(k) for k in ('status','raw_text','elapsed_s','generated_ids','utf8_valid','prompt_sha256')}
    data=[]
    for c in cases:
        item={k:c.get(k) for k in ('id','suite','category','question','evidence','prompt','evaluation_kind','unsupported_reason')}
        item['outputs']={str(round_no):{arm:outputs[(c['id'],round_no,arm)] for arm in ('zero','trained')}
                         for round_no in (1,2)}
        data.append(item)
    payload=base64.b64encode(json.dumps(data,ensure_ascii=False,separators=(',',':')).encode()).decode()
    html=HTML.replace('__BASE64_PAYLOAD__',payload)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as f:f.write(html)
    print(json.dumps({'cases':len(cases),'records':len(records),'html_sha256':sha(args.out),
                      'raw_replies_unchanged':True,'source_path':str(args.out)},ensure_ascii=False))


HTML='''<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RWKV State 成对回放</title><style>
:root{font-family:system-ui,-apple-system,"Noto Sans SC",sans-serif;color:#1d2630;background:#f2f5f8}*{box-sizing:border-box}body{margin:0}header{background:#162a3c;color:#fff;padding:18px 24px}header h1{font-size:21px;margin:0 0 6px}header p{margin:0;color:#c7d5df;font-size:13px}main{display:grid;grid-template-columns:minmax(240px,27%) 1fr;height:calc(100vh - 89px)}aside{background:#fff;border-right:1px solid #dbe2e8;overflow:auto;padding:14px}section#detail{overflow:auto;padding:20px}.controls{display:grid;gap:8px;position:sticky;top:0;background:#fff;padding-bottom:10px}input,select{font:inherit;padding:9px;border:1px solid #c7d1d9;border-radius:7px;width:100%}.case{border:1px solid #e1e7eb;border-radius:8px;padding:10px;margin:7px 0;cursor:pointer}.case:hover,.case.active{border-color:#327caa;background:#eef7fc}.case small{color:#60717e}.case strong{display:block;font-size:13px;line-height:1.45;margin-top:3px}.muted{color:#647582;font-size:13px}.card{background:#fff;border:1px solid #dbe2e8;border-radius:10px;padding:15px;margin-bottom:14px}.card h2{font-size:17px;margin:0 0 8px}.rounds{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.reply{background:#fff;border:1px solid #dbe2e8;border-radius:10px;padding:14px;min-width:0}.reply h3{margin:0 0 8px;font-size:15px}.status{font-size:12px;color:#657786}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-family:inherit;line-height:1.55;font-size:14px;margin:10px 0}.source{border-top:1px solid #e0e6ea;padding:10px 0}.source:first-of-type{border-top:0}.source summary{cursor:pointer;font-weight:600}.source a{font-size:12px}.pill{display:inline-block;background:#e4f1f9;color:#225d80;border-radius:12px;padding:2px 8px;font-size:12px;margin:3px 4px 0 0}.pill.bad{background:#fce8e5;color:#a02a1a}button{font:inherit;cursor:pointer}#count{margin:6px 0;color:#647582;font-size:12px}@media(max-width:900px){main{display:block;height:auto}aside{max-height:35vh;border-right:0;border-bottom:1px solid #dbe2e8}.rounds{grid-template-columns:1fr}}
</style></head><body><header><h1>RWKV State 成对回放</h1><p>相同输入，零 State 与训练 State；原始回答逐字显示。来源卡片是输入证据，不表示回答已正确。</p></header><main><aside><div class="controls"><select id="suite"></select><input id="query" placeholder="搜索题目或 ID"><div id="count"></div></div><div id="list"></div></aside><section id="detail"><p class="muted">选择一道题查看两轮原始回答与来源。</p></section></main><script id="payload" type="text/plain">__BASE64_PAYLOAD__</script><script>
const raw=atob(document.getElementById('payload').textContent.trim());const cases=JSON.parse(new TextDecoder().decode(Uint8Array.from(raw,c=>c.charCodeAt(0))));
const suite=document.getElementById('suite'),query=document.getElementById('query'),list=document.getElementById('list'),detail=document.getElementById('detail'),count=document.getElementById('count');
const suites=[...new Set(cases.map(x=>x.suite))];for(const name of ['全部',...suites]){const o=document.createElement('option');o.value=name;o.textContent=name;suite.appendChild(o)}if(suites.includes('fresh_github'))suite.value='fresh_github';let selected=null;
function el(tag,cls,text){const x=document.createElement(tag);if(cls)x.className=cls;if(text!==undefined)x.textContent=text;return x}
function shown(){const q=query.value.trim().toLowerCase();return cases.filter(c=>(suite.value==='全部'||c.suite===suite.value)&&(!q||(c.id+' '+c.question).toLowerCase().includes(q)))}
function drawList(){list.replaceChildren();const visible=shown();count.textContent=`${visible.length} / ${cases.length} 题`;for(const c of visible){const box=el('div','case'+(selected===c.id?' active':''));box.append(el('small','',`${c.suite} · ${c.category} · ${c.id}`),el('strong','',c.question||'无问题文本'));box.onclick=()=>{selected=c.id;drawList();drawDetail(c)};list.appendChild(box)}}
function drawReply(parent,label,r){const box=el('div','reply');box.append(el('h3','',label),el('div','status',`${r.status} · ${r.generated_ids?.length||0} tokens · ${r.elapsed_s?.toFixed?.(2)||'—'} 秒`));box.append(el('pre','',r.raw_text||''));const refs=[...new Set((r.raw_text||'').match(/\\[资料\\s*\\d+\\]/g)||[])];for(const ref of refs){const p=el('button','pill',ref);p.title='打开对应来源原文';p.onclick=()=>{const n=Number((ref.match(/\\d+/)||[])[0]);const source=document.getElementById('source-'+n);if(source){source.open=true;source.scrollIntoView({behavior:'smooth',block:'center'})}};box.appendChild(p)}parent.appendChild(box)}
function drawDetail(c){detail.replaceChildren();const intro=el('div','card');intro.append(el('h2','',c.question||c.id),el('div','muted',`${c.suite} · ${c.category} · ${c.id} · ${c.evaluation_kind||''}`));detail.appendChild(intro);for(const num of ['1','2']){detail.appendChild(el('h2','',`第 ${num} 轮`));const pair=el('div','rounds');drawReply(pair,'零 State',c.outputs[num].zero);drawReply(pair,'训练 State',c.outputs[num].trained);detail.appendChild(pair)}const box=el('div','card');box.appendChild(el('h2','','输入证据'));if(c.evidence?.length){for(let i=0;i<c.evidence.length;i++){const s=c.evidence[i],row=el('details','source'),label=s.label||'资料 '+(i+1),match=/^资料\\s*(\\d+)$/.exec(label);row.id='source-'+(match?Number(match[1]):i+1);row.appendChild(el('summary','',`${label} · ${s.title||s.source||s.id||''}`));const text=s.text||s.snippet||'';row.appendChild(el('pre','',text));if(typeof s.uri==='string'&&s.uri.startsWith('https://')){const a=el('a','','查看来源链接');a.href=s.uri;a.rel='noopener noreferrer';a.target='_blank';row.appendChild(a)}box.appendChild(row)}}else{box.append(el('p','muted','此题没有单独来源列表；展开原始输入检查完整证据。'))}const prompt=el('details','source');prompt.appendChild(el('summary','','完整原始输入'));prompt.appendChild(el('pre','',c.prompt||''));box.appendChild(prompt);detail.appendChild(box)}
suite.onchange=drawList;query.oninput=drawList;drawList();
</script></body></html>'''

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--eval',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
