"""Publish immutable repair traces alongside original real comparison results."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PUBLIC=ROOT/'llamaindex-retrieval/web/public/experiments'
PREVIEW=ROOT/'data/frontend-model-comparison-20260921/dist/experiments'

def visible_trace(value):
    # Raw prompt/output and graph stay visible; wire bytes/token arrays live in
    # the immutable archive and would otherwise be duplicated many times here.
    if isinstance(value, dict):
        return {k: visible_trace(v) for k, v in value.items()
                if k not in {'http', 'input_token_ids', 'output_token_ids', 'upstream_calls'}}
    if isinstance(value, list): return [visible_trace(v) for v in value]
    return value

def main():
    original=json.loads((PUBLIC/'github-natural-comparison-20260921.json').read_text())
    cases=[]
    for n,case in enumerate(original['cases']):
        answers=[case['answers'][0]]
        for name,label in [('writer-budget-repair-20260921','预算修复 · 同证据回放'),('compact-writer-repair-20260921','紧凑Writer · 未晋级'),('funnel-integration-20260921','漏斗v1 · 多对象提取诊断'),('funnel-atomic-v2-20260921','漏斗v2 · 单对象单字段'),('funnel-bound-units-v3-20260921','漏斗v3 · 原文单元绑定'),('funnel-structured-v4-20260921','漏斗v4 · 结构约束')]:
            f=ROOT/'data/quality-runs'/name/'run1'/f'{n:02d}.json'
            if not f.exists():f=f.with_name(f'{n:03d}.json')
            if not f.exists():continue
            row=json.loads(f.read_text());response=row.get('response',{});trace=response.get('generation',{});raw=response.get('answer',row.get('raw_text') or '')
            sources=response.get('sources',row.get('sources',[]))
            if name.startswith('compact-'):
                sources=[{'snippet':s['text'],'uri':None} for s in row['evidence']]
            graph=response.get('retrieval',{}).get('funnel');budget=row.get('trace',{}).get('evidence_budget')
            notes='已保存原始输出；生成结束不代表语义正确。'
            if name.startswith('compact-'):notes='候选失败：停止率改善，但真实题漏引用且空证据仍编造；不采用。'
            if budget:notes=f"实际使用{len(budget['included_source_ids'])}份，因预算未使用{len(budget['omitted_source_ids'])}份；原文未截断。"
            if graph:notes=f"本题未通过大型比较要求（实现者审读）。分层节点失败{len(graph.get('failures',[]))}项；事实{len(graph.get('facts',[]))}条。逐字校验不代表语义已独立核验。"
            answers.append({'label':label,'raw_text':raw,'finish_reason':trace.get('status',row.get('status')),'elapsed_s':trace.get('elapsed_ms',0)/1000 if response else row.get('elapsed_s',row.get('trace',{}).get('elapsed_ms',0)/1000),'notes':notes,'sources':[{'label':f'资料 {i}','text':s['snippet'],'url':s.get('uri')} for i,s in enumerate(sources,1)],'funnel':graph,'trace':visible_trace(response if response else row)})
        f=ROOT/'data/quality-runs/funnel-structured-v4-20260921/run1'/f'{n:03d}.writer-replay.json'
        if f.exists():
            row=json.loads(f.read_text())
            answers.append({'label':'漏斗v4 · 原Writer同事实回放','raw_text':row.get('raw_text') or '',
                'finish_reason':row['status'],'elapsed_s':row['trace']['elapsed_ms']/1000,
                'notes':'保留同一上游事实与判断，只更换最后Writer呈现；非新检索。',
                'sources':[{'label':f'资料 {i}','text':s['snippet'],'url':s.get('uri')} for i,s in enumerate(row['sources'],1)],
                'trace':visible_trace(row)})
        cases.append({'id':case['id'],'question':case['question'],'answers':answers})
    old=json.loads((ROOT/'llamaindex-retrieval/eval/model-size-paired-20260921/INPUTS.json').read_text())[:28]
    for n,case in enumerate(old):
        answers=[]
        baseline=ROOT/'data/quality-runs/writer-convergence-20260921/run2/baseline-round1'/f'{n:04d}.json'
        if baseline.exists():
            b=json.loads(baseline.read_text());answers.append({'label':'历史原Writer · 固定材料基线','raw_text':b['raw_text'],
                'finish_reason':b.get('finish_reason',b['status']),'elapsed_s':b.get('elapsed_ms',0)/1000,
                'notes':'旧题及旧答案原样保留；与新分层共享原始材料。','sources':case['evidence'],'trace':visible_trace(b)})
        for name,label in [('compact-writer-repair-20260921','紧凑Writer · 未晋级'),('funnel-integration-20260921','漏斗v1'),('funnel-atomic-v2-20260921','漏斗v2'),('funnel-bound-units-v3-20260921','漏斗v3'),('funnel-structured-v4-20260921','漏斗v4 · 结构约束')]:
            f=ROOT/'data/quality-runs'/name/'run1'/f'{n+8:03d}.json'
            if not f.exists():continue
            row=json.loads(f.read_text());response=row.get('response',{});gen=response.get('generation',{});graph=response.get('retrieval',{}).get('funnel')
            sources=response.get('sources',[])
            if name.startswith('compact-'):sources=[{'snippet':e['text']} for e in row['evidence']]
            answers.append({'label':label,'raw_text':response.get('answer',row.get('raw_text') or ''),
                'finish_reason':gen.get('status',row.get('status')),'elapsed_s':gen.get('elapsed_ms',row.get('elapsed_s',0)*1000)/1000,
                'notes':'完整保留旧题。执行/格式通过不代表事实、条件和引用正确；本候选未晋级，旧28语义未汇总为准确率。',
                'sources':[{'label':f'资料 {i}','text':source['snippet'],'url':source.get('uri')} for i,source in enumerate(sources,1)],
                'funnel':graph,'trace':visible_trace(response if response else row)})
        f=ROOT/'data/quality-runs/funnel-structured-v4-20260921/run1'/f'{n+8:03d}.writer-replay.json'
        if f.exists():
            row=json.loads(f.read_text());answers.append({'label':'漏斗v4 · 原Writer同事实回放','raw_text':row.get('raw_text') or '',
                'finish_reason':row['status'],'elapsed_s':row['trace']['elapsed_ms']/1000,
                'notes':'同一上游结果，末层独立回放；不是完整新链路回归通过。',
                'sources':[{'label':f'资料 {i}','text':source['snippet'],'url':source.get('uri')} for i,source in enumerate(row['sources'],1)],'trace':visible_trace(row)})
        cases.append({'id':case['id'],'question':case['question'],'answers':answers})
    output={'title':'真实比较修复 · 原始失败与分层过程' ,'summary':'固定原始资料回放，非新联网。分别展示预算修复、未通过的紧凑Writer、分层协议节点；旧失败和答案原文均保留。正式服务未切换。','cases':cases}
    data=json.dumps(output,ensure_ascii=False,indent=2)
    for directory in (PUBLIC,PREVIEW):directory.mkdir(parents=True,exist_ok=True);(directory/'funnel-repairs-20260921.json').write_text(data)
    print('Exported',sum(len(c['answers']) for c in cases),'answers',len(data.encode()),'bytes')
if __name__=='__main__':main()
