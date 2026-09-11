"""Build review drafts from verified sources and explicitly fictional scenarios.

No model labels, no embeddings, no semantic admission from mechanical checks.
"""
import collections,hashlib,json,random,sys
from pathlib import Path
P=Path(__file__).resolve().parent;PACKAGE=P.parents[2]
sys.path.insert(0,str(PACKAGE/'src'))
from llamaindex_retrieval.writer_prompt import writer_prompt_v2
from llamaindex_retrieval.rwkvos_batch import render_batch_prompt
from llamaindex_retrieval.state_tokens import Vocabulary,encode_training

def sha(b):return hashlib.sha256(b).hexdigest()
def dump(p,x):
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def linefile(p,rows):
 with p.open('x') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
def main():
 groups=json.loads((P/'SOURCES-REVISED.json').read_text())
 prose={r['page_id']:r['facts'] for r in json.loads((P/'PROSE-ANNOTATIONS-REVISED.json').read_text())}
 vocab=Vocabulary(PACKAGE/'statetune/assets/rwkv_vocab_v20230424.txt')
 units={};facts={}
 for g in groups:
  page=g['page_id'];units[page]=[]
  ff=prose[page] if g['kind']=='wiki_prose' else [dict(field=k,value=v) for k,v in g['fields']]
  facts[page]=ff
  for j,f in enumerate(ff):
   text=f['quote'] if 'quote' in f else '| 字段 | 资料记录值 |\n| --- | --- |\n| '+f['field']+' | '+f['value']+' |'
   origin={'document_id':g['document_id'],'page_id':page,'uri':g['uri'],'raw_row_sha256':g['raw_row_sha256'],'text_sha256':g['text_sha256'],'license':g['license'],'representation':'verbatim_prose' if 'quote' in f else 'rendered_original_infobox_field'}
   if 'quote' in f:
    assert g['text'][f['start']:f['end']]==text
    origin.update(start=f['start'],end=f['end'])
   else:origin.update(box_index=g['box_index'],decoded_key=f['field'],decoded_value=f['value'])
   units[page].append(dict(id=f"{g['document_id']}/writer-v2/f{j}",title=g['title'],uri=g['uri'],text=text,context_spans=[],fields=[],origin=origin))
 rows=[]
 def emit(group_id,split,kind,family,question,history,material,claims,missing=None):
  identity=f'writer_v2_{group_id}_{family}'
  rng=random.Random(identity);material=list(material);rng.shuffle(material)
  origins=[u['origin'] for u in material]
  evidence=[{**{k:v for k,v in u.items() if k!='origin'},'label':f'资料 {i}'} for i,u in enumerate(material,1)]
  labels={e['id']:i for i,e in enumerate(evidence,1)}
  target='\n'.join(f'{claim["field"]}：{claim["value"]}[资料 {labels[claim["source_id"]]}]。' for claim in claims)
  if missing:
   target+=('\n' if target else '')+missing
  assert target
  task=json.dumps({'history':history,'latest_question':question},ensure_ascii=False)
  # Include genuine requested task hints in half the rows; never answer values.
  fields=[question] if rng.randrange(2) else []
  body=writer_prompt_v2(task,evidence,fields)
  prompt,_=render_batch_prompt([{'role':'user','content':body}],'<think></think>');prompt+='\n'
  encoded=encode_training(prompt,target,vocab,8192)
  assert encoded['prompt_tokens']+2047<=8192
  rows.append(dict(id=identity,group_id=group_id,split=split,kind=kind,family=family,question=question,history=history,evidence=evidence,fields=fields,origins=origins,prompt=prompt,prompt_sha256=sha(prompt.encode()),target=target,target_sha256=sha(target.encode()),claims=[{**c,'citation':labels[c['source_id']]} for c in claims],missing_requirement=missing,prompt_tokens=encoded['prompt_tokens'],target_tokens_with_eos=encoded['target_tokens_with_eos'],author='root',independent_review_pending=True))
 for split in ['train','evaluation']:
  gg=[g for g in groups if g['split']==split]
  for i,g in enumerate(gg):
   page=g['page_id'];ff=facts[page];u=units[page];title=g['title'];other=units[gg[(i+1)%len(gg)]['page_id']][:2]
   claim=lambda j:dict(field=ff[j]['field'],value=ff[j]['value'],source_id=u[j]['id'])
   prefix=f'按提供资料对《{title}》的记载，'
   emit(page,split,g['kind'],'single',prefix+ff[0]['field']+'是什么？',[],u+other,[claim(0)])
   emit(page,split,g['kind'],'multiple',prefix+'分别说明'+ '、'.join(f['field'] for f in ff)+'。',[],u+other,[claim(j) for j in range(3)])
   history=[dict(role='user',content=f'我想了解《{title}》的{ff[0]["field"]}。'),dict(role='assistant',content='请确认你要查询的字段。')]
   emit(page,split,g['kind'],'history',f'更正：不问{ff[0]["field"]}，只回答《{title}》的{ff[1]["field"]}。',history,u+other,[claim(1)])
   emit(page,split,g['kind'],'partial',prefix+ff[0]['field']+'和'+ff[2]['field']+'分别是什么？',[],[u[0]]+other,[claim(0)],ff[2]['field']+'：当前资料未提供，无法确定。')
   empty=(i%2==0)
   emit(page,split,g['kind'],'empty' if empty else 'irrelevant',prefix+ff[0]['field']+'是什么？',[],[] if empty else other,[],('当前未提供参考资料，无法据此回答。' if empty else '现有资料未提供所问对象的相关信息，无法据此确定。'))
 # Explicitly fictional operational documents; no invented Wikipedia provenance.
 for split,count,offset in [('train',20,0),('evaluation',10,100)]:
  for i in range(count):
   n=i+offset;group=f'synthetic_{n:03}';a=f'杉舟-{n+31}';b=f'岚舟-{n+73}';v1=f'{6+n%7} ms';v2=f'{11+n%9} ms';day=f'2024-{1+n%12:02}-{1+n%27:02}'
   def doc(tag,text):return dict(id=f'{group}/{tag}',title=f'虚构测试项目{n}记录（{tag}）',uri=f'urn:rwkv:synthetic:{group}:{tag}',text=text,context_spans=[],fields=[],origin=dict(group_id=group,synthetic=True,author='root',text_sha256=sha(text.encode())))
   old=doc('v1',f'虚构场景，版本1。\n| 对象 | 响应时限 | 状态 |\n| --- | --- | --- |\n| {a} | {v1} | 未启用 |\n| {b} | 24 ms | 已启用 |')
   new=doc('v2',f'虚构场景，版本2。版本2替代版本1。\n| 对象 | 响应时限 | 生效日期 |\n| --- | --- | --- |\n| {b} | 19 ms | 2025-01-09 |\n| {a} | {v2} | {day} |')
   conflicting=doc('conflict',f'虚构场景，另一个来源也声称为版本2，未提供优先级。{a}的响应时限为43 ms。')
   c=lambda field,value,d:dict(field=field,value=value,source_id=d['id'])
   emit(group,split,'synthetic','table',f'按版本2，分别给出{a}和{b}的响应时限，以及{a}的生效日期。',[],[old,new],[c(a+'响应时限',v2,new),c(b+'响应时限','19 ms',new),c(a+'生效日期',day,new)])
   emit(group,split,'synthetic','versions',f'分别列出{a}在版本1和版本2的响应时限，不要混用版本。',[],[old,new],[c('版本1',v1,old),c('版本2',v2,new)])
   emit(group,split,'synthetic','history',f'更正对象为{b}，只问版本2的响应时限，撤回先前对象。',[dict(role='user',content=f'查询{a}版本1的响应时限。')],[old,new],[c(b+'版本2响应时限','19 ms',new)])
   emit(group,split,'synthetic','partial',f'版本2的{a}响应时限和维护负责人分别是什么？',[],[old,new],[c('响应时限',v2,new)],'维护负责人：资料未记载，无法确定。')
   emit(group,split,'synthetic','conflict',f'两份资料都称为版本2，{a}的响应时限究竟是多少？若不能确定，请说明冲突。',[],[new,conflicting],[c('一份版本2记录',v2,new),c('另一份版本2记录','43 ms',conflicting)],'两份记录冲突，未提供优先级，无法据此确定唯一响应时限。')
 counts=collections.Counter(r['split'] for r in rows);assert counts=={'train':600,'evaluation':200}
 assert len({r['id'] for r in rows})==800
 out=P/'draft';out.mkdir(exist_ok=False)
 for split in ['train','evaluation']:
  rr=[r for r in rows if r['split']==split];linefile(out/(split+'.drafts.jsonl'),rr)
  if split=='train':
   tokens=[{'id':r['id'],'prompt_sha256':r['prompt_sha256'],**encode_training(r['prompt'],r['target'],vocab,8192)} for r in rr]
   linefile(out/'train.tokens.jsonl',tokens)
 stats={'counts':dict(counts),'base_source_groups':{s:len({r['group_id'] for r in rows if r['split']==s}) for s in counts},'by_split_kind_family':dict(collections.Counter(r['split']+'/'+r['kind']+'/'+r['family'] for r in rows)),'max_prompt_tokens':max(r['prompt_tokens'] for r in rows),'max_target_tokens':max(r['target_tokens_with_eos'] for r in rows),'training_admitted':False,'independent_review_required':True,'test_is_not_blind_to_authors':True,'same_templates_across_splits':True,'files':{f.name:sha(f.read_bytes()) for f in out.iterdir()}}
 dump(out/'DRAFT.json',stats);print(json.dumps(stats,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
