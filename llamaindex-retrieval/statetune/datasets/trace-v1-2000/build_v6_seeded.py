"""Generate new-source variants from corrected real-call seeds; review required."""
import collections,hashlib,json,random,sys
from pathlib import Path
P=Path(__file__).resolve().parent;PACKAGE=P.parents[2];ROOT=PACKAGE.parent
sys.path.insert(0,str(PACKAGE/'src'))
from llamaindex_retrieval.writer_prompt import writer_prompt_v2
from llamaindex_retrieval.reader_prompt import binary_query_prompt,parse_binary_decision
from llamaindex_retrieval.rwkv_pipeline import planner_prompt,parse_plan
from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.rwkvos_batch import render_batch_prompt
from llamaindex_retrieval.state_tokens import Vocabulary,encode_training
OLD=P.parent/'writer-v2-expanded'
REJECTED=set('5955009 3129465 5348332 5025689 1801266 8240574 2902194 2902705 2902073 8747776 9042749 5569750 5318762 5685512 8890062 7482864 7704166 42975 6852237 6056596 1582169 3421480 1843007 1342454 5092943 6521659 1242328'.split())
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_bytes())
def write(p,x):
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def jsonl(p,rows):
 with p.open('x') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
def main():
 vocab=Vocabulary(PACKAGE/'statetune/assets/rwkv_vocab_v20230424.txt')
 catalogue=read(P/'PLAN.json');anchors={r['id']:r for r in catalogue['rows']}
 seed_path=P/'correction-seeds-v1/seeds.jsonl'
 seeds=[json.loads(line) for line in seed_path.read_text().splitlines()]
 seed_digest=sha(seed_path.read_bytes())
 seed_by_family=collections.defaultdict(list)
 for seed in seeds:seed_by_family[seed['failure_family']].append(seed)
 assert len(seeds)==17 and all(s['preserve_original_input'] for s in seeds)
 def parents(family,meta):
  candidates=seed_by_family[family]
  if family=='missing_evidence':
   setup=meta['causal_setup']
   if setup.startswith('empty'): names=['correction_empty']
   elif setup.startswith('four requested'):names=['correction_writer_partial_after_reader']
   elif setup.startswith('same topic'):names=['correction_planned_not_actual','correction_aggregate_not_each']
   else:names=['correction_missing_score']
   candidates=[s for s in candidates if s['id'] in names]
  elif family=='units_and_qualifiers':
   name='correction_fee_scope' if meta['causal_setup'].startswith('fee') else 'correction_can_units'
   candidates=[s for s in candidates if s['id']==name]
  assert candidates
  return [dict(seed_id=s['id'],seed_file_sha256=seed_digest,prompt_sha256=s['prompt_sha256'],corrected_output_sha256=s['target_sha256'],seed_design_guidance=s['generation_invariants'],guidance_scope='Seed-level design directions; per-row realized mechanism is causal_setup, not a claim that every direction is present') for s in candidates]
 groups=[g for g in read(OLD/'SOURCES-REVISED.json') if g['split']=='train']+[g for g in read(OLD/'ADDITIONAL-SOURCES.json') if g['page_id'] not in REJECTED]
 ann={g['page_id']:g['facts'] for file in ['PROSE-ANNOTATIONS-CONTEXT.json','ADDITIONAL-PROSE-ANNOTATIONS-REVISED.json'] for g in read(OLD/file)}
 groups.sort(key=lambda g:sha(('trace-curriculum:'+g['page_id']).encode()))
 units={}; facts={}
 for g in groups:
  page=g['page_id'];ff=ann[page] if g['kind']=='wiki_prose' else [dict(field=k,value=v) for k,v in g['fields']]
  facts[page]=ff;units[page]=[]
  for j,f in enumerate(ff):
   text=f.get('quote','| 对象 | 字段 | 资料记录值 |\n| --- | --- | --- |\n| '+g['title']+' | '+f['field']+' | '+f['value']+' |')
   origin={k:g[k] for k in ['document_id','page_id','uri','license','raw_row_sha256','text_sha256']}
   if 'quote' in f:origin.update(representation='verbatim_prose',start=f['start'],end=f['end'])
   else:origin.update(representation='rendered_original_infobox_field',box_index=g['box_index'],decoded_key=f['field'],decoded_value=f['value'])
   units[page].append(dict(id=f"{g['document_id']}/trace/f{j}",title=g['title'],uri=g['uri'],text=text,context_spans=f.get('context_spans',[]),fields=[],origin=origin))
 rows=[]
 def pack(identity,stage,family,body,target,meta):
  prompt,_=render_batch_prompt([{'role':'user','content':body}],'<think></think>')
  if stage in ('writer','resolver'):prompt+='\n'
  encoded=encode_training(prompt,target,vocab,8192)
  assert encoded['prompt_tokens']+2047<=8192,(identity,encoded['prompt_tokens'])
  parent_seeds=parents(family,meta)
  rows.append(dict(parent_corrected_seeds=parent_seeds,generation_method='new source or explicitly fictional controlled variant of corrected seed; wrong output excluded from model input and target',id=identity,split='train',stage=stage,primary_failure_family=family,anchor=anchors[family]['anchor'],causal_setup=meta.pop('causal_setup'),prompt=prompt,prompt_sha256=sha(prompt.encode()),target=target,target_sha256=sha(target.encode()),prompt_tokens=encoded['prompt_tokens'],target_tokens_with_eos=encoded['target_tokens_with_eos'],author='root',transport_protocol=('rwkvos_no_think_complete_no_final_lf' if stage=='planner' else 'rwkv_g1j_no_think_v1'),independent_review_pending=True,**meta))
 def writer(family,n,q,material,claims,missing='',history=None,setup='',force_first_label=False):
  identity=f'trace_writer_{family}_{n:04}';material=list(material);random.Random(identity).shuffle(material)
  if force_first_label:
   # Balance all six source positions, including1, independently of answer order.
   support=next(u for u in material if u['id']==claims[0]['source_id'])
   material.remove(support);material.insert(n%6,support)
  assert len({u['id'] for u in material})==len(material)
  evidence=[{**{k:v for k,v in u.items() if k!='origin'},'label':f'资料 {i}'} for i,u in enumerate(material,1)]
  label={e['id']:i for i,e in enumerate(evidence,1)}
  target='\n'.join(c['statement']+f"[资料 {label[c['source_id']]}]。" for c in claims)
  if missing:target+=('\n' if target else '')+missing
  history=history or [];fields=[];task=json.dumps({'history':history,'latest_question':q},ensure_ascii=False)
  pack(identity,'writer',family,writer_prompt_v2(task,evidence,fields),target,dict(question=q,history=history,evidence=evidence,fields=fields,origins=[u['origin'] for u in material],claims=[{**c,'citation':label[c['source_id']]} for c in claims],missing_requirement=missing,causal_setup=setup))
 def claim(g,j):
  f=facts[g['page_id']][j]
  return dict(object=g['title'],field=f['field'],value=f['value'],source_id=units[g['page_id']][j]['id'],statement=f"《{g['title']}》的{f['field']}：{f['value']}")
 def at(i):return groups[i%len(groups)]
 def U(g):return units[g['page_id']]
 def doc(group,tag,text):return dict(id=f'{group}/{tag}',title=f'虚构验证记录 {group}（{tag}）',uri=f'urn:rwkv:trace:{group}:{tag}',text=text,context_spans=[],fields=[],origin=dict(group_id=group,synthetic=True,author='root',text_sha256=sha(text.encode())))
 def C(field,value,u,object=''):
  return dict(field=field,value=value,object=object,source_id=u['id'],statement=field+'：'+value)
 # Same field, competing objects and distinct values; nontrivial citation attribution.
 pairs=[]
 for ai,a in enumerate(groups):
  for b in groups[ai+1:]:
   for aj,fa in enumerate(facts[a['page_id']]):
    for bj,fb in enumerate(facts[b['page_id']]):
     if fa['field']==fb['field'] and fa['value']!=fb['value']:
      pairs.append((a,aj,b,bj));break
    else:continue
    break
 pairs.sort(key=lambda v:sha((v[0]['page_id']+':'+v[2]['page_id']).encode()))
 assert len(pairs)>=250,len(pairs)
 for i,(a,aj,b,bj) in enumerate(pairs[:250]):
  field=facts[a['page_id']][aj]['field']
  q=f"仅根据提供的快照，先回答《{a['title']}》的{field}，再回答《{b['title']}》的{field}，逐项引用。"
  writer('citation_binding',i,q,U(a)+U(b),[claim(a,aj),claim(b,bj)],setup='same requested field across two source objects with distinct values, six evidence units; first supporting label balanced across1..6',force_first_label=True)
 # Full competing rows/three dates: set selection precedes date binding.
 themes=[('船舶','置作人工鱼礁','转交使用','退役日期','置礁日期'),('设备','回收再用','转交使用','停用日期','再用日期'),('档案','公开开放','内部移交','归档日期','开放日期')]
 for i in range(300):
  kind,wanted,other,event0,event1=themes[i%3];group=f'rowbinding-{i:03}';rng=random.Random(group)
  row_count=4+i%5; names=[f'{kind}{chr(65+j)}-{i+17}' for j in range(row_count)]; selected=set(rng.sample(range(row_count),1+(i//5)%(row_count-1)));table=[];answers=[]
  for j,name in enumerate(names):
   year=1990+(i%30)
   date0=f'{year+j//2}-{1+(i+j)%6:02}-{1+(i*3+j)%27:02}';date1=f'{year+j//2+(j%2)}-{7+(i+2*j)%6:02}-{1+(i+j*3)%27:02}'
   status=wanted if j in selected else other
   table.append(f'| {name} | {status} | {date0} | {date1 if j in selected else "未记载"} |')
   if j in selected:answers.append((name,date1))
  rng.shuffle(table)
  text=f'以下是明确虚构的登记快照。资料发布日期为{year+6}-12-09，不能作事件日期。\n| 对象 | 处置状态 | {event0} | {event1} |\n| --- | --- | --- | --- |\n'+'\n'.join(table)
  u=doc(group,'登记表',text);noise=doc(group,'参考文献',f'文献题名：{kind}登记表汇编。文献出版日期{year+5}-04-03，访问日期{year+6}-12-10。这两者不表示任何对象的{event1}。')
  claims=[C(f'{name}的{event1}',date,u,name) for name,date in answers]
  q=f'表中哪些{kind}被{wanted}？分别列出它们的{event1}，不要把{event0}、出版日或访问日混入。'
  writer('entity_field_binding',i,q,[u,noise],claims,setup=f'all {row_count} rows retained; {len(selected)} selected by disposition; three competing date roles incl citation date; answer uses event-specific date')
 # Separate technical and fee-policy scenarios; never join unrelated trace topics.
 for i in range(150):
  group=f'unit-scope-{i:03}'
  if i%2==0:
   width=[8,16,32,64,128][i%5];bits=[11,19,29,31][(i//2)%4];rate=2+i%13
   u=doc(group,'协议',f'虚构协议{i}：数据区上限{width}字节，标识符为{bits}位。只有线路长度不超过{12+i%17}米且双端接终端电阻时，数据率上限为{rate} Mbit/s。条件不满足时，本资料不保证达到该速率。')
   claims=[C('数据区上限',f'{width}字节',u),C('标识符长度',f'{bits}位',u),C('数据率及适用条件',f'{rate} Mbit/s；仅限线路长度不超过{12+i%17}米且双端接终端电阻',u)]
   writer('units_and_qualifiers',i,'按这份协议给出数据区上限、标识符长度、最高数据率及其适用条件。',[u],claims,setup='technical-only: bit/byte coexist and absolute rate is conditional; no invented unit conversion')
  else:
   fraction=['一半','四分之一','三分之二','30%','75%'][i%5];limit=210+i*3
   u=doc(group,'费用方案',f'虚构资助方案{i}：仅减免自修生考试费的{fraction}，不是全额免除；不应按申请人的学校类别一刀切剔除。每人资助上限为{limit}元，项目总预算为{2+i%9}百万元；总预算不等于个人上限。')
   claims=[C('减免比例和范围',f'自修生考试费的{fraction}；不是全额免除，不应按学校类别一刀切剔除',u),C('个人资助上限',f'{limit}元',u)]
   writer('units_and_qualifiers',i,'这项方案减免多少、适用范围有什么限制？每人的资助上限是多少？',[u],claims,setup='fee-policy-only: variable partial proportion, explicit non-exclusion rule and per-person ceiling versus total budget; analogue of6066934_02 qualifier loss')
 for i in range(300):
  a,b=at(i),at(i+103);f=facts[a['page_id']];ua,ub=U(a),U(b);variant=i%4
  q=f"《{a['title']}》的{f[0]['field']}是什么？"
  if variant==0:writer('missing_evidence',i,q,[],[],missing='当前未提供参考资料，无法据此确定。',setup='empty evidence; no fabricated fact or source label')
  elif variant==1:writer('missing_evidence',i,q,ub,[],missing='现有资料没有提供所问对象的相关信息，无法据此确定。',setup='only another object is presented; source title is part of evidence identity')
  elif variant==2:
   writer('missing_evidence',i,f"分别给出《{a['title']}》的{f[0]['field']}、{f[2]['field']}，以及《{b['title']}》的{facts[b['page_id']][0]['field']}和{facts[b['page_id']][2]['field']}。",[ua[0],ub[0]],[claim(a,0),claim(b,0)],missing=f"《{a['title']}》的{f[2]['field']}及《{b['title']}》的{facts[b['page_id']][2]['field']}：当前提供的资料未记载，无法确定。",setup='four requested facts; two supported and two withheld; review must rule out leakage in all source/context spans')
  else:
   group=f'nearby-missing-{i:03}';planned=7000+i*11;total=10+i%41
   u=doc(group,'计划',f'虚构项目{i}计划派发{planned}张使用券，预算总额为{total}亿元。未列出实际核销统计，未披露各市场的分项金额。')
   writer('missing_evidence',i,'实际使用了多少张券？各市场分别获得多少资金？',[u],[],missing='实际使用券数及各市场分项金额均未提供，无法确定；计划派发量不等于实际使用量，总预算也不能替代分项金额。',setup='same topic, tempting planned count and total amount; neither requested actual count nor disaggregated amounts supplied')
 for i in range(150):
  a,b=at(i+37),at(i+191);claims=[claim(g,j) for g in [a,b] for j in range(3)]
  q='请逐项回答以下六项，不增加其他背景：'+'；'.join(c['object']+'的'+c['field'] for c in claims)+'。'
  writer('multi_part_coverage',i,q,U(a)+U(b),claims,setup='six explicit requirements across two source objects; no dropped item or invented replacement')
 for i in range(150):
  a,b,c=at(i+11),at(i+97),at(i+223);ca,cb=claim(a,0),claim(b,0)
  old=claim(c,0);extra=claim(a,1)
  history=[dict(role='user',content=f"文件夹标题是《{c['title']}》，先准备它的{old['field']}，还想整理《{a['title']}》的{extra['field']}。先别填答案，我还在核对目录。"),dict(role='assistant',content='先以核对后的对象和字段为准。')]
  history.extend([dict(role='user',content=f"目录看清了，第一项应是《{a['title']}》的{ca['field']}；文件夹名是之前留下的，那个对象不用做。第二项先留着。"),dict(role='assistant',content='第一项按刚确认的目录处理。')])
  messages=[
   f"电子版已经整理好。刚才说的第二项{extra['field']}移到另一页，这里不回答它。",
   f"这页还需要《{b['title']}》的{cb['field']}，放在已核对的第一项后面。不要把这句话当作替换第一项。",
   f"桌上的照片不在资料里，照片顺序不能当作这些字段的值，也没有让你识别图片。",
   f"旧草稿还写着《{c['title']}》，那是文件名遗留；这页不恢复它，也不比较这几个对象哪个更好。",
   f"如果资料没记载就说缺少依据；这只是取值阅读，不要推荐后续方案。"]
  if i%2: messages[0],messages[1]=messages[1],messages[0]
  for content in messages:history.extend([dict(role='user',content=content),dict(role='assistant',content='按已确认的字段查证，不从工作安排推断答案。')])
  history.append(dict(role='assistant',content='也可以再补充对象背景、照片来源和其他字段。'))
  writer('history_correction',i,'不用刚才那些扩展。把这页最后留下的两项按顺序给我，各附来源。',U(a)+U(b)+U(c),[ca,cb],history=history,setup='15 history messages: initial object corrected, separate field removed and second object added in varying order; indirect latest request; irrelevant work logistics and unsolicited assistant expansion retained')
 for i in range(100):
  a=at(i+5);material=U(a)+[u for j in range(1,11) for u in U(at(i+5+j))]
  ca=claim(a,0)
  writer('bounded_answer_eos',i,f"只回答《{a['title']}》的{ca['field']}，给出依据后结束。",material,[ca],setup='33 complete evidence units with many unrequested facts; one required fact; short complete target followed by real EOS')
 # Reader direct-positive examples retain exact production binary JSON protocol.
 def reader(family,n,query,u,label,setup):
  source={k:u[k] for k in ['id','title','uri']};contexts=u['context_spans']
  target=json.dumps({'answer':label},ensure_ascii=False);assert parse_binary_decision(target)==(label=='YES')
  pack(f'trace_reader_{family}_{n:04}','resolver',family,binary_query_prompt([query],source,contexts,u['text']),target,dict(queries=[query],source=source,contexts=contexts,text=u['text'],origins=[u['origin']],expected_decision=label,causal_setup=setup,mechanism_relation=('direct_observed_failure_analogue' if family=='reader_false_negative' or n%6<3 else 'preventive_control')) )
 for i in range(150):
  g=at(i);j=i%3;f=facts[g['page_id']][j]
  reader('reader_false_negative',i,f"《{g['title']}》的{f['field']}是什么？",U(g)[j],'YES','single verbatim evidence unit directly states requested fact; preserve parent contexts and actual source identity')
 for i in range(50):
  group=f'reader-ratio-{i:03}';name=f'通道-{i+401}';ratio=2+i%7;value=11+i%31;unit=['Mbit/s','km/h','件/小时','升/分钟','帧/秒'][i%5]
  descriptions=[(f'{name}版本2的最高速率为旧版本的{ratio}倍，旧版本的绝对速率未记载。',f'{name}版本2的最高绝对速率是多少{unit}？','NO'),(f'{name}版本2比旧版本快{ratio}倍，最高绝对速率明确为{value} {unit}。',f'{name}版本2的最高绝对速率是多少{unit}？','YES'),(f'{name}版本2的最高速率为旧版本的{ratio}倍。',f'{name}版本2最高速率是旧版本的几倍？','YES'),(f'{name}版本1最高绝对速率为{value} {unit}，版本2尚未给数值。',f'{name}版本2最高绝对速率是多少{unit}？','NO'),(f'{name}版本1最高速率为9 {unit}，版本2为{value} {unit}。',f'{name}版本2最高绝对速率是多少{unit}？','YES'),(f'{name}版本2最高速率为{value} {unit}，仅测试速率。',f'{name}版本2的维护负责人是谁？','NO')]
  for j,(text,query,label) in enumerate(descriptions):reader('reader_relative_absolute',i*6+j,query,doc(group,f'case{j}',text),label,'controlled contrast: relative vs absolute, asked ratio, version mismatch, mixed versions or unprovided facet; single-unit decision cannot borrow other evidence')
 # Planner examples contain discarded history >6 requests, but only necessary current queries.
 settings=Settings(native_plan_protocol='queries_fields',native_max_queries=6,native_max_fields=12)
 for i in range(150):
  requested=[at(i+j*43) for j in range(1+i%6)]; c=at(i+277)
  active=[f"《{g['title']}》的"+'、'.join(f['field'] for f in facts[g['page_id']][:2]) for g in requested]
  original='原始任务：'+'；'.join(active)+'。所有这些组都需要。'
  history=[]
  if i%4:
   history=[dict(role='user',content=original)]
   for k in range(3):
    field=facts[c['page_id']][k]['field']
    history.extend([dict(role='user',content=f"下一页可能还要《{c['title']}》的{field}，先记在这里。"),dict(role='assistant',content='暂记，等你确认是否放到这次结果。'),dict(role='user',content=f"《{c['title']}》的{field}已经由同事处理，这次不用查；开头指定的组没有减少。")])
   history.extend([dict(role='user',content='打印安排和照片页码只是我们的工作记录，不是检索条件。开头那份清单仍是本次范围，按它逐组检索。'),dict(role='assistant',content='还可以顺便查对象的更多背景。')])
   q='按最后确认的任务检索。'
  else:q=original+'不要额外扩展要求。'
  target=json.dumps({'queries':active,'fields':[f"《{g['title']}》的{f['field']}" for g in requested for f in facts[g['page_id']][:2]]},ensure_ascii=False)
  parsed=parse_plan(target,settings);assert len(parsed['queries'])==1+i%6 and len(parsed['fields'])==2*(1+i%6)
  task=json.dumps({'history':history,'latest_question':q},ensure_ascii=False)
  pack(f'trace_planner_{i:04}','planner','planner_contract',planner_prompt(task,settings),target,dict(question=q,history=history,origins=[{'document_id':g['document_id'],'page_id':g['page_id'],'text_sha256':g['text_sha256']} for g in requested+[c]],active_requirements=parsed,causal_setup='variable1..6 retained objects and2..12 fields; long discarded-history cases plus direct-request controls; no copied duplicate queries'))
 assert len(rows)==len({r['id'] for r in rows})==2000
 assert collections.Counter(r['stage'] for r in rows)=={'writer':1400,'resolver':450,'planner':150}
 out=P/'draft-v6-seeded';out.mkdir(exist_ok=False)
 for stage in ['writer','resolver','planner']:
  rr=[r for r in rows if r['stage']==stage];jsonl(out/(stage+'.drafts.jsonl'),rr)
  jsonl(out/(stage+'.tokens.jsonl'),[{'id':r['id'],'prompt_sha256':r['prompt_sha256'],**encode_training(r['prompt'],r['target'],vocab,8192)} for r in rr])
 write(out/'DRAFT.json',dict(seed_file_sha256=seed_digest,seed_count=len(seeds),training_admitted=False,independent_semantic_and_mechanism_review_required=True,counts=dict(collections.Counter(r['stage'] for r in rows)),families=dict(collections.Counter(r['primary_failure_family'] for r in rows)),real_source_pool=len(groups),excluded_additional_source_pages=sorted(REJECTED),max_prompt_tokens=max(r['prompt_tokens'] for r in rows),max_target_tokens=max(r['target_tokens_with_eos'] for r in rows),files={f.name:sha(f.read_bytes()) for f in out.iterdir()},limitations=['synthetic controlled contrast scenarios are labeled, not real observed questions','source semantic approval does not establish defect mechanism coverage','all prompts use production stage format but benchmark inference still required','seed variants remain authored templates; real long-history improvement requires independent challenge evaluation','Reader version/field controls and unrelated-object Writer cases are preventive extensions, not exact observed failures']))
 print(json.dumps(read(out/'DRAFT.json'),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
