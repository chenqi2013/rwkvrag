"""Select isolated source groups before authoring tasks; no model calls."""
import collections,hashlib,json,random,re
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
CORPUS=ROOT/'data/corpora/finewiki-zh-5000'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def decode(s):return re.sub(r'\\u([0-9a-fA-F]{4})',lambda m:chr(int(m[1],16)),s)
def write(name,value):
 with (HERE/name).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
def main():
 excluded=json.loads((HERE.parent/'reader-v6-expanded/EXCLUSIONS.json').read_text())
 pages=set(excluded['excluded_page_ids']); paths=[]
 existing=json.loads((HERE/'SOURCES-REVISED.json').read_text())
 pages.update(g['page_id'] for g in existing)
 # Read identities from historical annotations/fixtures only, never their targets into training.
 def walk(x):
  if isinstance(x,dict):
   for k,v in x.items():
    if k=='page_id' and isinstance(v,(str,int)):pages.add(str(v))
    if k in ('document_id','id') and isinstance(v,str) and v.startswith('zhwiki/'):
     pages.add(v.split('/')[1].split(':')[0])
    walk(v)
  elif isinstance(x,list):
   for v in x:walk(v)
 for base in [ROOT/'llamaindex-retrieval/statetune/datasets',ROOT/'llamaindex-retrieval/eval']:
  for p in base.rglob('*.jsonl'):
   if HERE in p.parents or not any(w in p.name for w in ('inputs','fixtures','annotations')):continue
   try:
    for line in p.read_bytes().splitlines():walk(json.loads(line))
   except (ValueError,UnicodeError):continue
   paths.append(str(p.relative_to(ROOT)))
 rows=[json.loads(l) for l in (CORPUS/'original-rows.jsonl').read_bytes().splitlines()]
 random.Random(20260912).shuffle(rows)
 manifest={str(r['page_id']):r for r in map(json.loads,(CORPUS/'manifest.jsonl').read_bytes().splitlines())}
 used_titles={re.sub(r"\s+","",g["title"].lower()) for g in existing};used_texts={g["text_sha256"] for g in existing};box_groups=[];prose=[];families=collections.Counter()
 for g in existing:
  if g["kind"]=="wiki_infobox":families["|".join(sorted(k for k,v in g["fields"]))]+=1
 for r in rows:
  page=str(r['page_id']);title=r['title'];norm=re.sub(r'\s+','',title.lower());textsha=sha(r['text'].encode())
  if page in pages or norm in used_titles or textsha in used_texts:continue
  found=None
  try:boxes=json.loads(r['infoboxes'])
  except (KeyError,TypeError,ValueError):boxes=[]
  for bi,b in enumerate(boxes):
   vals=[(decode(k),decode(v)) for k,v in b.get('data',{}).items() if isinstance(v,str)]
   vals=[(k,v) for k,v in vals if 1<len(k)<18 and 0<len(v)<80 and not any(t in v+k for t in ['|','http','{','}','\\','↑','↓','待查','（英文）']) and not k.startswith(('•','（','官方名稱','網站'))]
   if len(vals)<3:continue
   fields=vals[:3];family='|'.join(sorted(k for k,v in fields))
   if families[family]>=4:continue
   found=(bi,fields,family);break
  if found and len(box_groups)<200:
   bi,fields,family=found;families[family]+=1
   g={'page_id':page,'title':title,'kind':'wiki_infobox','box_index':bi,'fields':fields}
   box_groups.append(g)
  elif 180<=len(r['text'])<=950 and len(prose)<40 and not title.startswith(('HD ','NGC ','C1','C2')):
   g={'page_id':page,'title':title,'kind':'wiki_prose','text':r['text'],'authored_facts_required':True}
   prose.append(g)
  else:continue
  pin=manifest[page]
  assert sha((CORPUS/pin['text_path']).read_bytes())==pin['text_sha256']==textsha
  g.update(document_id=r['id'],uri=pin['revision_url'],license=pin['license'],text_sha256=textsha,raw_row_sha256=sha(json.dumps(r,ensure_ascii=False,sort_keys=True).encode()),source_row=pin['source_row'])
  used_titles.add(norm);used_texts.add(textsha);pages.add(page)
 assert len(box_groups)==200 and len(prose)==40
 for g in box_groups+prose:g['split']='train'
 write('ADDITIONAL-SOURCES.json',box_groups+prose)
 write('ADDITIONAL-EXCLUSIONS.json',{'base':'reader-v6-expanded/EXCLUSIONS.json','historical_identity_files':paths,'excluded_page_ids':sorted(pages-set(g['page_id'] for g in box_groups+prose)),'identity_only_exclusion_scan':True})
 write('ADDITIONAL-SOURCE-SELECTION.json',{'counts':dict(collections.Counter((g['split']+'_'+g['kind']) for g in box_groups+prose)),'source_groups':240,'sources_sha256':sha((HERE/'ADDITIONAL-SOURCES.json').read_bytes()),'task_templates_not_independent_sources':True,'corpus_original_rows_sha256':sha((CORPUS/'original-rows.jsonl').read_bytes()),'independent_semantic_review_pending':True})
 print('Selected 200 additional structured and 40 prose sources; no targets yet')
if __name__=='__main__':main()
