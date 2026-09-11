"""Author corrections after independent rejection; preserve original files."""
import json,re,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[3]
def dec(s):return re.sub(r'\\u([0-9a-fA-F]{4})',lambda m:chr(int(m[1],16)),s)
def write(n,r):
 with (P/n).open('x') as f:json.dump(r,f,ensure_ascii=False,indent=2);f.write('\n')
def main():
 groups=json.loads((P/'SOURCES.json').read_text());raw={str(r['page_id']):r for r in map(json.loads,(ROOT/'data/corpora/finewiki-zh-5000/original-rows.jsonl').read_bytes().splitlines())}
 ann=json.loads((P/'PROSE-ANNOTATIONS.json').read_text());by={r['page_id']:r for r in ann}
 fixes={
 '7985587':[['选举日期','1985年1月15日','1985年巴西總統選舉于1985年1月15日在巴西举行'],['是否为最后一次由选举人团参与投票的总统选举','是','这是巴西最后一次由選舉人團參與投票的總統選舉'],['是否为军政府时期最后一次总统选举','是','也是巴西軍政府时期的最后一次總統選舉。']],
 '2861720':[['建设年份','1986年','凉伞站位于福建省南平市延平区太平镇，建于1986年。'],['距来舟站距离','64公里','距来舟站64公里，距福州站130公里。'],['是否办理包裹托运','不办理','客运办理旅客乘降、行李托运，不办理包裹托运；不办理货运业务。']],
 '6069162':[['3月2日巴库戒毒中心大火的死亡人数下限','至少26人','阿塞拜疆首都巴库市区一处戒毒中心突发大火，造成至少26人死亡，4人受伤。'],['3月8日宣布的进口钢铁关税','25%','對進口鋼鐵徵收25%的关税'],['3月8日宣布的进口铝关税','10%','對進口鋁徵收10%的關稅']],
 '3790333':[['雅乐体系制定时期','西周初年','雅乐的体系在西周初年制定'],['雅字的含义','正','"雅"就是"正"的意思'],['研究雅乐的基础','儒家经学和二十四史《礼乐志》','儒家经学和二十四史《礼乐志》是研究雅乐的基础。']]}
 for g in groups:
  page=g['page_id'];r=raw[page]
  if page in ('2899841','2901031','2902065','8716488'):
   boxes=json.loads(r['infoboxes']);bi=1 if page=='8716488' else 0;keys=['CAS号','PubChem CID','ChemSpider'] if bi==1 else ['赤經','赤緯','視星等（V）']
   data={dec(k):dec(v) for k,v in boxes[bi]['data'].items() if isinstance(v,str)}
   g.update(box_index=bi,fields=[[k,data[k]] for k in keys])
  if page in fixes:
   g.update(kind='wiki_prose',text=r['text'],authored_facts_required=True);g.pop('fields');g.pop('box_index')
   facts=[]
   for k,v,q in fixes[page]:
    assert q in r['text'];start=r['text'].index(q)
    facts.append(dict(field=k,value=v,quote=q,start=start,end=start+len(q)))
   ann.append(dict(page_id=page,author='root',facts=facts))
 # Preserve the subject governing the western country's figures.
 f=by['1244988']['facts'][0];q='西且弥国的治所在天山东于大谷，离长安8670里。西汉时332户，1926口，兵士738人。';f.update(quote=q,start=raw['1244988']['text'].index(q),end=raw['1244988']['text'].index(q)+len(q))
 f=by['924568']['facts'][2];q='学校的负责人福尔廷·奥古斯廷牧师承认，这所学校全部由他设计建造，建造期间没有雇用任何建筑工程师。他于11月8日被捕，被控过失杀人。';f.update(quote=q,start=raw['924568']['text'].index(q),end=raw['924568']['text'].index(q)+len(q))
 write('SOURCES-REVISED.json',groups);write('PROSE-ANNOTATIONS-REVISED.json',ann)
 write('SOURCE-CORRECTIONS.json',{'rejected_original_pages':sorted(list(fixes)+['2899841','2901031','2902065','8716488']),'same_page_split_preserved':True,'four_structured_sources_changed_to_prose':list(fixes),'original_files_preserved':True,'review_pending':True})
 print('Revised 8 sources; preserved splits and old drafts')
if __name__=='__main__':main()
