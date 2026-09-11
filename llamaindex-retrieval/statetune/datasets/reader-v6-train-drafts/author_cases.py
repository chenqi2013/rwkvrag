"""Explicit human-readable author judgments; quote lookup only supplies coordinates."""
import json
from copy import deepcopy
from pathlib import Path
from llamaindex_retrieval.statetune import digest, render, validate_case, write_rows
from llamaindex_retrieval.state_tokens import Vocabulary

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
SPECS = {
"7133379": [
 ("峰會徽標競賽的獲獎設計師得到多少盧布獎金？", {"E1": ["获奖徽章是由专业设计师，符拉迪沃斯托克（Vladivostok）居民叶夫根尼·波格勒布尼亚克（Yevgeny Pogrebnyak）创作的，他获得了65,000卢布的奖金。"]}),
 ("2008年4月的審計中，36個峰會建設項目有幾個得到正確記錄？", {"E2": ["审计表明，在与峰会有关的36个建设项目中，只有6个得到了正确记录。"]}),
 ("2009年3月下旬討論峰會計劃的會議，推遲了哪些建設項目？", {"E3": ["3月下旬，舒瓦洛夫在莫斯科会议，讨论首脑会议计划问题。会议证实首脑会议将在符拉迪沃斯托克举行，而不是在其他地方。会议期间，修建歌剧院和芭蕾舞剧院以及医疗中心的计划被推迟了，但是其余的建设应按计划继续进行。"]}),
 ("峰會前的青年論壇於哪幾天、在什麼校園舉行？", {"E4": ["该论坛于9月2日至4日在符拉迪沃斯托克举行的APEC-2012领导人周同时在远东联邦大学校园内举行。"]}),
 ("俄羅斯在哪次APEC會議提出主辦2012年峰會的願望？青年論壇結束後，代表們向領導人會議提交了什麼？", {"E1": ["在越南河内举行的2006年APEC会议上，俄罗斯提出了主办2012年APEC峰会的愿望。"], "E4": ["在论坛结束时，代表们发表了自己的宣言，并提交给亚太经合组织领导人会议的与会者。"]}),
 ("2012年9月9日峰會開幕時，俄羅斯島的實測氣溫是多少攝氏度？", {}),
 ("首筆4.37億盧布撥款具體按什麼比例分給各個承建企業？", {}),
],
"7340228": [
 ("上小川站最初在哪一天啟用？", {"E1": ["1925年（大正14年）8月15日：國有鐵道車站啟用[1]。"]}),
 ("上小川站2017年度的1日平均乘車人次是多少？", {"E2": ["| 2017年（平成29年） | 54               | [ 利用客数 18 ] |"]}),
 ("改問相鄰車站：上小川站在水郡線上夾在哪兩個車站之間？", {"E2": ["西金－上小川－袋田"], "E3": ["西金－上小川－袋田"]}),
 ("請同時找出上小川站的啟用日期，以及2017年度的1日平均乘車人次。", {"E1": ["1925年（大正14年）8月15日：國有鐵道車站啟用[1]。"], "E2": ["| 2017年（平成29年） | 54               | [ 利用客数 18 ] |"]}),
 ("從上小川站到水戶站的成人單程票價是多少日圓？", {}),
 ("上小川站在2019年每天有多少班客運列車停靠？", {}),
],
"8468551": [
 ("資料記載的2023年8月6日平原縣地震，震源深度是多少千米？", {"E1": ["山东省德州市平原县发生規模5.5級地震，震源深度10千米[4]。"]}),
 ("國家統計局在2023年8月哪一天宣布不再公布分年齡段青年失業率？", {"E2": ["## 8月15日\n- 國家統計局宣佈自8月起將不再公佈分年齡段青年失業率[24][25]。"]}),
 ("改問聖油229油輪：8月22日著火時，船上共有多少人？", {"E3": ["下午2時30分許，廣西欽州市北部灣海域一艘深圳市海昌华海运股份有限公司的“聖油229”油輪發生著火，造成2人死亡。下午4時45分許，火已撲滅。當時船上共17人[40][41]。"]}),
 ("分別找出平原縣地震的震源深度，以及聖油229油輪著火時的船上總人數。", {"E1": ["山东省德州市平原县发生規模5.5級地震，震源深度10千米[4]。"], "E3": ["下午2時30分許，廣西欽州市北部灣海域一艘深圳市海昌华海运股份有限公司的“聖油229”油輪發生著火，造成2人死亡。下午4時45分許，火已撲滅。當時船上共17人[40][41]。"]}),
 ("2023年8月6日平原縣地震一共損毀了多少棟住宅？", {}),
 ("8月23日鳳凰嶺隧道事故中，大巴撞擊隧道內牆時的車速是多少？", {}),
],
"8753422": [
 ("GingaMingaYo這首歌曲的調性、每分鐘拍數和時長分別是什麼？", {"E1": ["歌曲以降B小調作曲，速度為每分鐘122拍，時長3分35秒。"]}),
 ("GingaMingaYo的音樂錄影帶由哪兩位導演執導？", {"E2": ["該錄影帶由Zanybros的洪元基和沈智瀅執導"]}),
 ("不問導演了，改問日版CD單曲的唱片公司是哪一家？", {"E3": ["| 全球 | 2023年5月17日 | CD單曲            | 日版 | JVC建伍胜利娱乐        |"]}),
 ("請找出歌曲每分鐘的拍數，以及日版CD單曲的唱片公司。", {"E1": ["歌曲以降B小調作曲，速度為每分鐘122拍，時長3分35秒。"], "E3": ["| 全球 | 2023年5月17日 | CD單曲            | 日版 | JVC建伍胜利娱乐        |"]}),
 ("GingaMingaYo音樂錄影帶的拍攝總預算是多少韓元？", {}),
 ("GingaMingaYo韓版歌曲發行首日的全球串流播放總量是多少？", {}),
]}
TOPICS = {
"7133379": ["峰會概況、最初申辦與徽標", "2008年上半年審計與建設", "2008年下半年至2009年建設調整及批評", "青年論壇及未來之聲"],
"7340228": ["車站歷史、結構及早年乘車人次", "歷年乘車人次、周邊、巴士和相鄰車站", "巴士與相鄰車站的重疊原文及使用狀況引文", "參考資料尾部與官方外部連結"],
"8468551": ["8月5日至10日事件", "8月11日至21日事件", "8月21日尾段及22日至30日事件"],
"8753422": ["歌曲基本資料、作曲和宣傳", "MV導演、商業表現和榜單", "發行歷史表的日版CD資訊"]}
HISTORY = {
 ("7340228",3): [{"role":"user","content":"先查上小川站2017年的乘車人次。"}],
 ("8468551",3): [{"role":"user","content":"先查8月23日隧道大巴事故的乘員人數。"}],
 ("8753422",3): [{"role":"user","content":"先查GingaMingaYo音樂錄影帶的導演。"}],
}

def main(output=OUT):
    if any((output / name).exists() for name in ("annotations.jsonl", "AUTHOR-AUDIT.json")):
        raise ValueError("author outputs already exist; use a new output directory")
    vocab = Vocabulary(ROOT / "llamaindex-retrieval/statetune/assets/rwkv_vocab_v20230424.txt")
    drafts = {str(r["source"]["page_id"]): r for r in map(json.loads,
        (ROOT / "data/statetune/reader-v3-drafts/train.drafts.jsonl").read_text().splitlines())}
    annotations, audit = [], []
    for page, cases in SPECS.items():
        draft = drafts[page]
        assert draft["split"] == "train"
        for number, (question, evidence) in enumerate(cases, 1):
            row = deepcopy(draft)
            row.update(id=f"reader_v6_{page}_{number:02d}", reviewed=True,
                       reviewer="root-author-20260910", prompt_protocol="rwkv_g1j_no_think_v1",
                       question=question, history=HISTORY.get((page, number), []), active_tasks=[question])
            judgments = []
            for i, unit in enumerate(row["units"]):
                quotes = evidence.get(unit["id"], [])
                spans = []
                for quote in quotes:
                    start = unit["start"] + unit["text"].index(quote)
                    spans.append({"quote":quote,"quote_sha256":digest(quote),
                                  "source_span":{"start":start,"end":start+len(quote)}})
                judgments.append({"unit_id":unit["id"], "unit_sha256":unit["sha256"],
                    "supported": bool(quotes), "evidence":spans,
                    "reason": ("手工判断：本单元的"+TOPICS[page][i]+"中有逐字引文，直接支持当前任务的至少一项所求事实。"
                       if quotes else "手工判断：本单元叙述"+TOPICS[page][i]+"；没有提供当前任务所求的具体字段或事实，主题相关不构成该事实的证据。")})
            row["unit_judgments"] = judgments
            item = {k:row[k] for k in ["id","split","source","units","question","history","active_tasks","prompt_protocol"]}
            item["prompt"] = render(item);item["prompt_sha256"] = digest(item["prompt"])
            item["input_token_count"] = len(vocab.encode(item["prompt"]))
            selected = [u["id"] for u in row["units"] if u["id"] in evidence]
            gold = {"id":row["id"],"split":"train","reviewer":row["reviewer"],
                    "unit_judgments":judgments,"target":",".join(selected) if selected else "NONE",
                    "expected_unit_ids":selected,"bindings":{"source_text_sha256":digest(row["source"]["snippet"]),
                                                               "prompt_sha256":item["prompt_sha256"]}}
            encoded = validate_case(item, gold, vocab, 4096)
            annotations.append(row)
            audit.append({"id":row["id"],"target":gold["target"],"prompt_tokens":encoded["prompt_tokens"],
                          "sequence_tokens":len(encoded["input_ids"]),"source_sha256":digest(row["source"]["snippet"])})
    assert len(annotations) == 25
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "annotations.jsonl", annotations)
    (output / "AUTHOR-AUDIT.json").write_text(json.dumps({"cases":audit,"independent_review":"pending",
        "training_ready":False,"optimizer_updates":0,"new_evaluation_content_read":False},ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"authored":len(annotations),"max_tokens":max(x["sequence_tokens"] for x in audit),
                      "training_ready":False}))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    main(parser.parse_args().output)
