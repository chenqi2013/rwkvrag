"""Manual source-local judgments. Code locates literal quotes, never infers labels."""
import argparse
import json
from copy import deepcopy
from pathlib import Path

from llamaindex_retrieval.statetune import digest, write_rows

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
# Each explicit map is the author's semantic judgment, including overlapping support.
SPECS = {
"370260": [
 ("2008年奧運擊劍比賽在哪幾天、哪個場館舉行？", {"E1": "2008年夏季奧林匹克運動會的擊劍比賽由8月9日至17日在北京國家會議中心擊劍場舉行，本屆奧運的劍擊賽事共會產生10枚金牌。"}),
 ("女子個人佩劍的金、銀、銅牌選手分別代表哪個國家？", {"E3": "| 個人佩劍（詳細） | 玛丽埃尔·（美国）                                                                           | 萨达·（美国）                                           | 丽贝卡·（美国）                                                               |"}),
 ("仲滿在男子個人佩劍決賽的最終比分是多少？", {}),
 ("2008年奧運擊劍比賽現場累計入場觀眾有多少人？", {}),
],
"4271857": [
 ("TCWC Perth將熱帶低氣壓06U升為1級熱帶氣旋時，給它取了什麼名字？", {"E2": "TCWC Perth 將熱帶低氣壓06U升格為1級熱帶氣旋，並命名為瑪格達（Magda）。"}),
 ("烏魯伊在3月20日登陸的位置在哪裡？", {"E3": "強烈熱帶氣旋烏魯伊在昆士蘭州艾爾利海灘附近登陸。"}),
 ("勞倫斯造成的經濟損失總額是多少澳元？", {}),
 ("熱帶氣旋保羅造成多少棟住宅損毀？", {}),
],
"3667615": [
 ("《殺手少女》真人電影在日本哪一天上映？", {"E2": "真人版於2014年在美國上映，日本則於2015年4月11日上映。"}),
 ("KITE LIBERATOR漫畫版的原作、作畫和出版社分別是誰？", {"E3": "原作是梅津泰臣、作畫為小宮利公、出版社：KILL TIME COMMUNICATION。"}),
 ("《殺手少女》真人電影的全球票房總額是多少？", {}),
 ("KITE LIBERATOR漫畫單行本首刷印了多少冊？", {}),
],
"14255": [
 ("729年條目的大事記列出了日本的哪件事？", {"E1": "日本長屋王之變。"}),
 ("資料所列729年的阿拉伯帝國哈里發是誰？", {"E3": "阿拉伯帝国 - 希沙姆·本·阿卜杜勒-馬利克，哈里发（724年－743年）"}),
 ("729年唐朝登記的全國人口總數是多少？", {}),
 ("729年日本長屋王之變發生地當天的氣溫是多少？", {}),
],
"8295659": [
 ("列表中的保羅·高更在1903年的哪一天逝世？", {"E2": "### 5月8日\n- 保羅·高更。法國印象派畫家"}),
 ("1903年11月8日逝世、被稱為現代土壤科學奠基人的人物是誰？", {"E3": "### 11月8日\n- 瓦西里·多库切夫，俄羅斯地質學家和地理學家，現代土壤科學奠基人。"}),
 ("保羅·高更逝世時留下的遺產估值是多少法郎？", {}),
 ("瓦西里·多库切夫葬禮有多少人出席？", {}),
],
"6066934": [
 ("該預算案向精英運動員發展基金額外注資多少？", {"E2": "向精英運動員發展基金額外注資50億元。"}),
 ("關於文憑試考試費，盧偉國對自修生提出了什麼豁免建議？", {"E3": "經民聯立法會議員盧偉國認為豁免自修生一半考試費已經足夠杜絕搗亂情況，不應一刀切將自修生從計劃剔除。"}),
 ("免費海洋公園門票最後實際被使用了多少張？", {}),
 ("街市現代化計劃每個街市分別獲撥多少資金？", {}),
],
"100331": [
 ("210國道的起點和終點分別在哪裡？", {"E1": "210国道，也称为满防线、国道210线或G210线，是中国的一条国道，起点为内蒙古自治区包头市达尔罕茂明安联合旗满都拉镇，终点为广西壮族自治区防城港市的国道。"}),
 ("咸榆公路在哪一年動工、哪一年建成？", {"E4": "1932年动工、1935年建成“咸榆公路”宽6米至3米半，碾压土路。"}),
 ("210國道全線在2020年的日均車流量是多少？", {}),
 ("咸榆公路1935年竣工時的工程決算總額是多少？", {}),
],
"14253": [
 ("727年條目所列的法蘭克王國宮相是誰？", {"E2": "宫相 - 查理·马特（714年－741年）"}),
 ("727年出生欄記載了哪位唐朝將領？", {"E3": "## 出生\n- 李元谅，唐朝将领"}),
 ("李元谅的出生時辰是什麼？", {}),
 ("727年法蘭克王國徵收的全年稅款總額是多少？", {}),
],
"6477266": [
 ("受傷的后备警察部队隊員被送往哪裡的醫院？", {"E2": "傷者被送往斯利那加的軍事基地醫院。"}),
 ("哪位新加坡外交部長就普爾瓦馬襲擊表達哀悼？", {"E3": "新加坡：新加坡外交部長維文表示"}),
 ("襲擊後，印度政府向每個遇難者家庭實際發放多少賠償金？", {}),
 ("收治傷者的軍事基地醫院當天共有多少名值班醫生？", {}),
],
"5214431": [
 ("一致性條件公式中的rho代表什麼，算符使用哪種繪景？", {"E2": "这里的{\\displaystyle \\rho }表示初始的密度矩阵，且算符都使用海森堡绘景表示。"}),
 ("歐姆內斯後來用較少數學語言解釋這一詮釋時，拿哪個佯謬作例子？", {"E4": "这样就可以从形式上解释EPR佯谬中认为是可以一起提出的问题，实际上并不能一齐提出。"}),
 ("文中所說的退相干實驗一共使用了多少個電子？", {}),
 ("歐姆內斯提出較少數學語言的解釋方式時，首次演講的具體日期是哪一天？", {}),
],
"8647440": [
 ("1900年奧運男子馬拉松在哪一天舉行，賽程距離是多少公里？", {"E1": "在1900年夏季奧林匹克運動會田徑比賽中，男子馬拉松比賽於1900年7月19日舉行，共有來自五個國家的13名運動員參加。馬拉松賽程的距離為40.26公里。"}),
 ("男子馬拉松銀牌得主尚皮翁的完賽時間是多少？", {"E1": "| 銀牌 | 埃米爾·尚皮翁      | 法国   | 3:04:17 |", "E2": "| 銀牌 | 埃米爾·尚皮翁      | 法国   | 3:04:17 |"}),
 ("泰阿托奪冠後收到多少法郎獎金？", {}),
 ("這場馬拉松沿途一共設置了多少個飲水站？", {}),
],
"102495": [
 ("321國道佛山改線方案在哪一天獲广东省交通运输厅批復？", {"E1": "佛山市提出的“利用既有线路将G321、G325佛山段迁至区域边界”方案，于2011年9月5日获广东省交通运输厅批复。"}),
 ("321國道里程表中，都勻市距起點多少公里？", {"E2": "| 贵州省都匀市                 | 1198             |"}),
 ("321國道佛山改線工程的最終造價是多少人民幣？", {}),
 ("321國道全線在2020年的日均車流量是多少？", {}),
],
"3511317": [
 ("新華網採集的數據中，抗議者有多少比例受過高等教育？", {"E2": "77%受过高等教育"}),
 ("就巴西抗議運動發表看法、希望示威者不要利用足球表達訴求的國際足聯主席是誰？", {"E3": "国际足联主席约瑟夫·布拉特：“我可以理解巴西民众的不满情绪，但我希望示威者们不要利用足球来表达自己的诉求。"}),
 ("新華網抗議者數據的調查樣本一共有多少人？", {}),
 ("2013年巴西抗議運動造成的全國經濟損失總額是多少？", {}),
],
"4007069": [
 ("《“一國兩制”在香港特別行政區的實踐》白皮書由哪個機構、在哪一天發布？", {"E1": "《“一国两制”在香港特别行政区的实践》白皮書由中華人民共和國國務院新聞辦公室於2014年6月10日發布"}),
 ("公民黨代表在中聯辦門外放下的自製白皮書上寫了哪兩個字？", {"E3": "公民黨十多名代表，由黨魁梁家傑帶領，在中聯辦門外放下一本自製的白皮書，上面寫有「收回」兩字。"}),
 ("這份白皮書第一批印製了多少冊？", {}),
 ("撰寫這份白皮書所花費的總經費是多少？", {}),
],
"3927709": [
 ("聯黎部隊哪一天到達黎巴嫩，總部設在哪裡？", {"E2": "在1978年3月23日联黎部队到达黎巴嫩，在纳库拉建立了总部。"}),
 ("以色列軍隊1978年年末撤出黎巴嫩時，把陣地移交給了哪支部隊？", {"E3": "在1978年年末，以色列军队撤出了黎巴嫩，并将其位于黎巴嫩的阵地移交给他的盟友、正处于萨阿德·哈达德上校领导下的南黎巴嫩军。"}),
 ("聯黎部隊1978年在納庫拉建設總部的工程費用是多少？", {}),
 ("1978年南黎巴嫩流離失所者每戶平均得到多少救濟金？", {}),
],
"7775972": [
 ("U-107號在1919年以多少英鎊售予誰，這個價格是否包含發動機？", {"E2": "遂于1919年3月3日以2425英镑（不含发动机）的价格将U-107号售予乔治·科恩"}),
 ("U-107號在哪一年、哪座城市拆解報廢？", {"E1": "至1922年在斯旺西拆解报废。", "E2": "至1922年在斯旺西拆解报废。"}),
 ("U-107號最初建造的合同總價是多少馬克？", {}),
 ("拆解U-107號所得的廢鋼總重量是多少噸？", {}),
],
}


def main(output):
    if (output / "annotations.jsonl").exists():
        raise ValueError("refusing to overwrite annotations")
    annotations = []
    for relative in ("reader-v4-canonical", "reader-v6-train-drafts"):
        source = OUT.parent / relative / "annotations.jsonl"
        for old in map(json.loads, source.read_text().splitlines()):
            if old["split"] == "train":
                row = deepcopy(old)
                row["input_layout"] = "task_last"
                annotations.append(row)
    assert len(annotations) == 46
    for split in ("dev", "heldout"):
        drafts = ROOT / "data/statetune/reader-v6-next-balanced-drafts" / f"{split}.drafts.jsonl"
        for draft in map(json.loads, drafts.read_text().splitlines()):
            page = str(draft["source"]["page_id"])
            for number, (question, evidence) in enumerate(SPECS[page], 1):
                row = deepcopy(draft)
                row.update(id=f"reader_v6_eval_{page}_{number:02d}", reviewed=True,
                           reviewer="root-v6-eval-author-20260910", input_layout="task_last",
                           prompt_protocol="rwkv_g1j_no_think_v1", question=question,
                           history=[], active_tasks=[question])
                judgments = []
                for unit in row["units"]:
                    quote = evidence.get(unit["id"])
                    spans = []
                    if quote:
                        start = unit["start"] + unit["text"].index(quote)
                        spans.append({"quote": quote, "quote_sha256": digest(quote),
                                      "source_span": {"start": start, "end": start + len(quote)}})
                    judgments.append({"unit_id": unit["id"], "unit_sha256": unit["sha256"],
                                      "supported": bool(quote), "evidence": spans,
                                      "reason": ("作者逐段判断：引文直接提供当前问题所求事实；重叠段中的相同证据同样计入。"
                                                 if quote else "作者逐段判断：本单元没有给出当前问题要求的具体事实或数值；只有相关主题不能支持回答。")})
                row["unit_judgments"] = judgments
                annotations.append(row)
    assert len(annotations) == 110
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "annotations.jsonl", annotations)
    print(json.dumps({"cases": len(annotations), "layout": "task_last", "independent_review": "pending"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    main(parser.parse_args().output)
