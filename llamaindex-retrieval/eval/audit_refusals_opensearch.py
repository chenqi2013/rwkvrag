#!/usr/bin/env python3

import json
import urllib.request


QUERIES = {
    "俄罗斯解体": ["俄罗斯解体", "苏联解体", "苏联 解体 原因"],
    "第一个皇帝": ["中国历史 第一个皇帝", "秦始皇 第一个皇帝"],
    "2026世界杯决赛": ["2026年世界杯决赛", "2026 国际足联 世界杯 决赛 场馆"],
    "1811婚姻法令": ["1811年 精神病人 婚姻法令"],
    "奥的斯": ["奥的斯 发明", "奧的斯 發明"],
    "安全电梯": ["安全电梯 发明 奥的斯", "現代主義建築 安全電梯 奧的斯"],
    "长城关隘": ["长城 著名 关隘", "長城 關城 山海关 嘉峪关"],
    "RWKV": ["RWKV 模型 开发 公司"],
    "037型": ["037型 反潜 护卫艇", "037型 猎潜艇"],
    "山科友里": ["山科友里"],
    "2025和平奖": ["2025 诺贝尔和平奖 得主"],
    "东平": ["东平"],
    "CPUID": ["CPUID"],
    "阿尔法泽": ["阿尔法泽", "AlphaZero"],
    "香港回归": ["香港 回归 1997", "香港 主权移交"],
    "中国朝代": ["中国 朝代 列表"],
    "谭德塞任期": ["谭德塞 2022 任期 5年", "谭德塞 第二任期 结束"],
    "宇树科技": ["宇树科技", "Unitree Robotics"],
}


def request_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(request, timeout=60).read())


def main() -> None:
    output = {}
    for label, queries in QUERIES.items():
        label_hits = []
        for query in queries:
            response = request_json(
                "http://127.0.0.1:9200/rwkvrag-knowledge-v1/_search",
                {
                    "size": 10,
                    "_source": ["title", "document_id", "text", "metadata"],
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": [
                                "title^8",
                                "title_tokens^8",
                                "section^5",
                                "section_tokens^5",
                                "body^2",
                                "body_tokens^2",
                            ],
                            "operator": "or",
                        }
                    },
                },
            )
            label_hits.append(
                {
                    "query": query,
                    "hits": [
                        {
                            "score": hit.get("_score"),
                            "title": (hit.get("_source") or {}).get("title"),
                            "document_id": (hit.get("_source") or {}).get("document_id"),
                            "section": ((hit.get("_source") or {}).get("metadata") or {}).get("section"),
                            "text": str((hit.get("_source") or {}).get("text") or "")[:1600],
                        }
                        for hit in response.get("hits", {}).get("hits", [])
                    ],
                }
            )
        output[label] = label_hits
    with open("eval/search_history_refusal_opensearch_audit.json", "w") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
    for label, groups in output.items():
        print(f"\n## {label}")
        for group in groups:
            print("QUERY", group["query"])
            for hit in group["hits"][:5]:
                print(hit["title"], "|", hit["section"], "|", repr(hit["text"][:260]))


if __name__ == "__main__":
    main()
