"""Read-only presentation inspection. No model calls or application mutations."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from llama_index.core import Document
from llamaindex_retrieval.atomic_evidence import AtomicRequest, candidates, short_spans
from llamaindex_retrieval.reader_prompt import binary_query_prompt
from llamaindex_retrieval.schemas import SourceItem
from llamaindex_retrieval.verbatim_chunking import verbatim_nodes


def inspect(document_text, question, attribute):
    nodes = verbatim_nodes(Document(text=document_text, id_="layout-inspection"))
    table = next(n for n in nodes if n.metadata["content_type"] == "table")
    source = SourceItem(id=table.id_, document_id="layout-inspection", source="synthetic",
                        title="layout inspection", score=1, snippet=table.text,
                        metadata=table.metadata)
    request = AtomicRequest(object="棣山 C26", attribute=attribute, conditions="省电模式")
    selected, stats = candidates(source, request)
    local = short_spans(source.snippet)
    for span in local:
        span["metadata_context"] = table.metadata["context_spans"]
        parts = [c["text"] for c in span["context"]] + [c["text"] for c in span["metadata_context"]]
        joined = "".join(parts)
        header = next(c["text"] for c in table.metadata["context_spans"] if c["kind"] == "table_header")
        span["header_occurrences_in_context"] = joined.count(header)
        span["context_characters"] = sum(map(len, parts))
        span["header_text"] = header
    prompts = [{"text": s["text"], "contexts": [c["text"] for c in s["context"]],
                "prompt": binary_query_prompt([question], {"id": source.id, "title": source.title},
                                              [{"text": c["text"]} for c in s["context"]], s["text"])}
               for s in selected]
    return {"document": document_text, "source_text_sha256": sha256(document_text.encode()).hexdigest(),
            "table_text": table.text, "metadata": table.metadata, "local_spans": local,
            "candidate_stats": stats, "candidate_prompts": prompts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ordinary = "# 棣山 C26\n| 模式 | 续航（小时） |\n| --- | --- |\n| 省电 | 48 |\n"
    # A legal wide table; unique parent context fits 600 characters, duplicate attachment does not.
    columns = ["续航（小时）"] + [f"辅助字段{i:02d}" for i in range(1, 29)]
    wide = ("# 棣山 C26\n| 模式 | " + " | ".join(columns) + " |\n| --- | "
            + " | ".join("---" for _ in columns) + " |\n| 省电 | "
            + " | ".join(["48"] + ["0"] * 28) + " |\n")
    results = {"ordinary": inspect(ordinary, "棣山 C26的省电模式续航是多少？", "续航"),
               "wide": inspect(wide, "棣山 C26的省电模式续航是多少？", "续航")}
    row = next(s for s in results["ordinary"]["local_spans"] if "省电" in s["text"])
    assert row["header_occurrences_in_context"] == 2
    wide_result = results["wide"]
    wide_row = next(s for s in wide_result["local_spans"] if "省电" in s["text"])
    assert len(wide_row["text"]) <= 600
    assert sum(len(c["text"]) for c in wide_result["metadata"]["context_spans"]) <= 600
    assert wide_row["context_characters"] > 600
    assert not any("省电" in s["text"] for s in wide_result["candidate_prompts"])
    results["verification"] = {"ordinary_header_duplicated": True,
        "wide_data_row_dropped_before_model": True, "unique_parent_fits_budget": True,
        "model_calls": 0, "inference_effect_tested": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps({"verification": results["verification"],
        "ordinary_context_characters": row["context_characters"],
        "wide_context_characters": wide_row["context_characters"],
        "wide_unique_parent_characters": sum(len(c["text"]) for c in wide_result["metadata"]["context_spans"]),
        "wide_stats": wide_result["candidate_stats"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
