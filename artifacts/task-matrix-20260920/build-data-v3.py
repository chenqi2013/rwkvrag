"""Source-bound synthetic curriculum. Disjoint cases; no benchmark-derived targets."""
import hashlib
import json
from pathlib import Path
import random

from llamaindex_retrieval.reader_prompt import binary_query_prompt
from llamaindex_retrieval.rwkv_pipeline import conversation
from llamaindex_retrieval.rwkvos_batch import render_batch_prompt
from llamaindex_retrieval.schemas import ConversationMessage, SourceItem
from llamaindex_retrieval.state_tokens import Vocabulary, encode_training
from llamaindex_retrieval.task_matrix import (Cell, TaskPlan, assessment_prompt, followup_prompt,
    matrix_writer_prompt, plan_prompt, review_prompt)
from llamaindex_retrieval.writer_prompt import writer_prompt_v2

HERE = Path(__file__).resolve().parent
MODULE = HERE.parents[1]


def enc(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def build_case(split, number):
    rng = random.Random(f"matrix-v1-{split}-{number}")
    offset = {"train": 100, "validation": 500, "holdout": 900}[split]
    n = offset + number
    kind = number % 8
    objects = [f"岚舟{n} 1.0版", f"岚舟{n} 2.0版"] if kind == 1 else [f"青川{n}型", f"白岭{n}型"]
    dimensions = ["流量" if kind == 5 else "容量", "离线支持"]
    records = []
    for i, obj in enumerate(objects):
        value = (0 if kind == 3 and i == 0 else rng.randint(3, 89))
        unit = ("L/min" if i == 0 else "kg/min") if kind == 5 else "L"
        values = {dimensions[0]: f"{value} {unit}", dimensions[1]: "支持离线运行" if i == 0 else "不支持离线运行"}
        if kind == 2 and i == 1:
            values.pop(dimensions[0])
        text = f"{obj}：" + "；".join(f"{d}为{v}" for d, v in values.items()) + "。"
        if kind == 5:
            text += "资料没有提供介质密度，也没有可比性结论。"
        records.append({"object": obj, "values": values, "text": text, "eligible": True})
    if kind == 4:
        values = {dimensions[0]: "93 L"}
        records.append({"object": objects[0], "values": values,
                        "text": f"另一份同范围、无权威排序的记录：{objects[0]}容量为93 L。", "eligible": True})
    if kind == 6:
        records.insert(0, {"object": objects[0], "values": {dimensions[0]: "999 L"},
            "text": f"已废止草案：{objects[0]}容量暂填999 L，本记录不能作为核准值。", "eligible": False})
    if kind == 7:
        records = []
    rng.shuffle(records)
    sources = []
    for i, record in enumerate(records):
        identity = f"{split}-{number}-s{i + 1}"
        text = record["text"]
        source = SourceItem(id=f"{identity}@0:{len(text)}:{sha(text)[:12]}", document_id=identity,
            source="synthetic-matrix", title=f"记录{i+1}", uri=None, score=1.0, snippet=text,
            metadata={"parent_source_id": identity, "span_start": 0, "span_end": len(text), "span_sha256": sha(text)})
        sources.append(source)
    writer_sources = [s for s, r in zip(sources, records, strict=True) if r["eligible"]]
    labels = {s.id: i for i, s in enumerate(writer_sources, 1)}
    question = f"比较{'与'.join(objects)}的{'和'.join(dimensions)}，逐项给出来源，未记载或冲突请明确说明。"
    if kind == 5:
        question += "保留原单位，不要直接排名。"
    history = []
    if kind == 6:
        history = [{"role": "user", "content": f"比较{'与'.join(objects)}的容量、离线支持和价格。"},
                   {"role": "user", "content": "撤回价格，只保留容量和离线支持，采用核准资料。"}]
        question = "按最后确定的范围逐项比较，并列来源，采用核准资料。"
    cells = [Cell(id=f"c{i+1}", object=obj, dimension=dim, question=f"{obj}的{dim}是什么？")
             for i, (obj, dim) in enumerate((o, d) for o in objects for d in dimensions)]
    plan = TaskPlan(objects=objects, dimensions=dimensions, conditions=[], cells=cells)
    matrix, answers = [], []
    for cell in cells:
        matches = [(i, r) for i, r in enumerate(records) if r["eligible"] and r["object"] == cell.object and cell.dimension in r["values"]]
        values = {r["values"][cell.dimension] for _, r in matches}
        status = "missing" if not matches else "conflict" if len(values) > 1 else "supported"
        matrix.append({**cell.model_dump(), "status": status, "source_ids": [sources[i].id for i, _ in matches],
                       "assessment_status": "completed" if matches else "not_called"})
        if not matches:
            answers.append(f"{cell.object}的{cell.dimension}：资料不足，无法确定。")
        elif status == "conflict":
            answers.append(f"{cell.object}的{cell.dimension}存在冲突：" + "；".join(
                f"{r['values'][cell.dimension]}[资料 {labels[sources[i].id]}]" for i, r in matches) + "。无法确定唯一值。")
        else:
            i, r = matches[0]
            answers.append(f"{cell.object}的{cell.dimension}：{r['values'][cell.dimension]}[资料 {labels[sources[i].id]}]。")
    if kind == 5:
        answers.append("两者流量单位不同，资料未给密度，不能直接判断谁更大。")
    return {"id": f"{split}-{number:03}", "kind": kind, "question": question, "history": history,
        "plan": plan.model_dump(), "matrix": matrix, "sources": [s.model_dump() for s in writer_sources],
        "candidates": [s.model_dump() for s in sources],
        "records": records, "target": "\n".join(answers), "synthetic": True}


def training_rows(case):
    task = conversation(case["question"], [ConversationMessage(**m) for m in case["history"]])
    sources = [SourceItem(**s) for s in case["sources"]]
    plan = TaskPlan(**case["plan"])
    matrix = case["matrix"]
    yield "planner", plan_prompt(task, 8), enc(case["plan"])
    for cell, row in zip(plan.cells, matrix, strict=True):
        selected = [s for s in sources if s.id in row["source_ids"]]
        yield "planner", assessment_prompt(cell, selected), enc({"status": row["status"], "source_ids": row["source_ids"]})
        for source_dict, record in zip(case["candidates"], case["records"], strict=True):
            source = SourceItem(**source_dict)
            yes = record["eligible"] and record["object"] == cell.object and cell.dimension in record["values"]
            yield "resolver", binary_query_prompt([cell.question],
                {"id": source.metadata["parent_source_id"], "title": source.title, "uri": source.uri}, [], source.snippet), enc({"answer": "YES" if yes else "NO"})
    yield "writer", matrix_writer_prompt(task, matrix, sources), case["target"]
    # Non-matrix replay uses the very same source-bound facts and exact production prompt.
    evidence = [{"label": f"资料 {i}", "id": s.id, "title": s.title, "uri": s.uri,
                 "text": s.snippet, "context_spans": [], "fields": []} for i, s in enumerate(sources, 1)]
    yield "writer", writer_prompt_v2(task, evidence, [case["question"]]), case["target"]
    yield "planner", review_prompt(task, matrix, case["target"], sources), enc({"valid": True, "issues": []})
    # A missing-object answer is unambiguously incomplete, including empty-evidence tasks.
    omitted = case["target"].splitlines()[0]
    yield "planner", review_prompt(task, matrix, omitted, sources), enc({"valid": False, "issues": ["答案遗漏了其余所求项目。"]})
    # Swapped source numbers should not become accepted just because facts exist somewhere.
    if len(sources) > 1 and "[资料 1]" in case["target"]:
        wrong = case["target"].replace("[资料 1]", "[资料 999]")
        yield "planner", review_prompt(task, matrix, wrong, sources), enc({"valid": False, "issues": ["引用资料999没有对应来源。"]})
        source_objects = {s["id"]: r["object"] for s, r in zip(case["candidates"], case["records"], strict=True)}
        other = next(i for i, s in enumerate(sources, 1) if source_objects[s.id] != source_objects[sources[0].id])
        wrong = case["target"].replace("[资料 1]", f"[资料 {other}]")
        yield "planner", review_prompt(task, matrix, wrong, sources), enc({"valid": False, "issues": ["部分事实引用了不支持该对象的来源。"]})
    unresolved = [r for r in matrix if r["status"] != "supported"]
    previous = [{"cell_id": c.id, "query": c.question} for c in plan.cells]
    target = {"stop": not bool(unresolved), "queries": [
        {"cell_id": r["id"], "query": f"{r['object']} {r['dimension']} 正式说明 原文"} for r in unresolved]}
    yield "planner", followup_prompt(task, unresolved, previous), enc(target)


def main():
    output = HERE / "data-v3"
    output.mkdir(exist_ok=False)
    vocab = Vocabulary(MODULE / "statetune/assets/rwkv_vocab_v20230424.txt")
    pins = {}
    for split, count in (("train", 32), ("validation", 8), ("holdout", 8)):
        cases = [build_case(split, i) for i in range(count)]
        (output / f"{split}.cases.jsonl").write_text("".join(enc(c) + "\n" for c in cases))
        by_stage = {stage: [] for stage in ("planner", "resolver", "writer")}
        for case in cases:
            for index, (stage, prompt, target) in enumerate(training_rows(case)):
                rendered, _ = render_batch_prompt([{"role": "user", "content": prompt}], "<think></think", "complete")
                if stage != "planner":
                    rendered += "\n"
                row = {"id": f"{case['id']}-{index}", "case_id": case["id"], "stage": stage,
                    "prompt": rendered, "target": target, "prompt_sha256": sha(rendered), "target_sha256": sha(target),
                    **encode_training(rendered, target, vocab, max_tokens=4096)}
                by_stage[stage].append(row)
        for stage, rows in by_stage.items():
            p = output / f"{split}.{stage}.jsonl"
            p.write_text("".join(enc(row) + "\n" for row in rows))
            pins[p.name] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "rows": len(rows),
                           "max_tokens": max(len(r["input_ids"]) for r in rows)}
    (output / "MANIFEST.json").write_text(json.dumps({"synthetic": True, "independent_review": False,
        "split_by_case": True, "holdout_used_for_training": False, "seed": "matrix-v1",
        "files": pins, "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2))
    print(json.dumps(pins, indent=2))


if __name__ == "__main__":
    main()
