"""Planner-only scope curriculum; prior training replay plus new time/condition tasks."""
import hashlib
import json
from pathlib import Path

from llamaindex_retrieval.rwkv_pipeline import conversation
from llamaindex_retrieval.rwkvos_batch import render_batch_prompt
from llamaindex_retrieval.state_tokens import Vocabulary, encode_training
from llamaindex_retrieval.task_matrix import TaskPlan, plan_prompt

HERE = Path(__file__).resolve().parent


def case(split, i):
    n = {"train": 2000, "validation": 4000, "holdout": 6000}[split] + i
    kind = i % 8
    pairs = [("节点", "连接数", "缓存大小"), ("厂区", "耗电量", "产量"),
             ("装置", "额定电压", "工作温度"), ("车间", "故障次数", "维护时长")]
    noun, first, second = pairs[kind % 4]
    objects = [f"云杉{n}{noun}", f"海棠{n}{noun}"]
    dimensions = [first, second] if kind % 2 == 0 else [first]
    conditions = []
    if kind == 4:
        objects = [f"项目{n}签约", f"项目{n}验收"]
        dimensions = ["日期"]
        question = f"比较项目{n}的签约日期和验收日期，保留各自时间。"
    elif kind == 5:
        objects = [f"团队{n}上线实施", f"产品{n}对外公布"]
        dimensions = ["日期"]
        question = f"对比团队{n}上线实施日期与产品{n}对外公布日期，分别查询，不要混为一谈。"
    else:
        if kind == 6:
            conditions = [f"{2020+i%6}年", "东区", "核准版本"]
        elif kind == 7:
            conditions = ["额定负载", "环境温度25℃"]
        question = f"比较{'与'.join(objects)}的{'和'.join(dimensions)}。"
        if conditions:
            question += f"仅采用{'、'.join(conditions)}条件下的记录。"
    cells = [{"id": f"c{k+1}", "object": o, "dimension": d,
        "question": (f"在{'、'.join(conditions)}条件下，" if conditions else "") + f"{o}的{d}是什么？"}
        for k, (o, d) in enumerate((o, d) for o in objects for d in dimensions)]
    plan = TaskPlan(objects=objects, dimensions=dimensions, conditions=conditions, cells=cells)
    return {"id": f"scope-{split}-{i:03}", "question": question, "plan": plan.model_dump()}


def main():
    out = HERE / "data-plan-v3"
    out.mkdir(exist_ok=False)
    vocab = Vocabulary(HERE.parents[1] / "statetune/assets/rwkv_vocab_v20230424.txt")
    pins = {}
    for split, count in (("train", 128), ("validation", 8), ("holdout", 8)):
        rows = []
        if split == "train":
            rows = [json.loads(line) for line in (HERE / "data-v4/train.planner.jsonl").read_text().splitlines()]
        for i in range(count):
            c = case(split, i)
            prompt, _ = render_batch_prompt([{"role": "user", "content": plan_prompt(conversation(c["question"], []), 8)}], "<think></think", "complete")
            target = json.dumps(c["plan"], ensure_ascii=False, sort_keys=True)
            rows.append({"id": c["id"], "case_id": c["id"], "stage": "planner", "prompt": prompt,
                "target": target, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "target_sha256": hashlib.sha256(target.encode()).hexdigest(),
                **encode_training(prompt, target, vocab, max_tokens=4096)})
        path = out / f"{split}.planner.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        pins[path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(rows)}
    (out / "MANIFEST.json").write_text(json.dumps({"files": pins, "independent_review": False,
        "synthetic": True, "old_holdout_used_for_training": False,
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2))
    print(json.dumps(pins, indent=2))


if __name__ == "__main__":
    main()
