"""Planner-only scope curriculum; prior training replay plus new time/condition tasks."""
import hashlib
import json
from pathlib import Path

from llamaindex_retrieval.rwkv_pipeline import conversation
from llamaindex_retrieval.schemas import ConversationMessage
from llamaindex_retrieval.rwkvos_batch import render_batch_prompt
from llamaindex_retrieval.state_tokens import Vocabulary, encode_training
from llamaindex_retrieval.task_matrix import TaskPlan, plan_prompt

HERE = Path(__file__).resolve().parent


def case(split, i):
    n = {"train": 2000, "validation": 4000, "holdout": 6000}[split] + i
    kind = i % 12
    pairs = [("节点", "连接数", "缓存大小"), ("厂区", "耗电量", "产量"),
             ("装置", "额定电压", "工作温度"), ("车间", "故障次数", "维护时长")]
    noun, first, second = pairs[kind % 4]
    objects = [f"云杉{n}{noun}", f"海棠{n}{noun}"]
    dimensions = [first, second] if kind % 2 == 0 else [first]
    conditions = []
    history = []
    coverage = "grid"
    if kind == 4:
        objects = [f"项目{n}签约", f"项目{n}验收"]
        dimensions = ["日期"]
        question = f"比较项目{n}的签约日期和验收日期，保留各自时间。"
    elif kind == 5:
        objects = [f"团队{n}上线实施", f"产品{n}对外公布"]
        dimensions = ["日期"]
        question = f"对比团队{n}上线实施日期与产品{n}对外公布日期，分别查询，不要混为一谈。"
    elif kind == 8:
        objects = [f"枫桥{n}设备", f"沧浪{n}设备", f"月港{n}设备", f"桐溪{n}设备"]
        dimensions = ["额定功率", "冷却方式", "维护周期", "日志时区"]
        coverage = "listed"
        question = "分别查询" + "、".join(f"{o}的{d}" for o, d in zip(objects, dimensions)) + "，只回答这四项。"
    elif kind == 9:
        objects = objects[:1]
        dimensions = ["额定电压", "冷却方式", "最高温度"]
        question = f"逐项核对{objects[0]}的" + "、".join(dimensions) + "。"
    elif kind == 10:
        objects = [f"星川{n}系统", f"南叶{n}系统"]
        dimensions = ["发布时间", "许可证"]
        coverage = "listed"
        history = [{"role": "user", "content": f"查{objects[0]}的发布时间和{objects[1]}的许可证，再查价格。"},
                   {"role": "user", "content": "撤回价格，只保留前两项。"}]
        question = "按最后确定的范围回答。"
    elif kind == 11:
        objects = objects[:1]
        conditions = [f"https://example.invalid/manual/{n}", "2025年修订版"]
        question = f"仅按{'、'.join(conditions)}核对{objects[0]}的{'和'.join(dimensions)}。"
    else:
        if kind == 6:
            conditions = [f"{2020+i%6}年", "东区", "核准版本"]
        elif kind == 7:
            conditions = ["额定负载", "环境温度25℃"]
        question = f"比较{'与'.join(objects)}的{'和'.join(dimensions)}。"
        if conditions:
            question += f"仅采用{'、'.join(conditions)}条件下的记录。"
    pairs = list(zip(objects, dimensions)) if coverage == "listed" else [(o, d) for o in objects for d in dimensions]
    cells = [{"id": f"c{k+1}", "object": o, "dimension": d,
        "question": (f"在{'、'.join(conditions)}条件下，" if conditions else "") + f"{o}的{d}是什么？"}
        for k, (o, d) in enumerate(pairs)]
    plan = TaskPlan(coverage=coverage, objects=objects, dimensions=dimensions, conditions=conditions, cells=cells)
    return {"id": f"scope4-{split}-{i:03}", "question": question, "history": history, "plan": plan.model_dump()}


def main():
    out = HERE / "data-plan-v4"
    out.mkdir(exist_ok=False)
    vocab = Vocabulary(HERE.parents[1] / "statetune/assets/rwkv_vocab_v20230424.txt")
    pins = {}
    for split, count in (("train", 192), ("validation", 12), ("holdout", 12)):
        rows = []
        cases = []
        if split == "train":
            cases = [json.loads(line) for line in (HERE / "data-v4/train.cases.jsonl").read_text().splitlines()]
        cases += [case(split, i) for i in range(count)]
        for c in cases:
            task = conversation(c["question"], [ConversationMessage(**m) for m in c["history"]])
            prompt, _ = render_batch_prompt([{"role": "user", "content": plan_prompt(task, 8)}], "<think></think", "complete")
            target = json.dumps(TaskPlan(**c["plan"]).model_dump(), ensure_ascii=False, sort_keys=True)
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
