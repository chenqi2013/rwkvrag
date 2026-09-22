"""Bounded, resumable-by-new-run teacher generation with separate review calls.

Credentials never enter prompts, metadata or error text. Failed answers are retained,
never repaired or silently retried. Same-model review is not independent validation.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import time
import httpx
from contracts import ROOT, source_map, compile_item, review_map

FROZEN = ROOT / "llamaindex-retrieval/statetune/defect-batch-20260922"


def write(path, value):
    with path.open("x") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def now():
    return datetime.now(timezone.utc).isoformat()


class Teacher:
    def __init__(self, credentials, config, out, prior_cost):
        self.credentials, self.config, self.out = credentials, config, out
        self.spent, self.reserved, self.last_start = prior_cost, 0., 0.
        self.lock = asyncio.Lock()
        self.stopped = asyncio.Event()
        self.client = httpx.AsyncClient(timeout=config["timeout_s"], follow_redirects=False)

    async def call(self, job_id, phase, body):
        c = self.config
        reserve = (len(json.dumps(body, ensure_ascii=False).encode()) * c["peak_input_usd_per_million"]
                   + body["max_tokens"] * c["peak_output_usd_per_million"]) / 1_000_000
        async with self.lock:
            if self.stopped.is_set() or self.spent + self.reserved + reserve > c["cost_ceiling_usd"]:
                self.stopped.set()
                raise RuntimeError("Teacher run stopped or cost ceiling reached")
            self.reserved += reserve
            await asyncio.sleep(max(0, self.last_start + c["request_start_interval_s"] - time.monotonic()))
            self.last_start = time.monotonic()
        name = f"{job_id}-{phase}"
        request = {"created_at": now(), "endpoint": c["teacher_base_url"] + "/chat/completions",
                   "body": body, "reserved_upper_usd": reserve}
        write(self.out / "calls" / f"{name}.request.json", request)
        started = time.monotonic()
        charged, record = reserve, None
        try:
            response = await self.client.post(request["endpoint"], json=body,
                headers={"Authorization": "Bearer " + self.credentials["api_key"]})
            if response.status_code != 200:
                if response.status_code in {401, 402, 403, 429}:
                    self.stopped.set()
                record = {"http_status": response.status_code, "error": "Provider rejected request; credentials and provider error body omitted",
                          "response_body_sha256": hashlib.sha256(response.content).hexdigest()}
                raise RuntimeError(f"Teacher HTTP {response.status_code}")
            raw = response.json()
            usage = raw.get("usage", {})
            if "prompt_tokens" in usage and "completion_tokens" in usage:
                charged = (usage["prompt_tokens"] * c["peak_input_usd_per_million"]
                           + usage["completion_tokens"] * c["peak_output_usd_per_million"]) / 1_000_000
            record = {"http_status": 200, "response": raw}
            choice = raw["choices"][0]
            if choice["finish_reason"] != "stop":
                raise ValueError("Teacher output did not finish normally")
            return json.loads(choice["message"]["content"])
        except Exception as exc:
            if record is None:
                record = {"error_type": type(exc).__name__, "error": "Request incomplete; maximum reserved cost counted conservatively"}
            raise
        finally:
            assert record is not None
            record.update(ended_at=now(), elapsed_s=time.monotonic()-started, conservative_cost_usd=charged)
            write(self.out / "calls" / f"{name}.response.json", record)
            async with self.lock:
                self.reserved -= reserve
                self.spent += charged

    async def close(self):
        await self.client.aclose()


async def main(args):
    pins = json.loads((FROZEN / "PINS.json").read_text())
    for path, digest in pins.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
    c = json.loads((FROZEN / "CONFIG.json").read_text())
    credentials = json.loads(args.credentials.read_text())
    if credentials["base_url"] != c["teacher_base_url"] or credentials["model"] != c["teacher_model"]:
        raise ValueError("Teacher identity differs from frozen configuration")
    if args.credentials.stat().st_mode & 0o077:
        raise ValueError("Private configuration must have owner-only permissions")
    all_jobs = [json.loads(line) for line in (FROZEN / "JOBS.jsonl").read_text().splitlines()]
    jobs = all_jobs[args.start:args.stop]
    if not jobs or not 0 <= args.start < args.stop <= len(all_jobs):
        raise ValueError("Invalid job range")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    guard = (args.out.parent / "teacher.lock").open("a")
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prior_cost = 0.
    for path in args.out.parent.glob("*/calls/*.request.json"):
        receipt = path.with_name(path.name.replace(".request.json", ".response.json"))
        prior_cost += (json.loads(receipt.read_text())["conservative_cost_usd"] if receipt.exists()
                       else json.loads(path.read_text())["reserved_upper_usd"])
    args.out.mkdir(exist_ok=False)
    (args.out/"calls").mkdir()
    (args.out/"results").mkdir()
    write(args.out/"RUN.json", {"started_at": now(), "start": args.start, "stop": args.stop,
          "prior_conservative_cost_usd": prior_cost, "pins_sha256": hashlib.sha256((FROZEN/"PINS.json").read_bytes()).hexdigest(),
          "config": c, "job_ids": [j["id"] for j in jobs], "independent_review": False})
    teacher = Teacher(credentials, c, args.out, prior_cost)
    queue = asyncio.Queue()
    for job in jobs:
        queue.put_nowait(job)
    totals = {"jobs_recorded": 0, "items_structurally_and_teacher_accepted": 0, "failed_jobs": 0}

    def body(phase, value):
        return {"model": c["teacher_model"], "messages": [
            {"role": "system", "content": (FROZEN / ("generate.txt" if phase == "generation" else "review.txt")).read_text()},
            {"role": "user", "content": json.dumps(value, ensure_ascii=False)}],
            "response_format": {"type": "json_object"}, "thinking": {"type": c["thinking"]},
            "temperature": c[phase + "_temperature"], "max_tokens": c[phase + "_max_tokens"], "stream": False}

    async def worker():
        while not queue.empty() and not teacher.stopped.is_set():
            try:
                job = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            result = {"job_id": job["id"], "split": job["split"], "source_families": job["source_families"],
                      "material_mode": job["material_mode"], "status": "failed", "accepted": []}
            try:
                generation_input = {"material_mode": job["material_mode"],
                    "sources": [{k:s[k] for k in ["id", "title", "text"]} for s in job["sources"]]}
                draft = await teacher.call(job["id"], "generation", body("generation", generation_input))
                result["draft"] = draft
                sources = source_map(job, draft)
                checks, compiled = [], {}
                for index, item in enumerate(draft["items"]):
                    try:
                        compiled[index] = compile_item(item, sources)
                        checks.append({"index": index, "structural_pass": True})
                    except Exception as exc:
                        checks.append({"index": index, "structural_pass": False, "error": str(exc)})
                result["structural_checks"] = checks
                blind = {"material_mode": job["material_mode"],
                         "sources": [{k:s[k] for k in ["id", "title", "text"]} for s in sources.values()],
                         "items": [{k:v for k,v in item.items() if k not in {"rationale", "skill"}} for item in draft["items"]]}
                review = await teacher.call(job["id"], "review", body("review", blind))
                result["review"] = review
                decisions = review_map(review, len(draft["items"]))
                for index, student in compiled.items():
                    if decisions[index]["accept"]:
                        result["accepted"].append(dict(student, id=f"{job['id']}:{index}", index=index,
                            teacher_review=decisions[index], rationale=draft["items"][index]["rationale"],
                            skill=draft["items"][index]["skill"]))
                result["status"] = "reviewed"
            except Exception as exc:
                # Known local validation messages contain data, never credentials.
                result["error_type"] = type(exc).__name__
                result["error"] = str(exc) if isinstance(exc, (ValueError, RuntimeError, KeyError)) else "Request or data processing failed; inspect preserved receipt"
                totals["failed_jobs"] += 1
            write(args.out/"results"/f"{job['id']}.json", result)
            totals["jobs_recorded"] += 1
            totals["items_structurally_and_teacher_accepted"] += len(result["accepted"])
            print(dict(totals, conservative_cost_usd=round(teacher.spent, 4)), flush=True)
            queue.task_done()
    try:
        await asyncio.gather(*(worker() for _ in range(c["concurrency"])))
    finally:
        await teacher.close()
        write(args.out/"SUMMARY.json", dict(totals, planned_jobs=len(jobs), ended_at=now(),
              stopped=teacher.stopped.is_set(), conservative_cost_usd=teacher.spent,
              independent_review=False, training_exported=False))
        guard.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--credentials", type=Path, default=Path.home()/".config/rwkvrag/teacher-deepseek-20260922.json")
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--stop", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    asyncio.run(main(p.parse_args()))
