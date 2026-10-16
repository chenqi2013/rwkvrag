"""Question-directed parallel reading, followed by a selected-original Writer.

Models select source IDs; scheduling and validation never decide relevance or
repair an answer. This material-only candidate does not perform index retrieval.
"""

import asyncio
from copy import deepcopy
from hashlib import sha256
import json
from time import monotonic
from uuid import uuid4

from .schemas import AskResponse

PROTOCOL = "question-documents-v1"
READ = '阅读这一份完整原文，记录可能回答当前问题或连接问题对象的事实。保留主体与所属机构、关系对象、时间、版本、否定、未知项、数值及适用条件；表格保留每个相关对象对应的条件。不要把标题、来源编号或阅读要求当事实，不猜测。若无相关事实，findings为空。输出findings，每项source_id为本原文ID，statement为忠实记录，最多6项。'
WRITE = '仅根据已选原文回答问题。所有实质断言须受原文支持，保留时间、否定、职位、版本、数值和条件。材料不足时明确指出未知，不从关联关系猜测。用[材料ID]引用相应原文。只输出JSON：answer为直接回答，source_ids为实际支持回答的已选材料ID。'


def digest(text):
    return sha256(text.encode("utf-8")).hexdigest()


def obj(properties):
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def read_schema(source_id):
    return obj({"findings": {"type": "array", "maxItems": 6, "items": obj({
        "source_id": {"type": "string", "enum": [source_id]},
        "statement": {"type": "string"}})}})


def answer_schema(ids):
    items = {"type": "string", "enum": ids} if ids else {"type": "string"}
    return obj({"answer": {"type": "string"}, "source_ids": {
        "type": "array", "items": items, "maxItems": 6 if ids else 0}})


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def parse(raw, *, reader, allowed):
    """Strict structural/ID admission only; statements are not semantic labels."""
    value = json.loads(raw, object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ValueError("expected object")
    if reader:
        if set(value) != {"findings"} or not isinstance(value["findings"], list):
            raise ValueError("invalid findings")
        if len(value["findings"]) > 6:
            raise ValueError("too many findings")
        for finding in value["findings"]:
            if (not isinstance(finding, dict) or set(finding) != {"source_id", "statement"}
                    or not isinstance(finding["source_id"], str)
                    or finding["source_id"] not in allowed
                    or not isinstance(finding["statement"], str)):
                raise ValueError("invalid finding or source ID")
    elif (set(value) != {"answer", "source_ids"} or not isinstance(value["answer"], str)
          or not isinstance(value["source_ids"], list) or len(value["source_ids"]) > 6
          or any(not isinstance(i, str) or i not in allowed for i in value["source_ids"])):
        raise ValueError("invalid answer or source ID")
    return value


async def ask_documents(pipeline, question, materials, *, history=None):
    started = monotonic()
    # Each invocation owns its snapshots and records, even with overlapping asks.
    materials = [source.model_copy(deep=True) for source in materials]
    events = []
    selected = []
    budget = getattr(pipeline, "_request_budget", None)
    retrieval = {"mode": "provided_documents", "protocol": PROTOCOL,
                 "input_count": len(materials), "selected_ids": []}
    if budget is not None:
        budget.retrieval = retrieval

    def response(status, final=None):
        raw = final["raw_text"] if final is not None else None
        return AskResponse(answer=raw if raw is not None else "", sources=selected,
            retrieval=deepcopy(retrieval), generation={
                "pipeline": "rwkv", "protocol": PROTOCOL, "output_mode": "immutable",
                "status": status, "raw_model_answer": raw, "answer_modified": False,
                "parsed_answer": final.get("parsed") if final is not None else None,
                "model_calls": events, "model": pipeline.settings.native_model,
                "elapsed_ms": round((monotonic() - started) * 1000),
                "stage_concurrency": pipeline.settings.native_document_read_concurrency,
                "semantic_support_verified": False})

    if history:
        return response("unsupported_history")
    if len({s.id for s in materials}) != len(materials) or any(not s.id for s in materials):
        return response("invalid_materials")
    # Reject the entire request before any model work instead of dropping sources.
    if budget is not None and budget.max_calls is not None and len(materials) + 1 > budget.max_calls:
        budget.stop_status = "call_budget_exceeded"
        return response("call_budget_exceeded")

    async def call(instruction, data, schema, ids, *, stage, max_tokens):
        message = instruction + "\n" + json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        event = {"call_id": str(uuid4()), "stage": stage, "status": "pending",
                 "messages": [{"role": "user", "content": message}],
                 "evidence_ids": list(ids), "raw_text": None, "parsed": None,
                 "prompt_sha256": digest(message), "schema": deepcopy(schema)}
        events.append(event)
        call_started = monotonic()
        try:
            result = await pipeline.model.complete(event["messages"], structured_schema=schema,
                max_tokens=max_tokens, assistant_prefill="<think></think", temperature=0,
                seed=11, stage=stage, evidence_ids=ids, trace=event)
            event.update(result.trace)
            event.update(status=result.status, raw_text=result.raw_text,
                         finish_reason=result.finish_reason, parsed=None)
            if result.status == "completed":
                try:
                    event["parsed"] = parse(result.raw_text, reader=stage == "reader", allowed=ids)
                except (ValueError, TypeError) as error:
                    event["validation_error"] = str(error)
        except asyncio.CancelledError:
            event.update(status="cancelled", provider_execution_cancelled=None)
            raise
        except Exception as error:
            event.update(status="transport_error", error_type=type(error).__name__)
        finally:
            event.update(raw_sha256=digest(event["raw_text"] or ""),
                         elapsed_ms=round((monotonic() - call_started) * 1000))
        return event

    sem = asyncio.Semaphore(pipeline.settings.native_document_read_concurrency)

    def original(source):
        return {"id": source.id, "title": source.title, "text": source.snippet}

    async def read(source):
        async with sem:
            return await call(READ, {"source": original(source), "question": question},
                              read_schema(source.id), [source.id], stage="reader", max_tokens=768)

    tasks = [asyncio.create_task(read(source)) for source in materials]
    try:
        readings = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    # Stable input order; failed/empty readings select nothing, never all sources.
    selected.extend(source for source, event in zip(materials, readings)
                    if event.get("parsed") is not None and event["parsed"]["findings"])
    retrieval.update(selected_ids=[s.id for s in selected],
                     failed_readings=sum(e.get("parsed") is None for e in readings))
    if budget is not None:
        budget.writer_sources = selected
    if not selected and pipeline.settings.native_empty_evidence_policy == "fail":
        return response("reader_failed" if retrieval["failed_readings"] else "no_evidence")
    final = await call(WRITE, {"sources": [original(s) for s in selected], "question": question},
                       answer_schema([s.id for s in selected]), [s.id for s in selected],
                       stage="writer", max_tokens=2048)
    status = final["status"]
    if status == "completed":
        status = ("invalid_output" if final.get("parsed") is None else
                  "partial" if retrieval["failed_readings"] else "completed")
    return response(status, final)
