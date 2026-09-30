"""Experimental, callback-driven execution of the complete retrieval grid.

The callbacks own search, semantic reading and follow-up decisions. Scheduling
preserves every requested cell, including ones left pending by an error/budget.
This module is not wired into the production pipeline.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import time
from typing import Awaitable, Callable

from .retrieval_plan import RetrievalPlanV1


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    url: str


Search = Callable[[str], Awaitable[list[Document]]]
Read = Callable[[dict, list[Document]], Awaitable[str]]
Followup = Callable[[list[dict], list[dict]], Awaitable[str]]


def strict_json(raw):
    if not isinstance(raw, str):
        raise ValueError("model output must be text")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique)


def parse_read(raw, documents):
    value = strict_json(raw)
    if not isinstance(value, dict) or set(value) != {"status", "quotes"}:
        raise ValueError("read schema mismatch")
    status, quotes = value["status"], value["quotes"]
    if status not in {"supported", "insufficient", "conflict"} or not isinstance(quotes, list):
        raise ValueError("invalid evidence status")
    by_id = {d.id: d for d in documents}
    seen = set()
    for quote in quotes:
        if (not isinstance(quote, dict) or set(quote) != {"source_id", "quote"}
                or not isinstance(quote["source_id"], str)
                or not isinstance(quote["quote"], str)):
            raise ValueError("invalid quote shape")
        key = quote["source_id"], quote["quote"]
        if (key in seen or key[0] not in by_id or not key[1]
                or key[1] not in by_id[key[0]].text):
            raise ValueError("unknown, repeated or nonverbatim quote")
        seen.add(key)
    if ((status == "insufficient" and quotes)
            or (status == "supported" and not quotes)
            or (status == "conflict" and len(quotes) < 2)):
        raise ValueError("status/quote count mismatch")
    return value


async def execute_plan(plan: RetrievalPlanV1, search: Search, read: Read,
                       followup: Followup | None = None, *, max_searches=24,
                       max_reads=96, max_followup_rounds=3):
    if any(type(n) is not int or n < 0 for n in (max_searches, max_reads, max_followup_rounds)):
        raise ValueError("budgets must be nonnegative integers")
    cells = {c["id"]: {**c, "state": "not_searched", "evidence": [],
                         "attempted_queries": []} for c in plan.cells()}
    trace, attempts, registry = [], [], {}
    counts = {"searches": 0, "reads": 0, "followup_rounds": 0}
    seen_queries = set()
    stop_reason = "initial_round_complete"

    async def execute_query(obj, query, ids):
        if counts["searches"] >= max_searches:
            return False
        counts["searches"] += 1
        seen_queries.add((obj, query))
        attempt = {"object": obj, "query": query, "cell_ids": ids.copy()}
        attempts.append(attempt)
        for cid in ids:
            cells[cid]["attempted_queries"].append(query)
        started = time.monotonic()
        try:
            documents = await search(query)
            if not isinstance(documents, list) or any(not isinstance(d, Document) for d in documents):
                raise ValueError("invalid search documents")
            if (any(not isinstance(d.id, str) or not d.id or not isinstance(d.text, str)
                    or not d.text or not isinstance(d.url, str) for d in documents)
                    or len({d.id for d in documents}) != len(documents)):
                raise ValueError("empty/duplicate source identity")
            for document in documents:
                if document.id in registry and registry[document.id] != document:
                    raise ValueError("source identity reused for changed bytes")
            registry.update({d.id: d for d in documents})
            trace.append({"stage": "search", **attempt, "status": "ok",
                          "source_ids": [d.id for d in documents],
                          "elapsed_s": time.monotonic() - started})
        except Exception as error:
            # External exception messages may contain credentials or request URLs.
            trace.append({"stage": "search", **attempt, "status": "error",
                          "error_type": type(error).__name__,
                          "elapsed_s": time.monotonic() - started})
            for cid in ids:
                cells[cid]["state"] = "search_error"
            return True
        for cid in ids:
            cell = cells[cid]
            if not documents:
                cell["state"] = "search_returned_no_documents"
                continue
            cell["state"] = "not_read"
            if counts["reads"] >= max_reads:
                continue
            counts["reads"] += 1
            raw = None
            started = time.monotonic()
            try:
                raw = await read(json.loads(json.dumps({k: cell[k] for k in (
                    "id", "object", "dimension", "conditions")})), list(documents))
                decision = parse_read(raw, documents)
                cell["evidence"] = decision["quotes"]
                cell["state"] = ("read_without_verified_evidence" if decision["status"] == "insufficient"
                                 else decision["status"])
                trace.append({"stage": "read", "cell_id": cid, "raw": raw,
                              "raw_sha256": sha256(raw.encode()).hexdigest(),
                              "status": "validated", "source_ids": [d.id for d in documents],
                              "elapsed_s": time.monotonic() - started})
            except Exception as error:
                cell["state"] = "read_error" if raw is None else "invalid_read_output"
                trace.append({"stage": "read", "cell_id": cid, "raw": raw,
                              "status": cell["state"], "error_type": type(error).__name__,
                              "source_ids": [d.id for d in documents],
                              "elapsed_s": time.monotonic() - started})
        return True

    # A query is a per-object opportunity, not one request for every grid cell.
    for query in plan.initial_queries:
        ids = [cid for cid, c in cells.items() if c["object"] == query.object]
        if not await execute_query(query.object, query.query, ids):
            stop_reason = "search_budget_exhausted"
            break
    for _ in range(max_followup_rounds):
        pending = [c for c in cells.values() if c["state"] not in {"supported", "conflict"}]
        if not pending:
            stop_reason = "all_cells_have_model_selected_evidence"
            break
        if followup is None:
            break
        if counts["searches"] >= max_searches or counts["reads"] >= max_reads:
            stop_reason = "budget_exhausted_with_pending_cells"
            break
        counts["followup_rounds"] += 1
        raw = None
        try:
            # Copies prevent callbacks from mutating persisted execution state.
            raw = await followup(json.loads(json.dumps(pending)), json.loads(json.dumps(attempts)))
            decision = strict_json(raw)
            if not isinstance(decision, dict):
                raise ValueError("follow-up must be an object")
            if decision.get("action") == "stop":
                if set(decision) != {"action", "reason"} or not isinstance(decision["reason"], str) or not decision["reason"].strip():
                    raise ValueError("invalid stop decision")
                trace.append({"stage": "followup", "raw": raw, "status": "model_stop"})
                stop_reason = "model_stop_with_pending_cells"
                break
            if set(decision) != {"action", "queries"} or decision["action"] != "search":
                raise ValueError("invalid follow-up action")
            queries = decision["queries"]
            if not isinstance(queries, list) or not 1 <= len(queries) <= len(plan.objects):
                raise ValueError("invalid follow-up query count")
            round_objects = set()
            for query in queries:
                if not isinstance(query, dict) or set(query) != {"object", "query", "cell_ids"}:
                    raise ValueError("invalid follow-up query shape")
                obj, text, ids = query["object"], query["query"], query["cell_ids"]
                if (obj not in plan.objects or obj in round_objects or not isinstance(text, str)
                        or not 1 <= len(text.strip()) <= 300 or (obj, text) in seen_queries
                        or not isinstance(ids, list) or not ids
                        or any(not isinstance(cid, str) for cid in ids)
                        or len(ids) != len(set(ids))):
                    raise ValueError("unknown/repeated/empty follow-up query")
                round_objects.add(obj)
                if any(cid not in cells or cells[cid]["object"] != obj
                       or cells[cid]["state"] in {"supported", "conflict"} for cid in ids):
                    raise ValueError("follow-up refers to unrelated or resolved cell")
            trace.append({"stage": "followup", "raw": raw, "status": "validated"})
            # Preserve plan order for equal per-object opportunities within a round.
            for query in sorted(queries, key=lambda q: plan.objects.index(q["object"])):
                if not await execute_query(query["object"], query["query"], query["cell_ids"]):
                    stop_reason = "search_budget_exhausted"
                    break
            else:
                stop_reason = "followup_round_limit"
        except Exception as error:
            trace.append({"stage": "followup", "raw": raw, "status": "invalid_or_failed",
                          "error_type": type(error).__name__})
            stop_reason = "followup_invalid_or_failed"
            break
    pending = [cid for cid, c in cells.items() if c["state"] not in {"supported", "conflict"}]
    if not pending:
        stop_reason = "all_cells_have_model_selected_evidence"
    return {"cells": list(cells.values()), "pending_cell_ids": pending,
            "counts": counts, "stop_reason": stop_reason, "trace": trace,
            "sources": [{**asdict(d), "text_sha256": sha256(d.text.encode()).hexdigest()}
                        for d in registry.values()], "semantic_quality_verified": False}
