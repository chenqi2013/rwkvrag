"""Opt-in unified RWKV retrieval. Source adapters execute; RWKV decides meaning.

All mutable state is request-local. Reducers consume at most three unchanged
model outputs, explicitly marked as derived, not unseen original documents.
No background tasks, transport retries, answer rewriting or legacy fallback.
Explicit control revisions are separately budgeted model calls, never raw repairs.
"""
from __future__ import annotations

import asyncio
from itertools import zip_longest
from time import monotonic

from .citation_audit import TAG, VALID_TAG
from .model_client import model_client_options
from .native_rwkv import NativeRWKVClient
from .retrieval_gate import RetrievalGate, RetrievalQueries, query_schema
from .schemas import AskResponse, SearchRequest, SearchResponse, SourceItem
from .structured_native import StructuredNativeRWKVClient
from .unified_contracts import (
    PROTOCOL, Direction, Plan, Reading, Review, Task, digest, dumps, parse, parse_issues,
    permitted_channels, prompt, reduction_calls, task_input,
)
from .unified_materials import EvidenceCatalog, SourceAdapters
from .unified_plan_contract import bounded_plan_schema
from .unified_span_contract import bound_span_schema
from .unified_span_reading import SpanAnswer, SpanCatalog, SpanReading, span_prompt, synthesis_prompt
from .web_retrieval import WebSearchAdapter

_FATAL = {"http_error", "transport_error", "timeout", "invalid_response", "invalid_request"}
_CONTROLS = {"task", "direction", "plan", "plan_revision", "review"}


def citation_check(reading):
    tags = TAG.findall(reading.answer)
    matches = [VALID_TAG.fullmatch(tag) for tag in tags]
    labels = sorted({int(match[1]) for match in matches if match})
    return {"label_ids": labels,
            "unknown_label_ids": [n for n in labels if f"E{n}" not in reading.evidence_ids],
            "invalid_labels": [tag for tag, match in zip(tags, matches, strict=True) if not match],
            "missing_valid_citation": bool(reading.evidence_ids and not labels),
            "semantic_support_verified": False}


class UnifiedRetrievalPipeline:
    def __init__(self, settings, index, model=None, *, recorder=None, adapters=None, gate=None):
        # Revalidate mutated settings before constructing transports.
        settings.validate_unified_retrieval()
        if settings.native_unified is None:
            raise ValueError("unified retrieval requires an explicit policy")
        self.settings = settings
        self.web = WebSearchAdapter(settings)
        self.span_reading = settings.native_unified.reading_protocol == "span_ids_v1"
        self.cycle = settings.native_unified.control_protocol == "review_cycle_v1"
        self.protocol = ("rwkv-unified-v4" if self.cycle else
                         "rwkv-unified-v3" if self.span_reading else PROTOCOL)
        client = StructuredNativeRWKVClient if self.span_reading else NativeRWKVClient
        self.model = model or client(**model_client_options(settings, recorder=recorder))
        self.adapters = adapters or SourceAdapters(settings, index, self.web)
        self.gate = gate or (RetrievalGate(settings.retrieval_gate) if settings.retrieval_gate else None)

    async def aclose(self):
        await self.model.aclose()
        if self.gate is not None:
            await self.gate.aclose()

    async def ask(self, request):
        return await self._run(request, answer=True)

    async def search(self, request):
        # Same task, action, reading and follow-up core; no final-answer reducers.
        return await self._run(request, answer=False)

    async def ask_materials(self, question, materials, *, history=None):
        return await self._run(SearchRequest(question=question, history=history or []),
                               answer=True, materials=materials)

    async def _run(self, request, *, answer, materials=None):
        run = _Run(self, request, answer=answer, materials=materials)
        deadline = asyncio.timeout(run.timeout)
        try:
            async with deadline:
                await run.execute()
        except TimeoutError:
            if not deadline.expired():
                raise
            run.status = "request_timeout"
        # Deliberately do not use bounded_request's most-recent-Writer fallback.
        return run.response()


class _Run:
    def __init__(self, pipeline, request, *, answer, materials):
        self.pipeline, self.request = pipeline, request
        self.policy = pipeline.settings.native_unified
        self.want_answer, self.materials = answer, materials
        self.mode = "materials" if materials is not None else request.retrieval_mode
        self.allowed = permitted_channels(self.mode)
        self.messages = [{"id": f"M{i + 1}", **message.model_dump()}
                         for i, message in enumerate(request.history)]
        self.messages.append({"id": f"M{len(self.messages) + 1}", "role": "user",
                              "content": request.question})
        self.started = monotonic()
        self.calls, self.nodes, self.actions, self.plans, self.reviews = [], [], [], [], []
        self.leaves = []
        self.span_bindings = {}
        self.control_failures = []
        self.directions, self.plan_attempts, self.round_budgets = [], [], []
        self.round_number = None
        self.round_read_limit = None
        self.root = None
        self.task = None
        self.direction = None
        self.channels = []
        self.routing = None
        self.retrieval_needed = None
        self.status = "pending"
        self.stop_reason = None
        self.budget_events = []
        self.catalog = EvidenceCatalog(self.policy.unit_characters, pack_adjacent=pipeline.span_reading)
        self.admitted = 0
        self.max_calls = min(self.policy.max_calls,
                            pipeline.settings.native_request_max_calls or self.policy.max_calls)
        self.timeout = min(self.policy.timeout_seconds,
                           pipeline.settings.native_request_timeout_seconds or self.policy.timeout_seconds)
        self.model_failed = False

    def remaining(self):
        return {"actions": self.policy.max_actions - len(self.actions),
                "web_actions": self.pipeline.settings.web_search_max_queries - sum(
                    action["channel"] == "web" for action in self.actions),
                "read_units": self.policy.max_units - len(self.leaves),
                "model_calls": self.max_calls - self.admitted}

    def context(self):
        return {**task_input(self.messages),
                "task": self.task.model_dump() if self.task is not None else None,
                "allowed_channels": self.allowed, "selected_channels": self.channels,
                "direction": self.direction, "remaining": self.remaining(),
                **({"round": self.round_number, "round_read_units": self.round_read_limit,
                    "entry_scope": "closed_materials" if self.materials is not None else "conversation_and_permitted_retrieval"}
                   if self.pipeline.cycle else {})}

    def reduction_cost(self, count):
        # A v3 leaf is local findings, not a final answer. Even n=1 needs a
        # separately traced, budgeted synthesis. An empty frontier can only
        # report its execution gaps; it is not a source of factual evidence.
        return max(int(self.pipeline.span_reading), reduction_calls(count))

    async def call(self, purpose, payload, contract=None, *, evidence_ids=(),
                   parent_ids=(), terminal=False, token_limit=None, node=None, prompt_text=None,
                   revision_of=None, contract_schema=None):
        schema = contract.model_json_schema() if contract is not None and contract_schema is None else contract_schema
        text = prompt_text if prompt_text is not None else prompt(
            purpose, payload, contract, protocol=self.pipeline.protocol, contract_schema=schema)
        event = {"purpose": purpose, "protocol": self.pipeline.protocol, "parent_node_ids": list(parent_ids),
                 "terminal": terminal, "input_evidence_ids": list(evidence_ids),
                 "messages": [{"role": "user", "content": text}],
                 "raw_text": None, "status": "pending", "completion_attempted": False}
        if revision_of is not None:
            event.update(revision_of_call_id=revision_of.get("call_id"),
                         revision_of_raw_sha256=revision_of.get("raw_text_sha256"))
        if node is not None:
            event["node_id"] = node["id"]
            node["event_index"] = len(self.calls)
        self.calls.append(event)
        if self.model_failed or self.admitted >= self.max_calls:
            event["status"] = "blocked_model_failure" if self.model_failed else "call_budget_exceeded"
            return None, event
        reserve = (self.reduction_cost(len(self.leaves))
                   if self.want_answer and self.pipeline.span_reading and purpose in _CONTROLS else 0)
        if reserve and self.remaining()["model_calls"] <= reserve:
            event["status"] = "final_budget_reserved"
            self.stop_reason = "control_call_budget_exhausted"
            self.budget_events.append({"stage": purpose, "reason": event["status"],
                                       "reserved_model_calls": reserve})
            return None, event
        self.admitted += 1
        event["request_call_number"] = self.admitted
        try:
            result = await self.pipeline.model.complete(event["messages"],
                stage="unified_" + purpose, evidence_ids=tuple(evidence_ids), trace=event,
                max_tokens=token_limit or self.policy.control_tokens,
                assistant_prefill="<think></think", temperature=1.0, top_p=1.0, top_k=1, seed=11,
                **({"structured_schema": schema}
                   if self.pipeline.span_reading and contract is not None else {}))
            event.update(result.trace)
            event.update(raw_text=result.raw_text, status=result.status)
            if isinstance(result.raw_text, str):
                event["raw_text_sha256"] = digest(result.raw_text)
            self.model_failed |= result.status in _FATAL
            if result.status != "completed":
                return None, event
            if contract is None:
                return result.raw_text, event
            try:
                value = parse(result.raw_text, contract)
            except (ValueError, TypeError) as error:
                event["contract_status"] = "invalid_json_contract"
                event["contract_errors"] = parse_issues(error)
                return None, event
            event["contract_status"] = "parsed"
            return value, event
        except asyncio.CancelledError:
            event.update(status="cancelled", provider_execution_cancelled=None)
            raise
        except Exception as error:
            # Never serialize exception messages containing endpoint credentials.
            self.model_failed = True
            event.update(status="transport_error", error_type=type(error).__name__)
            return None, event

    def invalid(self, event, error):
        event.update(contract_status="invalid_binding", contract_error=str(error))

    async def execute(self):
        collected = await self.collect()
        # Early control failure stops acquisition, not a separately reserved
        # span final. Search/v2 retain their early-return contract.
        if collected or (self.want_answer and self.pipeline.span_reading):
            await self.finish()

    async def collect(self):
        if self.materials is None and self.pipeline.gate is not None:
            if not await self.decide_retrieval():
                return
            if self.retrieval_needed is False:
                self.stop_reason = "retrieval_not_required"
                return True
        if not (self.materials is not None and self.pipeline.span_reading):
            task, event = await self.call("task", task_input(self.messages), Task)
            if task is None:
                self.status = "task_failed"
                return
            try:
                task.check(self.messages)
            except ValueError as error:
                self.invalid(event, error)
                self.status = "task_failed"
                return
            self.task = task
        # Closed span reading already receives the complete original dialogue.
        # It needs neither a routing Task nor a generated conversation quote.
        if self.materials is not None:
            # The caller supplied a closed material set. This is an entry-point
            # capability binding, not a fabricated semantic/router model output.
            self.direction = "provided"
            self.routing = {"source": "explicit_materials_entry", "choice": "provided",
                            "allowed_channels": [], "model_decision": False}
        elif not await self.choose_direction():
            return
        if self.direction == "clarify":
            await self.clarify()
            return
        if self.direction == "provided" and (not self.pipeline.cycle or self.materials is not None):
            provided = self.materials
            if provided is None:
                # User-authored turns only; assistant history is not factual evidence.
                provided = [SourceItem(id=m["id"], document_id=m["id"], source="user_material",
                    title=m["id"], score=0.0, snippet=m["content"])
                    for m in self.messages if m["role"] == "user"]
            queues = self.add_materials(provided, self.request.question, "provided")
            await self.read(queues, reserve_review=False)
            self.stop_reason = self.stop_reason or "provided_materials_processed"
        else:
            await self.retrieve()
        return True

    async def finish(self):
        if self.root is not None:  # Already made an explicit clarification call.
            return
        if self.model_failed:
            self.status = "model_failed"
            return
        if self.status != "pending":
            if (self.pipeline.span_reading and (self.want_answer or self.leaves)
                    and self.status in {"task_failed", "plan_failed", "review_failed", "routing_failed"}):
                # No rejected control raw, fabricated direction, or new retrieval.
                # The final sees the original dialogue and mechanical gaps only.
                self.control_failures.append(self.status)
                self.stop_reason = self.stop_reason or "control_failed"
                self.status = "pending"
            else:
                return
        if not self.leaves and not (self.want_answer and self.pipeline.span_reading):
            self.status = "no_read_output"
            return
        if self.want_answer:
            await self.reduce()
        self.status = "model_failed" if self.model_failed else "completed"

    def add_materials(self, sources, query, action_id):
        queues = []
        for source in sources:
            try:
                queues.append(self.catalog.add(source, query, action_id))
            except ValueError as error:
                self.catalog.observations.append({"action_id": action_id, "source_id": source.id,
                    "status": "invalid_material", "error_code": str(error)})
        return queues

    def action_inputs(self):
        # Provider receipts may contain entire unselected snapshots. Controllers
        # receive execution metadata, not a hidden second material channel.
        return [{k: a[k] for k in ("id", "round", "channel", "query", "goal_ids", "status",
                                  "error_code", "error_type") if k in a} for a in self.actions]

    async def choose_direction(self):
        if self.pipeline.gate is not None and self.retrieval_needed is True:
            self.channels = list(self.allowed)
            self.direction = "hybrid" if len(self.channels) == 2 else self.channels[0]
            self.routing.update(choice=self.direction, allowed_channels=list(self.allowed),
                                channel_selection="all_explicitly_permitted_channels")
            return True
        choices = [choice for choice in ("provided", "knowledge_base", "web", "hybrid", "clarify")
                   if set(Direction(choice=choice).channels()) <= set(self.allowed)]
        payload = {**self.context(), "allowed_choices": choices}
        if self.pipeline.cycle:
            payload.update(previous_actions=self.action_inputs(), reviews=self.reviews)
        decision, event = await self.call("direction", payload, Direction)
        if decision is not None:
            try:
                decision.check(self.allowed)
            except ValueError as error:
                self.invalid(event, error)
                decision = None
        self.directions.append({"round": self.round_number if self.round_number is not None else 0,
            "call_id": event.get("call_id"), "completion_attempted": event["completion_attempted"],
            "status": "admitted" if decision else "rejected" if event["completion_attempted"] else "not_started",
            "choice": decision.choice if decision else None})
        if decision is None:
            self.status = "routing_failed"
            return False
        self.direction, self.channels = decision.choice, decision.channels()
        self.routing = {"source": "model", "choice": self.direction,
                        "call_id": event.get("call_id"), "model_decision": True}
        return True

    async def decide_retrieval(self):
        event = {"purpose": "retrieval_gate", "raw_text": None, "status": "pending",
                 "completion_attempted": False, "input_evidence_ids": []}
        self.calls.append(event)
        reserve = int(self.want_answer)
        if self.remaining()["model_calls"] <= reserve:
            event["status"] = "final_budget_reserved"
            self.status = "routing_failed"
            return False
        self.admitted += 1
        event["request_call_number"] = self.admitted
        try:
            self.retrieval_needed = await self.pipeline.gate.decide(self.messages, trace=event)
        except asyncio.CancelledError:
            event["status"] = "cancelled"
            raise
        except Exception as error:
            event.update(status="transport_error", error_type=type(error).__name__)
            self.model_failed = True
        if self.retrieval_needed is None:
            self.status = "routing_failed"
            return False
        self.direction = "hybrid" if self.retrieval_needed else "provided"
        self.routing = {"source": "retrieval_gate", "model_decision": True,
                        "needs_retrieval": self.retrieval_needed, "call_id": event.get("call_id"),
                        "choice": self.direction}
        return True

    async def clarify(self):
        self.status = "needs_clarification"
        if not self.want_answer:
            return
        self.root = self.new_node("clarify", [], 0)
        try:
            raw, result = await self.call("clarify", self.context(), terminal=True, node=self.root,
                                        token_limit=self.pipeline.settings.generation_max_tokens)
            if raw is None:
                self.status = "clarification_failed"
        finally:
            result = self.calls[self.root["event_index"]]
            self.root.update(raw_text=result.get("raw_text"), status=result["status"],
                             call_id=result.get("call_id"), raw_text_sha256=result.get("raw_text_sha256"))

    def work_capacity(self, control_calls):
        """Reserve one new reading, its Review, and the entire existing/new reduction."""
        reserve = self.reduction_cost(len(self.leaves) + 1) if self.want_answer else 0
        return self.remaining()["model_calls"] >= control_calls + 2 + reserve

    async def make_plan(self, round_number):
        previous = None
        revisions = self.policy.max_plan_revisions if self.pipeline.cycle else 0
        for attempt in range(revisions + 1):
            if self.pipeline.cycle and not self.work_capacity(1):
                self.stop_reason = "control_call_budget_exhausted"
                if previous is not None:
                    self.status = "plan_failed"
                return None
            payload = {**self.context(), "round": round_number,
                       "previous_actions": self.action_inputs(), "reviews": self.reviews}
            if previous is not None:
                rejected = {**{k: previous.get(k) for k in (
                    "call_id", "raw_text", "raw_text_sha256", "status", "contract_status",
                    "contract_errors", "contract_error")}, "kind": "rejected_control_output_not_evidence"}
                # Keep the rejected control bytes intact, before the current
                # task/budgets/actions/Review rather than as the trailing example.
                payload = {"rejected_plan": rejected, **payload}
            schema = None
            paired = self.pipeline.gate is not None and self.retrieval_needed is True
            if paired:
                remaining = self.remaining()
                try:
                    schema = query_schema(self.task, self.channels, actions=remaining["actions"],
                        web=remaining["web_actions"],
                        read_units=min(self.round_read_limit or remaining["read_units"], remaining["read_units"]))
                except ValueError:
                    self.stop_reason = "paired_retrieval_budget_exhausted"
                    return None
            elif self.policy.plan_contract == "bounded_v1":
                remaining = self.remaining()
                schema = bounded_plan_schema(self.task, self.channels,
                    remaining_actions=remaining["actions"], remaining_web=remaining["web_actions"],
                    read_units=min(self.round_read_limit, remaining["read_units"]))
            contract = RetrievalQueries if paired else Plan
            purpose = "plan_revision" if attempt else "plan"
            text = None
            if paired:
                text = ("生成检索查询queries；每条query会原样交给所有允许的资料渠道。"
                        "你决定查询文字和goal_ids，不选择渠道。未安排的目标列在unplanned_goal_ids。\n"
                        "仅按以下Schema输出JSON：\n" + dumps(schema) + "\n输入数据：\n" + dumps(payload))
            plan, event = await self.call(purpose, payload, contract,
                revision_of=previous, contract_schema=schema, prompt_text=text)
            query_plan = plan
            if paired and plan is not None:
                plan = plan.expand(self.channels)
                event["execution_contract"] = "queries_to_all_permitted_channels_v1"
            if previous is not None:
                event["revision_input_layout"] = "rejected_first_v1"
            if schema is not None:
                event.update(plan_contract="retrieval_queries_v1" if paired else self.policy.plan_contract,
                             plan_schema_sha256=digest(dumps(schema)))
            if plan is not None:
                try:
                    plan.check(self.task, self.channels, self.remaining()["actions"],
                        self.remaining()["web_actions"], {(a["channel"], a["query"]) for a in self.actions},
                        exclusive=self.pipeline.cycle)
                    if self.pipeline.cycle and len(plan.actions) > self.round_read_limit:
                        raise ValueError("round_read_budget_exceeded")
                except ValueError as error:
                    self.invalid(event, error)
                    plan = None
            self.plan_attempts.append({"round": round_number, "attempt": attempt,
                "call_id": event.get("call_id"), "raw_text_sha256": event.get("raw_text_sha256"),
                "status": "admitted" if plan is not None else "rejected",
                "contract_error": event.get("contract_error"),
                "contract_errors": event.get("contract_errors"), "native_status": event["status"]})
            if plan is not None:
                self.plans.append({"call_id": event.get("call_id"), **plan.model_dump(),
                                   **({"model_query_plan": query_plan.model_dump(),
                                       "execution_contract": "queries_to_all_permitted_channels_v1"} if paired else {})})
                return plan
            if self.model_failed:
                break  # No retries for native/transport failure.
            previous = event
        self.status = "plan_failed"
        return None

    async def acquire(self, plan, round_number):
        groups = []
        for action in plan.actions:
            receipt = {"id": f"A{len(self.actions) + 1}", "round": round_number,
                       **action.model_dump(), "status": "pending"}
            self.actions.append(receipt)
            started = monotonic()
            try:
                sources, detail = await self.pipeline.adapters.execute(action, self.request)
                receipt.update(status="completed", receipt=detail)
                groups.append(self.add_materials(sources, action.query, receipt["id"]))
            except asyncio.CancelledError:
                receipt.update(status="cancelled", provider_execution_cancelled=None)
                raise
            except Exception as error:
                receipt.update(status="failed", error_type=type(error).__name__)
                code = getattr(error, "safe_code", None)
                if code is not None:
                    receipt["error_code"] = code
            finally:
                receipt["elapsed_ms"] = round((monotonic() - started) * 1000)
        if not self.pipeline.cycle:
            return [queue for group in groups for queue in group]
        # Round-robin actions, then sources, then units: a large first action
        # must not mechanically starve all later actions under the round cap.
        return [queue for row in zip_longest(*groups) for queue in row if queue is not None]

    async def retrieve(self):
        for round_number in range(self.policy.max_rounds):
            if not self.remaining()["read_units"] or (not self.pipeline.cycle and not self.remaining()["actions"]):
                self.stop_reason = "retrieval_budget_exhausted"
                break
            self.round_number = round_number
            if self.pipeline.cycle:
                rounds_left = self.policy.max_rounds - round_number
                self.round_read_limit = (self.remaining()["read_units"] + rounds_left - 1) // rounds_left
                self.round_budgets.append({"round": round_number, "read_limit": self.round_read_limit,
                    "read_units_before": len(self.leaves), "calls_before": self.admitted})
                if round_number:
                    if not self.work_capacity(2):  # Direction + Plan, reading + Review + root.
                        self.stop_reason = "control_call_budget_exhausted"
                        break
                    if not await self.choose_direction():
                        return
                    if self.direction == "clarify":
                        await self.clarify()
                        return
            if self.pipeline.cycle and self.direction == "provided":
                provided = [SourceItem(id=m["id"], document_id=m["id"], source="user_material",
                    title=m["id"], score=0.0, snippet=m["content"]) for m in self.messages if m["role"] == "user"]
                queues = self.add_materials(provided, self.request.question, "provided")
            else:
                if not self.remaining()["actions"]:
                    self.stop_reason = "retrieval_budget_exhausted"
                    break
                plan = await self.make_plan(round_number)
                if plan is None:
                    return
                if not plan.actions:
                    self.stop_reason = "model_planned_no_actions"
                    break
                # Do not spend external quota when no reading+review slot can run.
                if not self.work_capacity(0):
                    self.stop_reason = "control_call_budget_exhausted"
                    self.budget_events.append({"stage": "acquisition", "reason": self.stop_reason,
                                               "not_executed_actions": plan.model_dump()["actions"]})
                    break
                queues = await self.acquire(plan, round_number)
            await self.read(queues, reserve_review=True, unit_limit=self.round_read_limit,
                            reserve_followup=self.pipeline.cycle and round_number + 1 < self.policy.max_rounds)
            if self.pipeline.cycle:
                self.round_budgets[-1].update(read_units_after=len(self.leaves), calls_after_read=self.admitted)
            if self.model_failed:
                return
            reserve = self.reduction_cost(len(self.leaves)) if self.want_answer else 0
            if self.remaining()["model_calls"] <= reserve:
                self.stop_reason = "control_call_budget_exhausted"
                break
            review, event = await self.call("review", {**self.context(),
                "leaf_outputs": [self.node_input(node) for node in self.leaves],
                "actions": self.action_inputs(),
                **({"inherited_span_bindings": self.inherited_span_inputs(list(dict.fromkeys(
                    s for n in self.leaves if n["status"] == "completed" for s in n.get("span_ids", []))))}
                   if self.pipeline.span_reading else {}),
                "unread_unit_ids": [key for key in self.catalog.units if key not in self.catalog.read]}, Review)
            if review is None:
                self.status = "review_failed"
                return
            self.reviews.append({"call_id": event.get("call_id"), **review.model_dump()})
            if review.next == "stop":
                self.stop_reason = "model_stop"
                break
            if round_number + 1 == self.policy.max_rounds:
                self.stop_reason = "round_budget_exhausted"
                if self.pipeline.cycle:
                    break
            if self.remaining()["model_calls"] < reserve + 3:
                self.stop_reason = "control_call_budget_exhausted"
                break

    async def read(self, queues, *, reserve_review, unit_limit=None, reserve_followup=False):
        candidates = list(self.catalog.candidates(queues))
        capacity = min(len(candidates), self.remaining()["read_units"],
                       unit_limit if unit_limit is not None else self.remaining()["read_units"])
        # If a two-round minimum fits, don't spend its control/read calls on
        # extra first-round leaves. Otherwise permit a bounded single round.
        future_root = self.reduction_cost(len(self.leaves) + 2) if self.want_answer else 0
        keep_followup = (reserve_followup and self.remaining()["read_units"] > 1
                         and self.remaining()["model_calls"] >= 6 + future_root)
        if self.pipeline.cycle and self.round_budgets:
            self.round_budgets[-1]["minimum_followup_capacity_reserved"] = keep_followup
        while capacity > 0:
            reduce_cost = self.reduction_cost(len(self.leaves) + capacity) if self.want_answer else 0
            required = capacity + reduce_cost + int(reserve_review)
            if keep_followup:
                future_root = self.reduction_cost(len(self.leaves) + capacity + 1) if self.want_answer else 0
                required = max(required, capacity + int(reserve_review) + 4 + future_root)
            if required <= self.remaining()["model_calls"]:
                break
            capacity -= 1
        async def read_one(evidence_id):
            source = self.catalog.units[evidence_id]
            payload = {**self.context(), "original_material": self.catalog.material(evidence_id)}
            if self.pipeline.span_reading:
                node = await self.read_span_node("leaf", payload, evidence_id=evidence_id,
                                                 token_limit=self.policy.leaf_tokens)
            else:
                node = await self.read_node("leaf", payload, [evidence_id],
                    originals=[source.id], token_limit=self.policy.leaf_tokens)
            self.catalog.read[evidence_id] = {"node_id": node["id"], "status": node["status"]}
            return node

        # Admit a bounded wave only after reserving all dependent reductions.
        # Results retain source order even when GPU requests finish out of order.
        width = self.policy.stage_concurrency
        for start in range(0, capacity, width):
            self.leaves.extend(await self.concurrent_stage(
                candidates[start:min(start + width, capacity)], read_one))
            if self.model_failed:
                break
        if len(candidates) > capacity:
            self.stop_reason = "read_budget_exhausted"
            self.budget_events.append({"stage": "reading", "reason": self.stop_reason,
                "candidate_units": len(candidates), "admitted_units": capacity,
                "not_admitted_unit_ids": candidates[capacity:]})

    async def concurrent_stage(self, items, operation):
        """Drain or cancel every admitted sibling before a dependency can run."""
        if self.policy.stage_concurrency == 1:
            return [await operation(item) for item in items]
        tasks = [asyncio.create_task(operation(item)) for item in items]
        try:
            return await asyncio.gather(*tasks)
        finally:
            # Request timeout/cancellation must not leave siblings generating
            # untracked results after response construction.
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def new_node(self, kind, parents, level):
        node = {"id": f"N{len(self.nodes) + 1}", "kind": kind, "level": level,
                "parents": [parent["id"] for parent in parents],
                "parent_call_ids": [parent.get("call_id") for parent in parents],
                "parent_raw_hashes": [parent.get("raw_text_sha256") for parent in parents],
                "raw_text": None, "status": "pending", "evidence_ids": [],
                "parsed_model_evidence_ids": None}
        self.nodes.append(node)
        return node

    async def read_node(self, purpose, payload, allowed, *, parents=(), originals=(),
                        token_limit=None, terminal=False):
        level = 1 + max((node["level"] for node in parents), default=-1)
        node = self.new_node(purpose, parents, level)
        node["allowed_evidence_ids"] = list(allowed)
        node["original_text_input_ids"] = list(originals)
        if terminal:
            self.root = node
        try:
            value, event = await self.call(purpose, payload, Reading, node=node,
                parent_ids=node["parents"], evidence_ids=originals,
                token_limit=token_limit, terminal=terminal)
            node.update(raw_text=event.get("raw_text"), call_id=event.get("call_id"),
                        raw_text_sha256=event.get("raw_text_sha256"), status=event["status"])
            if value is not None:
                # Record the model's parsed choice even if a later identity or
                # citation check rejects admission. None means unavailable.
                node["parsed_model_evidence_ids"] = list(value.evidence_ids)
                try:
                    value.check(allowed)
                    audit = citation_check(value)
                    node["citation_audit"] = audit
                    if audit["unknown_label_ids"] or audit["invalid_labels"] or audit["missing_valid_citation"]:
                        raise ValueError("citation_identity_failed")
                    node.update(evidence_ids=list(value.evidence_ids), parsed=value.model_dump())
                    event["contract_status"] = "valid_structure_and_binding"
                except ValueError as error:
                    self.invalid(event, error)
                    node["status"] = "invalid_contract"
            elif event["status"] == "completed":
                node["status"] = "invalid_contract"
            return node
        finally:
            # Reachable after cancellation too; only an explicitly designated root can be returned.
            if "event_index" in node:
                event = self.calls[node["event_index"]]
                node.update(raw_text=event.get("raw_text"), call_id=event.get("call_id"),
                            raw_text_sha256=event.get("raw_text_sha256"))
                if node["status"] == "pending":
                    node["status"] = event["status"]

    def inherited_span_inputs(self, span_ids):
        # Never expose lookup text/headings as if a reducer reread originals.
        keys = ("selected_span_id", "evidence_id", "source_id", "snapshot_id", "span",
                "catalog_sha256", "leaf_node_id", "leaf_call_id", "leaf_raw_sha256",
                "leaf_finding_indices", "title", "uri", "document_id", "source_channel")
        return {key: {**{k: self.span_bindings[key][k] for k in keys},
                      "original_text_in_this_prompt": False, "metadata_not_original_body": True,
                      "semantic_support_verified": False} for key in span_ids}

    async def read_span_node(self, purpose, payload, *, evidence_id=None, parents=(),
                             token_limit=None, terminal=False, reread_of=None):
        node = self.new_node(purpose, parents, 1 + max((p["level"] for p in parents), default=-1))
        node.update(span_ids=[], parsed_model_span_ids=None)
        leaf = evidence_id is not None
        catalog = SpanCatalog.build(payload["original_material"]) if leaf else None
        allowed = ([s["id"] for s in catalog.model_material()["body_spans"]] if leaf else
                   list(payload["inherited_span_bindings"]))
        node["allowed_span_ids"] = allowed
        originals = [self.catalog.units[evidence_id].id] if leaf else []
        node["original_text_input_ids"] = originals
        if leaf:
            node["original_evidence_id"] = evidence_id
        if reread_of is not None:
            node.update(reread_of_node_id=reread_of["id"], reread_of_call_id=reread_of.get("call_id"),
                        reread_of_raw_sha256=reread_of.get("raw_text_sha256"), reading_attempt=2)
        if terminal:
            self.root = node
        contract = SpanReading if leaf else SpanAnswer
        schema = bound_span_schema(contract, allowed)
        node["span_schema_sha256"] = digest(dumps(schema))
        text = (span_prompt(payload, catalog, rereading=reread_of is not None) if leaf else
                synthesis_prompt(payload, terminal=terminal, contract_schema=schema))
        try:
            value, event = await self.call(purpose, payload, contract,
                node=node, parent_ids=node["parents"], evidence_ids=originals,
                terminal=terminal, token_limit=token_limit, prompt_text=text, contract_schema=schema)
            event.update(span_schema_sha256=node["span_schema_sha256"],
                         **({"reread_of_node_id": reread_of["id"], "reading_attempt": 2}
                            if reread_of is not None else {}))
            node.update(raw_text=event.get("raw_text"), call_id=event.get("call_id"),
                        raw_text_sha256=event.get("raw_text_sha256"), status=event["status"])
            if value is None:
                if event["status"] == "completed":
                    node["status"] = "invalid_contract"
                return node
            selected = [f.span_id for f in value.findings] if leaf else list(value.span_ids)
            node["parsed_model_span_ids"] = selected
            try:
                if leaf:
                    bindings = value.bind(catalog)  # all-or-nothing; no salvaged findings
                    receipts = {}
                    for binding in bindings:
                        key = binding["selected_span_id"]
                        if key not in receipts:
                            source = self.catalog.units[binding["evidence_id"]]
                            receipts[key] = {**binding, "title": source.title, "uri": source.uri,
                                "document_id": source.document_id, "source_channel": source.source,
                                "leaf_node_id": node["id"],
                                "leaf_call_id": event.get("call_id"),
                                "leaf_raw_sha256": event.get("raw_text_sha256"),
                                "leaf_finding_indices": []}
                        receipts[key]["leaf_finding_indices"].append(binding["finding_index"])
                else:
                    value.check(allowed)
                    receipts = {key: self.span_bindings[key] for key in selected}
                ids = list(dict.fromkeys(b["evidence_id"] for b in receipts.values()))
                if not leaf:
                    # Span IDs alone declare references. Body literals/code are
                    # not a second, legacy citation namespace or admission gate.
                    node["citation_audit"] = {
                        "reference_protocol": "span_ids_v1", "selected_span_ids": selected,
                        "inline_citation_required": False, "inline_citation_interpreted": False,
                        "semantic_support_verified": False}
                self.span_bindings.update(receipts)
                node.update(span_ids=list(receipts), evidence_ids=ids, parsed=value.model_dump())
                event["contract_status"] = "valid_structure_and_binding"
            except ValueError as error:
                self.invalid(event, error)
                node["status"] = "invalid_contract"
            return node
        finally:
            if "event_index" in node:
                event = self.calls[node["event_index"]]
                node.update(raw_text=event.get("raw_text"), call_id=event.get("call_id"),
                            raw_text_sha256=event.get("raw_text_sha256"))
                if node["status"] == "pending":
                    node["status"] = event["status"]

    def node_input(self, node):
        eligible = node["status"] == "completed"
        # Failure bytes remain in the audit graph, not in downstream semantic
        # input. Status alone must not expose unbound content as possible facts.
        return {"node_id": node["id"], "call_id": node.get("call_id"), "status": node["status"],
                "raw_text": node["raw_text"] if eligible else None,
                "raw_text_sha256": node.get("raw_text_sha256"),
                "raw_withheld": not eligible,
                "model_selected_evidence_ids": node["evidence_ids"],
                **({"model_selected_span_ids": node.get("span_ids", [])}
                   if self.pipeline.span_reading else {}),
                "kind": "model_derived_output_not_original_evidence"}

    async def reduce(self):
        if self.pipeline.span_reading:
            await self.reduce_spans()
            return
        current = list(self.leaves)
        while len(current) > 1:
            next_level = []
            for start in range(0, len(current), 3):
                children = current[start:start + 3]
                allowed = list(dict.fromkeys(e for child in children for e in child["evidence_ids"]))
                if not any(child["status"] == "completed" for child in children):
                    node = self.new_node("merge", children, children[0]["level"] + 1)
                    node["status"] = "blocked_dependency"
                    if len(current) <= 3:
                        self.root = node
                else:
                    node = await self.read_node("merge", {**self.context(),
                        "children": [self.node_input(child) for child in children],
                        "inherited_bindings_not_original_text": {
                            e: self.catalog.binding(e) for e in allowed},
                        "execution_scope": self.execution_scope()}, allowed, parents=children,
                        token_limit=(self.pipeline.settings.generation_max_tokens if len(current) <= 3
                                     else self.policy.merge_tokens), terminal=len(current) <= 3)
                next_level.append(node)
                if self.model_failed:
                    return
            current = next_level
        if current:
            # n=1: the existing leaf raw is the root, not a program-written answer.
            self.root = current[0]

    async def reduce_spans(self):
        current = list(self.leaves)
        while True:
            terminal = len(current) <= 3
            next_level = []
            groups = [current[start:start + 3] for start in range(0, max(1, len(current)), 3)]

            async def reduce_group(children):
                eligible = [child for child in children if child["status"] == "completed"]
                allowed = list(dict.fromkeys(s for child in eligible for s in child.get("span_ids", [])))
                purpose = "final" if terminal else "merge"
                if not eligible and not terminal:
                    node = self.new_node(purpose, children, children[0]["level"] + 1)
                    node.update(status="blocked_dependency", span_ids=[], parsed_model_span_ids=None)
                else:
                    node = await self.read_span_node(purpose, {**self.context(),
                        "children": [self.node_input(child) for child in children],
                        "inherited_span_bindings": self.inherited_span_inputs(allowed),
                        "execution_scope": self.execution_scope()}, parents=children,
                        terminal=terminal,
                        token_limit=self.pipeline.settings.generation_max_tokens if terminal
                        else self.policy.merge_tokens)
                return node

            width = self.policy.stage_concurrency
            for start in range(0, len(groups), width):
                next_level.extend(await self.concurrent_stage(groups[start:start + width], reduce_group))
                if self.model_failed:
                    return
            if terminal:
                return
            current = next_level

    def execution_scope(self):
        return {"stop_reason": self.stop_reason,
                "task_unresolved": list(self.task.unresolved) if self.task else [],
                "read_units": len(self.catalog.read),
                "unread_units": len(self.catalog.units) - len(self.catalog.read),
                "failed_actions": [a["id"] for a in self.actions if a["status"] != "completed"],
                "failed_nodes": [n["id"] for n in self.nodes if n["status"] not in {"completed", "pending"}],
                "unplanned_goal_ids_by_round": [p["unplanned_goal_ids"] for p in self.plans],
                # Earlier gap reports remain in reviews; they are not a claim
                # that a subsequent model-reported resolution never happened.
                "model_reported_missing": self.reviews[-1]["missing"] if self.reviews else [],
                **({"control_failures": list(self.control_failures)} if self.pipeline.span_reading else {}),
                **({"round_budgets": self.round_budgets,
                    "plan_revision_calls": sum(c["purpose"] == "plan_revision" and c["completion_attempted"]
                                               for c in self.calls),
                    "plan_revision_events": sum(c["purpose"] == "plan_revision" for c in self.calls)}
                   if self.pipeline.cycle else {}),
                "budget_events": self.budget_events}

    def response(self):
        root = self.root
        evidence_ids = (root["evidence_ids"] if self.want_answer and root is not None else
                        list(dict.fromkeys(e for n in self.leaves for e in n["evidence_ids"])))
        sources = [self.catalog.units[e] for e in evidence_ids]
        scope = self.execution_scope()
        status = self.status
        if status == "completed":
            if self.want_answer and (root is None or root["status"] != "completed"):
                status = root["status"] if root is not None else "root_missing"
            elif not sources and self.retrieval_needed is not False:
                status = "no_evidence"
            elif (scope["task_unresolved"] or scope["unread_units"] or scope["failed_actions"] or scope["failed_nodes"]
                  or scope["model_reported_missing"] or self.control_failures
                  or (root and root.get("parsed", {}).get("missing"))
                  or self.stop_reason not in {"model_stop", "provided_materials_processed", "retrieval_not_required"}):
                status = "partial"
        retrieval = {"protocol": self.pipeline.protocol, "requested_mode": self.mode, "status": status,
            "task": self.task.model_dump() if self.task is not None else None,
            **({"task_execution": "not_required_closed_materials"}
               if self.materials is not None and self.pipeline.span_reading else {}),
            **({"task_execution": "not_required_retrieval_gate_false"}
               if self.retrieval_needed is False else {}),
            "routing": self.routing,
            "actions": self.actions, "plans": self.plans, "reviews": self.reviews,
            **({"directions": self.directions, "plan_attempts": self.plan_attempts}
               if self.pipeline.cycle else {}),
            "evidence_catalog": self.catalog.trace(), "execution_scope": scope,
            "candidate_count": len(self.catalog.units), "selected_count": len(sources),
            "semantic_support_verified": False}
        budget = {"max_calls": self.max_calls, "admitted_calls": self.admitted,
                  "stage_concurrency": self.policy.stage_concurrency,
                  "timeout_seconds": self.timeout, "elapsed_ms": round((monotonic() - self.started) * 1000),
                  "provider_execution_cancelled": None}
        if not self.want_answer:
            count = min(self.request.top_k or self.pipeline.settings.default_top_k,
                        self.pipeline.settings.max_top_k)
            return SearchResponse(results=sources[:count], retrieval={**retrieval,
                "returned": len(sources[:count]), "model_calls": self.calls,
                "nodes": self.nodes, "request_budget": budget,
                **({"selected_span_catalog": self.span_bindings} if self.pipeline.span_reading else {})})
        raw = root.get("raw_text") if root is not None else None
        return AskResponse(answer=raw if isinstance(raw, str) else "", sources=sources,
            retrieval=retrieval, generation={"pipeline": "rwkv", "protocol": self.pipeline.protocol,
                "status": status, "output_mode": "immutable", "answer_modified": False,
                "raw_model_answer": raw, "root_node_id": root["id"] if root else None,
                "answer_call_id": root.get("call_id") if root else None,
                "answer_representation": "model_raw_text" if self.direction == "clarify" else "model_raw_json",
                "model_calls": self.calls, "nodes": self.nodes, "request_budget": budget,
                "citation_map": {e[1:]: self.catalog.units[e].id for e in evidence_ids},
                "citation_audit": root.get("citation_audit") if root else None,
                **({"execution_scope": scope,
                    "span_bindings": {s: self.span_bindings[s] for s in root.get("span_ids", [])}
                                     if root is not None else {},
                    "root_contract": "SpanAnswer" if self.direction != "clarify" else None}
                   if self.pipeline.span_reading else {}),
                "semantic_support_verified": False})
