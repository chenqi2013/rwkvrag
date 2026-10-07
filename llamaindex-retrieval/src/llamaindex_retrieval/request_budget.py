"""Request-local call admission and wall-clock deadlines, not semantic decisions.

A shallow pipeline copy shares the original transport/semaphore, never counters.
Count logical client operations (including tokenize-only probes and routing), not
HTTP requests. Refused operations never reach a model client. The deadline also
covers retrieval and queueing; cancellation does not attest provider termination.
"""
import asyncio
from copy import copy, deepcopy
from functools import wraps
from time import monotonic
from uuid import uuid4

from .native_rwkv import NativeRWKVResult


class RequestBudget:
    def __init__(self, settings):
        self.max_calls = settings.native_request_max_calls
        self.timeout_seconds = settings.native_request_timeout_seconds
        self.started = monotonic()
        self.admitted = 0
        self.rejected = 0
        self.events = []
        self.retrieval = {}
        self.writer_sources = []
        self.writer_inputs = {}
        self.stop_status = None

    def admit(self, messages, stage, evidence_ids=(), trace=None):
        event = trace if trace is not None else {}
        event.update(call_id=str(uuid4()), stage=stage, status="pending",
                     messages=deepcopy(messages), evidence_ids=list(evidence_ids),
                     raw_text=None, completion_attempted=False)
        self.events.append(event)
        if self.max_calls is not None and self.admitted >= self.max_calls:
            self.rejected += 1
            self.stop_status = "call_budget_exceeded"
            event.update(status="call_budget_exceeded", request_call_admitted=False,
                         request_call_limit=self.max_calls)
            return event, False
        # There is no await between checking and reserving a slot.
        self.admitted += 1
        event.update(request_call_admitted=True, request_call_number=self.admitted)
        return event, True

    async def route(self, web, messages):
        event, admitted = self.admit(messages, "routing")
        if not admitted:
            return event
        try:
            result = await web.decide(messages)
            event.update(result)
            return event
        except asyncio.CancelledError:
            event.update(status="cancelled", cancellation_requested=True,
                         provider_execution_cancelled=None)
            raise
        except Exception as error:
            event.update(status="failed", error=type(error).__name__)
            raise

    def summary(self):
        return {"scope": "logical_model_operations_including_routing_and_token_probes",
                "max_calls": self.max_calls, "admitted_calls": self.admitted,
                "rejected_calls": self.rejected, "timeout_seconds": self.timeout_seconds,
                "elapsed_ms": round((monotonic() - self.started) * 1000),
                "stop_status": self.stop_status,
                "provider_execution_cancelled": None}


class BudgetedModel:
    def __init__(self, model, budget):
        self.model = model
        self.budget = budget

    def __getattr__(self, name):
        return getattr(self.model, name)

    async def complete(self, messages, **kwargs):
        event, admitted = self.budget.admit(messages, kwargs.get("stage", "reader"),
            kwargs.get("evidence_ids", ()), kwargs.get("trace"))
        position = len(self.budget.events) - 1
        if kwargs.get("stage") == "writer":
            self.budget.writer_inputs[position] = list(self.budget.writer_sources)
        if not admitted:
            return NativeRWKVResult("call_budget_exceeded", None, None, event)
        kwargs["trace"] = event
        try:
            result = await self.model.complete(messages, **kwargs)
            # Fake or custom clients may return a different dict. Retain later
            # parser annotations by binding the ledger to the returned trace.
            event.update(result.trace)
            result.trace.update(event)
            self.budget.events[position] = result.trace
            return result
        except asyncio.CancelledError:
            event.update(status="cancelled", cancellation_requested=True,
                         provider_execution_cancelled=None)
            raise
        except Exception as error:
            event.update(status="transport_error", error_type=type(error).__name__)
            raise


def bounded_request(method):
    @wraps(method)
    async def run(pipeline, *args, **kwargs):
        settings = pipeline.settings
        if (settings.native_request_max_calls is None
                and settings.native_request_timeout_seconds is None):
            return await method(pipeline, *args, **kwargs)
        budget = RequestBudget(settings)
        scoped = copy(pipeline)
        scoped._request_budget = budget
        scoped.model = BudgetedModel(pipeline.model, budget)
        deadline = asyncio.timeout(budget.timeout_seconds)
        try:
            async with deadline:
                response = await method(scoped, *args, **kwargs)
        except TimeoutError:
            if not deadline.expired():
                raise
            budget.stop_status = "request_timeout"
            # Preserve any raw Writer output already observed, even when a later
            # review or stage times out. No refusal, citation or answer is invented.
            writers = [(i, event) for i, event in enumerate(budget.events)
                       if event.get("stage") == "writer"]
            position, writer = next(((i, event) for i, event in reversed(writers)
                                     if isinstance(event.get("raw_text"), str)),
                                    writers[-1] if writers else (-1, {}))
            response = scoped._response(writer.get("raw_text"), budget.writer_inputs.get(position, []),
                budget.retrieval, budget.events, "request_timeout", budget.started,
                answer_trace=writer)
            response.generation["answer_call_id"] = writer.get("call_id")
        if budget.stop_status is not None:
            response.generation["status"] = budget.stop_status
        response.generation["request_budget"] = budget.summary()
        return response
    return run
