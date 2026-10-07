"""Immutable application task boundary, NOT a semantic correctness certificate.

Lookup queries are model-authored retrieval hints, not replacement user tasks.
This envelope is request-local data, never an RWKV recurrent State or cache.
"""
from dataclasses import dataclass
from hashlib import sha256
import json

PROTOCOL = "anchored_v1"
READER_PROTOCOL = "binary-task-v1"


def digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TaskAnchor:
    question: str
    task: str
    original_task_sha256: str
    origin: str
    rewrite_call_id: str | None

    @classmethod
    def from_resolution(cls, question, task, original_task, events):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Task anchor requires a nonempty question without repairing it")
        expected = json.dumps({"history": [], "latest_question": question}, ensure_ascii=False)
        if task != expected:
            raise ValueError("Task anchor requires the exact independent question envelope")
        if events:
            event = events[-1]
            if (event.get("purpose") != "current_question" or event.get("status") != "completed"
                    or event.get("parsed_question") != question
                    or not isinstance(event.get("call_id"), str) or not event["call_id"].strip()):
                raise ValueError("Task anchor requires a completed, parsed current-question trace")
            origin, call_id = "model_authored", event.get("call_id")
        else:
            if original_task != task:
                raise ValueError("History cannot be silently dropped when creating a task anchor")
            origin, call_id = "user_no_history", None
        return cls(question, task, digest(original_task), origin, call_id)

    def trace(self):
        return {"protocol": PROTOCOL, "origin": self.origin,
                "question_sha256": digest(self.question), "task_sha256": digest(self.task),
                "original_task_sha256": self.original_task_sha256,
                "rewrite_call_id": self.rewrite_call_id, "semantic_verified": False,
                "lookup_queries_are_answer_requirements": False}
