"""Isolated provenance contract; model owns field/scope/value/relation semantics."""
from hashlib import sha256
from typing import Literal

from pydantic import Field

from .choice_contract import Contract, Identity, Snapshot, Span, Text, indexed, unique
from .offline_replay import strict_json


class Fact(Contract):
    id: Identity
    object_id: Identity
    field: Text
    scope: Text
    claim: Text
    assessment: Literal["supported", "unsupported", "uncertain"]
    source_spans: tuple[Span, ...] = Field(min_length=1, max_length=8)


class Relation(Contract):
    id: Identity
    left_object_id: Identity
    right_object_id: Identity
    field: Text
    scope: Text
    left_fact_ids: tuple[Identity, ...] = Field(max_length=16)
    right_fact_ids: tuple[Identity, ...] = Field(max_length=16)
    status: Literal["less", "equal", "greater", "incomparable", "unknown", "conflict"]
    explanation: Text


class ComparisonRecord(Contract):
    protocol: Literal["comparison-contract-v1"]
    snapshots: tuple[Snapshot, ...] = Field(max_length=64)
    object_ids: tuple[Identity, ...] = Field(min_length=2, max_length=8)
    facts: tuple[Fact, ...] = Field(max_length=64)
    relations: tuple[Relation, ...] = Field(min_length=1, max_length=64)
    resolver_selected_ids: tuple[Identity, ...] = Field(max_length=64)
    writer_source_ids: tuple[Identity, ...] = Field(max_length=64)
    writer_relation_ids: tuple[Identity, ...] = Field(max_length=64)
    raw_answer: str = Field(max_length=32000)


def validate_comparison(record: ComparisonRecord):
    """Validate exact IDs/spans and Writer closure; never fix relation or prose."""
    snapshots = indexed(record.snapshots, "snapshot")
    facts = indexed(record.facts, "fact")
    relations = indexed(record.relations, "relation")
    for label in ("object_ids", "resolver_selected_ids", "writer_source_ids", "writer_relation_ids"):
        unique(getattr(record, label), label)
    for snapshot in snapshots.values():
        if sha256(snapshot.text.encode()).hexdigest() != snapshot.sha256:
            raise ValueError("snapshot hash mismatch")
    selected = set(record.resolver_selected_ids)
    if not selected <= {s.id for s in snapshots.values() if s.kind == "source"}:
        raise ValueError("unknown or non-source Resolver selection")
    for fact in facts.values():
        if fact.object_id not in record.object_ids:
            raise ValueError("unknown fact object")
        unique([(s.snapshot_id, s.start, s.end) for s in fact.source_spans], "fact span")
        for span in fact.source_spans:
            source = snapshots.get(span.snapshot_id)
            if (source is None or source.kind != "source"
                    or source.sha256 != span.snapshot_sha256
                    or not 0 <= span.start < span.end <= len(source.text)):
                raise ValueError("invalid fact source span")
    for relation in relations.values():
        if (relation.left_object_id == relation.right_object_id
                or relation.left_object_id not in record.object_ids
                or relation.right_object_id not in record.object_ids):
            raise ValueError("invalid relation objects")
        for side in ("left", "right"):
            refs = getattr(relation, side + "_fact_ids")
            unique(refs, "relation side fact")
            # Unknown is the only relation permitted to lack an entire side.
            if relation.status != "unknown" and not refs:
                raise ValueError("relation missing side")
            for identity in refs:
                fact = facts.get(identity)
                if (fact is None or fact.object_id != getattr(relation, side + "_object_id")
                        or fact.field != relation.field or fact.scope != relation.scope):
                    raise ValueError("foreign object, field or scope dependency")
                if relation.status != "unknown" and fact.assessment != "supported":
                    raise ValueError("decisive relation uses unconfirmed fact")
    closure = set()
    for identity in record.writer_relation_ids:
        if identity not in relations:
            raise ValueError("unknown Writer relation")
        relation = relations[identity]
        for fid in relation.left_fact_ids + relation.right_fact_ids:
            closure.update(s.snapshot_id for s in facts[fid].source_spans)
    if not closure <= selected:
        raise ValueError("Writer dependency was not selected by Resolver")
    if set(record.writer_source_ids) != closure:
        raise ValueError("Writer sources differ from dependency closure")
    return record


def parse_comparison(raw: str):
    strict_json(raw)
    return validate_comparison(ComparisonRecord.model_validate_json(raw))
