"""Structural contracts for model-owned choice judgments; no numeric solver.

This module is an isolated candidate. It never edits answers, infers a status,
selects a winner, or translates a transport failure into an unknown fact.
"""
from hashlib import sha256
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Text = Annotated[str, Field(min_length=1, max_length=2000, pattern=r"\S")]
Identity = Annotated[str, Field(min_length=1, max_length=128, pattern=r"\S")]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Snapshot(Contract):
    id: Identity
    kind: Literal["user", "source"]
    version: Identity
    text: str = Field(min_length=1, max_length=100000)
    sha256: Digest


class Span(Contract):
    snapshot_id: Identity
    snapshot_sha256: Digest
    start: int = Field(ge=0)
    end: int = Field(gt=0)


class Requirement(Contract):
    id: Identity
    kind: Literal["hard", "preference"]
    text: Text
    active: bool
    user_spans: tuple[Span, ...] = Field(min_length=1, max_length=8)
    supersedes: tuple[Identity, ...] = Field(max_length=8)


class Candidate(Contract):
    id: Identity
    name_and_scope: Text
    origin_spans: tuple[Span, ...] = Field(min_length=1, max_length=8)


class Observation(Contract):
    id: Identity
    candidate_id: Identity
    source_spans: tuple[Span, ...] = Field(min_length=1, max_length=8)


class Judgment(Contract):
    id: Identity
    candidate_id: Identity
    requirement_id: Identity
    status: Literal["satisfied", "not_satisfied", "unknown", "conflict"]
    explanation: Text
    observation_ids: tuple[Identity, ...] = Field(max_length=16)


class CandidateSummary(Contract):
    candidate_id: Identity
    status: Literal["eligible", "ineligible", "unresolved"]
    judgment_ids: tuple[Identity, ...] = Field(min_length=1, max_length=16)
    explanation: Text


class ChoiceRecord(Contract):
    protocol: Literal["choice-contract-v1"]
    snapshots: tuple[Snapshot, ...] = Field(min_length=1, max_length=64)
    requirements: tuple[Requirement, ...] = Field(min_length=1, max_length=24)
    candidates: tuple[Candidate, ...] = Field(min_length=1, max_length=8)
    observations: tuple[Observation, ...] = Field(max_length=64)
    judgments: tuple[Judgment, ...] = Field(max_length=64)
    summaries: tuple[CandidateSummary, ...] = Field(max_length=8)
    recommended_ids: tuple[Identity, ...] = Field(max_length=8)
    preference_ids: tuple[Identity, ...] = Field(max_length=8)
    raw_answer: str = Field(max_length=32000)


def unique(values, label):
    if len(values) != len(set(values)):
        raise ValueError("duplicate " + label)


def indexed(items, label):
    unique([item.id for item in items], label)
    return {item.id: item for item in items}


def validate_dependencies(record: ChoiceRecord):
    """Reject missing/foreign provenance and incomplete grids, never infer truth."""
    snapshots = indexed(record.snapshots, "snapshot ID")
    requirements = indexed(record.requirements, "requirement ID")
    candidates = indexed(record.candidates, "candidate ID")
    observations = indexed(record.observations, "observation ID")
    judgments = indexed(record.judgments, "judgment ID")
    for snapshot in snapshots.values():
        if sha256(snapshot.text.encode()).hexdigest() != snapshot.sha256:
            raise ValueError("snapshot content hash mismatch")

    def spans(refs, kinds):
        unique([(s.snapshot_id, s.start, s.end) for s in refs], "span")
        for ref in refs:
            source = snapshots.get(ref.snapshot_id)
            if (source is None or source.kind not in kinds
                    or source.sha256 != ref.snapshot_sha256
                    or not 0 <= ref.start < ref.end <= len(source.text)):
                raise ValueError("invalid span identity, kind, hash or character range")

    for requirement in requirements.values():
        spans(requirement.user_spans, {"user"})
        unique(requirement.supersedes, "superseded requirement")
        for old_id in requirement.supersedes:
            old = requirements.get(old_id)
            if old is None or old_id == requirement.id or old.active:
                raise ValueError("invalid superseded requirement")
    # Reject cycles even when all involved requirements have been withdrawn.
    visited = set()
    def visit(identity, trail):
        if identity in trail:
            raise ValueError("cyclic requirement revision")
        if identity in visited:
            return
        for old in requirements[identity].supersedes:
            visit(old, trail | {identity})
        visited.add(identity)
    for identity in requirements:
        visit(identity, set())
    for candidate in candidates.values():
        spans(candidate.origin_spans, {"user", "source"})
    for observation in observations.values():
        if observation.candidate_id not in candidates:
            raise ValueError("observation references unknown candidate")
        spans(observation.source_spans, {"source"})

    hard = {r.id for r in requirements.values() if r.active and r.kind == "hard"}
    if not hard:
        raise ValueError("choice contract requires at least one active hard condition")
    expected = {(c, r) for c in candidates for r in hard}
    actual = [(j.candidate_id, j.requirement_id) for j in judgments.values()]
    unique(actual, "candidate/condition judgment")
    if set(actual) != expected:
        raise ValueError("incomplete or expanded candidate/condition grid")
    for judgment in judgments.values():
        unique(judgment.observation_ids, "judgment observation")
        if judgment.status != "unknown" and not judgment.observation_ids:
            raise ValueError("decisive judgment requires evidence dependencies")
        for identity in judgment.observation_ids:
            observation = observations.get(identity)
            if observation is None or observation.candidate_id != judgment.candidate_id:
                raise ValueError("judgment references another candidate or absent observation")

    unique([s.candidate_id for s in record.summaries], "candidate summary")
    if {s.candidate_id for s in record.summaries} != set(candidates):
        raise ValueError("missing or foreign candidate summary")
    for summary in record.summaries:
        unique(summary.judgment_ids, "summary judgment")
        expected_ids = {j.id for j in judgments.values() if j.candidate_id == summary.candidate_id}
        if set(summary.judgment_ids) != expected_ids:
            raise ValueError("summary omitted or mixed candidate conditions")
    unique(record.recommended_ids, "recommended candidate")
    if not set(record.recommended_ids) <= candidates.keys():
        raise ValueError("recommended candidate absent")
    unique(record.preference_ids, "preference")
    allowed = {r.id for r in requirements.values() if r.active and r.kind == "preference"}
    if not set(record.preference_ids) <= allowed:
        raise ValueError("preference absent, withdrawn or hard condition")
    return record


def audit_consistency(record: ChoiceRecord):
    """Report incompatible model labels without correcting them or the answer.

    This checks label relationships only. It cannot establish whether a source
    supports a judgment or whether the prose actually matches these labels.
    """
    validate_dependencies(record)
    judgments = {j.id: j for j in record.judgments}
    issues = []
    for summary in record.summaries:
        statuses = {judgments[i].status for i in summary.judgment_ids}
        if summary.status == "eligible" and statuses != {"satisfied"}:
            issues.append({"code": "eligible_without_all_satisfied", "candidate_id": summary.candidate_id})
        if summary.status == "ineligible" and "not_satisfied" not in statuses:
            issues.append({"code": "ineligible_without_failed_condition", "candidate_id": summary.candidate_id})
        if summary.status == "unresolved" and statuses == {"satisfied"}:
            issues.append({"code": "unresolved_despite_all_satisfied", "candidate_id": summary.candidate_id})
        if summary.status == "unresolved" and "not_satisfied" in statuses:
            issues.append({"code": "unresolved_despite_failed_condition", "candidate_id": summary.candidate_id})
        if summary.candidate_id in record.recommended_ids and summary.status != "eligible":
            issues.append({"code": "recommendation_outside_eligible_candidates", "candidate_id": summary.candidate_id})
    return issues


def parse_record(raw: str):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    # First pass detects duplicate keys; validate_json permits JSON arrays for
    # immutable tuples without relaxing strict bool/int/string field types.
    json.loads(raw, object_pairs_hook=pairs)
    return validate_dependencies(ChoiceRecord.model_validate_json(raw))
