"""Explicit native State routing and metadata checks, not tensor attestation.

Refs are registered by an operator, not loaded/deleted by the answer pipeline.
An explicit null chooses no ref; it does not prove the server's default is zero.
"""
import json
import math
from types import MappingProxyType
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

StateRole = Literal['plan', 'reader', 'assessment', 'followup', 'review', 'writer', 'current_question']
StateRef = Annotated[str, StringConstraints(strict=True, pattern=r'^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$')]
PROTOCOL = 'vllm-rwkv.state-cache.v1'


class NativeStateRouting(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    model: str = Field(strict=True, min_length=1)
    roles: dict[StateRole, StateRef | None]

    @model_validator(mode='after')
    def explicit_base_roles(self):
        if self.model != self.model.strip() or not {'plan', 'reader', 'writer'} <= self.roles.keys():
            raise ValueError('State routing requires a model and explicit plan/reader/writer bindings (null for no ref)')
        return self


class NativeStateRouter:
    """Immutable private copy: concurrent requests never change a default ref."""
    def __init__(self, routing, *, model, prompt_protocol):
        # Pydantic's frozen models still contain mutable dicts; revalidate values
        # before taking our own immutable copy, even for an existing model instance.
        binding = NativeStateRouting.model_validate(
            routing.model_dump() if isinstance(routing, NativeStateRouting) else routing)
        if binding.model != model or prompt_protocol != 'g1j_plain':
            raise ValueError('State routing requires matching model and g1j_plain protocol')
        self.model = model
        self.roles = MappingProxyType(dict(binding.roles))

    def select(self, stage, role, *, check_only=False):
        if stage == 'writer_budget' and check_only:
            stage = 'writer'
        defaults = {'planner': 'plan', 'reader': 'reader', 'resolver': 'reader', 'writer': 'writer'}
        allowed = {'planner': {'plan', 'assessment', 'followup', 'review', 'current_question'},
                   'reader': {'reader', 'review'}, 'resolver': {'reader', 'review'}, 'writer': {'writer'}}
        if not isinstance(stage, str) or stage not in defaults:
            raise ValueError('No State role mapping for this stage')
        role = defaults[stage] if role is None else role
        if not isinstance(role, str) or role not in allowed[stage]:
            raise ValueError('State role is incompatible with the requested stage')
        if role not in self.roles:
            raise ValueError('State role has no explicit binding; no fallback allowed')
        ref = self.roles[role]
        return {'role': role, 'mode': 'base_no_ref' if ref is None else 'registered_ref',
                'state_ref': ref, 'model': self.model, 'metadata_checked': False,
                'state_consumption_verified': False, 'tensor_compatibility_verified': False}


def decode_metadata(body):
    """Fail on ambiguous/nonfinite metadata; never repair an engine response."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate State metadata key')
            result[key] = value
        return result

    def number(value):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError('Nonfinite State metadata value')
        return value

    return json.loads(body.decode('utf-8'), object_pairs_hook=pairs,
                      parse_constant=number, parse_float=number)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def check_model(data, model, context_tokens):
    cards = data.get('data')
    _require(isinstance(cards, list) and len(cards) == 1 and isinstance(cards[0], dict),
             'State routing requires exactly one advertised model')
    card = cards[0]
    _require(card.get('id') == model and type(card.get('max_model_len')) is int
             and card['max_model_len'] >= context_tokens, 'State model/context metadata mismatch')


def _worker(data, action):
    _require(data.get('protocol') == PROTOCOL and data.get('action') == action,
             'Unsupported State metadata protocol/action')
    workers = data.get('workers')
    _require(isinstance(workers, list) and len(workers) == 1 and isinstance(workers[0], dict),
             'State routing currently requires one worker')
    worker = workers[0]
    _require(worker.get('protocol') == PROTOCOL and worker.get('process_local') is True
             and worker.get('durable') is False, 'Unsupported State worker/lifetime metadata')
    return worker


def check_capabilities(data):
    worker = _worker(data, 'capabilities')
    _require(all(worker.get(k) is True for k in ('supported', 'enabled', 'import_supported')),
             'State import/read capability is not enabled')
    size = worker.get('snapshot_nbytes')
    _require(type(size) is int and size > 0, 'Missing State snapshot size')
    return size


def check_initial_ref(data, ref, snapshot_nbytes):
    worker = _worker(data, 'inspect')
    _require(worker.get('state_ref') == ref, 'Inspected State ref differs from requested ref')
    _require(type(worker.get('nbytes')) is int and worker['nbytes'] == snapshot_nbytes,
             'State snapshot size differs from serving configuration')
    _require(all(type(worker.get(k)) is int and worker[k] == 0 for k in
                 ('processed_token_count', 'pending_tail_token_count', 'scheduler_slots_reserved'))
             and worker.get('finish_reason') == 'initial',
             'State ref is not an initial snapshot; no conversation-state reuse')
    # state_dtype describes shifts in this API, not the recurrent matrix dtype.
    # No model checkpoint/source-State hash or atomic consumption receipt exists here.
