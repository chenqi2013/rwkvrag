"""Read-only historical transport audit. Never dispatches requests or edits answers.

The returned payload is historical evidence, not an executable request: State
references may have expired. Future runs must freeze fresh model/State bindings.
"""
import base64
import copy
import hashlib
import json


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs)


def extract_call(trace):
    """Verify saved bytes against projections before extracting one exact input."""
    for field in ("prompt", "raw_text"):
        if digest(trace[field]) != trace[field + "_sha256"]:
            raise ValueError(field + " hash mismatch")
    calls = [h for h in trace["http"] if h["stage"] == "chat_completion"]
    if len(calls) != 1:
        raise ValueError("expected one completion; cannot silently select a retry")
    call = calls[0]
    wire = base64.b64decode(call["request_body_base64"], validate=True)
    payload = strict_json(wire)
    if payload != call["payload"]:
        raise ValueError("wire request differs from recorded payload")
    if payload["messages"] != trace["messages"]:
        raise ValueError("message projection mismatch")
    for key, value in trace["parameters"].items():
        if payload.get(key) != value:
            raise ValueError("parameter projection mismatch: " + key)
    tokens = trace["prompt_token_ids"]
    if not tokens or any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError("invalid input tokens")
    if tokens != trace["chat_response"]["prompt_token_ids"]:
        raise ValueError("server input tokens differ from saved tokens")
    if len(tokens) != trace["usage"]["prompt_tokens"]:
        raise ValueError("input token count mismatch")
    return {
        "call_id": trace["call_id"], "stage": trace["stage"],
        "payload": copy.deepcopy(payload), "wire_sha256": hashlib.sha256(wire).hexdigest(),
        "prompt_sha256": trace["prompt_sha256"], "input_tokens": len(tokens),
        "token_sha256": digest(json.dumps(tokens, separators=(",", ":"))),
        "historical_state_binding": copy.deepcopy(trace["state_binding"]),
        "raw_text_sha256": trace["raw_text_sha256"],
    }


def audit_assessment_ids(trace):
    """Only audit parseability and membership, never convert failure to unknown."""
    if trace["stage"] != "assess":
        return []
    envelope = trace.get("envelope", {})
    span = envelope.get("answer_span")
    if not envelope.get("valid") or not span:
        return ["invalid_envelope"]
    start, end = span["start"], span["end"]
    if not 0 <= start < end <= len(trace["raw_text"]):
        return ["invalid_answer_span"]
    try:
        answer = strict_json(trace["raw_text"][start:end])
    except (ValueError, TypeError):
        return ["invalid_json"]
    if not isinstance(answer, dict) or not isinstance(answer.get("evidence_ids"), list):
        return ["invalid_evidence_ids"]
    ids = answer["evidence_ids"]
    if any(not isinstance(i, str) for i in ids):
        return ["invalid_evidence_ids"]
    issues = []
    if len(ids) != len(set(ids)):
        issues.append("duplicate_evidence_id")
    if not set(ids) <= set(trace["evidence_ids"]):
        issues.append("unknown_evidence_id")
    return issues
