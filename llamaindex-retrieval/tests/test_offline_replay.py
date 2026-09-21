import base64
import copy
import json

import pytest

from llamaindex_retrieval.offline_replay import audit_assessment_ids, digest, extract_call


def trace():
    payload = {"messages": [{"role": "user", "content": "原文"}], "top_k": 1}
    raw = '{"evidence_ids":["E2"]}'
    return {
        "call_id": "test", "stage": "assess", "prompt": "原文", "prompt_sha256": digest("原文"),
        "raw_text": raw, "raw_text_sha256": digest(raw), "messages": payload["messages"],
        "parameters": {"top_k": 1}, "prompt_token_ids": [1, 2],
        "chat_response": {"prompt_token_ids": [1, 2]}, "usage": {"prompt_tokens": 2},
        "state_binding": {"read_ref": "expired", "file_sha256": "historical"},
        "http": [{"stage": "chat_completion", "payload": payload,
                  "request_body_base64": base64.b64encode(json.dumps(payload).encode()).decode()}],
        "evidence_ids": ["E2"],
        "envelope": {"valid": True, "answer_span": {"start": 0, "end": len(raw)}},
    }


def test_exact_payload_and_no_mutation():
    t = trace()
    original = copy.deepcopy(t)
    result = extract_call(t)
    assert result["payload"] == t["http"][0]["payload"]
    result["payload"]["top_k"] = 7
    assert t == original
    assert result["historical_state_binding"]["read_ref"] == "expired"


@pytest.mark.parametrize("mutation", [
    lambda t: t.update(prompt="changed"),
    lambda t: t.update(raw_text="changed"),
    lambda t: t["http"][0]["payload"].update(top_k=2),
    lambda t: t["parameters"].update(top_k=2),
    lambda t: t["chat_response"].update(prompt_token_ids=[2, 1]),
    lambda t: t["usage"].update(prompt_tokens=3),
    lambda t: t["http"].append(copy.deepcopy(t["http"][0])),
    lambda t: t["http"][0].update(request_body_base64=base64.b64encode(b'{"a":1,"a":2}').decode()),
])
def test_corruption_is_rejected(mutation):
    t = trace()
    mutation(t)
    with pytest.raises(ValueError):
        extract_call(t)


def test_empty_evidence_does_not_become_unknown_or_get_repaired():
    t = trace()
    t["evidence_ids"] = []
    original = copy.deepcopy(t)
    assert audit_assessment_ids(t) == ["unknown_evidence_id"]
    assert t == original


def test_invalid_envelope_is_execution_issue():
    t = trace()
    t["envelope"]["valid"] = False
    assert audit_assessment_ids(t) == ["invalid_envelope"]
