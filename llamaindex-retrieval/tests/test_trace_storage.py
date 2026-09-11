import asyncio
from copy import deepcopy
import json
from unittest.mock import AsyncMock

from bson import BSON
from bson.codec_options import CodecOptions
import httpx
import pytest

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.repository import MongoRepository
from llamaindex_retrieval.rwkv_pipeline import RWKVPipeline
from llamaindex_retrieval.rwkvos_batch import RwkvosBatchClient
from llamaindex_retrieval.schemas import SourceItem


async def test_two_api_results_do_not_contain_other_callers_prompt_or_answer():
    receipts = []
    async def recorder(event, value):
        receipts.append((event, value))
    def handler(request):
        data = json.loads(request.content)
        return httpx.Response(200, json={"model": "model", "object": "chat.completion",
            "choices": [{"index": i, "message": {"role": "assistant", "content": f"answer-{i}"},
                         "finish_reason": "stop"} for i in range(len(data["contents"]))]})
    settings = Settings(_env_file=None, native_transport="rwkvos_batch",
                        native_writer_prefill="<think></think")
    async with RwkvosBatchClient(base_url="http://mock/v1", model="model", batch_wait_ms=10,
                                recorder=recorder, transport=httpx.MockTransport(handler)) as model:
        pipeline = RWKVPipeline(settings, None, model)
        materials = [SourceItem(id=str(i), document_id=str(i), title=str(i), source="test",
                                snippet=f"private-source-{i}", score=1) for i in range(2)]
        responses = await asyncio.gather(*(pipeline.ask_materials(f"question-{i}", [s])
                                          for i, s in enumerate(materials)))
    for i, response in enumerate(responses):
        serialized = response.model_dump_json()
        assert f"private-source-{i}" in serialized
        assert f"private-source-{1-i}" not in serialized
        assert f"answer-{1-i}" not in serialized
    assert len(receipts) == 2
    complete = receipts[-1][1]
    assert len(complete["payload"]["contents"]) == 2
    assert complete["response_body_complete"]


class MemoryBucket:
    def __init__(self):
        self.files = {}

    async def upload_from_stream_with_id(self, identity, name, data):
        self.files[identity] = bytes(data)

    async def open_download_stream(self, identity):
        return AsyncMock(read=AsyncMock(return_value=self.files[identity]))

    async def delete(self, identity):
        del self.files[identity]


@pytest.mark.parametrize("large_request", [False, True])
async def test_large_run_is_stored_under_bson_limit_and_restores_exactly(large_request):
    repo = MongoRepository.__new__(MongoRepository)
    repo.search_tests, repo.search_test_runs = AsyncMock(), AsyncMock()
    repo.payloads = MemoryBucket()
    repo.search_tests.find_one_and_update.return_value = {"id": "test", "run_count": 1}
    response = {"answer": "原始回答😀", "sources": [], "retrieval": {},
                "generation": {"pipeline": "rwkv", "status": "completed", "trace": "源" * 6_000_000}}
    request = {"question": "问题", "history": ["史" * (6_000_000 if large_request else 20)]}
    before = deepcopy(response)
    run = await repo.record_search_test_run(request, response)
    stored = repo.search_test_runs.insert_one.await_args.args[0]
    assert len(BSON.encode(stored)) < 16 * 1024 * 1024
    header = repo.search_tests.find_one_and_update.await_args.args[1]["$set"]
    assert len(BSON.encode(header)) < 16 * 1024 * 1024
    # MongoDB timestamps have millisecond precision for both inline and GridFS records.
    assert await repo._restore_run(stored) == BSON(BSON.encode(run)).decode(
        codec_options=CodecOptions(tz_aware=True))
    assert response == before
    if large_request:
        assert await repo._load_payload(header["request_payload_ref"]) == request


async def test_failed_record_insert_removes_its_large_payload():
    repo = MongoRepository.__new__(MongoRepository)
    repo.search_tests, repo.search_test_runs = AsyncMock(), AsyncMock()
    repo.payloads = MemoryBucket()
    repo.search_tests.find_one_and_update.return_value = {"id": "test", "run_count": 1}
    repo.search_test_runs.insert_one.side_effect = RuntimeError("database failure")
    with pytest.raises(RuntimeError):
        await repo.record_search_test_run({"question": "q"}, {"answer": "a" * 17_000_000})
    assert not repo.payloads.files


async def test_private_http_receipt_is_written_once_under_batch_event_identity():
    repo = MongoRepository.__new__(MongoRepository)
    repo.model_receipts = MemoryBucket()
    record = {"batch_id": "b", "request_body_base64": "cHJpdmF0ZQ=="}
    await repo.record_model_http("batch_completed", record)
    assert BSON(repo.model_receipts.files["b:batch_completed"]).decode() == record


async def test_unknown_test_id_does_not_leave_large_request_blob():
    repo = MongoRepository.__new__(MongoRepository)
    repo.search_tests = AsyncMock()
    repo.search_tests.find_one_and_update.return_value = None
    repo.payloads = MemoryBucket()
    assert await repo.record_search_test_run(
        {"question": "q", "history": ["a" * 17_000_000]}, {}, test_id="missing") is None
    assert not repo.payloads.files
