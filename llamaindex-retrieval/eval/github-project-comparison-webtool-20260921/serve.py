"""Isolated, explicitly operator-dispatched live web-search test provider."""
import asyncio
import json
import os
from pathlib import Path
import time
from uuid import uuid4
from llamaindex_retrieval.web_retrieval import SearchReaderAdapter

QUEUE = Path(os.environ["EXPERIMENT_SEARCH_QUEUE"])
QUEUE.mkdir(parents=True, exist_ok=True)
original_execute = SearchReaderAdapter._execute

async def execute(self, request, timeout):
    if request.get("action") != "search":
        return await original_execute(self, request, timeout)
    async with self.slots:
        key = uuid4().hex
        request_path = QUEUE / (key + ".request.json")
        request_path.write_text(json.dumps(request, ensure_ascii=False))
        response_path = QUEUE / (key + ".response.json")
        deadline = time.monotonic() + 180
        while not response_path.exists():
            if time.monotonic() > deadline:
                (QUEUE / (key + ".timeout")).touch()
                raise TimeoutError("test_search_dispatch_timeout")
            await asyncio.sleep(.3)
        raw = response_path.read_bytes()
        if len(raw) > 8 * 1024 * 1024:
            raise RuntimeError("web_response_too_large")
        return json.loads(raw)

SearchReaderAdapter._execute = execute
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("llamaindex_retrieval.api:app", host="127.0.0.1", port=18449)
