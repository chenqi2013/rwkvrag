"""Single-process experiment guard; production multi-instance fix is separate."""
import asyncio
import json
import os
import time
from pathlib import Path
from datetime import datetime, timezone
from llamaindex_retrieval.web_retrieval import SearchReaderAdapter

lock = asyncio.Lock()
last_end = 0.0
failed = False
original = SearchReaderAdapter._execute
log_path = Path(os.environ['EXPERIMENT_SEARCH_LOG'])

async def execute(self, request, timeout):
    global last_end, failed
    if request.get('action') != 'search':
        return await original(self, request, timeout)
    async with lock:
        if failed:
            raise RuntimeError('experiment_search_circuit_open')
        await asyncio.sleep(max(0, 6-(time.monotonic()-last_end)))
        event = {'query':request['query'], 'started_at':datetime.now(timezone.utc).isoformat()}
        try:
            result = await original(self, request, timeout)
            event.update(status='completed', returned=len(result.get('hits',[])))
            return result
        except Exception as exc:
            failed = True
            event.update(status='failed', error_type=type(exc).__name__, circuit_open=True)
            raise
        finally:
            last_end = time.monotonic()
            event['ended_at'] = datetime.now(timezone.utc).isoformat()
            with log_path.open('a') as f:
                f.write(json.dumps(event,ensure_ascii=False)+'\n')

SearchReaderAdapter._execute = execute
if __name__ == '__main__':
    assert len(json.loads(os.environ['TAVILY_API_KEYS'])) == 1
    assert os.environ['TAVILY_API_KEY'] == json.loads(os.environ['TAVILY_API_KEYS'])[0]
    import uvicorn
    uvicorn.run('llamaindex_retrieval.api:app',host='127.0.0.1',port=18451)
