"""Isolated real-browser chat regression. No model, retrieval or production APIs.

Uses an existing Windows Edge executable through WSL; installs nothing.
  .venv/bin/python web/tests/chat.browser.py --dist /absolute/build/dir \
    --output /new/result/dir --profile /mnt/c/Users/USER/AppData/Local/Temp/unique-profile
Requires httpx and websockets in the selected environment. All API responses below
are synthetic UI fixtures, not quality assessments. The owned browser is closed.
"""
import argparse
import asyncio
import base64
import json
import subprocess
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import websockets

REQUESTS, ERRORS, CHECKS = [], [], []
HOLDS = {}
TEXT = "这是离线界面夹具，不代表模型质量。\n\n使用 MODE_A() 启用解析，使用 MODE_B() 读取结果。\n逐字保留：[Source 99]"
ORIGINAL = "# 示例资料\r\n\r\nMODE_A()\r\nMODE_B()\r\n"


def fixture(question):
    answer = TEXT if question.strip() == '如何启用解析并读取结果？' else '离线夹具答复：' + question
    raw = json.dumps({'answer': answer, 'span_ids': ['E1.S1'], 'missing': []}, ensure_ascii=False, indent=2)
    return {'answer': raw, 'sources': [{'id': 'source1', 'document_id': 'doc1', 'title': '离线界面夹具来源',
        'source': 'provided_material', 'uri': None, 'score': 1, 'snippet': ORIGINAL, 'metadata': {}}],
        'retrieval': {}, 'generation': {'pipeline': 'rwkv', 'protocol': 'rwkv-unified-v4',
        'status': 'partial' if question == '模拟未完成' else 'completed', 'raw_model_answer': raw,
        'root_node_id': 'N1', 'answer_call_id': 'call1', 'root_contract': 'SpanAnswer', 'answer_representation': 'model_raw_json',
        'model_calls': [{'call_id': 'call1', 'purpose': 'final', 'completion_attempted': True}],
        'nodes': [{'id': 'N1', 'call_id': 'call1', 'kind': 'final', 'raw_text': raw, 'span_ids': ['E1.S1'],
                   'status': 'invalid_contract' if question == '模拟非法根' else 'completed'}],
        'span_bindings': {'E1.S1': {'source_id': 'source1', 'selected_original_text': ORIGINAL,
            'model_generated_quote': False, 'snapshot_id': 'fixture-snapshot', 'span': {'start': 0, 'end': len(ORIGINAL), 'sha256': 'a' * 64}}}}}


class CDP:
    def __init__(self, ws):
        self.ws, self.count = ws, 0

    async def call(self, method, params=None):
        self.count += 1
        await self.ws.send(json.dumps({'id': self.count, 'method': method, 'params': params or {}}))
        while True:
            value = json.loads(await asyncio.wait_for(self.ws.recv(), 30))
            if value.get('method') in ('Runtime.exceptionThrown', 'Log.entryAdded'):
                ERRORS.append(value)
            if value.get('id') == self.count:
                assert 'error' not in value, value
                return value.get('result', {})

    async def js(self, expression):
        result = await self.call('Runtime.evaluate', {'expression': expression, 'returnByValue': True, 'awaitPromise': True})
        assert 'exceptionDetails' not in result, result
        return result['result'].get('value')

    async def wait(self, expression):
        for _ in range(200):
            if await self.js(expression):
                return
            await asyncio.sleep(.1)
        raise AssertionError('DOM wait failed: ' + expression)

    async def click(self, selector):
        await self.js(f"(()=>{{const e=[...document.querySelectorAll({json.dumps(selector)})].find(e=>e.getClientRects().length);if(!e)throw Error('missing visible target');e.click()}})()")

    async def input(self, text):
        await self.js(f"(()=>{{const e=document.querySelector('.chat-composer textarea');e.focus();Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value').set.call(e,{json.dumps(text)});e.dispatchEvent(new Event('input',{{bubbles:true}}))}})()")

    async def send(self, text):
        await self.input(text)
        await self.click('[aria-label="发送消息"]')

    async def idle(self):
        await self.wait("!document.querySelector('[aria-label=\"停止等待\"]')")

    async def shot(self, name):
        value = await self.call('Page.captureScreenshot', {'format': 'png', 'captureBeyondViewport': False})
        (ARGS.output / (name + '.png')).write_bytes(base64.b64decode(value['data']))

    async def records(self):
        return await self.js("new Promise((resolve,reject)=>{const r=indexedDB.open('rwkvrag-chat-v1',1);r.onsuccess=()=>{const d=r.result,t=d.transaction('conversations'),q=t.objectStore('conversations').getAll();t.oncomplete=()=>{d.close();resolve(q.result)};t.onerror=()=>reject(t.error)}})")

    async def reload(self):
        old = await self.js('window.__chatTestLoad')
        await self.call('Page.reload', {'ignoreCache': True})
        await self.wait(f"window.__chatTestLoad!=={json.dumps(old)} && !!document.querySelector('.chat-composer textarea:not(:disabled)')")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, status, raw, mime='application/json'):
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass  # Expected when the client deliberately aborts a held fixture.

    def do_GET(self):
        if self.path == '/v1/admin/knowledge-bases':
            self.send(200, b'[{"id":"kb-test","name":"Fixture KB"}]')
            return
        if self.path.startswith('/v1/'):
            self.send(405, b'{}')
            return
        path = (ARGS.dist / self.path.split('?')[0].removeprefix('/admin/').lstrip('/')).resolve()
        if ARGS.dist not in path.parents or not path.is_file():
            path = ARGS.dist / 'index.html'
        self.send(200, path.read_bytes(), {'.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css'}.get(path.suffix, 'application/octet-stream'))

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get('content-length', 0)))
        if self.path != '/v1/ask':
            self.send(405, b'{}')
            return
        question = json.loads(raw)['question']
        number = len(REQUESTS)
        REQUESTS.append(json.loads(raw))
        (ARGS.output / f'request-{number:02}.json').write_bytes(raw)
        if question in HOLDS:
            assert HOLDS[question].wait(45), 'fixture hold timed out'
        value = {'detail': 'fixture HTTP failure'} if question == '模拟HTTP失败' else {'answer': '原始异常响应', 'sources': None} if question == '模拟异常响应' else fixture(question)
        body = json.dumps(value, ensure_ascii=False).encode()
        (ARGS.output / f'response-{number:02}.json').write_bytes(body)
        self.send(500 if question == '模拟HTTP失败' else 200, body)


def checked(name):
    CHECKS.append(name)
    print('PASS', name, flush=True)


async def exercise():
    with httpx.Client(timeout=2, trust_env=False) as client:
        for _ in range(100):
            try:
                version = client.get(f'http://127.0.0.1:{ARGS.cdp_port}/json/version').json()
                break
            except Exception:
                await asyncio.sleep(.2)
        else:
            raise TimeoutError('owned browser not ready')
        tab = next(t for t in client.get(f'http://127.0.0.1:{ARGS.cdp_port}/json/list').json() if t['type'] == 'page')
    async with websockets.connect(tab['webSocketDebuggerUrl'], max_size=32 * 1024 * 1024) as ws:
        c = CDP(ws)
        for method in ('Runtime.enable', 'Page.enable', 'Log.enable'):
            await c.call(method)
        await c.call('Emulation.setDeviceMetricsOverride', {'width': 1440, 'height': 1000, 'deviceScaleFactor': 1, 'mobile': False})
        await c.call('Page.addScriptToEvaluateOnNewDocument', {'source': "localStorage.setItem('rwkvrag-language','zh');window.__chatTestLoad=Math.random()"})
        await c.call('Page.navigate', {'url': f'http://127.0.0.1:{ARGS.port}/admin/#/chat'})
        await c.wait("!!document.querySelector('.chat-composer textarea:not(:disabled)')")
        await c.shot('desktop-welcome')
        assert not REQUESTS
        checked('chat mounts without generating or retrieving')
        await c.input('输入法确认')
        await c.js("document.querySelector('textarea').dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',isComposing:true,bubbles:true}))")
        await asyncio.sleep(.2)
        assert not REQUESTS
        checked('IME composition Enter does not submit')
        first = '  如何启用解析并读取结果？\n'
        await c.input(first)
        await c.js("(()=>{const f=document.querySelector('.chat-composer');f.dispatchEvent(new Event('submit',{bubbles:true,cancelable:true}));f.dispatchEvent(new Event('submit',{bubbles:true,cancelable:true}))})()")
        await c.wait("document.querySelector('.answer-body')!==null")
        await c.idle()
        assert len(REQUESTS) == 1 and REQUESTS[0]['question'] == first and REQUESTS[0]['history'] == []
        assert REQUESTS[0]['retrieval_mode'] == 'auto'
        assert await c.js("document.querySelector('.answer-body').textContent") == TEXT
        assert await c.js("document.querySelector('.chat-user-bubble').textContent") == first
        assert not await c.js("document.querySelector('.answer-body .citation-link')!==null")
        checked('exact user/body/literal text, automatic retrieval default and synchronous double-submit guard')
        await c.click('.answer-trace > summary')
        assert await c.js("document.querySelector('.raw-output').textContent") == fixture(first)['answer']
        await c.js("[...document.querySelectorAll('.cited-answer button')].find(e=>e.textContent==='E1.S1').click()")
        await c.wait("document.querySelector('.ant-drawer-open .citation-full-text')!==null")
        assert await c.js("document.querySelector('.ant-drawer-body section .citation-full-text').textContent") == ORIGINAL
        assert await c.js("JSON.parse(document.querySelector('.ant-drawer-body section pre').textContent).snapshot_id") == 'fixture-snapshot'
        assert await c.js("document.querySelector('.ant-drawer-body').textContent.includes('不代表全部被引用')")
        await asyncio.sleep(.4)
        await c.shot('source-drawer')
        await c.click('.ant-drawer-close')
        assert not await c.js("document.querySelector('.ant-drawer-body').textContent.includes('引用没有对应来源')")
        await asyncio.sleep(.5)
        await c.click('.answer-trace > summary')
        checked('raw and selected original provenance remain accessible, separate from full source')
        await c.send('那读取结果用哪个？')
        await c.wait("document.querySelectorAll('.chat-turn .answer-body').length===2")
        await c.idle()
        assert REQUESTS[1]['history'] == [{'role': 'user', 'content': first}, {'role': 'assistant', 'content': fixture(first)['answer']}]
        await c.shot('desktop-multiturn')
        for _ in range(100):
            rows = await c.records()
            if rows and len(rows[0]['turns']) == 2 and rows[0]['turns'][-1]['status'] == 'received':
                break
            await asyncio.sleep(.1)
        original_id = rows[0]['id']
        await c.reload()
        await c.wait("document.querySelectorAll('.chat-turn .answer-body').length===2")
        assert len(REQUESTS) == 2
        checked('real two-turn payload and IndexedDB reload without resend')
        await c.click('[aria-label="检索设置"]')
        await c.wait("document.querySelectorAll('.chat-settings .ant-select-disabled').length===2")
        await c.click('.ant-modal-close')
        checked('scope and knowledge-base controls lock once a conversation starts')
        for question in ('模拟HTTP失败', '模拟非法根', '模拟未完成', '模拟异常响应'):
            await c.send(question)
            await c.idle()
            assert len(REQUESTS[-1]['history']) == 4
        assert await c.js("document.querySelectorAll('.chat-history-notice').length") == 4
        assert await c.js("document.body.textContent.includes('原始异常响应')")
        await c.send('失败后的正常追问')
        await c.idle()
        assert len(REQUESTS[-1]['history']) == 4
        checked('HTTP failure, invalid root, partial and invalid envelope kept but excluded from later history')
        await c.click('.chat-sidebar .chat-new')
        await c.wait("document.querySelectorAll('.chat-turn').length===0")
        await c.send('独立会话问题')
        await c.idle()
        assert REQUESTS[-1]['history'] == []
        await c.js(f"[...document.querySelectorAll('.chat-sidebar .chat-history-select')].find(e=>e.title==={json.dumps(first.strip())}).click()")
        await c.wait("document.querySelectorAll('.chat-turn').length===7")
        checked('new/select conversations isolate histories and retain old failures')
        HOLDS['模拟迟到响应'] = threading.Event()
        await c.send('模拟迟到响应')
        await c.wait("document.querySelector('.chat-pending')!==null")
        await c.click('.chat-sidebar .chat-new')
        await c.wait("document.body.textContent.includes('另一对话正在请求')")
        await c.click('[aria-label="停止等待"]')
        await c.send('停止等待后的新会话')
        await c.idle()
        HOLDS['模拟迟到响应'].set()
        await asyncio.sleep(.5)
        assert REQUESTS[-1]['question'] == '停止等待后的新会话' and REQUESTS[-1]['history'] == []
        assert await c.js("document.querySelectorAll('.chat-turn').length") == 1
        assert await c.js("document.querySelector('.answer-body').textContent") == '离线夹具答复：停止等待后的新会话'
        rows = await c.records()
        old = next(row for row in rows if row['id'] == original_id)
        assert old['turns'][-1]['status'] == 'cancelled' and 'response' not in old['turns'][-1]
        checked('switch + abort + late response cannot contaminate new conversation')
        HOLDS['后台会话回复'] = threading.Event()
        await c.send('后台会话回复')
        await c.wait("document.querySelector('.chat-pending')!==null")
        await c.click('.chat-sidebar .chat-new')
        HOLDS['后台会话回复'].set()
        await c.idle()
        assert await c.js("document.querySelectorAll('.chat-turn').length") == 0
        rows = await c.records()
        assert any(r['turns'] and r['turns'][-1].get('response') == fixture('后台会话回复') for r in rows)
        checked('uncancelled response settles its originating conversation, not active one')
        HOLDS['刷新中断'] = threading.Event()
        await c.send('刷新中断')
        await c.wait("document.querySelector('.chat-pending')!==null")
        await asyncio.sleep(.3)
        count = len(REQUESTS)
        await c.reload()
        await c.wait("document.body.textContent.includes('等待已中断')")
        HOLDS['刷新中断'].set()
        await asyncio.sleep(.3)
        assert len(REQUESTS) == count
        checked('reload marks pending interrupted without generation retry')
        await c.click('.chat-sidebar .chat-history-row.active .chat-delete')
        await c.wait("document.querySelector('.ant-modal-confirm-btns, .ant-modal-footer')!==null")
        await c.click('.ant-modal-footer .ant-btn-primary')
        await c.wait("!document.querySelector('.ant-modal-wrap:not([style*=\"display: none\"])')")
        await c.reload()
        assert not any(any(t['question'] == '刷新中断' for t in r['turns']) for r in await c.records())
        checked('explicit deletion persists and does not resurrect interrupted conversation')
        await c.call('Emulation.setDeviceMetricsOverride', {'width': 390, 'height': 844, 'deviceScaleFactor': 1, 'mobile': True})
        await c.wait("document.documentElement.scrollWidth<=innerWidth")
        await c.shot('mobile-chat')
        await c.click('[aria-label="打开会话列表"]')
        await c.wait("document.querySelector('.ant-drawer-open .chat-mobile-drawer')!==null")
        await asyncio.sleep(.4)
        await c.shot('mobile-conversations')
        await c.click('.chat-mobile-drawer .chat-new')
        await c.wait("document.querySelectorAll('.chat-turn').length===0")
        await asyncio.sleep(.5)
        await c.shot('mobile-welcome')
        checked('390px layout, mobile drawer and new conversation')
        await c.click('[aria-label="检索设置"]')
        await c.wait("document.getElementById('chat-scope')?.getClientRects().length>0")
        point = await c.js("(()=>{const r=document.querySelector('.chat-settings .ant-select').getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()")
        await c.call('Input.dispatchMouseEvent', {'type': 'mousePressed', 'button': 'left', 'clickCount': 1, **point})
        await c.call('Input.dispatchMouseEvent', {'type': 'mouseReleased', 'button': 'left', 'clickCount': 1, **point})
        await c.wait("document.querySelector('.ant-select-item-option[title=\"仅网络\"]')!==null")
        await c.click('.ant-select-item-option[title="仅网络"]')
        await c.click('.ant-modal-close')
        await c.send('显式网络范围夹具')
        await c.idle()
        assert REQUESTS[-1]['retrieval_mode'] == 'web' and REQUESTS[-1]['history'] == []
        checked('explicit source scope is sent unchanged; no real provider is called')
        # Emulate a second tab committing an edit. The first tab must not overwrite it.
        await asyncio.sleep(.3)
        active = await c.js("sessionStorage.getItem('rwkvrag-chat-active-v1')")
        await c.js(f"new Promise(resolve=>{{const o=indexedDB.open('rwkvrag-chat-v1',1);o.onsuccess=()=>{{const d=o.result,t=d.transaction('conversations','readwrite'),s=t.objectStore('conversations'),r=s.get({json.dumps(active)});r.onsuccess=()=>{{const v=r.result;v.revision++;v.draft='other tab draft';s.put(v)}};t.oncomplete=()=>{{d.close();resolve(true)}}}}}})")
        await c.input('this tab draft')
        await c.js("document.querySelector('textarea').blur()")
        await c.wait("document.body.textContent.includes('本页后续更改仅在内存中')")
        rows = await c.records()
        assert next(r for r in rows if r['id'] == active)['draft'] == 'other tab draft'
        checked('stale-tab write conflict warns visibly and preserves newer stored data')
        await c.call('Emulation.setDeviceMetricsOverride', {'width': 1440, 'height': 1000, 'deviceScaleFactor': 1, 'mobile': False})
        await c.js("location.hash='/search'")
        await c.wait("document.querySelector('.app-shell textarea')!==null")
        await c.js("(()=>{const e=document.querySelector('textarea');Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value').set.call(e,'旧检索页兼容夹具');e.dispatchEvent(new Event('input',{bubbles:true}))})()")
        await c.js("[...document.querySelectorAll('button')].find(e=>e.textContent.includes('检索并生成答案')).click()")
        await c.wait("document.querySelector('.answer-body')!==null")
        assert REQUESTS[-1]['question'] == '旧检索页兼容夹具' and REQUESTS[-1]['history'] == []
        assert await c.js("document.querySelector('.citation-source-card').getClientRects().length>0")
        assert await c.js("document.querySelector('.answer-body').textContent") == '离线夹具答复：旧检索页兼容夹具'
        await c.js("location.hash='/chat'")
        await c.wait("document.querySelector('.chat-composer textarea')?.value==='other tab draft'")
        checked('existing Search Lab remains usable and returning to chat restores the stored draft')
        assert not [e for e in ERRORS if e.get('method') == 'Runtime.exceptionThrown'], ERRORS
        return {'browser': version, 'checks': CHECKS, 'requests': len(REQUESTS), 'events': ERRORS,
                'native_calls': 0, 'production_requests': 0, 'synthetic_fixture_only': True}


def main():
    ARGS.output.mkdir(parents=True, exist_ok=False)
    ARGS.profile.mkdir(parents=True, exist_ok=False)
    server = ThreadingHTTPServer(('127.0.0.1', ARGS.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    profile = subprocess.check_output(['wslpath', '-w', str(ARGS.profile)], text=True).strip()
    log = (ARGS.output / 'edge.log').open('x')
    process = subprocess.Popen([ARGS.edge, '--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
        '--disable-background-networking', '--disable-component-update', '--disable-sync', '--disable-extensions',
        '--enable-automation', f'--user-data-dir={profile}', f'--remote-debugging-port={ARGS.cdp_port}', 'about:blank'], stdout=log, stderr=subprocess.STDOUT)
    result = {}
    try:
        result = asyncio.run(exercise())
    except Exception as error:
        result = {'error': type(error).__name__, 'detail': str(error), 'traceback': traceback.format_exc(), 'checks': CHECKS, 'events': ERRORS}
        raise
    finally:
        for event in HOLDS.values():
            event.set()
        server.shutdown()
        server.server_close()

        async def close():
            try:
                with httpx.Client(timeout=3, trust_env=False) as client:
                    v = client.get(f'http://127.0.0.1:{ARGS.cdp_port}/json/version').json()
                async with websockets.connect(v['webSocketDebuggerUrl']) as ws:
                    c = CDP(ws)
                    arguments = await c.call('Browser.getBrowserCommandLine')
                    assert f'--user-data-dir={profile}' in arguments['arguments']
                    await ws.send(json.dumps({'id': 999, 'method': 'Browser.close'}))
            except Exception as error:
                result['cleanup_error'] = str(error)
        asyncio.run(close())
        try:
            result['browser_exit'] = process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            result['browser_exit'] = 'not_observed'
        log.close()
        (ARGS.output / 'RESULT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dist', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--edge', default='/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')
    parser.add_argument('--port', type=int, default=18594)
    parser.add_argument('--cdp-port', type=int, default=18595)
    ARGS = parser.parse_args()
    ARGS.dist = ARGS.dist.resolve()
    main()
