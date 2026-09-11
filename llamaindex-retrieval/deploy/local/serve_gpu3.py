"""Single-user RWKV CUDA service on the authorized GPU3, immutable inference receipts."""
import argparse
import base64
import importlib
import hashlib
import socket
import subprocess
import json
import os
from pathlib import Path
import signal
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = Path('/home/chase/rwkvrag')
sys.path.insert(0, str(ROOT/'llamaindex-retrieval/src/llamaindex_retrieval'))
from state_tokens import Vocabulary
GPU = 'GPU-a9570da2-547a-c2b3-0cab-7bbdc1a8a8b0'

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def write(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())

def host_guard():
    if socket.gethostname() != 'rwkv-82' or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU:
        raise ValueError('only authorized physical GPU3 is allowed')
    if Path('/etc/machine-id').read_text().strip() != 'bcd164d5ad3a4ab3b0790412e32e69f3':
        raise ValueError('host machine identity changed')
    info = subprocess.run(['nvidia-smi', '-i', GPU, '--query-gpu=index,uuid,memory.used',
        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=20).stdout.strip()
    index, uuid, memory = [x.strip() for x in info.split(',')]
    if (index,uuid) != ('3',GPU): raise ValueError('GPU identity mismatch')
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'],
        check=True, capture_output=True, text=True, timeout=20).stdout
    if any(x.split(',')[0].strip() == GPU for x in apps.splitlines()):
        raise ValueError('GPU3 occupied; no process is stopped automatically')
    return {'hostname':socket.gethostname(), 'physical_index':3, 'uuid':GPU, 'memory_used_mib_before':int(memory)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--sha256', required=True)
    args = ap.parse_args()
    assert sha(args.config) == args.sha256
    cfg = json.loads(args.config.read_text())
    for name, digest in cfg['bindings'].items():
        assert sha(ROOT/name) == digest, name
    assert cfg['model'] == 'rwkv7-g1j-2.9b-20260831-ctx16384', 'Deployment is restricted to the user-selected 2.9B model'
    assert cfg['checkpoint_sha256'] == '966f3420f833532aae3fb1fd6326533b08d43d23b7b03eaa2f0694a30b64a239'
    gpu = host_guard()
    out = ROOT/cfg['output']/time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())/str(os.getpid())
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    def stop(*_):
        # socketserver catches Exception raised inside a request; shutdown must
        # propagate through that handler to the outer resource cleanup.
        raise SystemExit('inference service stopped')
    signal.signal(signal.SIGTERM, stop)
    server = None
    calls = 0
    try:
        runtime = ROOT/cfg['runtime']
        assert sha(runtime/'RUNTIME-SOURCE.json') == 'c0feaa35b8db51f3a67c5e7aabc7b58b54e532e4f5b681020ef27803691feb6c'
        for name, pin in json.loads((runtime/'RUNTIME-SOURCE.json').read_text())['files'].items():
            assert sha(runtime/'rwkv'/name) == pin['sha256']
        assert sha(Path(cfg['checkpoint'])) == cfg['checkpoint_sha256']
        os.environ.update(TORCH_CUDA_ARCH_LIST='12.0',
                          TORCH_EXTENSIONS_DIR=str(ROOT/'data/experiments/practical-rag-20260910/compile-cache'),
                          MAX_JOBS='4', RWKV_V7_ON='1', RWKV_CUDA_ON='1', RWKV_JIT_ON='1',
                          TORCH_FORCE_WEIGHTS_ONLY_LOAD='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.set_grad_enabled(False)
        assert torch.cuda.device_count() == 1
        sys.path.insert(0, str(runtime))
        module = importlib.import_module('rwkv.model')
        assert Path(module.__file__).resolve() == runtime/'rwkv/model.py'
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        model = module.RWKV(model=cfg['checkpoint'][:-4], strategy='cuda bf16')
        assert (model.n_layer, model.n_embd, model.n_head, model.head_size) == (32,2560,40,64)
        vocab = Vocabulary(runtime/'rwkv/rwkv_vocab_v20230424.txt')
        versions = {k: t._version for k,t in model.z.items()}
        write(out/'READY.json', {'config':cfg,'config_sha256':args.sha256,'gpu':gpu,
             'pid':os.getpid(),'backend':'official rwkv0.8.30 CUDA WKV7 sequence kernel and recurrent decode',
             'torch':torch.__version__,'matmul_allow_tf32':False,'cudnn_allow_tf32':False,
             'optimizer_updates':0,'sampling':'argmax','state_axes':'H,V,K; no transpose'})

        def generate(prompt, limit, state_id):
            nonlocal calls
            ids = vocab.encode(prompt)
            if len(ids) + limit - 1 > 16384:
                raise ValueError('context budget exceeded; input not truncated')
            calls += 1
            name = f'{calls:04}'
            write(out/(name+'.started.json'), {'prompt':prompt,'input_ids':ids,'max_tokens':limit,
                  'state_id':state_id,'state_sha256':None,
                  'zero_initial_state':not bool(state_id),'prompt_sha256':__import__('hashlib').sha256(prompt.encode()).hexdigest()})
            tick = time.monotonic()
            working = []
            for i in range(model.n_layer):
                wkv = torch.zeros(model.n_head,model.head_size,model.head_size)
                working.extend([torch.zeros(model.n_embd,dtype=torch.bfloat16,device='cuda:0'),
                                wkv.to('cuda:0').clone(),torch.zeros(model.n_embd,dtype=torch.bfloat16,device='cuda:0')])
            tokens = []
            reason = 'length'
            completed = False
            try:
                logits, working = model.forward(ids, working)
                for j in range(limit):
                    if not torch.isfinite(logits).all():
                        raise ValueError('nonfinite logits')
                    token = int(logits.argmax())
                    tokens.append(token)
                    if token != 0 and token not in vocab.by_id:
                        raise ValueError('model emitted an undefined vocabulary token')
                    if token == 0:
                        reason = 'stop'
                        break
                    if j+1 < limit:
                        logits, working = model.forward([token], working)
                raw = b''.join(vocab.by_id[t] for t in tokens if t != 0)
                text = raw.decode('utf-8')
                completed = True
                return text, reason
            finally:
                pieces = []
                invalid_token = None
                for token in tokens:
                    if token == 0:
                        continue
                    if token not in vocab.by_id:
                        invalid_token = token
                        break
                    pieces.append(vocab.by_id[token])
                raw = b''.join(pieces)
                write(out/(name+'.completed.json'), {'output_token_ids':tokens,'raw_bytes_base64':base64.b64encode(raw).decode(),
                      'generation_completed':completed,
                      'invalid_token':invalid_token,'raw_bytes_scope':'complete' if invalid_token is None else 'decodable_prefix_only',
                      'actual_eos':bool(tokens and tokens[-1]==0),'finish_reason':reason,
                      'raw_text':raw.decode('utf-8') if invalid_token is None and _valid_utf8(raw) else None,
                      'elapsed_seconds':time.monotonic()-tick,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})

        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                self.connection.settimeout(1200)
            def log_message(self, *_):
                pass
            def send(self, status, body):
                raw = json.dumps(body,ensure_ascii=False).encode()
                self.send_response(status)
                self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            def do_GET(self):
                if self.path == '/health':
                    self.send(200,{'ready':True,'model':cfg['model'],'state_sha256':None,'calls':calls})
                else:
                    self.send(404,{'error':'unknown path'})
            def do_POST(self):
                try:
                    n = int(self.headers.get('Content-Length','0'))
                    if not 0 < n <= 2_000_000:
                        raise ValueError('invalid request size')
                    body = json.loads(self.rfile.read(n))
                    if self.path == '/v1/tokens/count':
                        self.send(200,{'tokens':len(vocab.encode(body['text']))})
                        return
                    if self.path != '/v1/batch/completions':
                        self.send(404,{'error':'unknown path'})
                        return
                    if (body.get('model') != cfg['model'] or body.get('top_p') != 0
                        or body.get('alpha_presence') != 0 or body.get('alpha_frequency') != 0
                        or body.get('stream') is not False or body.get('stop_tokens') != [0]
                        or body.get('state_id') is not None):
                        raise ValueError('unsupported model/state/sampling request')
                    contents, limit = body['contents'], body['max_tokens']
                    if not isinstance(contents,list) or not 1 <= len(contents) <= 8 or not all(isinstance(p,str) and p for p in contents):
                        raise ValueError('invalid batch contents')
                    if type(limit) is not int or not 1 <= limit <= 2048:
                        raise ValueError('output limit')
                    choices=[]
                    for i,prompt in enumerate(contents):
                        text,reason=generate(prompt,limit,body.get('state_id'))
                        choices.append({'index':i,'message':{'role':'assistant','content':text},'finish_reason':reason})
                    self.send(200,{'object':'chat.completion','model':cfg['model'],'choices':choices})
                except (ValueError, KeyError, UnicodeError) as exc:
                    self.send(422,{'error':str(exc)})

        HTTPServer.request_queue_size = 32
        server = HTTPServer(('127.0.0.1',18423),Handler)
        server.timeout = 1
        while True:
            server.handle_request()
    finally:
        if server:
            server.server_close()
        unchanged = None
        if 'model' in locals() and 'versions' in locals():
            unchanged = all(t._version == versions[k] for k,t in model.z.items())
        write(out/'STOPPED.json', {'elapsed_seconds':time.monotonic()-start,'generation_calls':calls,
              'base_tensor_versions_unchanged':unchanged,'base_tensor_count':len(versions) if 'versions' in locals() else None,
              'optimizer_updates':0,'training_executed':False})


def _valid_utf8(raw):
    try:
        raw.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False


if __name__ == '__main__':
    main()
