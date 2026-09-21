# RWKV fp32io16 startup contract

User requirement: use the RWKV kernel with `fp32io16`, never the FP16 recurrent-State path.

For the audited local optimized engine, `--mamba-ssm-cache-dtype float32` allocates FP32 recurrent State and selects `infer_tmix_wkv7_recurrent_fp32io16_forward_varlen`. `--dtype float16` describes weights/IO, which are the `io16` in this kernel mode; it does **not** request the FP16 recurrent kernel. The model code is exclusively backed by FlashRWKV2 0.1.0a13.

`launch.py` defaults to a CPU-side preflight and verifies the actual engine source hashes, provider version and required APIs. Changed sources or unavailable APIs stop startup; there is no FP16 fallback. Startup is restricted to the already authorized host/GPU3 and free experiment port18426. No existing service is stopped by this launcher.

On the model host, with this directory copied intact:

```sh
/home/chase/rwkvrag/data/services/vllm-decode-20260920/engine/.venv/bin/python launch.py --mode check --report /path/to/new-preflight.json
/home/chase/rwkvrag/data/services/vllm-decode-20260920/engine/.venv/bin/python launch.py --mode start --report /path/to/new-start-preflight.json
```

Use a new report path each time. Preflight success is **not** a model startup, kernel execution trace or quality result. Readiness, actual execution and quality require subsequent checks. No model was started by this configuration update.

The previous typed-funnel service command also set State to float32. Earlier reports calling the whole run “FP16” conflated IO precision and recurrent State; those frozen reports and results remain unchanged. This profile makes the distinction explicit for new runs.
