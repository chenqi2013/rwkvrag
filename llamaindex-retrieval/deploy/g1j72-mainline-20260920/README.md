# 7.2B comparison/choice mainline endpoint

User-selected mainline model: G1j 7.2B 20260831, using the user's previously
verified local vllm-rwkv snapshot. Dedicated served name `rwkvrag-g1j72-mainline`,
loopback port18426, context16384, FP16 weights, FP32 recurrent state, zero State.
Do not load 2.9B Reader/Writer states. Existing application18440 and regression18446
remain on their original configuration; this directory is an inference profile,
not a completed application migration.

Remote root: `/home/chase/rwkvrag/data/services/g1j72-mainline-20260920`.
`launch.py` checks GPU3's exact UUID, host identity, 2,444 engine file hashes,
model metadata and six weight-shard hashes before loading. It reuses the engine,
environment and compile cache from `vllm-decode-20260920`; it writes new deployment
records under this new root and preserves old experiment snapshots.

User systemd unit: `rwkvrag-g1j72-mainline-20260920.service`, started18:09:55
2026-09-20 Asia/Shanghai, temporary12-hour runtime (expected expiry06:09:55 Sept21).
Environment matches the prior verified deployment: authorized CUDA UUID,
TORCH_CUDA_ARCH_LIST12.0, previous TORCH_EXTENSIONS_DIR, MAX_JOBS4,
OMP_NUM_THREADS4, VLLM_WORKER_MULTIPROC_METHODspawn and previous SCM version.
Only authorized GPU3 is visible. Model memory reservation is not a minimum
consumer-hardware requirement.

## Smoke checks and actual findings

All three transport requests completed with matching prompt token IDs/counts and
`finish_reason=stop`. Exact artifact tokenizer/chat template is used with
`rwkv_generation_prompt=fake_think`; submit token IDs without special-token
insertion. Explicit top-1 configuration is in `profile.json`.

- Seen Reader material returned `INSUFFICIENT` as expected.
- Small comparison repeated the supplied values correctly, but omitted the
  requested source labels.
- **Constrained choice failed:** the model accepted a1000g device under a900g
  upper limit and selected it for longer battery life. Its complete original
  response is preserved. Endpoint readiness is not comparison/choice correctness.

See [raw checks and deployment evidence](../../../artifacts/g1j72-mainline-20260920/).
These tiny fixed-material fixtures are transport checks, not RAG evaluation or a
claim of general quality improvement.

## Integration boundary

The application's existing native transport has a different prompt/BOS and
default sampling contract. Do not change only its base URL to this endpoint.
Application integration must preserve the actual artifact tokenizer/template,
explicit supported sampling controls, raw model output and termination bounds.
After that, validate the comparison and hard-condition choice workflows with7.2B.

This service shares GPU3 with the running2.9B regression. Loading/warmup and smoke
traffic overlap that batch; keep its inputs/configuration unchanged and describe
its timing as observational, not a controlled speed comparison.
