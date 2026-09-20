# Local engine deployment, 2026-09-20

The deployment uses a snapshot of `/home/chase/GitHub/vllm-rwkv`, including the user's existing uncommitted changes. Source identity is the 2,444-file manifest frozen in commit `55ba45e8`, rather than the package version banner. The upstream repository was not pulled and the local engine working tree was not modified.

Remote root: `/home/chase/rwkvrag/data/services/vllm-decode-20260920` on `rwkv-8222` (`rwkv-82`). Only GPU3, UUID `GPU-a9570da2-547a-c2b3-0cab-7bbdc1a8a8b0`, is authorized and exposed to the process. Endpoint: loopback port 18426.

## Startup attempt 1

Unit: `rwkvrag-vllm-decode-20260920.service`. All source files, model metadata and six model shards passed hash checks. Torch 2.13.0+cu130 and FlashRWKV2 0.1.0a13 were imported from the copied environment. The engine loaded all weights, then failed during lazy FlashRWKV2 extension compilation: setuptools-scm could not infer a version because the source snapshot excludes `.git`. No regression requests were executed. The complete log is preserved as `startup-attempt1.log`.

## Startup attempt 2

Unit: `rwkvrag-vllm-decode-20260920-attempt2.service`, started at 14:08:31 Asia/Shanghai, runtime limit 7,200 seconds. The frozen launcher, engine source, model and inference flags are unchanged. Added environment variable:

```
SETUPTOOLS_SCM_PRETEND_VERSION=0.1.dev2072+g4b5cebdc2.d20260901
```

This supplies the existing installed package version to the build metadata resolver; it is not a claim that the banner identifies the current source snapshot. The source manifest remains authoritative. The launcher repeats all identity checks before starting inference.

`DEPENDENCIES.json` additionally binds 133 files in FlashRWKV2, the RWKV transformers implementation and tokenizers; a separate remote comparison found zero mismatches before startup attempt 2.

Attempt 2 compiled all 89 FlashRWKV2 objects successfully and loaded the extension. Model loading including first-time compilation took 1,102.329 seconds. Initialization then failed its memory-profile consistency assertion: free VRAM increased from 57.7 to 64.46 GiB. The older isolated 7.2B service reached its pre-existing runtime limit at 14:13:05 and released GPU memory during this initialization window (confirmed in that unit's journal). No regression requests were executed. See `startup-attempt2.log`.

## Startup attempt 3

Unit: `rwkvrag-vllm-decode-20260920-attempt3.service`. Same launcher, environment and inference settings as attempt 2. Reuses the successfully compiled extension cache after the old service has expired. All identity guards run again; no memory-profile guard was bypassed.

Started at 14:31:36; API startup completed at 14:35:20 Asia/Shanghai. Expected runtime limit: 16:31:36. Cached model/extension loading took 6.603 seconds and engine profiling/cache/warmup 8.94 seconds, excluding the preceding full shard verification. Health passed before the first regression request.

`RUNTIME-BEFORE.json` confirms all 2,444 engine files and 133 dependency files match their manifests, and identifies the actual FlashRWKV2 library mapped into EngineCore PID 1203470. The API server PID is 1203049. This is an isolated test deployment, not a switch of the main application.
