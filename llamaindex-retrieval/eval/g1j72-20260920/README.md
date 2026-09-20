# G1j 7.2B inference-only evaluation

See `docs/g1j72-test-report-20260920.md` for results and limitations. No production
source files or indexes are modified. The frozen input contracts and every failed
output are retained; there are no answer repairs or training calls.

## Runtime

- Existing 2.9B service: local/remote loopback `18423`.
- Isolated 7.2B service: local/remote loopback `18425`.
- `GET /health`; `POST /v1/batch/completions`; `POST /v1/tokens/count`.
- Only authorized physical GPU 3; server checks host identity and GPU UUID.
- `SERVER.json` pins the existing NAS checkpoint, runtime source and server code.
- Service unit: `rwkvrag-g1j72-eval-20260920.service`, remote user manager.
- Tunnel unit: `rwkvrag-g1j72-tunnel-20260920.service`, local user manager.
- Both are transient test units with a 4-hour runtime limit, not persistent
  production services. Reuse requires checking readiness and available GPU memory.

## Reproduction from repository root

Use a new output directory for each run. The runner refuses to overwrite an
existing run, validates frozen file hashes, and preserves exact prompts/outputs.

```bash
rtk proxy llamaindex-retrieval/.venv/bin/python \
  llamaindex-retrieval/eval/g1j72-20260920/run_probe.py \
  --arm direct \
  --endpoint http://127.0.0.1:18425 \
  --model rwkv7-g1j-7.2b-20260831-ctx16384 \
  --output data/quality-runs/g1j72-repeat/direct-72-zero
```

For the 2.9B baseline, use port `18423` and model
`rwkv7-g1j-2.9b-20260831-ctx16384`. The deployed-State reference additionally uses
`--state-id writer-trace-300`. Do not load that State into the 7.2B service.

For source-local extraction and relation composition, use `--arm staged --group
controlled`. For the hand-selected verbatim span control, use `--arm oracle_quotes
--group controlled`. These two arms use zero initial State.

`run_relations.py` independently tests the frozen relation prompt with the
hand-selected spans. It accepts `--endpoint`, `--model`, and `--output`. Its
supplemental preregistration is `PLAN-relations.json`; it is explicitly a diagnostic
on exposed examples, not a new blind holdout.

## Receipts

Client traces and matched native token receipts are archived under
`artifacts/g1j72-20260920/raw-traces.tar.gz`. `TRACE-MANIFEST.json` hashes every client
JSON file; `EXECUTION-SUMMARY.json` records the exact client/native match, token
counts and finish reasons. `MANUAL-REVIEW.json` contains per-case judgments and
failure notes. No aggregate score should be treated as production acceptance.
