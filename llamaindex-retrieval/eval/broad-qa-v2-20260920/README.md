# Broad QA v2: explicit corpus coverage

The first run was stopped after discovery of a harness error: it used the nonexistent
`default` knowledge-base scope. All 66 recorded requests and the original 478 cases
are retained, including cancellation records. Its results are not a retrieval baseline.

Read-only corpus inspection then established that neither historical 200-question
Wiki suite has its exact reference titles in the current 5,000-page corpus. The old
full corpus is unavailable in the current index. Changing only the scope cannot
restore this old retrieval coverage. This remains a documented gap.

This version executes **all 478 original questions**, in the same frozen order:

- 400 historical Wiki questions with their unchanged retained reference excerpts,
  through `/v1/material-ask`: Writer/source-grounding checks, not retrieval tests.
- 6 basic, 20 diverse and 8 native Wiki questions through actual `/v1/ask`, using
  `wiki-5000-20260908`, validated to contain 45,960 chunks. Any absent expected article
  in the basic/diverse suites is a corpus gap, not successful semantic abstention.
- All original 44 fixed-material cases unchanged, including the three empty inputs
  expected to produce HTTP 422. These remain coverage failures in the denominator.

All original questions, histories, labels and reference text remain unchanged.
No production code, prompt, State or index changes. Concurrency 2, timeout 240s,
no client retries; existing application-internal behavior is traced unchanged.
This is a new, explicitly narrower execution scope for 400 questions, not a claim
that their original full-corpus end-to-end coverage has been restored. Restoring
that corpus and rerunning the full retrieval suite remains required before claiming
full historical end-to-end regression coverage.

`prepare.py` records source hashes and corpus preflight. `run.py` preserves raw
requests/responses and follows v1 diagnostics. Those diagnostics do not establish
semantic accuracy. The previously frozen 66-case manual review selection applies
to this run, including failures; it is agent review, not independent adjudication.

Run after committing this directory:

```bash
PYTHONPATH=llamaindex-retrieval/src llamaindex-retrieval/.venv/bin/python \
  llamaindex-retrieval/eval/broad-qa-v2-20260920/run.py \
  --output data/quality-runs/broad-qa-v2-20260920/run1
```
