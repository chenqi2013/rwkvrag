# Reader output-label ablation — preregistration

Status before first run: inputs, two profiles, prompt transformation and scoring frozen in Git. This experiment follows docs/EXPERIMENTS.md; it does not train a State or change an application configuration.

## Sole variable

Same 7.2B checkpoint, zero State, strict-v1 instruction, question, original excerpt, context, source ID, native envelope, greedy sampling, 4096-token input limit, 32-token output budget, stop token 0, concurrency 1 and timeouts.

- Baseline labels: YES / NO.
- Candidate labels: ANSWERABLE / INSUFFICIENT.

Only the instruction's exact label tokens and its necessary “output NO” declaration change. No demonstrations or additional semantic instructions are added. The data portion is byte-identical, including YES/NO or JSON appearing in adversarial source text. Longer label tokenization is part of the intervention; output format failures remain failures under the same budget.

No claim is made that neutral labels remove all semantic ambiguity. The purpose is to measure whether they change the known confusion between an explicitly negative answer and missing evidence.

## Materials and schedule

64 cases per arm:

- 40 seen cases from evidence-support-20260920, marked development **only for runner compatibility**. They are regression cases, not unseen validation or new tuning data.
- 24 newly authored validation cases, 12 positive/negative pairs: explicit negatives and unknowns, affirmative support, zero quantities, time, region, model variant, table header, attribute mismatch and injection.

The new cases were labelled and self-reviewed before inference. They have not been independently reviewed. They include related failure families, so they are not a representative or independently blind benchmark. Pair members stay in the same split. No training is performed and no tuning occurs between the splits.

Run the baseline profile first and then the candidate, using the same frozen case order and sequential calls. Preserve per-call timestamps; latency is observational because the two arms run at different times. Output words can have different token lengths; do not interpret timing as general model speed.

## Metrics and stopping

Report the same confusion matrix, precision, recall, invalid count and complete-pair accuracy, separately for seen/new material. The predeclared small-sample gate is zero invalid output, zero false positives, recall >= 90%, pair accuracy >= 90%, complete execution, with regressions disclosed. Invalid output is never mapped to insufficient, and the parser never accepts the other arm's labels as a fallback.

Abort/relabel a run if model identity changes, service fails preflight or frozen input/code hashes change. No prompt editing or retries based on mid-run results. A later protocol change gets its own directory, bindings and results.

## Reproduce

```bash
PYTHONPATH=llamaindex-retrieval/src llamaindex-retrieval/.venv/bin/python \
  llamaindex-retrieval/eval/reader-label-20260920/run.py \
  --output data/quality-runs/reader-label-20260920/run1
```

The runner is a separately frozen copy of the prior evaluation runner, changing only the parser dispatch and provenance for this experiment. The metric implementation and strict-v1 prompt are imported from the unchanged earlier protocol. It records the prior deployment's 7.2B weight SHA as provenance rather than pretending the health endpoint verifies that SHA.

No application promotion is automatic. Once these new cases are inspected, they too become seen regression material for future experiments.
