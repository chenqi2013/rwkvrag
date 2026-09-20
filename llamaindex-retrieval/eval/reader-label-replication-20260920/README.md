# Label-only replication: preregistration before first inference

The user requested verification before sharing a claimed improvement. This follow-up takes priority over the previously planned table-layout experiment. It does not change table layout, prompts, training, checkpoints or application deployment.

## Intervention and material

Import the unchanged strict-v1 prompt and the previously frozen instruction-only YES/NO → ANSWERABLE/INSUFFICIENT mapping. Never replace label text inside a question, excerpt or context. Same 7.2B checkpoint, zero State, native envelope, temperature 0, output budget 32, input limit 4096, stop token 0, concurrency 1 and timeouts.

160 new synthetic cases: 80 positive and 80 negative, 80 minimal pairs, 20 template families, four entity/value variants per family. Ten categories: negation, affirmation, zero/unknown, time, region, object, attribute/unit, table, revision and quoted instructions. Two families use English questions/evidence; all instructions remain unchanged Chinese prompts. Old 40/24 cases are not included in the primary result.

Labels and rationales are authored and self-reviewed by the coding agent before inference. No independent human or second-agent review has occurred. This is a controlled synthetic replication, not a blind independent benchmark or production distribution. The expected labels never enter the prompt. Family variants are correlated and are not counted as independent semantic scenarios.

## Schedule and fixed analysis

Two rounds of both arms: 640 calls, but only **160 unique cases**. Within each round arms are interleaved, first-arm order is balanced across cases, and arm order reverses in round 2. All calls are sequential. Case order and schedule are frozen in schedule.json. Round 1 is primary; round 2 measures repeatability and is never pooled as another 160 independent cases.

Report confusion matrices including invalid outputs, recall, precision, per-category results, pair accuracy, recovered/regressed cases, and exact classification agreement between rounds. Invalid output is not NO/INSUFFICIENT and remains an error. No retries or repairing raw output. Health identity is checked before running and after each 40 calls. Stop on failed health identity, preserve incomplete results, and do not silently resume as the same clean run.

Uncertainty: paired accuracy differences are resampled by whole template family (20 clusters), 10,000 bootstrap draws, random seed 20260920, 2.5/97.5 percentile interval. All families are equal size. Also report leave-one-family-out minimum delta. These intervals describe sensitivity to the chosen synthetic families, not population generalization or an independently validated causal mechanism. No external significance claim or multiple subcategory significance tests.

## Predeclared limited-improvement criteria

Before seeing results, a limited positive report about this setting requires all of:

1. Round 1 overall accuracy (including invalid) improves at least 5 percentage points.
2. Its family-cluster interval lower bound is above zero.
3. Removing any one family leaves a positive overall difference.
4. Neither round increases false positives or invalid output.
5. Round 2 accuracy also improves; each arm agrees with its first round on at least 99% of cases.

Passing only supports a qualified report of improvement on this particular controlled experiment. It does not establish that labels are the sole root cause, transfer to other prompts/models, or production readiness. The original stricter pilot gate (zero invalid, zero false positives, recall and pair accuracy >=90%) is still reported separately. All category regressions and individual failures remain visible even if aggregate criteria pass. If criteria fail, report the actual tradeoff rather than claiming a general improvement.

## Reproduce

```bash
PYTHONPATH=llamaindex-retrieval/src llamaindex-retrieval/.venv/bin/python \
  llamaindex-retrieval/eval/reader-label-replication-20260920/run.py \
  --output data/quality-runs/reader-label-replication-20260920/run1
```

The directory must be new. Dataset, schedule, scripts, imported protocols, runtime client and this preregistration are hashed before the first call. Weight provenance refers to the prior verified deployment; the health endpoint does not itself return or prove a checkpoint SHA. Full original traces and HTTP receipts are retained. The application and prior experiment files stay unchanged.
