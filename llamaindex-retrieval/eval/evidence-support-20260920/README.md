# Evidence support experiment — frozen before first run

This experiment evaluates whether one saved excerpt supports answering one specific question. It does not retrieve, normalize a fact, decide which conflicting source is true, generate an answer, or update Wiki. It changes no production configuration.

## Why a separate check

The previous source review preserved exact evidence but found geography mismatch, table-header false positives, and negative-QA false negatives. Exact quoting and correct semantic support are separate requirements. A future stronger checkpoint should be tested against the same contract before replacing a component.

## Frozen experiment

- 40 synthetic cases, 20 minimal pairs, balanced positive/negative labels.
- 10 categories: object, attribute, time, region, negation, table, unknown/zero, units, revision, injection.
- Each pair stays in one split. Development and validation contain different named entities and wording; 20 cases each. They are still related synthetic templates, not a representative production sample.
- Cases and labels were authored by the coding agent and require independent human review before serving as a production benchmark.
- All profiles and both prompt variants are fixed before any inference. No tuning between development and validation in this run. Once results are inspected, these cases become regression material, not a permanently unseen test set.
- The source ID is an opaque content digest. Labels, rationale, category and split are never added to model prompts.

## Label meaning

YES means the excerpt plus attached context directly supplies information sufficient to answer the requested property, with every explicit object/time/region/mode/version condition matched. It does not mean the answer to a yes/no question is affirmative: an explicit prohibition or non-support is valid evidence. Numeric zero is a recorded value. Unknown quantity does not establish zero. A table header alone is insufficient. A withdrawn draft does not establish a requested current official value. Unit dimensions must agree.

The task is deliberately stricter than the legacy prompt's permission to support only part of a query. Comparing canonical vs strict-v1 measures this prompt/task-contract mismatch as well as model compliance; it is not a claim that the old Reader was trained for the new contract.

## Four profiles

| Profile | Checkpoint + State | Prompt |
| --- | --- | --- |
| reader29-canonical | 2.9B + Reader450 | Existing binary support prompt |
| reader29-strict | 2.9B + Reader450 | Strict-v1, same YES/NO JSON shape |
| zero72-canonical | 7.2B + zero State | Existing binary support prompt |
| zero72-strict | 7.2B + zero State | Strict-v1 |

Within each checkpoint/State pair, only the prompt changes. The cross-model comparison changes checkpoint **and** State, so it cannot isolate parameter count. No 2.9B State is loaded on 7.2B. Each call uses temperature 0, a 32-token output budget, exact native prompt envelope and stop token 0. Model health and available State hashes are saved for each profile.

## Metrics and predeclared pilot gate

Report TP, FP, TN, FN, invalid output, precision, recall, false-positive rate, false-negative rate and pair accuracy, by split and category. Invalid positive cases remain in the recall denominator; invalid outputs are not silently converted to NO. Pair accuracy requires both the positive and negative member to be correct.

The **small-sample pilot gate** requires complete evaluation, no invalid output, zero false positives, recall >= 90%, and pair accuracy >= 90%. Passing this gate authorizes more evaluation, not production deployment. Zero observed false positives in 10 validation negatives is not evidence of a low real-world false-positive rate.

No post-processing fixes model JSON, no retry, no response rewriting, no profile selection based only on an aggregate score. A faster profile that misses evidence or a high-recall profile that admits wrong regions has a different failure mode, not an automatic win.

## Reproduce

From the repository root, with both local model tunnels available:

```bash
PYTHONPATH=llamaindex-retrieval/src llamaindex-retrieval/.venv/bin/python \
  llamaindex-retrieval/eval/evidence-support-20260920/run.py \
  --output data/quality-runs/evidence-support-20260920/run1
```

The output directory must be new. Inputs, prompts, implementation hashes and git commit are bound before the first call. Every result and unmodified trace is written separately. Failed model calls remain invalid cases. Full HTTP receipts stay in the local ignored output directory; a reviewed public artifact can omit receipts without changing raw saved results.

New RWKV deployments can be compared by adding a profile with a checkpoint-compatible State or no State. A different transport requires an adapter that returns unmodified output, finish reason, timing and provenance; it must not change the scoring contract.

## What comes after this experiment

Use disagreements to prepare independently reviewed Reader training examples. Keep the selected excerpts and old model labels immutable; attach any new model's support assessment separately. Test new State or prompting on newly frozen material before wiring it into atomic evidence, then evaluate the full retrieval pipeline. Only after reliable support checks should normalized claims feed relation building and hierarchical summaries.
