# Local vllm-rwkv decoding regression: preregistration

The user requested testing their locally optimized engine and updating the server
source. Use an isolated copy of the current local working tree, including existing
uncommitted changes; never replace it with upstream or modify the user's tree.

## Scope and material

Compare top-1 with the deployed model artifact's Fake Think sampling configuration
on the same engine, weight artifact, zero State, FP16 weights/FP32 recurrent state,
input token IDs, prompt rendering, stop policy, and 32-token output budget.
The 160 cases are the **already-seen regression** materials from the frozen label
replication (20 template families; 80 positive/80 negative). No independent gold
review, unseen validation, end-to-end quality claim, or deployment promotion.

Use the unchanged strict-v1 instructions with ANSWERABLE/INSUFFICIENT. Render once
using the artifact's canonical chat template in fake_think mode. Its incomplete
`<think></think` prefix requires the model to emit `>`; the parser checks that
boundary explicitly then parses the remaining JSON without repairs. This renderer,
engine and dtype differ from the prior simple server: do not attribute a cross-run
difference to sampling. Table presentation is unchanged within this comparison.

## Conditions

Both arms: temperature=1.0, presence_penalty=0, frequency_penalty=0,
penalty_decay=0.996 (the actual artifact value), max_tokens=32, EOS=0 and the
artifact's stop strings. Disable implicit generation config inheritance in the
isolated server and send all effective controls explicitly.

- top1: top_k=1, top_p=1; seeds 11 and 101 as repeatability checks.
- fake: top_k=32, top_p=0.28; seeds 11,29,47,71,101.

1120 calls, but only 160 cases. This is a decoding-policy comparison changing
top_k and top_p together, not a temperature ablation. Requests are sequential;
conditions rotate per case in a frozen shuffled schedule. No retries, output
repair, constrained grammar, state import/reuse, extra thinking, or best-of-N.
Health/runtime failures stop and preserve partial results. Invalid formats and
length exits remain errors, never converted to negative decisions.

## Analysis frozen before inference

Top1 seed11 is the primary baseline; seed101 checks exact token/output and label
stability. Report each of five fake seeds separately plus the mean of their case
accuracy, recall, FP, invalid count, and pair accuracy. Seeds are not new semantic
samples. Report per-category changes and failures, output lengths and latency.

Bootstrap paired per-case accuracy deltas (candidate mean over seeds minus the
primary baseline) by all 20 whole template families, 10,000 draws, fixed seed
20260920. Report the 95% percentile interval conditional on these synthetic
families, not a production-population guarantee.

A limited positive regression claim requires: mean accuracy gain >=5 percentage
points; family interval lower bound >0; no fake seed increases FP or invalid
outputs; every fake seed has accuracy >= baseline; and both top1 runs have identical
classification results. Otherwise report the tradeoff, not a winner. Always show
the original pilot gate (zero FP/invalid, recall and pair accuracy >=90%) separately.
Even passing does not authorize claiming an unseen validation or production gain.

Bind source/installed dependency hashes, model metadata and checked shard hashes,
rendered prompts/token IDs, schedule, scripts and Git commit before the first case.
Use only physical GPU3 on rwkv-8222 and loopback port18426. Do not interrupt
unrelated services. Runtime/compile failures are infrastructure findings, not
model semantic failures.
