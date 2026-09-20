# Reader evidence presentation: seen-material regression

## Hypothesis and scope

Test whether literal multiline presentation makes the existing object/header/row
relationship easier to read than the frozen single-line JSON serialization.
This is a presentation-policy comparison, not a position-only ablation. It changes
escaping and boundary syntax together. Do not claim it proves the root cause.

Use all 160 already-seen cases from the decoding regression, including the failures
and successful cases. They are synthetic, self-reviewed and not independently
adjudicated. No unseen validation or production promotion is possible in this run.
Do not edit the main application's Reader, selector, retrieval or Writer.

## Fixed conditions

Use the same live local vllm-rwkv snapshot, dependency and compiled-library hashes,
G1j 7.2B artifact, zero State, FP16 weights/FP32 recurrent state, canonical
fake_think template, strict-v1 instruction and ANSWERABLE/INSUFFICIENT labels.
Keep temperature=1, top_k=1, top_p=1, seed=11, presence/frequency=0,
penalty_decay=.996, output budget=32 and existing EOS/stop policy.
All input IDs are frozen before execution, with input+output <=4096.

Two arms: `json` reproduces the previous baseline body and token IDs exactly;
`multiline` replaces only its final serialized data object. Field order remains
source, question, contexts, text; values remain byte-for-byte Unicode strings.
Each field uses a length-declared boundary and an exact raw value. Context count
and indices describe structure only. No inferred object, unit, answer, example,
additional semantic instruction, reordered evidence or normalized fact is added.
Preparation checks lossless reconstruction of every field and identical instruction.

Two rounds per arm, 640 sequential calls. Fixed shuffled case order, reversed in
round 2; arm order alternates per case and flips in round 2. No retries, best-of,
output repair, extra thinking, grammar constraints or state import. Invalid
outputs remain errors; infrastructure errors abort and preserve partial evidence.

## Preregistered analysis

Round 1 is primary, round 2 measures repeatability rather than adding samples.
Report TP/FP/TN/FN/invalid, recall, accuracy and pair accuracy overall and by
category, exact raw/token agreement and per-case improvements/regressions.
Compare the new JSON baseline against the preceding top1 seed11 run; any drift
must be visible, and prevents the limited positive regression claim below.

Bootstrap paired case accuracy differences by 20 whole template families,
10,000 draws, seed 20260920, 95% percentile interval. This interval is conditional
on these synthetic families and not a production-population guarantee.

A limited positive regression claim requires ALL of:
- >=5 percentage points overall accuracy gain in the primary round;
- family interval lower bound >0;
- no increase in FP or invalid outputs in either round;
- no accuracy regression in any of the ten categories in either round;
- improved table-positive recall in both rounds;
- identical classification between rounds in each arm;
- identical raw/token JSON baseline versus the previous frozen top1 run.

Separately report the existing pilot gate: zero FP/invalid, >=90% positive recall
and pair accuracy. Even if a limited improvement is found, neither seen data nor
this binary decision experiment validates production or complete comparison answers.
After analysis, stop this experiment. A new dataset and experiment are required
for task-instruction changes, normalization, independent assessments or molecules.

Only physical GPU3 on rwkv-8222 is authorized. Verify source/dependency/library
bindings before and after. If the temporary service expired, record and redeploy
the same snapshot rather than silently switching to the main model.
