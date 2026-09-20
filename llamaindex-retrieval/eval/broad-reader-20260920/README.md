# Broad additive Reader regression

The user requires broad coverage and preservation of previous questions. This
experiment adds 960 synthetic cases in 40 semantic template families and executes
every row of the historical 40, 64 and 160-case support sets. The 64-case file
contains the earlier 40: retain those memberships, report 264 historical rows but
224 unique historical case IDs. Total 1,224 rows / 1,184 unique case IDs. The 960
new cases are 12 positive/negative pairs per family, not 960 independent families.
Names/numbers and three question phrasings vary within each family.

## Labels and coverage

Cases and labels are agent-authored and self-reviewed before inference; no
independent adjudication. New materials were not used for previous tuning. After
this execution they are exposed regression data, not a reusable blind set.
All source rows and original membership are retained in cases.json. Synthetic
minimal pairs contrast present versus missing/wrong object, attribute or scope.

Coverage includes definition, location, responsible person, time, quantities,
units, recorded zero, explicit negation, affirmative facts, complete lists,
procedure ordering, stated causes, ranges, exact versus approximate values,
model identity, regional/mode scope, withdrawn revisions, two-object comparison,
conflicting records, tables, English, cross-language facts, explicit referents,
long distractors, untrusted instructions and conditional exceptions.

This remains a single-evidence support task. Full actual-retrieval questions and
fixed-material answers run separately in broad-qa-20260920. No claim that a binary
Reader test validates arithmetic, open-ended answers, citation entailment or a
complete conflict-resolution architecture.

## Frozen comparison

Reuse the exact JSON/multiline protocols from reader-presentation-20260920 without
editing them. Same local vllm-rwkv, G1j 7.2B zero State, FP16 weights/FP32 recurrent
state, canonical fake_think template, labels and 32-token output budget.
temperature=1, top_k=1, top_p=1, seed=11, penalties=0, decay=.996, same stop policy.
No per-type switching, semantic patches, State training or prompt searches.

Two rounds per arm, 4,896 sequential calls. Seeded shuffled case order reversed
in round 2; per-case arm order flips. Every input is tokenized and frozen; data
round-trips losslessly, input+output <=4096. Original historical prompt source IDs
are preserved; unique run IDs and pair IDs are only bookkeeping, not model input.
No retry or output repair. Infrastructure errors abort and preserve partial calls;
invalid outputs stay in the denominator. No case is dropped for bad performance.

## Reporting and gates

Report each cohort, category and family separately, both rounds, confusion counts,
invalid outputs, recall and pair accuracy, and every improved/regressed row.
Report historical duplicate memberships explicitly; do not inflate unique samples.
Round 1 is primary; round 2 only checks stability. Verify historical baselines
against previous runs where identical conditions exist (the historical160 set).

The candidate is already known to have drawbacks on older data. No positive
promotion claim unless: all historical baseline-correct cases remain correct in
both rounds; no cohort/category increases false positives or invalid outputs;
neither arm drifts between rounds; new-cohort accuracy gain >=5 points; and the
new 40-family paired bootstrap interval is entirely above zero (10,000 draws,
seed 20260920). Report the original zero-FP/invalid, recall/pair>=90% pilot gate
separately. No ordinary-question regression may be traded away for conflict gain.
No production promotion or independent semantic certification occurs in this run.

HISTORY-RETENTION.json binds all 352 pre-existing tracked eval/test files selected
by .json/.jsonl/.py/.md from baseline 7e7d33c8. They must remain present and unchanged.
This retention ledger does not mean every historical training/output file was
rerun. Explicit execution coverage is the case manifests of this and the QA suite.
