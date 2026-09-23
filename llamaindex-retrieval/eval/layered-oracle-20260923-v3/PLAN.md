# Layered oracle V3: one hard condition at a time

V2 gave one project at a time but still showed both hard conditions and the GPU
preference. It therefore did not reproduce the v10 condition node, which sees
one hard requirement and its matching field cells. V3 changes this one axis:
each of the same 15 **seen** fictional projects yields two Gold-field probes,
one for current Windows native support and one for current MIT/Apache-2.0
license. Each call sees only the chosen field's facts, including version status.

The model must output exactly one of `satisfied`, `not_satisfied`, `unknown`, or
`conflict`. Empty and `not_checked` inputs are `unknown`; conflicting current
records are `conflict`; legacy records cannot override current records.

Use the same native 7.2B base, fp32io16 kernel, zero/release-V1 resolver State,
greedy decoding and two rounds. Freeze and push inputs before the independent
GPU3 run; preserve V1/V2 and the old queue. Score 30 distinct conditions, 120
raw records. This is an already seen Gold-field development diagnostic, not
actual extraction, retrieval, production prompt replay or a held-out test.
