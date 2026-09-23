# Layered oracle diagnosis V1

## Question and boundary

Find which isolated capability fails on new, explicitly fictional multi-project material:
raw source text → bound facts; gold facts → eligibility/recommendation; gold decision plus selected verbatim evidence → final Writer response. This is a development diagnostic, not a production accuracy estimate, live retrieval test, or proof of an internal model state. The existing 24 frozen GitHub cases remain the release check and are not edited or used as training.

## Fixed controls and labels

- Four author-written scenarios have 2, 3, 4 and 6 projects. All names, source texts, facts and decisions are in `SCENARIOS.json`; they are distinct from the prior fresh GitHub repository set. Project-count, conflicting same-version claims, untested status, missing license, similar project names, and an older-version distractor are covered. The author of the cases is also the initial reviewer; no independent semantic audit is claimed.
- For each scenario, three prompts are prepared before decoding. Extraction receives only raw source text and project names. Comparison receives only author-checked facts and requirements. Writer receives the author-checked decision and selected literal current-version evidence. These are distinct interventions; a pass in one stage does not certify the complete pipeline.
- All 12 cases use the same 7.2B base and native fp32io16 runtime as the current State evaluation. Each case runs zero State and the matching final role-specific trained State in two rounds, reversing arm order in round two. Exact prompts, input tokens, hashes, generated IDs, raw bytes/text, termination and elapsed time are retained. Greedy argmax; extraction limit 1024, comparison 512, Writer 2048 output tokens. These limits differ by stage and must not be pooled into one max-token rate.
- Only the initial State changes within each zero/trained pair. The roles are Resolver for extraction/comparison and Writer for final prose. No new training, no real retriever, no semantic post-processing or answer repair.

## Scores and interpretation

- Extraction: exact tuple `(project, attribute, value, source_id, status)`, true/false positives, false negatives, precision and recall, plus project/source/value/status swap hints. Invalid JSON stays invalid. Each complete tuple is counted once. Gold source membership is author-checked, not independently blinded.
- Comparison: exact partition into eligible/rejected/unresolved, project-level status accuracy and exact recommended set. Invalid/foreign/duplicate names stay failures.
- Writer: raw finish reason, output cap, invalid or malformed citation labels, long-span repetition, and mentioned projects. These are mechanical indicators; semantic fidelity and whether a citation supports a claim require separate raw-text review, not an automatic pass.
- Per-project counts and paired changes must accompany whole-case success. A zero whole-case score is not evidence that no subskill improved. Extraction failure does not stop independent comparison and Writer probes because failures can coexist.
- No predetermined 95–98% pass threshold is asserted for this small diagnostic set. Any later production gate needs independently reviewed real material and the existing ordinary-question regression.

## Sequence

1. Validate author gold, source IDs and decision partition; prepare immutable `inputs/cases.jsonl` and `inputs/PINS.json` before model calls.
2. Run the existing frozen evaluation queue without alteration. Run this new paired diagnostic as an independent GPU3 job with its own output directory and pinned model/runtime configuration.
3. Score all 48 raw records, inspect Writer originals, and report limitations. Do not change this version after observing results; create V2 for any fixes or wider probes.

The current architecture rules still apply: no code-generated semantic decision, no source-specific patches, no rewriting or adding citations to RWKV raw output. Writer sees only the explicitly selected literal evidence.
