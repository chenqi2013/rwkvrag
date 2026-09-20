# Broad existing-question application regression

User requirement: do not remove, disable or narrow existing questions to improve
conflict/comparison scores. This suite preserves original case rows and origins.
Historical input artifacts are retained in a separate hash ledger; new tests are
additions. No existing fixtures, question text, gold labels or application flags
are edited. No train/holdout claim is inferred from old filenames.

## Scope frozen before requests

Run both historical 200-question diverse Wiki sets, the 6 basic and 20 diverse
questions, and 8 native Wiki questions through the current application's actual
knowledge-base-only /v1/ask endpoint. Preserve user history when present.
Also run all 8 writer holdout, 16 task-matrix validation/holdout and 20 atomic
fixtures through /v1/material-ask. These are explicitly fixed-material Writer
checks, not retrieval/Reader checks. Atomic fixture questions are mechanically
derived from their original object/attribute/conditions; originals are retained.
Three historical fixed-material cases have zero materials. Preserve their empty
lists: the public API currently requires at least one material, so a 422 rejection
is an API-contract coverage failure, not a model semantic refusal. Do not insert
dummy evidence, skip them, or silently reroute them to retrieval.

478 planned application requests: 434 actual-retrieval plus 44 fixed-material.
No new prompt, State, model, routing, question filter or repair setting is applied.
The endpoint keeps its existing internal behavior; any internal retries/repairs
are observable application behavior, not silently disabled by this harness.
The harness uses concurrency 2, timeout 240 seconds, and no request retries.
All failures remain in the denominator. Failed requests do not stop remaining
questions; persistent service outage is visible rather than replaced with answers.

Capture raw request/response, latency, original fixture and source file hashes.
Record nonsecret app settings hash, model identity and index alias before/after.
Index alias drift or settings drift prevents a stable-runtime claim; it does not
delete the affected questions or overwrite results. /v1/ask's normal history
recording is retained, so these executions also remain inspectable in the app.

## Analysis limits

Report execution status, per-suite/type/style totals, empty answers, generation
statuses, refusal text indicators, citation presence/range, and expected source
or reference-term diagnostics. These are automated diagnostics, not independent
semantic accuracy or citation-entailment certification. Preserve the raw answers
for review. Existing gold evidence/facts are retained, not rewritten to match
answers. Do not use a general LLM grader to convert uncertain cases to passed.

No candidate has been deployed. This is the current application's broad baseline,
not a before/after claim against the isolated 7.2B Reader experiment. Reader
presentation gets a separate broader paired experiment with frozen scripts/data.
Both halves must be reported; success on one does not cancel failures on the other.
