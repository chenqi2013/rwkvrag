# Typed funnel v5: fixed-material node validation

User authorized repair. Previous commit b6f39e7e is backed up on GitHub.

Hypothesis: an explicit observed flag, field-specific JSON value types, preserved
units and a source quote reduce field substitution and missing zero/negation.
The variable is the atomic output protocol as a whole (prompt + schema), not a
claim that one isolated word or type annotation caused any improvement.

## Stage A (this freeze)

- Retain all 13 previous failed atomic cases, exact source text. ATOMS.json adds
  concrete field types and manually written expected value/unit/presence.
- Same 7.2B zero State, locally optimized vLLM, greedy top_k=1, seed=11,
  16384 context, native g1j_plain, concurrency 1, output 512 in both arms.
  This output reserve differs from historical 384; both new arms share 512.
- Run v4 atomic prompt/schema and v5 atomic prompt/schema serially for each case.
- Save every exact prompt, wire response, token IDs, parsed result and failure.
- Candidate exact value/type/unit/presence accuracy is reported separately from
  grammar success. Gate: >=12/13 correct, both zero cases and all 4 boolean cases
  correct, no new invented value on the unknown case. Implementer-written and
  reviewed; these are seen regressions, not an independent generalization score.
- Never change frozen inputs, gold, prompts or bindings during execution.
- Failure ends this version's quality claim. Any revision gets a new version.

## Later stages

Freeze task revision and candidate qualification tests separately; retain all 36
previous funnel questions, then all 188 ordinary/comparison regression members.
Model decisions, not Python keyword rules, interpret withdrawal and negation.
Run retrieval and live web separately only after fixed-material evidence passes.
No training or production promotion is authorized by a completed run alone.

## Runtime

Only rwkv-8222 physical GPU3, UUID GPU-a9570da2-547a-c2b3-0cab-7bbdc1a8a8b0.
Separate service port18426/unit rwkvrag-typed-funnel-20260921. Production18440 and
the current SearchReader credentials are untouched. Stop the experimental unit
at the end; preserve its startup configuration and logs.
