# Layered oracle V2: seen protocol diagnostics

This uses the four **already seen** fictional scenarios from V1 unchanged. It is
a development diagnostic, not a new held-out test and not a real retrieval run.
All V1 inputs, prompts, raw answers, and scoring remain frozen.

Three independent interventions are evaluated against their matching V1 stage:

1. Extract: ask directly for a JSON array, preserving the same five fields and
   source text. Exact schema and exact fact tuples are separate outcomes.
2. Qualify: give each project only its own Gold facts and hard conditions, then
   request one of three status words. This isolates local judgment from a large
   cross-project decision and removes recommendation from the qualification call.
3. Writer: use the existing `writer_prompt_v2` plus the same kind of compact
   Gold assessment passed by `funnel_final_projection`. This tests whether the
   standard prompt can produce citations; it is **not byte-identical** to a
   production call because the Gold decision replaces the actual Resolver output.

Both zero and release-V1 trained FP32 role states are evaluated twice with the
same native 7.2B base, fp32io16 recurrent kernel, greedy decoding, fixed input
order, and unchanged evaluator. Only an independent service on authorized
physical GPU 3 may run these inputs. Do not modify running historical queues.

Before promotion, require semantic review of every distinct Writer answer,
per-project status accuracy, complete original ordinary-question regression,
and live web + knowledge retrieval on real multi-project queries. Prompt or
format improvements on these seen fictional cases alone cannot be promoted.
