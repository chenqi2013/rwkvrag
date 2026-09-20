# Restored complete-article retrieval baseline

Frozen before the first request. Preserve all 434 original retrieval questions:
200 verified Wiki, 200 second-batch Wiki, 6 basic, 20 diverse, 8 native/history.
Original questions, history, case records and gold are unchanged. Only the
knowledge-base selection changes to the restored isolated corpus. All requests
use actual `/v1/ask`; no reference answers are supplied as materials.

Corpus: 5,577 complete article/version records, 58,594 verified verbatim nodes.
Keep the existing 5,000 background records plus all matching FineWiki versions
and four explicitly versioned Wikipedia supplements. All 434 questions have
corresponding articles; all 400 old Wiki canonical identities match. This is a
new controlled corpus, not a byte-identical reconstruction of the deleted index.
The 83 nonliteral historical references remain marked; gold is never rewritten.

Main settings are copied with only Mongo database, index and upload directory
changed. Same 2.9B model, Reader450/Writer300, prompts, decoding and budgets.
Application endpoint 18446, model endpoint 18423; concurrency 4, timeout 600 s,
no harness retries. Existing internal model repair behavior remains enabled.
No latency/quality causal comparison against v2's fixed-reference run is claimed.
No source, model, prompt or configuration edits during execution.

`COVERAGE.json` maps original identities to version-specific document IDs.
Source presence is an automated diagnostic, not evidence entailment or answer
accuracy. All failures, length exits, unsupported citations and missing facts
remain recorded. The 48-case review is frozen before outputs: the same 14 Wiki
cases, all 8 native/history and all 26 basic/diverse cases. Review is by the agent,
not independent adjudication, and must not be extrapolated to 434-case accuracy.

```sh
PYTHONPATH=llamaindex-retrieval/src llamaindex-retrieval/.venv/bin/python \
  llamaindex-retrieval/eval/restored-retrieval-v2-20260920/run.py \
  --output data/quality-runs/restored-retrieval-v2-20260920/run1
```

Original artifacts and corpus manifests are hash-bound in `ORIGINS.json`.
Raw calls preserve full responses and model traces. Keep this run separate from
the completed 478-case v2 and the invalid/aborted v1.

V1 could not start because identical article bodies violated file-table content uniqueness. No model requests ran. V2 shares physical files by exact text SHA256, retains all 5,577 article/version records and all 58,594 indexed nodes, and keeps unchanged text hashes and question inputs. New isolated index/database v2; old failed artifacts retained.
