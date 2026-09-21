# Typed funnel repair evidence — 2026-09-21

Report: [results and limitations](../../docs/archive/2026-09/typed-funnel-repair-20260921.md).

- `nodes.tar.gz`: all typed v5/v6/v7/v8 paired atomic outputs, task-node variants, requirement events and grammar-isolation probes, and service startup metadata.
- `full-v8.tar.gz`: complete36 input outcomes,2220 calls and implementer diagnostic review.
- Each `*-manifest.json` records archive SHA256 and each original file SHA256. Archive members were read back and compared byte-for-byte via hashes. No historic result was patched.
- Frozen input/prompt/code bindings are in the matching `llamaindex-retrieval/eval/*-20260921` directories and pre-execution commits.
- Real-project questions here replay frozen selected source snapshots; this run did not fetch new web results.

- `full-v10-natural.tar.gz` and `full-v10-ordinary.tar.gz`: complete36 candidate outcomes,2391 calls; no raw answer changed.
- `validation.tar.gz`: structural audits, v10 review, transparent follow-up numeric corrections to both reviews, completed-run summary, browser checks/screenshot and stopped-service evidence.
- Validation:288 backend checks,34 frontend checks;108 rendered answers compared with raw records. No candidate promotion: node budget exhaustions835→0, but Writer length failures1→5.
