# Table presentation inspection

This directory contains a read-only structural reproduction, not a model evaluation.
It runs the current verbatim chunker and atomic candidate builder on two synthetic
Markdown tables. Assertions verify duplicated header context and an answer row
excluded by the 600-character context budget despite unique parent context fitting.
No inference, prompt repair, deployment or production code changes occur.

```bash
PYTHONPATH=llamaindex-retrieval/src llamaindex-retrieval/.venv/bin/python \
  llamaindex-retrieval/eval/table-presentation-inspection-20260920/inspect_layout.py \
  --output /tmp/table-presentation-inspection.json
```

Use a new output path when reproducing; do not overwrite archived artifacts.
Results and the proposed subsequent experiment are in the
[inspection report](../../../docs/archive/2026-09/table-presentation-inspection-20260920.md).
The known-failure-view artifact is an excerpt from the prior immutable run plus
an untested raw-text display preview; it is not a new model response.
