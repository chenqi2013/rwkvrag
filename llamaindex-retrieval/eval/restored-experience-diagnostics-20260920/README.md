# Post-run experience diagnostics

Read the complete frozen 434-case run after exporting its full evidence bundle:

```sh
python llamaindex-retrieval/eval/restored-experience-diagnostics-20260920/analyze.py
```

Writes a new `EXPERIENCE-DIAGNOSTICS.json` and refuses to overwrite one. Reports
request wall time separately from overlapping model-call queue/active time,
Reader work volume, exact repeated call inputs, repeated source units under
different task groups, and Writer length/budget records.

These are operational observations under four concurrent regression requests,
not semantic accuracy, an interactive latency promise, or proof that removing
calls is safe. The configured context window is not a verified server limit.
No model requests, prompt changes, source changes or answer rewriting occur.
