# V8 separate quantity binding

V7 completed13 pairs: 11/13 exact, all core values/booleans/unknown correct, but two count units omitted. It did not pass the full gate.

V8 keeps v7 extraction byte-identical and adds one separately traced model call for every known quantity (not a question/unit-specific branch). The model selects a complete verbatim scalar including physical/count unit. Code validates substring membership and parses decimal syntax; it does not infer the unit. The first raw output is retained and never edited.

Same13 cases/gold/model/State/sampling and v4 paired baseline. Extra call max128 tokens is an explicit latency/cost change; this is an architectural unit-binding test, not a prompt-word comparison. Same gate >=12/13 with both zero cases and allboolean cases correct, unknown not invented. Seen regressions, implementer gold, not independent accuracy. Preserve old failures. No production promotion from node results.
