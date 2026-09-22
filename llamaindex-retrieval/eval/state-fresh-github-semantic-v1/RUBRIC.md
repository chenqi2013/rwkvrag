# New GitHub fixed-evidence semantic review (frozen before model output)

Input membership is the 24 cases and pinned official README prefixes in
`../state-fresh-github-20260922-v1/`. The reviewer must use those exact input
sources, not current websites or background knowledge. This rubric supplies no
training target. It applies to both zero-State and trained-State replies in both
rounds. The reviewer is the implementer; this is not independent adjudication.

For each reply, record `full`, `partial`, `wrong`, or `runtime_failure`, with
separate notes for a missing requested project/criterion, unsupported claim,
wrong source citation, misread limitation, answer repetition, and output cap.
`full` requires every requested project and criterion, source support for each
material claim, citations to the correct input source, and no unsupported
superiority or feature-parity assertion. `partial` means useful supported facts
but at least one requested point missing. `wrong` includes a material factual
error, invented evidence, misattributed citation, or an answer that does not
address the question. A syntactically valid citation or normal stop is never
enough for `full`. Preserve every raw answer; a reviewer label never repairs it.

## Case-specific checks

| Case prefix | Required checks |
| --- | --- |
| `http_clients-comparison-01` | Cover Requests, HTTPX, aiohttp, and urllib3 separately; distinguish explicit sync/async claims from README silence. |
| `http_clients-comparison-02` | Make the async + possible sync requirement explicit; identify README support for HTTPX's two APIs and aiohttp's async client; do not infer unsupported sync APIs. |
| `http_clients-comparison-03` | Identify only README-explicit HTTP/2 support, then assess async from the same evidence; silence is not proof of lack. |
| `http_clients-comparison-04` | Check whether these README excerpts actually establish a direct urllib3–Requests relationship; do not supply a dependency claim from background knowledge. Refuse a universal speed ranking without comparable measurements. |
| `http_clients-ordinary-01` through `04` | State the named project's README-defined purpose and relevant features; cite its own source and do not borrow a neighbor project's claims. |
| `analytics_stack-comparison-01` | Cover pandas, Polars, DuckDB, and Arrow; separate DataFrame/data analysis, SQL analytical database, and columnar data format claims. |
| `analytics_stack-comparison-02` | Address CSV, SQL, and continued Python use as separate requirements; distinguish direct README evidence from integration details requiring verification. |
| `analytics_stack-comparison-03` | Address each project's explicit Python DataFrame API statement, without assigning the same API to all four. |
| `analytics_stack-comparison-04` | Reject an unmeasured fastest-on-the-user's-data conclusion; distinguish project performance claims from a controlled local benchmark. |
| `analytics_stack-ordinary-01` through `04` | Give only the named project's README position and uses, with its own citation. |
| `python_quality-comparison-01` | Cover Ruff, Black, isort, and mypy separately across formatting, import ordering, and static type checking. |
| `python_quality-comparison-02` | Give a source-grounded combination for the three requirements and name the compatibility/coverage questions that still need testing. |
| `python_quality-comparison-03` | Test Ruff versus Black, isort, and mypy separately; do not turn overlap or a project claim into complete equivalence. |
| `python_quality-comparison-04` | Distinguish code style tools from static typing, cover all four, and avoid an unmeasured speed conclusion. |
| `python_quality-ordinary-01` through `04` | State the named project's README position and uses with its own citation; no feature borrowing. |

Compare zero and trained replies within the same case and round first. Report
counts for full/partial/wrong/runtime failure by ordinary versus comparison,
both rounds separately; list every case newly worsened by trained State. A
case counts as a stable gain only if trained State improves without introducing
a wrong claim in either round. These 24 cases test fixed supplied evidence;
they do not test search, source recall, current web accuracy, or deployment.
