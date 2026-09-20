# Complete-article regression corpus restoration

Authorized after the user confirmed the old server corpus was deleted.
FineWiki upstream: https://huggingface.co/datasets/HuggingFaceFW/finewiki
Revision: `8bd13e72e6a002407649b3e898535f42ceb1aeb9`.

`download.py` saves the upstream tree and verifies all five Chinese parquet
shards against upstream SHA256. `select.py` keeps complete original records
for test article titles and simplified/traditional lookup variants, plus all
5,000 previous background records. All versions are retained; only identical
canonical rows are deduplicated with all upstream positions recorded. Answers
are not used to choose a source version. Selected-v2 adds multi-object titles
that the initial selection omitted; the initial selection is retained locally.

`supplement.py` saves four absent articles from Wikipedia at pinned revisions:
CPUID 93517304, センラ 94343040, 成县 86994369, 蒙古航空 87726745.
Raw API responses, HTML, wikitext and Markdown with conversion metadata remain
available. The conversion uses markdownify 1.2.3 in an isolated tools environment.
These sources are new frozen supplements, not the deleted original artifacts.

`build_index.py` uses the unchanged application chunker, verifies all 58,594
indexed spans against complete originals, and creates only its own index,
database and file registrations. The first build attempt failed at the read-only
OpenSearch existence check (positional API argument); the second uses keyword
arguments and accepts only an identical preexisting settings file. No partial
index was discarded. Final corpus: 5,577 version records / 5,429 article IDs.

Provenance and licenses are in each manifest entry (Wikipedia contributors,
CC-BY-SA-4.0; FineWiki dataset authors). Preserve attribution and links when
redistributing. Local shards are intentionally not committed to Git; manifests,
upstream hashes and a compressed selected corpus provide reproducibility.

This reconstruction restores article coverage, not the original corpus size,
rank distribution, deleted index bytes or original benchmark conditions.
