"""Package reviewed, separate stage datasets; this never starts GPU training."""
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def main():
    draft = P / 'draft-v6-seeded'
    review_path = P / 'SEEDED-INDEPENDENT-REVIEW.json'
    review = load(review_path)
    assert review['complete'] is True and review['reviewer'] == '/root/v3_semantic_review'
    approved = {d['id']: d for d in review['decisions']}
    assert len(approved) == 2000 and all(d['verdict'] == 'approve' for d in approved.values())
    for name, digest in review['files'].items():
        assert sha((draft / name).read_bytes()) == digest
    out = P / 'release-v1'
    out.mkdir(exist_ok=False)
    summary = []
    for stage, sizes in [('writer', [100, 300, 600, 1400]), ('resolver', [450]), ('planner', [150])]:
        rows = lines(draft / (stage + '.drafts.jsonl'))
        tokens = {r['id']: r for r in lines(draft / (stage + '.tokens.jsonl'))}
        for r in rows:
            d = approved[r['id']]
            assert d['stage'] == stage == r['stage']
            assert d['prompt_sha256'] == r['prompt_sha256'] and d['target_sha256'] == r['target_sha256']
        # Deterministic proportional prefixes: nested subsets before any new model output.
        families = {}
        for row in rows:
            families.setdefault(row['primary_failure_family'], []).append(row)
        for group in families.values():
            group.sort(key=lambda r: sha(('seeded-subset:' + r['id']).encode()))
        ordered = []
        used = {family: 0 for family in families}
        while len(ordered) < len(rows):
            family = max(families, key=lambda f: ((len(ordered) + 1) * len(families[f]) / len(rows) - used[f], f)
                         if used[f] < len(families[f]) else (-1e9, f))
            ordered.append(families[family][used[family]])
            used[family] += 1
        for size in sizes:
            subset = ordered[:size]
            folder = out / f'{stage}-{size}'
            folder.mkdir()
            decisions = [dict(id=r['id'], decision='approve', stage=stage,
                              prompt_sha256=r['prompt_sha256'], target_sha256=r['target_sha256']) for r in subset]
            with (folder / 'train.tokens.jsonl').open('x') as handle:
                for row in subset:
                    handle.write(json.dumps(tokens[row['id']], ensure_ascii=False) + '\n')
            write(folder / 'INDEPENDENT-REVIEW.json', dict(
                provenance='Root transcription/subset of independent decisions; not authenticated signature',
                author='root', reviewer=review['reviewer'], independent_of_author=True,
                parent_review_sha256=sha(review_path.read_bytes()), decisions=decisions,
                scope=review['scope'], limitations=review['limitations']))
            manifest = dict(schema='rwkv_trace_release_v1', training_data_admitted=True,
                checkpoint_sha256='966f3420f833532aae3fb1fd6326533b08d43d23b7b03eaa2f0694a30b64a239',
                stage=stage, train_count=size,
                stage_prompt={'writer': 'writer_prompt_v2', 'resolver': 'binary_query_prompt', 'planner': 'planner_prompt_queries_fields'}[stage],
                prompt_protocol='rwkvos_no_think_complete_no_final_lf' if stage == 'planner' else 'rwkv_g1j_no_think_v1',
                max_sequence_tokens=8192, max_generation_tokens=2048,
                vocabulary_sha256=sha((P.parents[1] / 'assets/rwkv_vocab_v20230424.txt').read_bytes()),
                dev_or_heldout_content_included=False,
                exposure_note='Corrected exposed cases are generation seeds only; these rows use disjoint source material or disclosed fictional records.',
                rows=decisions, files={f.name: sha(f.read_bytes()) for f in folder.iterdir()},
                seed_review_sha256=sha((P / 'correction-seeds-v1/INDEPENDENT-REVIEW.json').read_bytes()),
                training_started=False)
            write(folder / 'RELEASE.json', manifest)
            summary.append(dict(stage=stage, count=size, release=str((folder / 'RELEASE.json').relative_to(P)),
                                sha256=sha((folder / 'RELEASE.json').read_bytes())))
    write(out / 'INDEX.json', dict(final_combined_training_rows=2000, seeds_excluded_from_training_count=17,
          writer_subsets_nested=True, stage_states_separate=True, training_started=False, releases=summary))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
