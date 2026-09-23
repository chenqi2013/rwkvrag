#!/usr/bin/env bash
# Run the three already-frozen paired evaluation shards after the fresh shard.
set -euo pipefail

root=/home/chase/rwkvrag/data/experiments/state-progression-20260922
python=/home/chase/GitHub/RWKV-LH/data/runtime/engines/vllm-rwkv-67f0c5996c50/.venv/bin/python
fresh_unit=rwkvrag-state-fresh-v1-eval.service
fresh_out="$root/eval-release-v1-fresh-github"

while systemctl --user is-active --quiet "$fresh_unit"; do
    sleep 20
done
test "$(systemctl --user show "$fresh_unit" -p Result --value)" = success
test ! -e "$fresh_out/FAILED.json"
"$python" - "$fresh_out" <<'PY'
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
summary = json.loads((root / 'SUMMARY.json').read_text())
if summary['records'] != 96 or summary['planned_records'] != 96:
    raise SystemExit('Fresh shard incomplete; evaluation queue stopped')
PY

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-a9570da2-547a-c2b3-0cab-7bbdc1a8a8b0
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

run_shard() {
    local name=$1
    local digest=$2
    "$python" "$root/source-v2/evaluate.py" \
        --config "$root/source-v2/TRAINING-v2.json" \
        --training "$root/train-release-v1" \
        --cases "$root/eval-release-v1-inputs/$name.jsonl" \
        --cases-sha256 "$digest" \
        --vocab "$root/evaluation-assets-v1/rwkv_vocab_v20230424.txt" \
        --vocab-sha256 8324476023347dec2964625ccb2075c864d250a9c6d9a74f36daba628de8c008 \
        --out "$root/eval-release-v1-$name"
}

run_shard seen-188 6fe80a4845fb667eff540d306fea528e321fe7783d01b3c4cfe69932d11f1a2d
run_shard seen-retrieval-and-handoff 8d9623b7512750cd9bf2f32d9ec97551d28dc2886af5a9edd2e5d3f69acaa913
run_shard source-separated-dev-holdout 99596d759e6269759a2d0b03bec3d665acb33c24182ea363dbe2d37ac6fef81f
