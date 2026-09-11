#!/usr/bin/env bash
# Run once on rwkv-8222 after GPU3 is released. Never queues, retries or resumes.
set -euo pipefail
set -o noclobber
PROJECT=/home/chase/rwkvrag
PYTHON=/home/chase/chase/RWKV-PEFT/.venv/bin/python
CONFIG=llamaindex-retrieval/eval/reader-v3-gpu3-20260910/PILOT.json
CONFIG_SHA=8a7ad960b1e744cb8cdc26b5f013eb9b07743316ab95f1380ac5fb4b1f882837
RUN=data/experiments/reader-v3-gpu3-20260910
export CUDA_VISIBLE_DEVICES=GPU-a9570da2-547a-c2b3-0cab-7bbdc1a8a8b0
export PYTHONDONTWRITEBYTECODE=1
cd "$PROJECT"
# Explicit host/GPU admission before creating run logs. Does not reserve the GPU.
"$PYTHON" -c 'import sys; sys.path.insert(0,"llamaindex-retrieval/statetune"); from preflight_state import host_guard; host_guard()'
mkdir -p "$RUN"
timeout --signal=TERM --kill-after=15s 1800s "$PYTHON" llamaindex-retrieval/statetune/train_state.py   --config "$CONFIG" --config-sha256 "$CONFIG_SHA" --output "$RUN/train"   > "$RUN/train.console.log" 2>&1
TRAIN_SHA=$("$PYTHON" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$RUN/train/COMPLETED.json")
timeout --signal=TERM --kill-after=15s 900s "$PYTHON" llamaindex-retrieval/statetune/evaluate_state.py   --config "$CONFIG" --config-sha256 "$CONFIG_SHA" --training-run "$RUN/train"   --training-receipt-sha256 "$TRAIN_SHA" --output "$RUN/dev"   > "$RUN/dev.console.log" 2>&1
