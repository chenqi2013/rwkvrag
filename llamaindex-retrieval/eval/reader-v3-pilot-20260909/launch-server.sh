#!/usr/bin/env bash
# Run once on rwkv-8222 after GPU2 is released. Never queues, retries or resumes.
set -euo pipefail
set -o noclobber
PROJECT=/home/chase/rwkvrag
PYTHON=/home/chase/chase/RWKV-PEFT/.venv/bin/python
CONFIG=llamaindex-retrieval/eval/reader-v3-pilot-20260909/PILOT.json
CONFIG_SHA=0c856d856e28e53cd01d7bc00611db1abb8311f03b7667904415d1ceb23920da
RUN=data/experiments/reader-v3-pilot-20260909
export CUDA_VISIBLE_DEVICES=GPU-5d61943c-0955-e221-92a8-318915f5a3a0
export PYTHONDONTWRITEBYTECODE=1
cd "$PROJECT"
# Explicit host/GPU admission before creating run logs. Does not reserve the GPU.
"$PYTHON" -c 'import sys; sys.path.insert(0,"llamaindex-retrieval/statetune"); from preflight_state import host_guard; host_guard()'
mkdir -p "$RUN"
timeout --signal=TERM --kill-after=15s 1800s "$PYTHON" llamaindex-retrieval/statetune/train_state.py   --config "$CONFIG" --config-sha256 "$CONFIG_SHA" --output "$RUN/train"   > "$RUN/train.console.log" 2>&1
TRAIN_SHA=$("$PYTHON" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$RUN/train/COMPLETED.json")
timeout --signal=TERM --kill-after=15s 900s "$PYTHON" llamaindex-retrieval/statetune/evaluate_state.py   --config "$CONFIG" --config-sha256 "$CONFIG_SHA" --training-run "$RUN/train"   --training-receipt-sha256 "$TRAIN_SHA" --output "$RUN/dev"   > "$RUN/dev.console.log" 2>&1
