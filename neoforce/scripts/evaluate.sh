#!/usr/bin/env bash
# NeoForce evaluation: force MAE / RMSE and contact mIoU on the test split.
# Usage: bash scripts/evaluate.sh <data_root> <checkpoint> [output_json]
set -euo pipefail

DATA_ROOT=${1:?"usage: bash scripts/evaluate.sh <data_root> <checkpoint> [output_json]"}
CHECKPOINT=${2:?"usage: bash scripts/evaluate.sh <data_root> <checkpoint> [output_json]"}
OUTPUT=${3:-eval_result/neoforce.json}
CONFIG=${CONFIG:-configs/neoforce.json}

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

python -m neoforce.evaluate \
  --checkpoint "$CHECKPOINT" \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT" \
  --split splits/test.jsonl \
  --output "$OUTPUT" \
  "${@:4}"
