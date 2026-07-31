#!/usr/bin/env bash
# NeoForce training. Usage: bash scripts/train.sh <data_root> [num_gpus] [output_dir]
set -euo pipefail

DATA_ROOT=${1:?"usage: bash scripts/train.sh <data_root> [num_gpus] [output_dir]"}
NUM_GPUS=${2:-8}
OUTPUT_DIR=${3:-outputs/neoforce}
CONFIG=${CONFIG:-configs/neoforce.json}
PORT=${PORT:-29500}

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

torchrun \
  --nnodes=1 --nproc_per_node="$NUM_GPUS" \
  --rdzv_backend=c10d --rdzv_endpoint="127.0.0.1:$PORT" \
  -m neoforce.train \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT" \
  --train-split splits/train.jsonl \
  --eval-split  splits/val.jsonl \
  --force-stats splits/force_stats_train.json \
  --output-dir "$OUTPUT_DIR" \
  "${@:4}"
