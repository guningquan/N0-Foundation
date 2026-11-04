#!/usr/bin/env bash
# Keep only the final checkpoint of a run (user request 2026-11-01): wait until the training process
# has exited and <final> exists with its model file, then delete the other step directories.
#   usage: prune_after_train.sh <ckpt_dir> <exp_name> [final (20000)]
DIR=$1; EXP=$2; FINAL=${3:-20000}
while pgrep -f "^.*python.* scripts/train_n0vtla.py .*--exp-name=$EXP( |$)" >/dev/null; do sleep 300; done
[[ -f $DIR/$FINAL/model.safetensors ]] || { echo "[$(date '+%F %T')] no $DIR/$FINAL/model.safetensors; not pruning"; exit 1; }
for s in "$DIR"/*/; do s=${s%/}; [[ $(basename "$s") == "$FINAL" ]] || { rm -rf "$s"; echo "[$(date '+%F %T')] removed $s"; }; done
