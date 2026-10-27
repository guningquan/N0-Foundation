#!/usr/bin/env bash
# S3 + S4 + S6: formal collection (3 tasks x 100, GPUs 2,3 by default), then per task in the background:
# LeRobot conversion + norm stats (convert.sh) and 1A contact statistics (stats_1a.sh), CPU only.
#   usage: formal_data.sh [workers (2)] [gpus (2,3)]   -- one worker per GPU, see collect.sh
# Safe to re-run: collect.sh resumes each task from the episodes already saved.
set -uo pipefail
cd "$(dirname "$0")"
WORKERS=${1:-2}; GPUS=${2:-2,3}
LOG=../logs/formal; mkdir -p "$LOG"
for TASK in dual_bowl_place_stack dual_cup_handover_place dual_screw_sleeve; do
  START=$(date +%s)
  bash collect.sh "$TASK" "$GPUS" "$WORKERS"
  echo "[formal_data] $TASK collected in $(( $(date +%s) - START ))s"
  ( bash convert.sh "$TASK" && bash stats_1a.sh "$TASK" ) > "$LOG/prep_$TASK.log" 2>&1 &
done
wait
echo "[formal_data] all done"
