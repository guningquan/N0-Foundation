#!/usr/bin/env bash
# Teacher-forced check of a trained policy on expert demos (offline_check.py).
#   usage: offline_check.sh b0|1a_rule <task> [gpu (3)] [port (5570)] [seeds ("0 1 2")]
set -euo pipefail
source "$(dirname "$0")/tasks.sh"
VARIANT=$1; TASK=$2; GPU=${3:-3}; PORT=${4:-5570}; SEEDS=${5:-0 1 2}
SHORT=$(task_short "$TASK"); ASSET=$(task_asset "$TASK"); CKPT_STEP=${CKPT_STEP:-20000}
case "$VARIANT" in
  b0) CONFIG=sim_dual_arm_tactile; CKPT=$N0VTLA_1A/checkpoints/sim_dual_arm_tactile/b0sim_$SHORT/$CKPT_STEP ;;
  1a_rule) CONFIG=sim_dual_1a_rule; CKPT=$N0VTLA_1A/checkpoints/sim_dual_1a_rule/1a_rulesim_$SHORT/$CKPT_STEP
           export VTLA_1A_STATS=$N0VTLA_1A/assets/sim_dual_1a_rule/$ASSET/tactile_contact_stats.json ;;
esac
LOG=$SIM_RUNS/logs/eval/offline_${VARIANT}_$SHORT; source /mnt/ssd1/guningquan/miniforge3/etc/profile.d/conda.sh
set +u; conda activate vtla; set -u
( cd "$N0VTLA_1A" && export N0VTLA_DATA_HOME=$N0VTLA_1A/models HF_HOME=$N0VTLA_1A/models/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && \
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$N0VTLA_1A VTLA_ASSET_ID=$ASSET \
  exec python scripts/serve_zmq.py --config "$CONFIG" --ckpt "$CKPT" --addr "tcp://*:$PORT" --default-prompt "$(task_prompt "$TASK")" ) > ${LOG}_server.log 2>&1 &
SERVER_PID=$!; trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
set +u; conda deactivate; conda activate UniVTAC; set -u
python "$(dirname "$0")/offline_check.py" "$RAW_ROOT/$TASK/collect_vtla/hdf5" $SEEDS --port "$PORT" \
  --prompt "$(task_prompt "$TASK")" --out ${LOG}.jsonl
