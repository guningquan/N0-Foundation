#!/usr/bin/env bash
# Train one or more variants on the same two GPUs, one after the other, and evaluate each finished model on ONE
# idle GPU (the user's cap: at most 3 GPUs for all of these jobs together). Keeps only the final checkpoint.
#   usage: train_chain.sh <task> <train_gpus> <variant> [<variant> ...]      variants: b0 1a_rule 1a_score
#   env:   DATA_ROOT  ASSET_ID  EXP_SUFFIX (as in train.sh / eval.sh)  TAG_SUFFIX (_formal)  TOTAL (100)
#          START_SEED (1000000)  PORT (5565)  SKIP_GPUS (1,4,7: other people's GPUs)
# An eval GPU must not be in SKIP_GPUS, nor a training GPU while another variant still has to train, and must use
# < 2000 MiB and < 30 %; if none is idle, the script waits for one.
set -uo pipefail
source "$(dirname "$0")/tasks.sh"
TASK=$1; GPUS=$2; shift 2
SHORT=$(task_short "$TASK"); SFX=${EXP_SUFFIX:-}; TAG=${TAG_SUFFIX:-_formal}
log() { echo "[$(date '+%F %T')] $*"; }

idle_gpu() {   # <extra comma-separated GPUs to skip>
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F', ' -v skip=",$1,${SKIP_GPUS:-1,4,7}," 'index(skip, "," $1 ",") == 0 && $2 < 2000 && $3 < 30 {print $1}' |
    head -1
}

evaluate() {   # <variant> <port> <GPUs to skip>: one GPU, one watchdog (an earlier eval may still run)
  local v=$1 port=$2 g
  until g=$(idle_gpu "$3") && [[ -n $g ]]; do log "no idle GPU for the $v eval; waiting"; sleep 600; done
  SERVER_GPU=$g CLIENT_GPU=$g PORT=$port START_SEED=${START_SEED:-1000000} TOTAL=${TOTAL:-100} \
    TAG_SUFFIX=$TAG setsid bash "$(dirname "$0")/eval.sh" "$v" "$TASK" \
    > "$SIM_RUNS/logs/eval/${v}_$SHORT${TAG}_driver.log" 2>&1 < /dev/null &
  local pg=$!
  setsid bash "$(dirname "$0")/eval_watchdog.sh" "$v" "$TASK" "$g" "$port" "$TAG" \
    $(( ${START_SEED:-1000000} + ${TOTAL:-100} - 1 )) "$pg" \
    > "$SIM_RUNS/logs/eval/${v}_$SHORT${TAG}_watchdog.log" 2>&1 < /dev/null &
  log "$v eval on GPU $g (pgid $pg), tag ${v}_$SHORT$TAG"
}

port=${PORT:-5565}
while (( $# )); do
  v=$1; shift
  case $v in
    b0) cfg=sim_dual_arm_tactile ;; 1a_rule) cfg=sim_dual_1a_rule ;; 1a_score) cfg=sim_dual_1a_score ;;
    *) log "unknown variant $v"; exit 1 ;;
  esac
  dir=$N0VTLA_1A/checkpoints/$cfg/${v}sim_$SHORT$SFX
  log "training $v $TASK on GPUs $GPUS -> $dir"
  TRAIN_GPUS=$GPUS bash "$(dirname "$0")/train.sh" "$v" "$TASK" > "$SIM_RUNS/logs/train/${v}_$SHORT$SFX.log" 2>&1 < /dev/null
  [[ -f $dir/20000/model.safetensors ]] || { log "no $dir/20000/model.safetensors; stopping"; exit 1; }
  bash "$(dirname "$0")/prune_after_train.sh" "$dir" "${v}sim_$SHORT$SFX"
  log "$v done"
  sleep 60   # let the GPUs drain
  evaluate "$v" "$port" "$( (( $# )) && echo "$GPUS")"; port=$((port + 1))
done
