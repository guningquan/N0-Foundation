#!/usr/bin/env bash
# Watchdog for one eval.sh run: if the client log has not changed for STALL_MIN minutes (the sim hung,
# e.g. inside the UIPC gel solver), kill the run, record the hung seed, and relaunch from the next seed.
#   usage: eval_watchdog.sh <variant> <task> <gpu> <port> <tag_suffix> <end_seed> <pgid>
#   env:   STALL_MIN (20)  CKPT_STEP (20000)
# Each relaunch uses TAG_SUFFIX <tag_suffix>_r<k>, so its logs and result directory are separate; hung
# seeds go to logs/eval/<tag>_hung_seeds.txt. Exits when the watched run ends on its own.
set -uo pipefail
source "$(dirname "$0")/tasks.sh"
VARIANT=$1; TASK=$2; GPU=$3; PORT=$4; SUFFIX=$5; END_SEED=$6; PGID=$7
STALL_MIN=${STALL_MIN:-20}
SHORT=$(task_short "$TASK"); BASE_TAG=${VARIANT}_${SHORT}${SUFFIX}
LOG_DIR=$SIM_RUNS/logs/eval; HUNG=$LOG_DIR/${BASE_TAG}_hung_seeds.txt
tag=$BASE_TAG; k=0
log() { echo "[$(date '+%F %T')] $*"; }
log "watching $tag (pgid $PGID, gpu $GPU, end seed $END_SEED, stall ${STALL_MIN} min)"
while true; do
  sleep 60
  pgrep -g "$PGID" >/dev/null || { log "$tag ended on its own"; exit 0; }
  client=$LOG_DIR/${tag}_client.log
  [[ -f $client ]] || continue                       # server still loading
  age=$(( $(date +%s) - $(stat -c %Y "$client") ))
  (( age < STALL_MIN * 60 )) && continue
  seed=$(tr '\r' '\n' < "$client" | grep -oP 'Setting seed: \K[0-9]+' | tail -1)
  log "$tag: client log idle ${age}s at seed ${seed:-?}; killing pgid $PGID"
  kill -- -"$PGID" 2>/dev/null; sleep 15; pkill -9 -g "$PGID" 2>/dev/null; sleep 5
  [[ -n $seed ]] || { log "no seed found in $client; giving up"; exit 1; }
  echo "$seed $(date '+%F %T') $tag" >> "$HUNG"
  (( seed >= END_SEED )) && { log "hung seed was the last one; done"; exit 0; }
  k=$((k + 1)); tag=${BASE_TAG}_r$k
  SERVER_GPU=$GPU CLIENT_GPU=$GPU PORT=$PORT START_SEED=$((seed + 1)) TOTAL=$((END_SEED - seed)) \
    TAG_SUFFIX=${SUFFIX}_r$k CKPT_STEP=${CKPT_STEP:-20000} \
    setsid bash "$(dirname "$0")/eval.sh" "$VARIANT" "$TASK" > "$LOG_DIR/${tag}_driver.log" 2>&1 < /dev/null &
  sleep 5
  PGID=$(ps -o pgid= -p $! | tr -d ' ')
  log "relaunched as $tag from seed $((seed + 1)) (pgid $PGID)"
done
