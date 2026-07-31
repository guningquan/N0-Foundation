#!/bin/bash
# Eval driver. Usage: TASK=grasp_classify PROMPT="Grasp the object" PORT=29541 TAG=notac_gc_0613 bash run_eval.sh
set +u
TASK=${TASK:?}; PROMPT=${PROMPT:?}; PORT=${PORT:-29541}; TAG=${TAG:?}
OUT=${EVAL_OUT:-./eval_result/runs/$TAG}
mkdir -p $OUT
# Environment (override via env vars for non-default installs)
CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-UniVTAC}
ISAACSIM_SETUP=${ISAACSIM_SETUP:-$HOME/isaacsim/setup_conda_env.sh}
source "$CONDA_SH" && conda activate "$CONDA_ENV" && source "$ISAACSIM_SETUP" 2>/dev/null
# Wait until the server port responds
for i in $(seq 1 60); do timeout 3 bash -c "echo > /dev/tcp/${EVAL_HOST:-localhost}/$PORT" 2>/dev/null && break; sleep 10; done
cd "$(cd "$(dirname "$0")/.." && pwd)"; python eval/eval_official_ee.py $TASK demo \
  --server_host ${EVAL_HOST:-localhost} --server_port $PORT --prompt "$PROMPT" \
  --start_seed 100 --total_num 20 2>&1 | tee $OUT/eval.log
echo EVAL_DONE_$TAG >> $OUT/eval.log
