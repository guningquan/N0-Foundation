#!/usr/bin/env bash
# S6: 1A contact statistics (tau, tau_min, P(n_a)) on one task's formal LeRobot dataset, CPU only.
#   usage: stats_1a.sh <task>
#   output: N0-VTLA-1A/assets/sim_dual_1a_rule/<asset>/tactile_contact_stats.json (+ plots in N0-VTLA-1A/logs/)
set -euo pipefail
source "$(dirname "$0")/tasks.sh"
TASK=$1; ASSET=$(task_asset "$TASK")
source /mnt/ssd1/guningquan/miniforge3/etc/profile.d/conda.sh
set +u; conda activate vtla; set -u   # conda activate.d scripts read unset variables
cd "$N0VTLA_1A"
export PYTHONPATH=$N0VTLA_1A N0VTLA_DATA_HOME=$N0VTLA_1A/models HF_HOME=$N0VTLA_1A/models/huggingface \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
python scripts/tactile_contact_stats.py --repo-id "$LEROBOT_ROOT/$TASK" \
  --content-box 28 196 0 224 --n0 10 --top-q 0.05 --pool 4 --tau-floor otsu --num-workers 8 \
  --out "assets/sim_dual_1a_rule/$ASSET/tactile_contact_stats.json" --plot-dir "logs/contact_stats_$ASSET"
