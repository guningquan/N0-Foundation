#!/usr/bin/env bash
# S4: convert one task's NeoSim HDF5 episodes to LeRobot and compute its norm stats (N₀-VTLA：仿真实验.md).
#   usage: convert.sh <task> [src_config (collect_vtla)]
#   env:   DATA_ROOT ($LEROBOT_ROOT)  ASSET_ID (task asset), as in train.sh: a re-collection goes to a new
#          root and asset so the old dataset and norm stats (needed by old checkpoints) are kept
set -euo pipefail
source "$(dirname "$0")/tasks.sh"
TASK=$1; SRC_CFG=${2:-collect_vtla}
ASSET=${ASSET_ID:-$(task_asset "$TASK")}; DATA_ROOT=${DATA_ROOT:-$LEROBOT_ROOT}
source /mnt/ssd1/guningquan/miniforge3/etc/profile.d/conda.sh
set +u; conda activate vtla; set -u   # conda activate.d scripts read unset variables
cd "$N0VTLA"
export PYTHONPATH=$N0VTLA
python scripts/convert_neosim_to_lerobot.py \
  --src "$RAW_ROOT/$TASK/$SRC_CFG/hdf5" \
  --dst "$DATA_ROOT/$TASK" \
  --prompt "$(task_prompt "$TASK")" --fps 60 --overwrite --max-episodes 100
# norm stats -> assets/sim_dual_arm_tactile/<asset>/norm_stats.json (train.sh's offline env)
export VTLA_DATASET_PATH=$DATA_ROOT/$TASK VTLA_ASSET_ID=$ASSET
export N0VTLA_DATA_HOME=$N0VTLA/models HF_HOME=$N0VTLA/models/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/compute_norm_stats.py --config-name sim_dual_arm_tactile
ls -l "assets/sim_dual_arm_tactile/$ASSET/norm_stats.json"
