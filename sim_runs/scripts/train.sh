#!/usr/bin/env bash
# S5 / S7: train B0-sim or 1A-sim on GPUs 2,3 (N₀-VTLA：仿真实验.md), from the N0-VTLA-1A worktree.
#   usage: train.sh b0|1a_rule|1a_score <task> [extra train.sh args, e.g. --num-train-steps=200]
#   env:   DATA_ROOT ($LEROBOT_ROOT)  ASSET_ID (task asset)  EXP_SUFFIX ("")
#          PHASE_A_STEPS (2000)  PHASE_RAMP_STEPS (500)  TRAIN_GPUS (2,3)  BATCH (32)  SAVE_INTERVAL (5000)
# Both variants use the same recipe: batch 32 on 2 GPUs (16/GPU), 20k steps, phase A 2000 + ramp 500.
# train_pytorch.py keeps every saved checkpoint (22.6 GB each; keep_period is not used), hence 5000.
set -euo pipefail
source "$(dirname "$0")/tasks.sh"
VARIANT=$1; TASK=$2; shift 2
SHORT=$(task_short "$TASK")
DATA_ROOT=${DATA_ROOT:-$LEROBOT_ROOT}
ASSET=${ASSET_ID:-$(task_asset "$TASK")}
TRAIN_GPUS=${TRAIN_GPUS:-2,3}
# Both variants train from the N0-VTLA-1A worktree (branch dir1a): with input_res_mode "off" its code
# path is HEAD's, so B0 and 1A differ only in the config, and the main tree (the user's uncommitted
# real-robot edits) is never imported. Norm stats are computed in the main tree (S4, convert.sh).
REPO=$N0VTLA_1A
case "$VARIANT" in
  b0)      export CONFIG_NAME=sim_dual_arm_tactile ;;
  1a_rule) export CONFIG_NAME=sim_dual_1a_rule
           export VTLA_1A_STATS=$N0VTLA_1A/assets/sim_dual_1a_rule/$ASSET/tactile_contact_stats.json ;;
  # 1A-SCORE: the rule statistics plus the frozen scorer (VTLA_1A_SCORER, default assets/sim_dual_1a_score/<asset>/)
  1a_score) export CONFIG_NAME=sim_dual_1a_score
           export VTLA_1A_STATS=$N0VTLA_1A/assets/sim_dual_1a_rule/$ASSET/tactile_contact_stats.json
           export VTLA_1A_SCORER=${VTLA_1A_SCORER:-$N0VTLA_1A/assets/sim_dual_1a_score/$ASSET/contact_scorer.pt} ;;
  *) echo "unknown variant $VARIANT" >&2; exit 1 ;;
esac
# 1A keeps the B0 action space: both configs use the same norm stats
mkdir -p "$REPO/assets/$CONFIG_NAME/$ASSET"
cp -n "$N0VTLA/assets/sim_dual_arm_tactile/$ASSET/norm_stats.json" "$REPO/assets/$CONFIG_NAME/$ASSET/"
cmp "$N0VTLA/assets/sim_dual_arm_tactile/$ASSET/norm_stats.json" "$REPO/assets/$CONFIG_NAME/$ASSET/norm_stats.json"
source /mnt/ssd1/guningquan/miniforge3/etc/profile.d/conda.sh
set +u; conda activate vtla; set -u   # conda activate.d scripts read unset variables
cd "$REPO"
export CUDA_VISIBLE_DEVICES=$TRAIN_GPUS NPROC_PER_NODE=$(echo "$TRAIN_GPUS" | tr ',' '\n' | wc -l)
export VTLA_DATASET_PATH=$DATA_ROOT/$TASK
export VTLA_ASSET_ID=$ASSET
export VTLA_PRETRAINED_CHECKPOINT=$BASE_CKPT
export VTLA_DEFAULT_PROMPT="$(task_prompt "$TASK")"
export VTLA_PHASE_A_STEPS=${PHASE_A_STEPS:-2000} VTLA_PHASE_RAMP_STEPS=${PHASE_RAMP_STEPS:-500}
export EXP_NAME=${VARIANT}sim_$SHORT${EXP_SUFFIX:-} WANDB_MODE=offline
echo "[train] repo=$REPO config=$CONFIG_NAME exp=$EXP_NAME data=$VTLA_DATASET_PATH asset=$VTLA_ASSET_ID" \
     "gpus=$CUDA_VISIBLE_DEVICES phaseA=$VTLA_PHASE_A_STEPS/$VTLA_PHASE_RAMP_STEPS extra=$*"
CHECK_ONLY=1 bash train.sh
bash train.sh --overwrite --batch-size=${BATCH:-32} --save-interval=${SAVE_INTERVAL:-5000} "$@"
