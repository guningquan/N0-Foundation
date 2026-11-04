#!/usr/bin/env bash
# Open-loop replay of expert demos through the eval client's execution path (N₀-VTLA：仿真实验.md §10.7).
#   usage: replay.sh <task> [gpu (3)] [start_seed (0)] [n (5)]
set -euo pipefail
source "$(dirname "$0")/tasks.sh"
TASK=$1; GPU=${2:-3}; START=${3:-0}; N=${4:-5}
source /mnt/ssd1/guningquan/miniforge3/etc/profile.d/conda.sh
set +u; conda activate UniVTAC; set -u
cd "$NEOSIM"
export CUDA_VISIBLE_DEVICES=$GPU UNIVTAC_HEADLESS=1 UNIVTAC_STEPS_PER_ACTION=2 EVAL_TAG=replay
export OMNI_KIT_ACCEPT_EULA=YES VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json VK_CLAMP_MAXALLOC=1
export EVAL_STEP_LIM=$(task_step_lim "$TASK") REPLAY_HDF5_DIR=$RAW_ROOT/$TASK/collect_vtla/hdf5
python eval/replay_n0vtla_dual_joint.py "$TASK" fasteval --prompt "$(task_prompt "$TASK")" --exec_horizon 16 \
  --start_seed "$START" --total_num "$N" --headless --enable_cameras
