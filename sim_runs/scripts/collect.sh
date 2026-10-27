#!/usr/bin/env bash
# S3: collect scripted-expert demos for one task on GPUs 2/3 (N₀-VTLA：仿真实验.md).
#   usage: collect.sh <task> [gpus (2,3)] [workers (2)] [task_config (collect_vtla)] [episodes (100)]
# parallel_collect_data.py splits the workers over the listed GPUs. Re-running resumes: episodes already
# in hdf5/ count toward the target, and seeds restart after the largest one tried (hdf5 names, metadata.json).
# One worker per GPU: with 3 per GPU the UIPC reset (PCG, a host sync per iteration) slowed to minutes per
# sim step under GPU time-slicing and hit reset_time_limit; without errors it was only 10-60% faster per GPU.
# --restart-on-error: a worker whose reset timed out stays broken (every later reset times out), so it is
# replaced by a fresh process; collect_vtla.yml also raises short reset limits to 1200 s.
set -euo pipefail
source "$(dirname "$0")/tasks.sh"
TASK=$1; GPUS=${2:-2,3}; WORKERS=${3:-2}; CFG=${4:-collect_vtla}; TARGET=${5:-100}
OUT=$RAW_ROOT/$TASK/$CFG
DONE=$(ls "$OUT/hdf5" 2>/dev/null | grep -c '\.hdf5$' || true)
NEXT=$(python3 - "$OUT" <<'EOF'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
seeds = [int(p.stem) for p in out.glob("hdf5/*.hdf5")]
try:
    seeds += [int(k) for k in json.loads((out / "metadata.json").read_text())]
except (OSError, ValueError):
    pass
print(max(seeds) + 1 if seeds else 0)
EOF
)
if (( DONE >= TARGET )); then echo "[collect] $TASK: $DONE/$TARGET episodes already in $OUT/hdf5"; exit 0; fi
source /mnt/ssd1/guningquan/miniforge3/etc/profile.d/conda.sh
set +u; conda activate UniVTAC; set -u   # conda activate.d scripts read unset variables
cd "$NEOSIM"
export OMNI_KIT_ACCEPT_EULA=YES VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
# driver 595 reports maxMemoryAllocationSize = UINT64_MAX, which crashes Isaac Sim 4.5 RTX: clamp it
# with the opt-in user layer in sim_runs/vk_layer (~/.local/share/vulkan/implicit_layer.d)
export VK_CLAMP_MAXALLOC=1
mkdir -p "$SIM_RUNS/logs/collect"
echo "[collect] $TASK: $DONE/$TARGET done; collecting $(( TARGET - DONE )) more from seed $NEXT, $WORKERS workers on GPUs $GPUS"
CUDA_VISIBLE_DEVICES=$GPUS python scripts/parallel_collect_data.py "$TASK" "$CFG" --workers="$WORKERS" \
  --episodes=$(( TARGET - DONE )) --start-seed="$NEXT" --restart-on-error >> "$SIM_RUNS/logs/collect/${TASK}.log" 2>&1
echo "[collect] $TASK: $(ls "$OUT/hdf5" 2>/dev/null | wc -l) episodes in $OUT/hdf5"
