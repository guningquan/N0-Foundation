# Environment Lock

Exact version matrix for a known-good evaluation/collection node. `requirements.lock` at the repo root is the complete `pip freeze` (274 packages) of a working `UniVTAC` conda environment — when this document and the lock file disagree, **the lock file wins**.

One-shot setup: `setup.sh`.

## System layer (hard prerequisites)

| Item | Value | Notes |
|---|---|---|
| OS | **Ubuntu 22.04.5 LTS** (glibc 2.35) | ⚠️ Avoid 20.04: glibc 2.31 forces source builds of several wheels (torch_scatter etc.) |
| GPU / driver | RTX 4090 / NVIDIA **580.x** (≥535 works; minor-compat covers CUDA 12.x) | |
| ffmpeg | 4.4.2 (`sudo apt install ffmpeg`) | apt-level; not covered by pip freeze |
| Toolchain | build-essential / cmake / git | required to compile TacEx (libuipc) |

## conda env `UniVTAC`, Python **3.10.20**

### Core hard constraints (any mismatch breaks the stack)

| Package | Version | Install |
|---|---|---|
| torch | **2.7.0+cu128** | pip, cu128 index |
| torchvision | 0.22.0+cu128 | pip, cu128 index |
| numpy | **1.26.4** | pip |
| isaacsim | **4.5.0.0** (with all `isaacsim-*` extensions) | pip (on 22.04 the pip install works; no binary needed) |
| IsaacLab | **v2.1.1** (commit `90b79bb2d`, editable `source/*`, five extensions) | git clone + `./isaaclab.sh -i` |
| nvidia_curobo | 0.7.8 (NVlabs/curobo `@d64c4b0`, editable) | git clone + `pip -e` |
| warp-lang | 1.12.1 | pip |
| pyuipc | 0.9.0 (**local build**: `third_party/TacEx/source/tacex_uipc/build/python`) | compile libuipc |
| tacex / tacex_assets / tacex_uipc / tacex_tasks | 0.1.0 (**this repo's `third_party/TacEx`**, editable) | `pip -e source/*` |
| torch_scatter | 2.1.2 (pt28cu128 wheel) | special wheel; see setup.sh |
| transformers 5.5.4 / trimesh 4.11.5 / opencv-python 4.11.0.86 / scipy 1.15.3 | | pip |

### Lines in `requirements.lock` that are machine-local paths — do **not** `pip install -r` directly; setup.sh installs them separately:

- `-e git+https://github.com/isaac-sim/IsaacLab@90b79bb2d...` (×5: isaaclab / _assets / _mimic / _rl / _tasks)
- `-e git+https://github.com/NVlabs/curobo.git@d64c4b0...#egg=nvidia_curobo`
- `pyuipc @ file:///.../third_party/TacEx/source/tacex_uipc/build/python` (local build)
- `-e .../openpi-client` (WebSocket policy client for remote inference; only needed to run `eval/`. A vendored copy ships in `eval/remote_infer_lib/`.)

## Verification

```bash
conda activate UniVTAC
python -c "import torch, isaaclab, tacex_uipc; print(torch.__version__)"   # 2.7.0+cu128
python scripts/collect_data.py insert_hole demo   # one successful episode = environment OK
```
