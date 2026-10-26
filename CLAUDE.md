# N0-Foundation（fork：guningquan/N0-Foundation，分支 n0vtla-sim）

NeoSim 在 `neosim/`，在本服务器上已装好、可直接用。先读 `neosim/docs/QuickStart_Server_zh.md`；从头安装与排错见 `neosim/docs/ServerSetupNotes.md`。

本服务器上必须遵守：

- **只用 GPU 2、3。**其他卡上是别人的任务。`collect_data.sh` / `parallel_collect.sh` 的 GPU 参数会直接设成 `CUDA_VISIBLE_DEVICES`，填物理编号 2 或 3。
- **运行任何仿真前：**
  ```bash
  source /mnt/ssd1/guningquan/miniforge3/etc/profile.d/conda.sh
  set +u; conda activate UniVTAC; set -u
  export OMNI_KIT_ACCEPT_EULA=YES VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json VK_CLAMP_MAXALLOC=1
  ```
  不设 `VK_CLAMP_MAXALLOC=1`，带相机的仿真会在启动时段错误（驱动 595 的问题）。
- **不要运行 `neosim/setup.sh`：**它会删掉并重建 `UniVTAC` 环境。
- **装包：**一律加 `-c /mnt/ssd2/guningquan/Programs_server/sim_runs/constraints_neosim.txt`，保持 torch 2.7.0+cu128；不要用 `pip install -r requirements.txt`。
- **不在 git 里的目录：**`neosim/third_party/TacEx`、`neosim/assets`、`neosim/data`、`neosim/eval_result`。不要把它们加进 git。
- **自检命令：**`bash collect_data.sh insert_hole demo 2 0 0 1`，1 条成功即环境正常；Isaac Sim 启动约 80 秒。
