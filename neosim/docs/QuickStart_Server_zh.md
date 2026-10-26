# NeoSim 快速使用（本服务器，`n0vtla-sim` 分支）

这份说明写给之后要在这台服务器上用 NeoSim 的任务：环境已经装好、能跑，照着做就能直接用。在别的机器上从头安装，见 [ServerSetupNotes.md](./ServerSetupNotes.md)（英文，记录了全部安装问题和修复）。

本说明的内容都已在 2026-10-26 实际验证过：采集、转换、训练、部署、仿真评测各跑通了一遍。

## 1. 东西都在哪里

| 内容 | 位置 | 是否在 git 里 |
|---|---|---|
| NeoSim 代码 | `/mnt/ssd2/guningquan/Programs_server/N0-Foundation/neosim` | 是。分支 `n0vtla-sim`，远程 `fork` = `git@github.com:guningquan/N0-Foundation.git`；`origin` 仍是官方仓库 |
| Python 环境 | conda 环境 `UniVTAC`（`/mnt/ssd1/guningquan/miniforge3/envs/UniVTAC`）：Python 3.10、Isaac Sim 4.5、Isaac Lab 2.1.1、cuRobo、TacEx/libuipc、torch 2.7.0+cu128 | — |
| Isaac Lab、cuRobo 源码（可编辑安装） | `/mnt/ssd2/guningquan/Programs_server/sim_deps/` | 否 |
| 改版 TacEx（含 libuipc 及编译产物） | `neosim/third_party/TacEx/`，来自 `anlorla/UniVTAC` 的 `NeoSim` 分支 | 否 |
| 资产 | `neosim/assets/`，以及 `third_party/TacEx/.../tacex_assets/data/`，来自 HF `NeoteAI/NeoSim-Assets` | 否 |
| 驱动 595 的 Vulkan 补丁层 | 已注册：`~/.local/share/vulkan/implicit_layer.d/VkLayer_n0vtla_clamp_maxalloc.json`，库文件在 `/mnt/ssd2/guningquan/Programs_server/sim_runs/vk_layer/`；源码副本在 `neosim/tools/vk_clamp_maxalloc/` | 源码在 |
| 装包约束文件（锁定 torch 等版本） | `/mnt/ssd2/guningquan/Programs_server/sim_runs/constraints_neosim.txt`，由 `requirements.lock` 生成，生成命令见 ServerSetupNotes 第 2 节 | 否 |
| N₀-VTLA 实验的一键脚本（采集、转换、训练、评测、统计） | `/mnt/ssd2/guningquan/Programs_server/sim_runs/scripts/` | 否 |

## 2. 每次使用前（必做）

```bash
source /mnt/ssd1/guningquan/miniforge3/etc/profile.d/conda.sh
set +u; conda activate UniVTAC; set -u        # 脚本开了 set -u 时必须这样包一下，见下面第三条
export OMNI_KIT_ACCEPT_EULA=YES               # 不弹许可协议确认
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export VK_CLAMP_MAXALLOC=1                    # 必须设置，原因见下面第一条
cd /mnt/ssd2/guningquan/Programs_server/N0-Foundation/neosim
```

- **`VK_CLAMP_MAXALLOC=1` 不能省。**服务器驱动是 595.84，不设这个变量时，任何带相机的仿真启动就会段错误。这个变量打开第 1 节里的 Vulkan 补丁层；不设时补丁层不起作用。
- **只用 GPU 2、3，其他卡上是别人的任务。**采集脚本的 `--gpu` 参数会直接设成 `CUDA_VISIBLE_DEVICES`，填的是物理编号：照官方文档填 `0` 会占到别人的卡。
- **严格模式下要包一层。**环境里的编译器激活脚本会读未定义的变量，所以在开了 `set -u` 的脚本里，`conda activate` 和 `conda deactivate` 都要用 `set +u` 和 `set -u` 包起来。
- **启动慢。**Isaac Sim 每次启动约 80 秒。

## 3. 常用操作

**环境自检：**采到 1 条成功演示即说明环境正常。

```bash
bash collect_data.sh insert_hole demo 2 0 0 1      # 参数依次为：任务 配置 GPU 起始种子 最大种子 条数
```

**采集：**

```bash
bash collect_data.sh <task> <config> 2              # 单进程
bash parallel_collect.sh <task> <config> 2,3 4      # 4 个进程，按 GPU 轮流分配
```

- **任务：**`envs/*.py`。双臂 8 个（`dual_*`），单臂 12 个。
- **配置（`task_config/`）：**
  - `demo`：5 条，含 `gel_particle`；
  - `collect`：100 条；
  - `collect_vtla`：100 条，不含 `gel_particle`，`save_dir` 指向 `/mnt/ssd2/guningquan/Dataset/neosim_raw`；
  - `fasteval`：评测用，存储频率低。
- **输出：**`<save_dir>/<task>/<config>/hdf5/<seed>.hdf5`。并行采集时，已采的种子以 hdf5 文件名为准，不会写 `suc_map.txt`。
- **速度：**双臂任务一条约 6–8 分钟（`collect_vtla` 约每秒 5.6 个仿真步；`demo` 约 3.8 个）。
- **大小：**双臂任务每条约 280 MB（`collect_vtla`）或 490 MB（`demo`）。

**双臂 HDF5 的结构：**

| 键 | 内容 |
|---|---|
| `observation/{head,wrist,wrist_b}/rgb` | 每帧一段 JPEG 字节，原生 270×480。`cv2.imdecode` 解出来就是仿真原始的 RGB 顺序，不要再翻转 |
| `tactile/{left_tactile,right_tactile,left_tactile_b,right_tactile_b}/rgb` | 触觉图，240×320，同样是 JPEG；另有 `rgb_marker`、`depth`、`marker`、`pose` |
| `embodiment/joint`、`embodiment_b/joint` | 两只臂各 T×9：7 个关节 + 2 个手指关节；夹爪取第一个手指关节，即下标 7 |

- 每 2 个仿真步存一帧，数据是 60 Hz。
- 自由空间里触觉图与首帧完全相同，差分为 0。

**转成 LeRobot 数据集**（供 N₀-VTLA、openpi 一类训练用）：用 `N0-VTLA/scripts/convert_neosim_to_lerobot.py`。触觉默认无损编码；状态和动作是 16 维（每臂 7 个关节 + 夹爪）。

**评测（策略服务端 + 仿真客户端）：**

- **N₀-VTLA：**
  - 客户端是 `eval/eval_n0vtla_dual_joint_cl.py`，协议在 `eval/remote_infer_lib/n0vtla_zmq_client.py`，服务端是 `N0-VTLA/scripts/serve_zmq.py`。示例命令见 ServerSetupNotes 第 5 节，一键脚本是 `sim_runs/scripts/eval.sh`。
  - 一个服务端只能接一个客户端，因为服务端每回合只保存一份触觉基准帧。
  - 默认设置：每块执行完整的 16 步，每个动作走 2 个仿真步，种子从 1,000,000 起，逐种子写入 `results.jsonl`。
  - 步数上限按仿真步计算。每个动作走 1 步还是 2 步会影响成功率，正式评测前要先定。
- **其他策略：**
  - 远程服务端：参考 `eval/` 里现有的客户端，有 π0.5、giga、xr0、twam 等，走 WebSocket；
  - 本地加载：按 `docs/Deploy.md`，在 `policy/<名字>/deploy_policy.py` 里实现接口，用 `scripts/eval_policy.py` 评测。

## 4. 不要做的事

- **不要再运行 `setup.sh`。**它的 `conda create -y` 会删掉并重建整个 `UniVTAC` 环境。
- **不要用 `pip install -r requirements.txt`。**这个文件里的版本互相矛盾。
- **装任何包都要加约束文件：**`-c /mnt/ssd2/guningquan/Programs_server/sim_runs/constraints_neosim.txt`。不加的话，pip 可能把 torch 升级到 CUDA 13 版本。
- **不要卸载 `nvidia-*-cu12` 一类的 CUDA 运行库包。**它们和别的包共用文件名，卸载会删掉 torch 需要的库。

## 5. 显存参考（RTX A6000，48 GB）

| 进程 | 每个约占 |
|---|---|
| Isaac Sim 仿真，采集或评测客户端 | 8.5 GB |
| N₀-VTLA 推理服务端 | 8.7 GB |
| N₀-VTLA 训练（每卡 batch 16） | 36 GB/卡 |

- 一张卡可以同时开 4–5 个仿真进程。
- 训练时这张卡基本占满，不要同时在上面跑仿真。
- N₀-VTLA 的一个 checkpoint 约 22.6 GB（含优化器状态），正式训练要注意保存间隔。
