# sim_runs：N₀-VTLA 仿真实验的运行脚本

N₀-VTLA 在 NeoSim（`neosim/`）里做仿真实验用的脚本，以及让 Isaac Sim 4.5 在这台服务器上跑起来的补丁。实验步骤（S0–S8）见 `N₀-VTLA：仿真实验.md`（这台服务器上在 `Programs_server/tactile_research_four_tracks/`，不在本仓库），环境搭建见 `neosim/docs/QuickStart_Server_zh.md`。

| 路径 | 内容 |
|---|---|
| `scripts/tasks.sh` | 任务表和路径（`ROOT`、`RAW_ROOT`、`LEROBOT_ROOT`），其他脚本都会 source 它 |
| `scripts/s0_phase*.sh` | S0：搭建 UniVTAC 环境（TacEx、vcpkg、libuipc、setup.sh 失败后的修复） |
| `scripts/collect.sh`、`collect_all.sh`、`formal_data.sh` | S3：用脚本专家采集演示，可断点续采；`formal_data.sh` 采完每个任务后在后台做转换和统计 |
| `scripts/convert.sh` | S4：HDF5 转 LeRobot，并计算 norm stats |
| `scripts/stats_1a.sh` | S6：1A 的接触统计（只用 CPU） |
| `scripts/train.sh` | S5 / S7：训练 B0-sim / 1A-sim |
| `scripts/eval.sh`、`compare_eval.py` | S2 / S5 / S8：在 NeoSim 里闭环评测，并比较评测结果 |
| `constraints_neosim.txt` | UniVTAC 环境的 pip 版本约束 |
| `vk_layer/` | Vulkan 补丁层的源码和 manifest：驱动 595 报告的 `maxMemoryAllocationSize` 是 UINT64_MAX，会让 Isaac Sim 4.5 崩溃，这一层把它截断 |
| `vcpkg-overlays/tinygltf/` | tinygltf 2.9.3 的 vcpkg overlay port（换成当前的源码包哈希） |

`logs/`、`videos/` 和编译出的 `.so` 不入库。

## 换机器时

- **路径写死了**：`scripts/tasks.sh` 和 `s0_phase*.sh` 里的 `ROOT=/mnt/ssd2/guningquan/Programs_server`，以及数据目录，要先改掉。脚本默认 `sim_runs` 就在 `$ROOT/sim_runs`。
- **Vulkan 补丁层**：先编译出 `.so`：
  ```bash
  gcc -shared -fPIC -O2 -o vk_layer/libVkLayer_n0vtla_clamp_maxalloc.so vk_layer/clamp_maxalloc.c -lpthread
  ```
  然后把 `vk_layer/VkLayer_n0vtla_clamp_maxalloc.json` 拷到 `~/.local/share/vulkan/implicit_layer.d/`，把里面的 `library_path` 改成 `.so` 的绝对路径。运行 Isaac Sim 时设置 `VK_CLAMP_MAXALLOC=1`。只有驱动报告 UINT64_MAX 的机器才需要这一层。
