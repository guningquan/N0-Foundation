# sim_runs：N₀-VTLA 仿真实验的运行脚本

N₀-VTLA 在 NeoSim（`neosim/`）里做仿真实验用的脚本，以及让 Isaac Sim 4.5 在这台服务器上跑起来的补丁。实验步骤（S0–S8）见 `N₀-VTLA：仿真实验.md`（这台服务器上在 `Programs_server/tactile_research_four_tracks/`，不在本仓库），环境搭建见 `neosim/docs/QuickStart_Server_zh.md`。

| 路径 | 内容 |
|---|---|
| `scripts/tasks.sh` | 任务表和路径（`ROOT`、`RAW_ROOT`、`LEROBOT_ROOT`），其他脚本都会 source 它 |
| `scripts/s0_phase*.sh` | S0：搭建 UniVTAC 环境（TacEx、vcpkg、libuipc、setup.sh 失败后的修复） |
| `scripts/collect.sh`、`collect_all.sh`、`formal_data.sh` | S3：用脚本专家采集演示，可断点续采；`formal_data.sh` 采完每个任务后在后台做转换和统计 |
| `scripts/convert.sh` | S4：HDF5 转 LeRobot，并计算 norm stats |
| `scripts/stats_1a.sh` | S6：1A 的接触统计（只用 CPU） |
| `scripts/train.sh` | S5 / S7：训练 B0-sim / 1A-RULE-sim / 1A-SCORE-sim（`b0`、`1a_rule`、`1a_score`） |
| `scripts/eval.sh`、`compare_eval.py` | S2 / S5 / S8：在 NeoSim 里闭环评测，并比较评测结果 |
| `scripts/train_chain.sh` | 在同两张卡上依次训练几个变体，每个训完只留最后一个 checkpoint，再用 1 张空闲卡评测（总共最多 3 张卡）：`train_chain.sh <task> <train_gpus> <variant...>` |
| `scripts/prune_after_train.sh`、`eval_watchdog.sh` | 训练进程退出后删掉中间 checkpoint；评测卡死（客户端日志 20 分钟不动）时记下种子，从下一个种子重启 |
| `scripts/replay.sh`（`neosim/eval/replay_n0vtla_dual_joint.py`）、`offline_check.sh` / `.py` | 诊断：把专家关节轨迹开环回放进评测客户端；在专家演示上做 teacher-forced 的离线检查 |
| `scripts/joint_spread.py`、`ik_branch_test.py` | 诊断 7 自由度臂零空间分支：演示之间的关节离散度；cuRobo 规划选到的分支（Cup Handover，见 `neosim/task_config/collect_vtla_ikrank.yml`） |
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
