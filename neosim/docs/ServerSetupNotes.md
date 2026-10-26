# Server Setup Notes: NeoSim on a Shared Multi-GPU Server

These notes record what it took to get NeoSim running for the N₀-VTLA simulation experiments on a shared server, on top of [Installation.md](./Installation.md). The machine: Ubuntu 22.04.4, 8× RTX A6000, NVIDIA driver 595.84, no `sudo`, and other users' jobs on most GPUs (only GPUs 2 and 3 were used).

Every step below was run on that machine. The acceptance test at the end (`collect_data.sh insert_hole demo`, one successful episode) passed after these fixes.

## 1. Parts missing from the release

**TacEx source.** `setup.sh` expects the modified TacEx in `third_party/TacEx`, but it is not in this repository. The NeoSim link on the released `n0_VTLA_dual_bowl_place_stack` checkpoint card points to `anlorla/UniVTAC` (branch `NeoSim`), whose `third_party/TacEx` contains the libuipc source as a plain directory:

```bash
git clone --depth 1 -b NeoSim https://github.com/anlorla/UniVTAC.git /path/to/UniVTAC-NeoSim
mkdir -p third_party
cp -r /path/to/UniVTAC-NeoSim/third_party/TacEx third_party/
```

**Assets.** The README link is still a TODO; the assets are on Hugging Face as `NeoteAI/NeoSim-Assets`:

```bash
hf download NeoteAI/NeoSim-Assets --repo-type dataset --local-dir /path/to/NeoSim-Assets
cp -r /path/to/NeoSim-Assets/assets .
cp -r /path/to/NeoSim-Assets/tacex_data/. third_party/TacEx/source/tacex_assets/tacex_assets/data/
```

## 2. `setup.sh` pitfalls, in the order they appear

1. **`conda create -y` deletes an existing env.** Step 2 of `setup.sh` answers "remove the existing environment?" with yes. Install the libuipc toolchain (step 6 below) after `setup.sh`, not before. Without `sudo`, run it with `SKIP_APT=1`.
2. **The RL frameworks upgrade torch.** Isaac Lab's "all rl-frameworks" step installs the newest skrl / stable-baselines3, which pull torch 2.14 (CUDA 13). Restore the pins from `requirements.lock`, used as a constraints file:

   ```bash
   grep -v -E '^-e |@ file://|^torch_scatter @|^robomimic @' requirements.lock > constraints.txt
   pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 \
     --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple -c constraints.txt
   pip install skrl==2.0.0 stable_baselines3==2.8.0 rl_games==1.6.1 rsl-rl-lib==2.3.3 -c constraints.txt
   ```

   If you uninstall the CUDA 13 wheels, force-reinstall the CUDA 12 ones afterwards (`pip install --force-reinstall --no-deps nvidia-cudnn-cu12==9.7.1.26 nvidia-nccl-cu12==2.26.2 ...`, versions as in the lock). The two families share file names such as `nvidia/cudnn/lib/libcudnn.so.9` and `nvidia/nccl/lib/libnccl.so.2`, so the uninstall deletes the CUDA 12 copies.
3. **`isaaclab` core fails to install.** Its dependency `flatdict==4.0.1` needs `pkg_resources`, which setuptools 81+ no longer has:

   ```bash
   pip install "setuptools<81"
   pip install flatdict==4.0.1 --no-build-isolation
   pip install -e /path/to/IsaacLab/source/isaaclab -c constraints.txt
   ```

4. **PATH order.** If your shell profile puts `/usr/local/cuda-XX/bin` or `~/.local/bin` before the conda env, builds pick up the system `nvcc`/`cmake`. Run `export PATH=$CONDA_PREFIX/bin:$PATH` before building.
5. **cuRobo.** Build with a CUDA toolkit that matches torch (`CUDA_HOME=/usr/local/cuda-12.8` for `torch 2.7.0+cu128`). The conda CUDA keeps its headers under `targets/x86_64-linux/include`, where torch's extension builder does not look:

   ```bash
   CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=8.6 pip install -e /path/to/curobo --no-build-isolation -c constraints.txt
   ```

6. **libuipc (`tacex_uipc`).**
   - Toolchain: `conda env update -n UniVTAC --file third_party/TacEx/source/tacex_uipc/libuipc/conda/env.yaml` (CMake 3.26, GCC 11.4, CUDA 12.4).
   - vcpkg: clone and bootstrap it in `$HOME`, then `export CMAKE_TOOLCHAIN_FILE=$HOME/Toolchain/vcpkg/scripts/buildsystems/vcpkg.cmake`.
   - `export VCPKG_OVERLAY_PORTS=$PWD/tools/vcpkg-overlays`. GitHub's archive of tinygltf v2.9.3 (pinned by libuipc's vcpkg baseline) no longer matches the recorded SHA512. The overlay is vcpkg's own 2.9.3 port with only the hash updated; it was checked against an independent download.
   - `tacex_uipc/setup.py` hard-codes `-j4`. Pre-building with the same CMake arguments saves time; the `pip install -e` afterwards only relinks:

     ```bash
     mkdir -p third_party/TacEx/source/tacex_uipc/build && cd third_party/TacEx/source/tacex_uipc/build
     cmake ../libuipc -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DUIPC_BUILD_PYBIND=1 -DUIPC_DEV_MODE=1 \
       -DUIPC_BUILD_GUI=0 -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release \
       -DVCPKG_OVERLAY_PORTS=$VCPKG_OVERLAY_PORTS
     cmake --build . -j32
     ```

   - The conda-forge binutils/gcc activation scripts read unset variables. Scripts that use `set -u` must wrap the activation: `set +u; conda activate UniVTAC; set -u`.
7. **Install all four TacEx packages and align the env with the lock.** If one of the four packages fails, pip installs none of them. `setup.sh` also never installs NeoSim's own dependencies (`transforms3d`, `pytorch_kinematics`, ...), and `requirements.txt` contains conflicting pins, so install the packages the lock lists that are missing or differ, against the same constraints:

   ```bash
   pip install -e third_party/TacEx/source/tacex_assets -e third_party/TacEx/source/tacex \
     -e third_party/TacEx/source/tacex_uipc -e third_party/TacEx/source/tacex_tasks -c constraints.txt
   ```

8. **The final self-check in `setup.sh` cannot pass in plain Python.** `import tacex_uipc` imports `isaacsim.core`, which only exists inside a running app. The real acceptance test is:

   ```bash
   bash collect_data.sh insert_hole demo <gpu> 0 0 1   # one successful episode
   ```

## 3. Driver 595: RTX segfault at startup

**Symptom:** every app with cameras segfaults while creating the hydra engine (`librtx.scenedb.plugin.so`). The Kit log shows `SceneDbContext : Buffer size instance limit -256`.

**Cause:** driver 595.84 (a New Feature Branch driver) reports `maxMemoryAllocationSize = 0xffffffffffffffff`, and Isaac Sim 4.5's RTX scene DB truncates it to a signed 32-bit size. NVIDIA's recommended fix is a Production Branch driver (the authors' node used 580.x).

**Workaround without root:** `tools/vk_clamp_maxalloc` is an opt-in implicit Vulkan layer. It clamps that one property to `0xFFFFFFFF` and passes everything else through.

```bash
bash tools/vk_clamp_maxalloc/install.sh
export VK_CLAMP_MAXALLOC=1
VK_CLAMP_MAXALLOC=1 vulkaninfo 2>&1 | grep -m1 clamp_maxalloc   # [clamp_maxalloc] ... -> 0xffffffff
```

With the layer active, the scene DB limit becomes positive. A headless app with cameras then starts, and data collection and policy evaluation run normally. Without `VK_CLAMP_MAXALLOC=1` the layer does nothing.

## 4. Runtime environment

```bash
conda activate UniVTAC
export OMNI_KIT_ACCEPT_EULA=YES                                  # no interactive EULA prompt
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json  # several Vulkan ICDs are installed
export VK_CLAMP_MAXALLOC=1                                       # driver 595 only, see section 3
```

The `--gpu` argument of `collect_data.sh` and `parallel_collect.sh` becomes `CUDA_VISIBLE_DEVICES`, so it is a physical GPU id. On a shared machine, never leave it at the documented `0`.

Measured on an RTX A6000: about 5.6 sim steps/s with `task_config/collect_vtla.yml`, i.e. 6–8 minutes per dual-arm episode (a little slower with `demo.yml`, which also stores `gel_particle`). Isaac Sim startup takes about 80 s.

## 5. N₀-VTLA evaluation client

- **Files:**
  - `eval/eval_n0vtla_dual_joint_cl.py`: the dual-arm joint-space evaluation loop, derived from `eval/eval_pi05_dual_joint_cl.py`;
  - `eval/remote_infer_lib/n0vtla_zmq_client.py`: the ZMQ + msgpack protocol of N0-VTLA's `scripts/serve_zmq.py`. It can be imported without Isaac Sim.
- **Protocol details:**
  - The four pads are renamed to the server's `*_tactile` keys (`left_tactile`, `right_tactile`, `left_tactile_b`, `right_tactile_b` → tactile views a–d). The server silently drops keys that do not end in `_tactile`.
  - Images are sent as PNG at native resolution, flipped to BGR first, because the server decodes with cv2 and flips back.
  - Every episode starts with a `reset`, which clears the server's tactile baseline.
- **One server per client:** the server keeps a single per-episode tactile baseline.
- **Defaults:**
  - the full 16-step chunk is executed;
  - 2 sim steps per action (`UNIVTAC_STEPS_PER_ACTION`), since the data stores a frame every 2 sim steps;
  - `step_lim` counts sim steps;
  - seeds start at 1,000,000, far from collection seeds;
  - one line per seed is written to `<run_dir>/results.jsonl`, including errored seeds.

```bash
# server (N0-VTLA env, GPU 3)
VTLA_ASSET_ID=n0_dual_bowl_place_stack_norm python scripts/serve_zmq.py --config sim_dual_arm_tactile \
  --ckpt checkpoints/n0_VTLA_dual_bowl_place_stack --addr "tcp://*:5557" \
  --default-prompt "Use both arms to place and stack the bowls"
# client (UniVTAC env, GPU 2)
CUDA_VISIBLE_DEVICES=2 UNIVTAC_HEADLESS=1 EVAL_TAG=released python eval/eval_n0vtla_dual_joint_cl.py \
  dual_bowl_place_stack fasteval --server_port 5557 --prompt "Use both arms to place and stack the bowls" \
  --start_seed 1000000 --total_num 3 --headless --enable_cameras
```

Sanity result with the released checkpoint at 2 sim steps per action: 2 of 3 seeds succeeded, and the failure hit `step_lim`.

**Open question:** `step_lim` was calibrated at 1.05× the slowest expert episode, so 2 sim steps per action leaves the policy little slack. The authors' π0.5 client defaults to 1 sim step per action, which moves the arm twice as fast in simulated time but doubles the action budget. Settle this before formal evaluation, and use the same value for every model compared.

## 6. Collection config

`task_config/collect_vtla.yml` is `collect.yml` with two changes:
- `gel_particle` is removed from the tactile outputs, which roughly halves the episode size (about 280 MB instead of 490 MB per dual-arm episode);
- `save_dir` points to `/mnt/ssd2/guningquan/Dataset/neosim_raw`. Edit it for another machine.

With `parallel_collect.sh`, the collected seeds are the `hdf5/<seed>.hdf5` file names (no `suc_map.txt` is written).
