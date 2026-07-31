# Creating an Insertion Task from Mesh Assets

This guide walks through turning a URDF/mesh asset into a simulation-ready insertion task (a peg inserted into a socket with a hole), from asset preparation to data collection. USB is used as the running example; the pipeline applies to any peg/socket pair. See `envs/insert_HDMI.py` and `envs/insert_USB.py` for reference implementations.

## Prerequisites

- Objects are simulated as UIPC rigid/soft bodies on **tetrahedral (tet) meshes**. A usable USD must carry four attributes on `/<name>/mesh` — `tet_points / tet_indices / tet_surf_points / tet_surf_indices` — at **real-world scale (meters)**, as a **single rigid body** (URDF joints are not supported).
- Tet meshes are computed with **fTetWild** (via the `wildmeshing` package), which robustly converts arbitrary triangle surfaces into tet volumes using an error envelope and winding numbers, tolerating dirty or non-watertight input. Tets are **baked offline into the USD**; otherwise they are computed at load time, which can take minutes for dense meshes.
- Environment: `conda activate UniVTAC` (sets up libcuda and the EULA; Isaac will not start without it).

## Pipeline

**1. Select the part.** Parse the asset's `result.json` and remove all articulated parts (UIPC does not support joints), keeping only the body to be inserted.

**2. Rebuild a simplified solid (recommended).** Compose boxes with `trimesh` instead of cleaning up scanned meshes:
- **Peg**: body + rectangular plug via boolean union, **plug facing −Z**, centered, converted mm→m (adapt dimensions in `scripts/asset_tools/build_usb.py`). Use real-world standard dimensions.
- **Socket**: outer block with a hole and lead-in chamfer; hole = plug cross-section + **per-side clearance (must be > 0.5 mm; 1.5 mm recommended — smaller clearances jam)** (`scripts/asset_tools/build_usb_slot.py`).

**Checkpoint 1** — `python scripts/view_obj.py /tmp/asset/PEG.obj` (orbit with the mouse). Verify shape, dimensions, and hole alignment before continuing.

**3. Convert to USD and bake tets.**

```bash
# OBJ -> surface USD (SKIP_TET=1 skips built-in tetgen; -i accepts a directory; --headless for no UI)
SKIP_TET=1 python scripts/convert.py -i /tmp/asset -o assets/objects \
  --collision-approximation convexDecomposition --headless

# Bake tets -> npz (one process per asset; args: <in.obj> <out.npz> <edge_length_r> <epsilon>)
PY=$(conda run -n UniVTAC which python)
$PY scripts/asset_tools/bake_tet.py /tmp/asset/PEG.obj  /tmp/PEG_tet.npz  0.026 0.001
$PY scripts/asset_tools/bake_tet.py /tmp/asset/SLOT.obj /tmp/SLOT_tet.npz 0.055 0.001

# Write tets into the USD (uses Isaac's bundled pxr; args: <target.usd> <tet.npz>)
EXT=<conda-env>/lib/python3.10/site-packages/isaacsim/extscache/omni.usd.libs-*
PYTHONPATH=$EXT LD_LIBRARY_PATH=$EXT/bin:$EXT/pxr/.libs \
  $PY scripts/asset_tools/write_tet_to_usd.py assets/objects/PEG.usd /tmp/PEG_tet.npz
```

Two tet parameters: `edge_length_r` controls **tet count (simulation cost)**; `epsilon` controls **surface smoothness** (tighten toward 0.001 if the surface looks rough).

**Checkpoint 2** — `python scripts/view_usd.py assets/objects/PEG.usd` (requires a display). Verify the in-simulation appearance: smooth surface, sufficiently fine tets.

**4. Build the task.** Copy an existing `envs/insert_*.py` and adapt asset names and dimension constants (`create_actors`, `_reset_actors`, `pre_move`, `_play_once`, `check_success`). Copy a `task_config/*.yml` as the config (video recording requires `video_frequency > 0` plus camera/tactile observations).

> Recording starts from the "already grasped" state: grasping happens in `pre_move` (reset phase, not recorded); only `_play_once` is recorded.

**5. Motion planning.** `place_actor` aligns the object's local +Z with the hole's +Z, so the plug automatically faces down. For a long hand-held peg, set `self.planner_ignore_actors = {'peg'}` in the task — otherwise the planner treats the held object as an obstacle and planning always fails.

**Checkpoint 3** — run the task end to end:

```bash
pkill -9 -f collect_data.py     # clear stale instances (multiple Isaac processes contend for the GPU)
python scripts/collect_data.py <task> <cfg>   # <task> = module under envs/, <cfg> = YAML under task_config/
# Set DISPLAY for a live GUI; render_frequency: 0 in the cfg = headless recording; max_seed: 0 = single trial
# Video: data/<task>/<cfg>/video/0_success.mp4 (records _play_once only)
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `errno=28 / No space left on device` | inotify limit (not disk): `sudo sysctl -w fs.inotify.max_user_watches=1048576` |
| `Could not load libcuda.so` | `conda activate UniVTAC` was not run |
| Loading stalls for minutes with idle GPU | Tets not baked into the USD → redo step 3 |
| Rough / wavy surface | `epsilon` too loose → rebake with 0.001 |
| Insertion jams | Hole clearance too small (> 0.5 mm required) / misalignment / insufficient downward push |
| `Arm motion planning failed` | Held peg treated as an obstacle → `planner_ignore_actors` |
| `World is not valid` | Initial IPC penetration: do not pre-insert the peg or overlap it with the gripper |
| Tet baking `double free` | Run one process per asset |
| pxr `ImportError: libtf.so` | Point `LD_LIBRARY_PATH` at the extscache `bin/` directory |

## Script Reference

| Script | Purpose |
|---|---|
| `scripts/asset_tools/build_usb.py` | Generate a simplified peg (template; adapt dimensions) |
| `scripts/asset_tools/build_usb_slot.py` | Generate a socket with hole (hole = plug + clearance + chamfer) |
| `scripts/asset_tools/bake_tet.py` | Bake tets (fTetWild) → npz |
| `scripts/asset_tools/write_tet_to_usd.py` | Write tets into a USD |
| `scripts/view_obj.py` / `view_usd.py` / `view_task.py` | Inspect an OBJ / a USD / a task scene |
| `scripts/convert.py` / `collect_data.py` | OBJ→USD conversion / run tasks, collect data, record video |
