# eval/ — Policy-server evaluation clients (NeoSim's primary path)

Clients here run the simulator locally and query a **remote policy server** over a
WebSocket protocol: each step (or chunk boundary) the client packs observations
(head/wrist cameras + tactile + robot state) and receives an action chunk back.
The vendored client library lives in `remote_infer_lib/`
(`WebsocketClientPolicy`, msgpack-numpy transport; openpi-style).

## Design note (why this exists alongside `policy/`)

`policy/` loads lightweight models (ACT/ViTAL, tens of MB) **in-process** with the
simulator. Large VLA policies cannot take that path, for two hard reasons:

1. **Interpreter environments are mutually exclusive.** The simulator is pinned to the
   Isaac stack (isaacsim 4.5 / isaaclab 0.41 / its own torch), while modern VLA
   inference needs a newer torch + diffusers + model-specific code trees. The two
   dependency sets cannot coexist in one Python environment, so in-process loading
   fails at import time.
2. **Hardware separation.** Simulation runs on a rendering GPU node; multi-GB policy
   checkpoints are served from separate (often remote) GPU servers. A network protocol
   is the only common interface — and most baselines (Pi0.5, XR0, LingBot-VA, giga)
   already ship as openpi-style policy servers.

Given that, the evaluation loop in these clients mirrors the upstream local-eval
harness (`scripts/eval_policy.py`), with exactly one substitution: `policy.eval()` →
a server round-trip. This was deliberate — upstream files stay untouched, and the loop
behavior (success latching, early-stop, step limits) stays aligned with the local path
by construction. There are **no imports in either direction** between `eval/` and
`policy/`; the only shared code is the task environments (`envs/`).

## File map — one client per (server protocol × arms × action space)

Every client shares the same skeleton: the evaluation loop, config loading and
logging live in `_common.py` (`eval_policy_full` / `eval_policy_lite` — client-specific
behavior is injected as `infer_fn` / `reset_fn`). A client differs from the reference
**only** in three adapter concerns:
**obs packing** (cameras/tactile keys), **state encoding** (EE-rot6d / joint dims),
and **action decoding + protocol** (direct-IK vs joint set; streaming kv-cache vs
stateless). See each file's docstring for its exact deviation list, and
`docs/IKExecution.md` for the EE→joint execution path.

| Client | Server protocol | Arms | Action space |
|---|---|---|---|
| `eval_twam_ee_cl.py` **(reference)** | TWAM streaming (kv-cache) | 1 | EE rot6d-10 → direct IK |
| `eval_twam_ee_dual_cl.py` | TWAM streaming | 2 | EE rot6d-20 → direct IK |
| `eval_lbva_ee_cl.py` | LingBot-VA (video-only) | 1 | EE rot6d-10 |
| `eval_lbva_ee_dual_cl.py` | LingBot-VA (video-only) | 2 | EE rot6d-20 |
| `eval_pi05_single_joint_cl.py` | openpi (stateless) | 1 | joint-8, direct set |
| `eval_pi05_dual_joint_cl.py` | openpi (stateless) | 2 | joint-16, direct set |
| `eval_giga_dual_joint_cl.py` | giga (stateless) | 2 | joint-16, direct set |
| `eval_xr0_ee_cl.py` / `eval_xr0_ee_dual_cl.py` | XR0 | 1 / 2 | EE |
| `eval_official_ee.py` | TWAM | 1 | EE via `take_action('ee')` — `run_eval.sh` entry |
| `run_eval.sh` | driver: waits for the server port, runs 20 held-out seeds, tees a log | | |

> The byte-identical pieces (eval loop / config / logging) are consolidated in
> `_common.py`; the remaining per-client functions (obs packing, state encoding,
> action execution, `vta_eval`-style protocol) intentionally stay in each file so
> every published number's exact inference path remains independently auditable.

## Quick start: running a client ↔ server evaluation

1. **Start your policy server** (your model repo's serve entry) with the checkpoint under
   test. It must listen on a WebSocket port and implement the protocol below.
2. **Reach the server from the sim machine.** Same host: nothing to do. Remote server:
   `ssh -N -L <port>:localhost:<port> <server-host>` in a tmux (add `-g` on a gateway
   host to let other LAN machines connect through it).
3. **Run the client:**

```bash
conda activate UniVTAC
python eval/eval_twam_ee_cl.py <task> <task_config> \
    --server_host localhost --server_port <port> \
    --prompt "<training prompt, verbatim>" \
    --start_seed 1000000 --total_num 20 --headless --enable_cameras
```

Worked example — the exact command that reproduces the lift_can 90% baseline:

```bash
python eval/eval_twam_ee_cl.py lift_can demo \
    --server_host 192.168.50.18 --server_port 31610 \
    --prompt "Lift the can" --start_seed 100 --total_num 20 --headless --enable_cameras
```

4. **Results**: `eval_result/<policy>/<task>/<cfg>/<run>/metadata.json` (per seed:
   `result` / `cost_step` / `cost_time` / `rel_pose`) + per-episode videos + `log.log`;
   a `Final Result: N/20` line prints at the end. The client logs the resolved
   `tac_key` and prompt at startup — **check both before trusting any score**.

## Protocol (what the server must implement)

All messages are msgpack dicts over a WebSocket (`remote_infer_lib`).

```python
# inference request, each chunk:
{'obs':           {<camera keys>: uint8 HxWx3, ...},
 'tactile':       {<tactile keys>: uint8 HxWx3, ...},
 'current_state': [float x state_dim],   # EE rot6d10+gripper (single) / 20-dim (dual)
 'prompt':        str}                    # training prompt, verbatim
# response: {'action': ndarray (C, F, H)}
```

Keys — single-arm: `observation.images.top / wrist_l` + `tactile_a / tactile_b`;
dual-arm: `top / wrist_l / wrist_r` + `tactile_ll / tactile_lr / tactile_rl / tactile_rr`.

**Closed-loop re-grounding** (chunk end, streaming servers): the client sends
`{'obs': <video keyframes>, 'tactile': <tactile keyframes>, 'state': <executed action>,
'current_state': ..., 'compute_kv_cache': True, 'imagine': False, 'prompt': ...}`.
A failed kv-cache commit **aborts the episode** (recorded as `error`) — silently
continuing would degrade the policy to open-loop and quietly poison the score.

**Reset** (episode start): `{'reset': True, 'prompt': ...}`.

## Defaults (generic protocol — override only for experiments)

| Knob | Default | Meaning |
|---|---|---|
| `UNIVTAC_TICKS_PER_SLOT` | **2** | sim steps per action slot; 2 = one dataset row (`save_frequency: 2`), matching the training-data cadence |
| tactile key | env `UNIVTAC_TAC_KEY` > task_config `policy_tac_key` > **`gel_particle`** | MUST match the training representation; `demo.yml` pins `rgb` (legacy lines), `repr_demo.yml` pins `gel_particle` |
| `--prompt` | **required** | training prompt verbatim (`tasks.jsonl`); no silent fallback |
| `cfg.step_lim` | per-task, **data-calibrated** (max archived success steps ×1.05) | `EVAL_STEP_LIM` env = explicit override |
| `--start_seed` | 1000000 | held-out range (collection seeds start at 0) |
| `--total_num` | 20 | standard episode count |
| `--exec_horizon` | 0 | execute the full returned chunk |
| `UNIVTAC_TAC_KEYFRAMES` | 0 (all frames) | trim tactile keyframes at re-grounding |
| `UNIVTAC_PANEL` | `rgb_marker` | tactile image tiled into recorded videos (display only) |

## Troubleshooting

- **Binary-Isaac nodes**: Isaac's bundled `websockets` (12.x) shadows the env's and
  lacks the sync-client `ping_interval` kwarg. Fix: prepend a shim directory containing
  only a symlink to the env's `websockets` package — do **not** prepend the whole
  site-packages (its numpy breaks Isaac's prebundled scipy). pip-installed Isaac nodes
  are unaffected.
- **Wrong tactile key / wrong prompt** = silently poisoned scores; both are printed at
  startup — verify them against the training config before trusting numbers.
- `GLFW initialization failed` warnings under `--headless` are expected and harmless.
- Ten consecutive `error` episodes abort the run (is the policy server down?).
