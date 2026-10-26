#!/usr/bin/env python3
"""Dual-arm JOINT N0-VTLA eval client — derived from eval_pi05_dual_joint_cl.py:
  - obs = 3 cameras head/wrist/wrist_b + 4 tactile pads left/right_tactile(_b) (native resolution, PNG)
    + 16-dim joint state + prompt, over ZMQ REQ + msgpack (remote_infer_lib/n0vtla_zmq_client.py)
  - tactile key = env UNIVTAC_TAC_KEY > task_config policy_tac_key > rgb (what convert_neosim_to_lerobot.py
    trains on; NeoSim's generic default is gel_particle). MUST match the training representation: it is
    logged at startup, and a key the task config does not record stops the run before Isaac starts
  - server = N0-VTLA scripts/serve_zmq.py --config sim_dual_arm_tactile (cam_high/cam_left_wrist/cam_right_wrist
    + tactile_a..d = left_tactile/right_tactile/left_tactile_b/right_tactile_b)
  - action = 16-dim joint (d[:8]=arm A [7 joints+gripper] / d[8:16]=arm B), direct set_arm (no IK);
    chunk H=16, executed in full (--exec_horizon 0; the client refuses any other horizon), 2 sim steps per
    action (UNIVTAC_STEPS_PER_ACTION=2)
  - take_action_cnt counts SIM STEPS (as take_action and eval_twam_ee_dual_cl.drive_dual do), so the per-task
    step_lim bounds the same sim time for any UNIVTAC_STEPS_PER_ACTION: step_lim/2 actions at 2
  - reset = {"cmd": "reset"} at the start of EVERY episode: the server re-captures the tactile baseline tac_0
    on the next predict. A failed reset fails the episode (_common swallows it, which would silently keep
    the previous episode's baseline)
  - ONE client per server process: the server holds a single per-episode tactile baseline, so two clients on
    one port corrupt each other's contact signal. Use one serve_zmq.py (own port) per client
  - start_seed 1000000 (above every collection seed); results under eval_result/n0vtla_dual/; per-seed
    <run_dir>/results.jsonl that also lists errored seeds (those are skipped by the success count), wedge
    aborts included: a stalled sim is not a policy failure
  - gel debug dump: rgb of every pad at the first chunk at or after take_action_cnt 0/75/300/900 of every
    episode -> <run_dir>/gel_dump/<seed>_<cnt>_<pad>.png
Everything else (joint state, to_u8, [GEL] depth print, success-check fix, EVAL_STEP_LIM) matches the pi05 dual client verbatim.
"""
import sys, os, time, json, yaml, argparse, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root (envs/, task_config/)
from _common import log_line as _log_line, get_config
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dual-arm JOINT N0-VTLA eval")
parser.add_argument("task_name", type=str)
parser.add_argument("task_config", type=str)
parser.add_argument("--server_host", default="127.0.0.1")
parser.add_argument("--server_port", type=int, default=5557)
parser.add_argument("--prompt", required=True)  # MUST be the training prompt verbatim (tasks.jsonl); a wrong prompt silently poisons scores
parser.add_argument("--exec_horizon", type=int, default=0)   # execute first N steps of each chunk, then re-infer; 0 = the whole chunk (serve_zmq.py requires it)
parser.add_argument("--start_seed", type=int, default=1000000)   # held-out range: collection seeds start at 0
parser.add_argument("--max_seed", type=int, default=-1)
parser.add_argument("--total_num", type=int, default=20)
parser.add_argument("--print_only", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.start_seed == -1: args_cli.start_seed = 1000000   # -1 keeps meaning "default"
args_cli.enable_cameras = True
if os.environ.get('UNIVTAC_GUI'): args_cli.livestream = 0; args_cli.headless = False  # GUI on local X: headless kit breaks PhysX scene queries -> gel attachment sweep blind
elif os.environ.get('UNIVTAC_HEADLESS'): args_cli.livestream = 0; args_cli.headless = True
else: args_cli.livestream = 2
args_cli.num_envs = 1

STEPS_PER_ACTION = int(os.environ.get("UNIVTAC_STEPS_PER_ACTION", "2"))  # training data stores one frame every 2 sim steps (save_frequency=2) -> 2 matches the time base
task_config, task_config_file = get_config(args_cli.task_config, default_root=Path('./task_config'), type='yaml')
# Tactile image sent to the policy. Priority: explicit env override > task_config policy_tac_key > rgb (the
# representation convert_neosim_to_lerobot.py trains on). MUST match the training representation.
TAC_KEY = os.environ.get('UNIVTAC_TAC_KEY') or task_config.get('policy_tac_key') or 'rgb'
SETTINGS = (f"task={args_cli.task_name} | cfg={args_cli.task_config} | server={args_cli.server_host}:{args_cli.server_port} | "
            f"prompt='{args_cli.prompt}' | tac_key={TAC_KEY} | exec_horizon={args_cli.exec_horizon or 'full chunk'} | "
            f"steps_per_action={STEPS_PER_ACTION} (step_lim counts sim steps) | "
            f"start_seed={args_cli.start_seed} | max_seed={args_cli.max_seed} | total_num={args_cli.total_num} | "
            f"EVAL_STEP_LIM={os.environ.get('EVAL_STEP_LIM') or '-'} | EVAL_TAG={os.environ.get('EVAL_TAG', 'run')} | "
            f"display={'gui' if os.environ.get('UNIVTAC_GUI') else 'headless' if os.environ.get('UNIVTAC_HEADLESS') else 'livestream'}")
print(f"[n0vtla-dual] settings: {SETTINGS}", flush=True)
_tac_recorded = (task_config.get("observations") or {}).get("tactile") or []
if TAC_KEY not in _tac_recorded:   # stop before the server wait and Isaac: every episode would error on the missing key
    sys.exit(f"[n0vtla-dual] tactile key '{TAC_KEY}' is not recorded by {task_config_file} (observations.tactile={_tac_recorded})")

import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))   # remote_infer_lib lives alongside this file
from remote_infer_lib.n0vtla_zmq_client import N0VTLAZmqClient, CAM_WIRE_KEYS, TAC_WIRE_KEYS, ACTION_HORIZON
if 0 < args_cli.exec_horizon < ACTION_HORIZON:
    print(f"[n0vtla-dual] WARNING: --exec_horizon {args_cli.exec_horizon} executes only part of each {ACTION_HORIZON}-step chunk; "
          f"serve_zmq.py requires the whole chunk (the gripper-close commands sit in its tail)", flush=True)
print(f"[n0vtla-dual] connecting server {args_cli.server_host}:{args_cli.server_port} ...", flush=True)
client = N0VTLAZmqClient(host=str(args_cli.server_host), port=int(args_cli.server_port))
client.wait_for_server()   # answers a reset; the server may still be loading the checkpoint
print(f"[n0vtla-dual] connected: {client.addr} (reset ok)", flush=True)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import torch, importlib

def to_u8(t):
    x = t[..., :3]; x = x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x); x = np.squeeze(x)
    if x.dtype != np.uint8:
        # float buffers arrive in BOTH [0,1] (cameras) and [0,255] (taxim tactile_rgb)
        # ranges; a blind *255 overflows the latter to a solid-white image.
        _sc = 1.0 if float(x.max()) > 1.5 else 255.0
        x = (x * _sc).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(x)

def _arm_joint8(rm):
    qpos = np.asarray(rm.get_qpos(), dtype=np.float64).reshape(-1)
    s = np.zeros(8, dtype=np.float32); s[:7] = qpos[:7]; s[7] = float(rm.get_gripper_qpos())
    return s

def joint_state_dual(task):   # 16 dims: arm A [7 joints+gripper] + arm B
    return np.concatenate([_arm_joint8(task._robot_manager), _arm_joint8(task._robot_manager_b)]).astype(np.float32)

def _drive_one_joint(rm, j8, force=True):   # set joints directly (pi05 actions are joint targets, no IK)
    rm.set_arm(torch.as_tensor(np.asarray(j8[:7], dtype=np.float32), device=rm.robot.device), force=force)  # torch tensor, not numpy
    rm.set_gripper(float(j8[7]), force=force)

def drive_dual_joint(task, ja, jb, force=True):
    _drive_one_joint(task._robot_manager, ja, force)
    _drive_one_joint(task._robot_manager_b, jb, force)
    # cnt counts sim steps (ticks), as take_action and eval_twam_ee_dual_cl.drive_dual do: step_lim is data-calibrated
    # in sim steps, so it bounds the same sim time for any STEPS_PER_ACTION
    for _ in range(STEPS_PER_ACTION):
        task._step()
        task.take_action_cnt += 1
        if task.take_action_cnt >= task.cfg.step_lim:
            break

class WedgeAbort(RuntimeError):
    """The sim stalled (5 consecutive actions >10 s): the episode loop records the seed as errored."""

def _gel_status(observation):
    out = []
    for name, d in (observation.get("tactile") or {}).items():
        dep = d.get("depth")
        if dep is None:
            out.append(name + "=NA")
            continue
        dd = dep[50:-50, 50:-50]
        try:
            mn = float(dd.min())
            mx = float(dd.max())
            out.append("%s=(%.4f,%.4f)" % (name, mn, mx))
        except Exception:
            out.append(name + "=ERR")
    return " ".join(out)

_GEL_DUMP_AT = (0, 75, 300, 900)   # take_action_cnt marks; each dumps at the first chunk boundary at or after it
_GEL_DUMPED = set()                # (seed, mark) pairs already dumped

def _gel_dump(task, observation):   # rgb of every pad -> <run_dir>/gel_dump/<seed>_<cnt>_<pad>.png, every episode
    import cv2
    c, seed = task.take_action_cnt, task.cfg.seed
    due = [m for m in _GEL_DUMP_AT if m <= c and (seed, m) not in _GEL_DUMPED]
    if not due:
        return
    _GEL_DUMPED.update((seed, m) for m in due)
    out = task.save_root / "gel_dump"   # per run dir: concurrent clients do not overwrite each other
    out.mkdir(parents=True, exist_ok=True)
    for name, d in (observation.get("tactile") or {}).items():
        img = d.get("rgb")
        if img is None:
            continue
        cv2.imwrite(str(out / f"{seed}_{c}_{name}.png"), to_u8(img)[..., ::-1])   # to_u8: a blind *255 whitens the [0,255] tactile buffer

def n0vtla_eval(task, observation):
    print(f"[GEL] cnt={task.take_action_cnt} {_gel_status(observation)}", flush=True)
    _gel_dump(task, observation)
    cam, tac = observation["observation"], observation["tactile"]
    state = joint_state_dual(task)
    # all 3 cameras + all 4 tactile pads (a missing one raises); the client renames them to the server's wire keys
    action = client.predict(state, args_cli.prompt,
                            cams={k: to_u8(cam[k]["rgb"]) for k in CAM_WIRE_KEYS},
                            tacs={k: to_u8(tac[k][TAC_KEY]) for k in TAC_WIRE_KEYS})   # (H=16, 16)
    print(f'[DBG] state={np.round(state,3).tolist()}', flush=True)
    print(f'[DBG] act0 ={np.round(np.asarray(action[0]).reshape(-1),3).tolist()}', flush=True)
    print(f'[DBG] actN ={np.round(np.asarray(action[-1]).reshape(-1),3).tolist()}', flush=True)
    n_exec = min(args_cli.exec_horizon if args_cli.exec_horizon > 0 else action.shape[0], action.shape[0])
    import time as _t
    _slow = getattr(task, "_giga_slow_cnt", 0)
    for i in range(n_exec):
        d = np.asarray(action[i]).reshape(-1)     # (16,)
        _t0 = _t.time()
        drive_dual_joint(task, d[:8], d[8:16])
        _dt = _t.time() - _t0
        _slow = _slow + 1 if _dt > 10.0 else 0
        task._giga_slow_cnt = _slow
        if _slow >= 5:
            # a stalled sim is not a policy failure: raise, so the seed is recorded as errored with its real
            # take_action_cnt and stays out of the success count and the paired statistics
            print(f"[WEDGE-ABORT] {_slow} consecutive actions >10s (last {_dt:.1f}s) -> seed recorded as errored", flush=True)
            raise WedgeAbort(f"{_slow} consecutive actions >10s (last {_dt:.1f}s)")
        # success-check fix: this client steps via _step and bypasses take_action, so check_success is never called and eval_success stays False; check explicitly
        if task.check_success():
            task.eval_success = True
        if getattr(task, "eval_success", False): return
        if task.take_action_cnt >= task.cfg.step_lim: return

log_path = Path('./log')
def log(msg):
    _log_line(msg, log_path, args_cli.print_only)


def _write_result(path, seed, success, steps, time_s, error=None):
    with open(path, 'a') as f:   # appended per seed, so an interrupted run keeps the finished seeds
        f.write(json.dumps({'seed': seed, 'success': bool(success), 'steps': int(steps), 'time_s': round(time_s, 2),
                            'error': error is not None, 'error_msg': error}) + '\n')


def _eval_policy_lite_jsonl(task, start_seed, max_seed, test_total_num, instructions,
                            itype='seen', *, infer_fn, reset_fn, log):
    """_common.eval_policy_lite with three changes: a reset_fn failure fails the episode instead of being
    swallowed (a missed server reset keeps the previous episode's tactile baseline), the wedge counter starts
    at 0 in every episode, and every seed that ran, errored ones (wedge aborts too) included, gets one line in
    <run_dir>/results.jsonl (steps = take_action_cnt, in sim steps).
    A seed cut short by Ctrl-C gets no line, so it is never mistaken for a failure."""
    results_path = task.save_root / 'results.jsonl'
    test_num, succ_num, err_num, seed = 0, 0, 0, start_seed
    consec_err = 0
    while test_num < test_total_num and (max_seed == -1 or seed <= max_seed):
        test_num += 1; succ = False; started = False; eval_start = time.perf_counter(); task.mode = 'eval'
        try:
            task.reset(seed=seed, instructions=instructions[itype])
            started = True   # before this, take_action_cnt may still be the previous episode's
            task._giga_slow_cnt = 0   # BaseTask._reset_idx does not know this counter: do not carry it into a new episode
            task.mean_steps = task.cfg.step_lim
            reset_fn()
            while task.take_action_cnt < task.cfg.step_lim:
                infer_fn(task, task._get_observations())
                if task.eval_success: succ = True; break
                if task.check_early_stop(): break
        except Exception as e:
            log(f"[{test_num:<3d}] Seed {seed} exception: {e}\n{traceback.format_exc()}")
            _write_result(results_path, seed, False, task.take_action_cnt if started else 0,
                          time.perf_counter() - eval_start, f"{type(e).__name__}: {e}")
            task.clean_cache(result='error'); test_num -= 1; err_num += 1
            consec_err += 1
            if consec_err >= 10:
                log(f"Aborting: {consec_err} consecutive errors (policy server down?)")
                break
        else:
            consec_err = 0
            if succ: succ_num += 1
            _write_result(results_path, seed, succ, task.take_action_cnt, time.perf_counter() - eval_start)
            task.clean_cache(result='success' if succ else 'failed')
            log(f"[{test_num:<3d}] Seed {seed} {'success' if succ else 'failed'} "
                f"after {time.perf_counter()-eval_start:.1f}s. Total {succ_num}/{test_num}.")
        finally: seed += 1
    return {'test_num': test_num, 'succ_num': succ_num, 'err_num': err_num}


def eval_policy(task, start_seed, max_seed, test_total_num, instructions, itype='seen'):
    return _eval_policy_lite_jsonl(
        task, start_seed, max_seed, test_total_num, instructions, itype,
        infer_fn=n0vtla_eval,
        reset_fn=client.reset,   # {"cmd": "reset"}: the next predict re-captures the tactile baseline
        log=log)


def main():
    global log_path
    task_file_name = args_cli.task_name
    try: instructions, _ = get_config(task_file_name, default_root=Path('./instructions'), type='json')
    except Exception: instructions = {'seen': ['Empty'], 'unseen': ['Empty']}
    task_module = importlib.import_module(f"envs.{task_file_name}")
    curr_time = time.strftime(r'%Y-%m-%d_%H:%M:%S')
    env_cfg = task_module.TaskCfg()
    env_cfg.save_dir = Path('eval_result') / 'n0vtla_dual' / task_file_name / task_config_file.stem / (os.environ.get('EVAL_TAG','run')+'_'+curr_time)
    env_cfg.decimation = task_config.get("decimation", env_cfg.decimation)
    env_cfg.obs_data_type = task_config.get("observations", {})
    env_cfg.save_frequency = task_config.get("save_frequency", env_cfg.save_frequency)
    env_cfg.video_frequency = task_config.get("video_frequency", env_cfg.video_frequency)
    env_cfg.scene.num_envs = 1
    if args_cli.device is not None: env_cfg.sim.device = args_cli.device
    if os.environ.get('EVAL_STEP_LIM'): env_cfg.step_lim = int(os.environ['EVAL_STEP_LIM'])
    task = task_module.Task(env_cfg, mode='eval')
    log_path = task.save_root / "log.log"
    log(f"N0VTLA-DUAL {SETTINGS} | step_lim: {task.cfg.step_lim} sim steps = "
        f"{-(-task.cfg.step_lim // STEPS_PER_ACTION)} actions | run_dir: {task.save_root}")
    start_seed = args_cli.start_seed
    results = eval_policy(task, start_seed, args_cli.max_seed, args_cli.total_num, instructions, 'seen')
    log(f"Final: {results['succ_num']}/{results['test_num']} success ({results['err_num']} errored seeds skipped; "
        f"per-seed results in {task.save_root / 'results.jsonl'}).")
    task.close(); client.close(); simulation_app.close()

if __name__ == "__main__":
    main()
