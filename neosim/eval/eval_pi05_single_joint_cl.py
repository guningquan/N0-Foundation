#!/usr/bin/env python3
"""Single-arm JOINT pi05 eval client — single-arm variant of eval_pi05_dual_joint_cl.py:
  - obs = openpi format (2 cameras top/wrist_l + observation.state[8-dim joint] + prompt), no wrist_r, no tactile
  - server = openpi serve_pi05_single (dataset keys -> repack -> FlexivInputs -> pi05 -> 8-dim Out)
  - action = 8-dim joint (d[:7]=7 joints / d[7]=gripper), direct set_arm (no IK); delta mask (7,-1) handled server-side
  - no compute_kv_cache (pi05 is not streaming)
Everything else (eval_policy/main/reset/seeds/success) matches the dual client verbatim.
"""
import sys, os, time, json, yaml, argparse, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root (envs/, task_config/)
from _common import log_line as _log_line, get_config, eval_policy_lite as _eval_policy_lite
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Single-arm JOINT pi05 eval")
parser.add_argument("task_name", type=str)
parser.add_argument("task_config", type=str)
parser.add_argument("--server_host", default="127.0.0.1")
parser.add_argument("--server_port", type=int, default=8000)
parser.add_argument("--prompt", required=True)  # MUST be the training prompt verbatim (tasks.jsonl); a wrong prompt silently poisons scores
parser.add_argument("--exec_horizon", type=int, default=25)   # execute first N steps of each chunk, then re-infer
parser.add_argument("--start_seed", type=int, default=-1)
parser.add_argument("--max_seed", type=int, default=-1)
parser.add_argument("--total_num", type=int, default=20)
parser.add_argument("--print_only", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
if os.environ.get('UNIVTAC_GUI'): args_cli.livestream = 0; args_cli.headless = False  # GUI on local X: headless kit breaks PhysX scene queries -> gel attachment sweep blind
elif os.environ.get('UNIVTAC_HEADLESS'): args_cli.livestream = 0; args_cli.headless = True
else: args_cli.livestream = 2
args_cli.num_envs = 1

import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))   # remote_infer_lib lives alongside this file
from remote_infer_lib.websocket_client_policy import WebsocketClientPolicy
print(f"[pi05-single] connecting server {args_cli.server_host}:{args_cli.server_port} ...", flush=True)
client = WebsocketClientPolicy(host=str(args_cli.server_host), port=int(args_cli.server_port))
print(f"[pi05-single] connected: {client.get_server_metadata()}", flush=True)

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

def joint_state_single(task):   # 8 dims: 7 joints + gripper (single arm: only _robot_manager, no _robot_manager_b)
    return _arm_joint8(task._robot_manager).astype(np.float32)

def _drive_one_joint(rm, j8, force=True):   # set joints directly (pi05 actions are joint targets, no IK)
    rm.set_arm(torch.as_tensor(np.asarray(j8[:7], dtype=np.float32), device=rm.robot.device), force=force)  # torch tensor, not numpy
    rm.set_gripper(float(j8[7]), force=force)

STEPS_PER_ACTION = int(os.environ.get("UNIVTAC_STEPS_PER_ACTION", "1"))  # training data uses 2 sim steps per frame (save_frequency=2); set 2 to match the time base

def drive_single_joint(task, j8, force=True):
    _drive_one_joint(task._robot_manager, j8, force)
    for _ in range(STEPS_PER_ACTION):
        task._step()
    task.take_action_cnt += 1

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

_GEL_DUMPED = set()

def _gel_dump(task, observation):
    import cv2, os
    import numpy as _np
    c = task.take_action_cnt
    if c not in (0, 75, 300, 900) or c in _GEL_DUMPED:
        return
    _GEL_DUMPED.add(c)
    os.makedirs("/tmp/gel_dump", exist_ok=True)
    for name, d in (observation.get("tactile") or {}).items():
        img = d.get("rgb")
        if img is None:
            continue
        x = img.cpu().numpy() if hasattr(img, "cpu") else _np.asarray(img)
        x = _np.squeeze(x)[..., :3]
        if x.dtype != _np.uint8:
            x = (x * 255).clip(0, 255).astype(_np.uint8)
        cv2.imwrite("/tmp/gel_dump/%d_%s.png" % (c, name), x[..., ::-1])

def pi05_eval(task, observation):
    print(f"[GEL] cnt={task.take_action_cnt} {_gel_status(observation)}", flush=True)
    _gel_dump(task, observation)
    cam = observation["observation"]
    state = joint_state_single(task)
    obs = {'observation.images.top':    to_u8(cam["head"]["rgb"]),
           'observation.images.wrist_l': to_u8(cam["wrist"]["rgb"]),
           'observation.state':          state,
           'prompt':                     args_cli.prompt}
    ret = client.infer(obs)
    action = np.asarray(ret['actions'])          # (H, 8)
    if action.ndim == 1: action = action[None]
    print(f'[DBG] state={np.round(state,3).tolist()}', flush=True)
    print(f'[DBG] act0 ={np.round(np.asarray(action[0]).reshape(-1),3).tolist()}', flush=True)
    print(f'[DBG] actN ={np.round(np.asarray(action[-1]).reshape(-1),3).tolist()}', flush=True)
    n_exec = min(args_cli.exec_horizon if args_cli.exec_horizon > 0 else action.shape[0], action.shape[0])
    import time as _t
    _slow = getattr(task, "_giga_slow_cnt", 0)
    for i in range(n_exec):
        d = np.asarray(action[i]).reshape(-1)     # (8,)
        _t0 = _t.time()
        drive_single_joint(task, d)
        _dt = _t.time() - _t0
        _slow = _slow + 1 if _dt > 10.0 else 0
        task._giga_slow_cnt = _slow
        if _slow >= 5:
            print(f"[WEDGE-ABORT] {_slow} consecutive steps >10s (last {_dt:.1f}s) -> episode failed", flush=True)
            task.take_action_cnt = task.cfg.step_lim
            return
        # success-check fix: this client steps via _step and bypasses take_action, so check_success is never called and eval_success stays False; check explicitly
        if task.check_success():
            task.eval_success = True
        if getattr(task, "eval_success", False): return
        if task.take_action_cnt >= task.cfg.step_lim: return

log_path = Path('./log')
def log(msg):
    _log_line(msg, log_path, args_cli.print_only)


def eval_policy(task, start_seed, max_seed, test_total_num, instructions, itype='seen'):
    return _eval_policy_lite(
        task, start_seed, max_seed, test_total_num, instructions, itype,
        infer_fn=pi05_eval,
        reset_fn=lambda: client.infer(dict(reset=True, prompt=args_cli.prompt)),
        log=log)


def main():
    global log_path
    task_file_name = args_cli.task_name
    task_config, task_config_file = get_config(args_cli.task_config, default_root=Path('./task_config'), type='yaml')
    try: instructions, _ = get_config(task_file_name, default_root=Path('./instructions'), type='json')
    except Exception: instructions = {'seen': ['Empty'], 'unseen': ['Empty']}
    task_module = importlib.import_module(f"envs.{task_file_name}")
    curr_time = time.strftime(r'%Y-%m-%d_%H:%M:%S')
    env_cfg = task_module.TaskCfg()
    env_cfg.save_dir = Path('eval_result') / 'pi05_single' / task_file_name / task_config_file.stem / (os.environ.get('EVAL_TAG','run')+'_'+curr_time)
    env_cfg.decimation = task_config.get("decimation", env_cfg.decimation)
    env_cfg.obs_data_type = task_config.get("observations", {})
    env_cfg.save_frequency = task_config.get("save_frequency", env_cfg.save_frequency)
    env_cfg.video_frequency = task_config.get("video_frequency", env_cfg.video_frequency)
    env_cfg.scene.num_envs = 1
    if args_cli.device is not None: env_cfg.sim.device = args_cli.device
    if os.environ.get('EVAL_STEP_LIM'): env_cfg.step_lim = int(os.environ['EVAL_STEP_LIM'])
    task = task_module.Task(env_cfg, mode='eval')
    log_path = task.save_root / "log.log"
    log(f"PI05-SINGLE Task: {task_file_name} | cfg: {task_config_file.stem} | server {args_cli.server_host}:{args_cli.server_port} | prompt='{args_cli.prompt}'")
    start_seed = 100 if args_cli.start_seed == -1 else args_cli.start_seed
    results = eval_policy(task, start_seed, args_cli.max_seed, args_cli.total_num, instructions, 'seen')
    log(f"Final: {results['succ_num']}/{results['test_num']} success.")
    task.close(); simulation_app.close()

if __name__ == "__main__":
    main()
