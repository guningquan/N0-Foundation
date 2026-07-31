#!/usr/bin/env python3
"""Standalone EE eval — COPIES official scripts/eval_policy.py logic VERBATIM.

Only two (required) deviations from the official harness:
  1. WebsocketClientPolicy connects **before** the Isaac AppLauncher starts
     (websockets.sync.connect deadlocks if called after Isaac sim is up).
  2. Actions are executed via take_action(ee, action_type='ee') (official openpi uses 'qpos').
Everything else matches the official flow: expert_check qualification (play_once),
reset+instructions, _get_observations, eval loop, check_success/eval_success/check_early_stop, seeds.json.
"""
import sys
import os
import time
import json
import yaml
import argparse
import traceback
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root (envs/, task_config/)
from _common import log_line as _log_line, get_config, eval_policy_full as _eval_policy_full

import argparse as _ap
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Eval Policy (EE, standalone, official-aligned)")
parser.add_argument("task_name", type=str)
parser.add_argument("task_config", type=str)
parser.add_argument("--server_host", default="127.0.0.1")
parser.add_argument("--server_port", type=int, default=29542)
parser.add_argument("--prompt", required=True)  # MUST be the training prompt verbatim (tasks.jsonl); a wrong prompt silently poisons scores
parser.add_argument("--exec_horizon", type=int, default=0)
parser.add_argument("--expert_check", action="store_true")
parser.add_argument("--start_seed", type=int, default=-1)
parser.add_argument("--max_seed", type=int, default=-1)
parser.add_argument("--total_num", type=int, default=20)
parser.add_argument("--print_only", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = True; args_cli.livestream = 0
args_cli.num_envs = 1

# ============ STEP 1: connect to the policy server (must precede Isaac; sync ws connect deadlocks otherwise) ============
import numpy as np
from scipy.spatial.transform import Rotation as R
from remote_infer_lib.websocket_client_policy import WebsocketClientPolicy

print(f"[ee-eval] connecting policy server {args_cli.server_host}:{args_cli.server_port} (before Isaac) ...", flush=True)
client = WebsocketClientPolicy(host=str(args_cli.server_host), port=int(args_cli.server_port))
print(f"[ee-eval] connected: {client.get_server_metadata()}", flush=True)

# ============ STEP 2: launch Isaac ============
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import importlib

# -------- EE state helpers --------
def to_u8(t):
    x = t[..., :3]
    x = x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
    x = np.squeeze(x)
    if x.dtype != np.uint8:
        # float buffers arrive in BOTH [0,1] (cameras) and [0,255] (taxim tactile_rgb)
        # ranges; a blind *255 overflows the latter to a solid-white image.
        _sc = 1.0 if float(x.max()) > 1.5 else 255.0
        x = (x * _sc).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(x)

# ===== XR0 EE eval helpers (replace pi05/rot6d) =====
ACTION_PARTS = (("left_ee_pos", slice(0,3)), ("left_ee_aa", slice(3,6)), ("left_gripper", slice(6,7)),
                ("left_joint", slice(7,13)), ("right_ee_pos", slice(14,17)), ("right_ee_aa", slice(17,20)),
                ("right_gripper", slice(20,21)), ("right_joint", slice(21,27)))

def aa2rotm(aa):
    aa = np.asarray(aa, dtype=np.float32); angle = float(np.linalg.norm(aa))
    axis = aa/(angle+1e-10); x,y,z = axis.tolist()
    K = np.array([[0,-z,y],[z,0,-x],[-y,x,0]], dtype=np.float32); I = np.eye(3, dtype=np.float32)
    return I + np.sin(angle)*K + (1-np.cos(angle))*(K@K)

def split_action(a):
    a = np.asarray(a, dtype=np.float32); return {n: a[..., sl] for n,sl in ACTION_PARTS}

def recover_action(action, rs):
    parts = split_action(action); t = {}
    for side in ("left","right"):
        rotm = np.asarray(rs[side+"_ee_rotm"], np.float32).reshape(3,3)
        pos = np.asarray(rs[side+"_ee_pos"], np.float32).reshape(3)
        grip = np.asarray(rs[side+"_gripper_pos"], np.float32).reshape(1)
        joint = np.asarray(rs[side+"_arm_joint"], np.float32).reshape(6)
        t[side+"_ee_pos"] = pos[None] + parts[side+"_ee_pos"] @ rotm.T
        t[side+"_ee_rotm"] = np.stack([rotm @ aa2rotm(d) for d in parts[side+"_ee_aa"]], 0)
        t[side+"_gripper_pos"] = grip[None] + parts[side+"_gripper"]
        t[side+"_arm_joint"] = joint[None] + parts[side+"_joint"]
    return t

def xr0_state32(task):
    rm = task._robot_manager
    j = np.asarray(rm.get_qpos(), np.float32).reshape(-1)[:6]
    g = float(rm.get_gripper_qpos())
    s = np.zeros(32, np.float32); s[6] = g; s[7:13] = j
    return s

def xr0_robot_state(task):
    rm = task._robot_manager; pose = rm.get_ee_pose()
    p = np.asarray(pose.p, np.float64).reshape(-1)[:3]
    q = np.asarray(pose.q, np.float64).reshape(-1)   # wxyz
    rotm = R.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()
    j = np.asarray(rm.get_qpos(), np.float32).reshape(-1)[:6]
    g = float(rm.get_gripper_qpos())
    return {"left_ee_pos": p, "left_ee_rotm": rotm.reshape(9), "left_gripper_pos": np.array([g], np.float32),
            "left_arm_joint": j, "right_ee_pos": np.zeros(3), "right_ee_rotm": np.eye(3).reshape(9).astype(np.float32),
            "right_gripper_pos": np.zeros(1, np.float32), "right_arm_joint": np.zeros(6, np.float32)}

_gpatched = {"done": False}
def _ensure_gripper_patch(task):
    if _gpatched["done"]: return
    rm = task._robot_manager; _o = rm.plan_gripper
    def _pg(pos, type='percent', _o=_o):
        if isinstance(pos, torch.Tensor): pos = float(pos.reshape(-1)[0])
        elif hasattr(pos,'__len__'): pos = float(pos[0])
        else: pos = float(pos)
        return _o(pos, type)
    rm.plan_gripper = _pg; _gpatched["done"] = True

STEPS_PER_ACTION = int(os.environ.get("UNIVTAC_STEPS_PER_ACTION", "1"))  # training save_frequency=2 -> set 2 to match the time base (2 physics steps per action)
_DUMP = []  # XR0_DUMP: per-chunk waypoint trajectory diagnostics
if os.environ.get("XR0_EXEC_HORIZON"):   # execute only the first N steps then re-infer (receding horizon), skipping distal retraction
    args_cli.exec_horizon = int(os.environ["XR0_EXEC_HORIZON"])

def vta_eval(task, observation):
    """XR0: obs = 2 cameras + 32-dim state; model outputs (30,32) deltas; recover_action restores absolute EE; driven via take_action('ee') (sim-internal IK)."""
    _ensure_gripper_patch(task)
    cam = observation["observation"]
    obs = {"observation.images.ego":    to_u8(cam["head"]["rgb"]),
           "observation.images.wrist_l": to_u8(cam["wrist"]["rgb"]),
           "observation.state":          xr0_state32(task),
           "prompt":                     args_cli.prompt}
    ret = client.infer(obs)
    if "actions" not in ret:
        raise RuntimeError(f"server error: {ret}")
    action = np.asarray(ret["actions"], dtype=np.float32)      # (30,32)
    rs = xr0_robot_state(task)
    tg = recover_action(action, rs)
    pos, rotm, grip = tg["left_ee_pos"], tg["left_ee_rotm"], tg["left_gripper_pos"]
    if os.environ.get("XR0_DUMP"):
        import numpy as _np
        _DUMP.append({"cnt": int(task.take_action_cnt),
                      "cur": _np.asarray(rs["left_ee_pos"], _np.float32).reshape(3).copy(),
                      "wp":  pos.copy().astype(_np.float32),          # (30,3) recovered absolute waypoints
                      "raw_pos_delta": action[:, 0:3].copy(),         # (30,3) raw model pos deltas
                      "raw_grip": action[:, 6].copy()})               # (30,) gripper delta
        if len(_DUMP) <= 8:
            _dd = pos - rs["left_ee_pos"].reshape(1,3)                # per-waypoint displacement from current pose
            _dist = _np.linalg.norm(_dd, axis=1)
            print(f"[WP cnt={task.take_action_cnt}] |disp| w0={_dist[0]:.4f} w10={_dist[10]:.4f} w20={_dist[20]:.4f} w29={_dist[29]:.4f} "
                  f"max@{int(_dist.argmax())} | retreats_midway={'YES' if _dist.argmax()<25 and _dist[29]<_dist[:25].max()-0.005 else 'no'}", flush=True)
            _np.save(f"/tmp/xr0_wp_{os.getpid()}.npy", _np.array(_DUMP, dtype=object), allow_pickle=True)
    if task.take_action_cnt == 0:
        print(f"[DBG] state6={np.round(xr0_state32(task)[7:13],3).tolist()} ee0={np.round(pos[0],3).tolist()}", flush=True)
    for i in range(pos.shape[0]):
        if args_cli.exec_horizon and i >= args_cli.exec_horizon:
            break
        q = R.from_matrix(rotm[i]).as_quat()                   # xyzw
        ee = torch.tensor([float(pos[i,0]),float(pos[i,1]),float(pos[i,2]),
                           float(q[3]),float(q[0]),float(q[1]),float(q[2]), float(grip[i,0])], dtype=torch.float32)
        task.take_action(ee, action_type='ee_ik')
        # time-base alignment: pad (STEPS_PER_ACTION-1) extra physics steps per action so objects/contacts settle as during training
        for _ in range(STEPS_PER_ACTION - 1):
            if getattr(task, "eval_success", False) or task.take_action_cnt >= task.cfg.step_lim:
                break
            task._step()
            if task.check_success():
                task.eval_success = True
        if getattr(task, "eval_success", False):
            return
        if task.take_action_cnt >= task.cfg.step_lim:
            return

# -------- log (copied from official harness) --------
log_path = Path('./log')
def log(msg):
    _log_line(msg, log_path, args_cli.print_only)


# -------- eval_policy(): mirrors the scripts/eval_policy.py loop semantics (success latching / step limits identical); inference is a policy-server round-trip. Self-contained: no dependency on policy/ (see eval/README.md) --------
def eval_policy(task, expert_check, start_seed, max_seed, test_total_num, instructions, instruciton_type='seen'):
    return _eval_policy_full(
        task, expert_check, start_seed, max_seed, test_total_num, instructions, instruciton_type,
        infer_fn=vta_eval,
        reset_fn=lambda: client.infer(dict(reset=True, prompt=args_cli.prompt)),
        log=log)


def main():
    global log_path
    task_file_name = args_cli.task_name
    task_config, task_config_file = get_config(
        args_cli.task_config, default_root=Path('./task_config'), type='yaml')

    # instructions: as in official harness, read from instructions/{task}.json; fall back to Empty
    try:
        instructions, _ = get_config(task_file_name, default_root=Path('./instructions'), type='json')
    except Exception:
        instructions = {'seen': ['Empty'], 'unseen': ['Empty']}

    task_module = importlib.import_module(f"envs.{task_file_name}")
    curr_time = time.strftime(r'%Y-%m-%d_%H:%M:%S')

    env_cfg = task_module.TaskCfg()
    env_cfg.save_dir = Path('eval_result') / 'xr0' / task_file_name / task_config_file.stem / curr_time
    env_cfg.decimation = task_config.get("decimation", env_cfg.decimation)
    env_cfg.obs_data_type = task_config.get("observations", {})
    env_cfg.save_frequency = task_config.get("save_frequency", env_cfg.save_frequency)
    env_cfg.video_frequency = task_config.get("video_frequency", env_cfg.video_frequency)
    env_cfg.random_texture = task_config.get("random_texture", False)
    env_cfg.scene.num_envs = 1
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    import os as _os

    if _os.environ.get('EVAL_STEP_LIM'):

        env_cfg.step_lim = int(_os.environ['EVAL_STEP_LIM'])

    task = task_module.Task(env_cfg, mode='eval')
    log_path = task.save_root / "log.log"
    log(f"Task: {task_file_name} | config: {task_config_file.stem} | server {args_cli.server_host}:{args_cli.server_port}")
    log(f"prompt='{args_cli.prompt}' exec_horizon={args_cli.exec_horizon} expert_check={args_cli.expert_check}")

    start_seed = 1000000 if args_cli.start_seed == -1 else args_cli.start_seed
    results = eval_policy(
        task=task, expert_check=args_cli.expert_check,
        start_seed=start_seed, max_seed=args_cli.max_seed,
        test_total_num=args_cli.total_num, instructions=instructions,
        instruciton_type='seen')
    log(f"Final Result: {results['succ_num']}/{results['test_num']}"
        f"({results['succ_num']/max(1,results['test_num'])*100:.2f}%) success.")

    task.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
