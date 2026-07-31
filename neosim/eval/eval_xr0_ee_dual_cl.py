#!/usr/bin/env python3
"""XR0 dual-arm EE eval client — dual-arm extension of the single-arm XR0 client (eval_xr0_ee_cl.py).
Dual-arm adaptation only:
  1. state32 fills both arms (A=left dims 6, 7:14; B=right dims 20, 21:28).
  2. robot_state reads ee/joint/gripper for both arms (recover_action is already dual-arm generic).
  3. obs = 3 cameras ego/wrist_l/wrist_r = cam head/wrist/wrist_b (matches training ego/wrist_left/wrist_right order).
  4. Drive: per-arm solve_ik+set_arm+set_gripper, then a single _step (mirrors take_action ee_ik, for both arms).
save_dir matches 7j = eval_result/xr0/<task>/...
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

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="XR0 dual-arm EE eval (official-aligned)")
parser.add_argument("task_name", type=str)
parser.add_argument("task_config", type=str)
parser.add_argument("--server_host", default="localhost")
parser.add_argument("--server_port", type=int, default=39001)
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
if os.environ.get("XR0_EXEC_HORIZON"):   # execute only the first N steps then re-infer, skipping distal retraction (matches single-arm)
    args_cli.exec_horizon = int(os.environ["XR0_EXEC_HORIZON"])

# ============ STEP 1: connect to the policy server (must precede Isaac; sync ws connect deadlocks otherwise) ============
import numpy as np
from scipy.spatial.transform import Rotation as R
from remote_infer_lib.websocket_client_policy import WebsocketClientPolicy

print(f"[xr0-dual] connecting policy server {args_cli.server_host}:{args_cli.server_port} (before Isaac) ...", flush=True)
client = WebsocketClientPolicy(host=str(args_cli.server_host), port=int(args_cli.server_port))
print(f"[xr0-dual] connected: {client.get_server_metadata()}", flush=True)

# ============ STEP 2: launch Isaac ============
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import importlib
from envs.utils.transforms import Pose

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

# ===== XR0 EE eval helpers (32-dim dual-arm) =====
ACTION_PARTS = (("left_ee_pos", slice(0,3)), ("left_ee_aa", slice(3,6)), ("left_gripper", slice(6,7)),
                ("left_joint", slice(7,14)), ("right_ee_pos", slice(14,17)), ("right_ee_aa", slice(17,20)),
                ("right_gripper", slice(20,21)), ("right_joint", slice(21,28)))

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
        joint = np.asarray(rs[side+"_arm_joint"], np.float32).reshape(7)
        t[side+"_ee_pos"] = pos[None] + parts[side+"_ee_pos"] @ rotm.T
        t[side+"_ee_rotm"] = np.stack([rotm @ aa2rotm(d) for d in parts[side+"_ee_aa"]], 0)
        t[side+"_gripper_pos"] = grip[None] + parts[side+"_gripper"]
        t[side+"_arm_joint"] = joint[None] + parts[side+"_joint"]
    return t

def _arm_qg(rm):
    j = np.asarray(rm.get_qpos(), np.float32).reshape(-1)[:7]
    g = float(rm.get_gripper_qpos())
    return j, g

def xr0_state32(task):
    """32-dim state: A=left (gripper dim6, joints dims7:14), B=right (gripper dim20, joints dims21:28)."""
    s = np.zeros(32, np.float32)
    jL, gL = _arm_qg(task._robot_manager)
    jR, gR = _arm_qg(task._robot_manager_b)
    s[6] = gL; s[7:14] = jL
    s[20] = gR; s[21:28] = jR
    return s

def _arm_rs(rm):
    pose = rm.get_ee_pose()
    p = np.asarray(pose.p, np.float64).reshape(-1)[:3]
    q = np.asarray(pose.q, np.float64).reshape(-1)   # wxyz
    rotm = R.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()
    j = np.asarray(rm.get_qpos(), np.float32).reshape(-1)[:7]
    g = float(rm.get_gripper_qpos())
    return p, rotm, g, j

def xr0_robot_state(task):
    pL, rL, gL, jL = _arm_rs(task._robot_manager)
    pR, rR, gR, jR = _arm_rs(task._robot_manager_b)
    return {"left_ee_pos": pL, "left_ee_rotm": rL.reshape(9), "left_gripper_pos": np.array([gL], np.float32),
            "left_arm_joint": jL, "right_ee_pos": pR, "right_ee_rotm": rR.reshape(9),
            "right_gripper_pos": np.array([gR], np.float32), "right_arm_joint": jR}

_gpatched = {"done": False}
def _ensure_gripper_patch(task):
    if _gpatched["done"]: return
    for rm in (task._robot_manager, task._robot_manager_b):
        _o = rm.plan_gripper
        def _pg(pos, type='percent', _o=_o):
            if isinstance(pos, torch.Tensor): pos = float(pos.reshape(-1)[0])
            elif hasattr(pos,'__len__'): pos = float(pos[0])
            else: pos = float(pos)
            return _o(pos, type)
        rm.plan_gripper = _pg
    _gpatched["done"] = True

def _drive_one_ee_ik(rm, ee, force=True):
    """Mirrors take_action ee_ik direct IK: solve_ik -> set_arm; hold pose on failure. No _step here (the caller steps)."""
    target_pose = Pose(p=ee[:3], q=ee[3:7])
    ik = rm.solve_ik(target_pose)
    if ik['status'] == 'Success':
        rm.set_arm(ik['position'], force=force)
    rm.set_gripper(ee[-1], force=force)

def take_action_dual(task, ee_a, ee_b, force=True):
    """Each arm sets joints via direct IK, then a single _step (one action = one step, arms synchronized)."""
    if task.take_action_cnt >= task.cfg.step_lim or task.eval_success:
        return
    task.take_action_cnt += 1
    _drive_one_ee_ik(task._robot_manager, ee_a, force)
    _drive_one_ee_ik(task._robot_manager_b, ee_b, force)
    task._step()
    if task.check_success():
        task.eval_success = True

def vta_eval(task, observation):
    """XR0 dual-arm: obs = 3 cameras + 32-dim dual-arm state; model outputs (30,32) deltas; recover restores absolute EE; both arms driven via ee_ik."""
    _ensure_gripper_patch(task)
    cam = observation["observation"]
    obs = {"observation.images.ego":     to_u8(cam["head"]["rgb"]),
           "observation.images.wrist_l": to_u8(cam["wrist"]["rgb"]),
           "observation.images.wrist_r": to_u8(cam["wrist_b"]["rgb"]),
           "observation.state":          xr0_state32(task),
           "prompt":                     args_cli.prompt}
    ret = client.infer(obs)
    if "actions" not in ret:
        raise RuntimeError(f"server error: {ret}")
    action = np.asarray(ret["actions"], dtype=np.float32)      # (30,32)
    rs = xr0_robot_state(task)
    tg = recover_action(action, rs)
    Lp, Lr, Lg = tg["left_ee_pos"], tg["left_ee_rotm"], tg["left_gripper_pos"]
    Rp, Rr, Rg = tg["right_ee_pos"], tg["right_ee_rotm"], tg["right_gripper_pos"]
    if task.take_action_cnt == 0:
        s = xr0_state32(task)
        print(f"[DBG] L_joint6={np.round(s[7:13],3).tolist()} R_joint6={np.round(s[21:27],3).tolist()} "
              f"L_ee0={np.round(Lp[0],3).tolist()} R_ee0={np.round(Rp[0],3).tolist()}", flush=True)
    n = Lp.shape[0]
    for i in range(n):
        if args_cli.exec_horizon and i >= args_cli.exec_horizon:
            break
        qL = R.from_matrix(Lr[i]).as_quat()                    # xyzw
        qR = R.from_matrix(Rr[i]).as_quat()
        eeL = torch.tensor([float(Lp[i,0]),float(Lp[i,1]),float(Lp[i,2]),
                            float(qL[3]),float(qL[0]),float(qL[1]),float(qL[2]), float(Lg[i,0])], dtype=torch.float32)
        eeR = torch.tensor([float(Rp[i,0]),float(Rp[i,1]),float(Rp[i,2]),
                            float(qR[3]),float(qR[0]),float(qR[1]),float(qR[2]), float(Rg[i,0])], dtype=torch.float32)
        take_action_dual(task, eeL, eeR)
        if getattr(task, "eval_success", False):
            return
        if task.take_action_cnt >= task.cfg.step_lim:
            return

# -------- log --------
log_path = Path('./log')
def log(msg):
    _log_line(msg, log_path, args_cli.print_only)


# -------- eval_policy(): copied verbatim from the 7j single-arm client --------
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
    log(f"XR0-DUAL Task: {task_file_name} | config: {task_config_file.stem} | server {args_cli.server_host}:{args_cli.server_port}")
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
