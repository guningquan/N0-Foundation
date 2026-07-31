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
from _common import log_line as _log_line, get_config

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
if os.environ.get('UNIVTAC_GUI'):
    args_cli.livestream = 0
    args_cli.headless = False
elif os.environ.get('UNIVTAC_HEADLESS'):
    args_cli.livestream = 0
    args_cli.headless = True
else:
    args_cli.livestream = 2
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

def ee_to_rot6d_state(task):
    pose = task._robot_manager.get_ee_pose()
    p_xyz = np.asarray(pose.p, dtype=np.float64).reshape(-1)[:3]
    q_wxyz = np.asarray(pose.q, dtype=np.float64).reshape(-1)
    Rm = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
    r6 = Rm[:, :2].T.reshape(-1)
    grip = float(task._robot_manager.get_gripper_qpos())
    s = np.zeros(20, dtype=np.float32); s[:3] = p_xyz; s[3:9] = r6; s[9] = grip
    return s

def rot6d10_to_ee(d):
    xyz = np.asarray(d[:3], dtype=np.float64)
    c1 = np.asarray(d[3:6], dtype=np.float64); c2 = np.asarray(d[6:9], dtype=np.float64)
    b1 = c1 / (np.linalg.norm(c1) + 1e-8)
    c2 = c2 - (b1 @ c2) * b1; b2 = c2 / (np.linalg.norm(c2) + 1e-8)
    b3 = np.cross(b1, b2)
    Rm = np.stack([b1, b2, b3], axis=1)
    q_xyzw = R.from_matrix(Rm).as_quat()
    grip = float(d[9])
    return [float(xyz[0]), float(xyz[1]), float(xyz[2]),
            float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]), grip]

_patched = {"done": False}
def _ensure_gripper_patch(task):
    if _patched["done"]:
        return
    _rm = task._robot_manager; _o = _rm.plan_gripper
    def _pg(pos, type='percent', _o=_o):
        if isinstance(pos, torch.Tensor):
            pos = float(pos.reshape(-1)[0])
        elif hasattr(pos, '__len__'):
            pos = float(pos[0])
        else:
            pos = float(pos)
        return _o(pos, type)
    _rm.plan_gripper = _pg
    _patched["done"] = True

_VTA_FIRST = True   # set True after each seed reset (chunk0 skips frame 0)

def _pack_video(c):
    return {'observation.images.top': to_u8(c["head"]["rgb"]),
            'observation.images.wrist_l': to_u8(c["wrist"]["rgb"])}

def _pack_tac(t):
    return {'observation.images.tactile_a': to_u8(t["left_tactile"]["rgb"]),
            'observation.images.tactile_b': to_u8(t["right_tactile"]["rgb"])}

def _env_int(name, default):
    try:
        return int(os.environ.get(name, str(default)) or default)
    except ValueError:
        return int(default)

def _exec_slots(total_slots):
    requested = _env_int('UNIVTAC_EXEC_SLOTS', 0)
    return min(total_slots, requested) if requested > 0 else total_slots

def _ticks_per_slot():
    return max(1, _env_int('UNIVTAC_TICKS_PER_SLOT', 2))  # 2 = one dataset row (save_frequency=2) per action; matches the historical eval protocol

def _select_tactile_keyframes(tactile_frames):
    keep = _env_int('UNIVTAC_TAC_KEYFRAMES', 0)
    if keep > 0:
        return tactile_frames[-keep:]
    return tactile_frames

def vta_eval(task, observation):
    """Closed loop: infer -> execute action chunk (collect keyframes per frame) -> compute_kv_cache re-grounds on real observations + action history."""
    global _VTA_FIRST
    _ensure_gripper_patch(task)
    cam = observation["observation"]; tac = observation["tactile"]
    chunk_start_state = ee_to_rot6d_state(task).tolist()
    obs = {'obs': _pack_video(cam),
           'current_state': chunk_start_state, 'prompt': args_cli.prompt}
    ret = client.infer(obs)
    action = np.asarray(ret['action'])          # (C, F, H)
    try:   # diagnostics (opt-in: set UNIVTAC_TGT_DUMP=1): dump per-chunk target values to check monotonicity
        import os as _os
        assert _os.environ.get('UNIVTAC_TGT_DUMP'), 'tgtdump disabled'
        _os.makedirs('/tmp/tgtdump', exist_ok=True)
        with open('/tmp/tgtdump/%d.txt' % _os.getpid(), 'a') as _f:
            for _i in range(action.shape[1]):
                for _j in range(action.shape[2]):
                    _v = np.asarray(action[:, _i, _j]).reshape(-1)[:10]
                    _f.write(' '.join('%.4f' % _x for _x in _v) + '\n')
    except Exception: pass
    F = int(action.shape[1]); H = int(action.shape[2])
    key_frame_list = []; tac_frame_list = []
    apf = max(1, H // 4)   # 4 video keyframes per frame; tactile follows the same cadence, trimmable via UNIVTAC_TAC_KEYFRAMES
    start_f = 1 if _VTA_FIRST else 0
    _VTA_FIRST = False
    for i in range(start_f, F):
        for j in range(_exec_slots(H)):
            d = np.asarray(action[:, i, j]).reshape(-1)
            ee = torch.tensor(rot6d10_to_ee(d[:10]), dtype=torch.float32)
            for _ in range(_ticks_per_slot()):
                task.take_action(ee, action_type='ee_ik')
                if getattr(task, "eval_success", False):
                    break
                if task.take_action_cnt >= task.cfg.step_lim:
                    break
            if getattr(task, "eval_success", False):
                return
            if task.take_action_cnt >= task.cfg.step_lim:
                return
            if (j + 1) % apf == 0:
                kobs = task._get_observations()
                key_frame_list.append(_pack_video(kobs["observation"]))
    # closed-loop grounding: write real new observations (keyframes) + the actually executed action chunk into the KV cache
    if key_frame_list:
        try:
            client.infer({'obs': key_frame_list,
                          'state': action, 'current_state': chunk_start_state,
                          'compute_kv_cache': True, 'imagine': False,
                          'prompt': args_cli.prompt})
        except Exception as _e:
            # silent kv failure = the policy silently degrades to open-loop; a poisoned
            # episode must abort (recorded as error) per the "kv failures == 0" validity rule.
            print(f"[cl] compute_kv_cache FAILED -> abort seed: {_e}", flush=True)
            raise RuntimeError(f"compute_kv_cache commit failed: {_e}")

# -------- log (copied from official harness) --------
log_path = Path('./log')
def log(msg):
    _log_line(msg, log_path, args_cli.print_only)


# -------- eval_policy(): mirrors the scripts/eval_policy.py loop semantics (success latching / step limits identical); inference is a policy-server round-trip. Self-contained: no dependency on policy/ (see eval/README.md) --------
def eval_policy(task, expert_check, start_seed, max_seed, test_total_num, instructions, instruciton_type='seen'):
    from shutil import ExecError
    test_num, succ_num, seed = 0, 0, start_seed
    consec_err = 0
    seed_path = task.save_root.parent / 'seeds.json'
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    if seed_path.exists():
        with open(seed_path, 'r') as f:
            seed_status = json.load(f)
    else:
        seed_status = {}

    while test_num < test_total_num and (max_seed == -1 or seed <= max_seed):
        if not seed_status.get(str(seed), True):
            seed += 1
            continue

        if expert_check and str(seed) not in seed_status:
            test_start = time.perf_counter()
            task.mode = 'eval_test'
            try:
                task.reset(seed=seed)
                task.play_once()
                if not task.check_success() or not task.plan_success:
                    raise ExecError(f'seed {seed} Expert check failed, check {task.check_success()}, plan {task.plan_success}.')
                else:
                    seed_status[seed] = True
                    with open(seed_path, 'w') as f:
                        json.dump(seed_status, f)
                test_cost = time.perf_counter() - test_start
                task.clean_cache(result='test_success')
                log(f'Expert check succ, seed {seed}, cost {test_cost:.2f}s')
            except Exception as e:
                test_cost = time.perf_counter() - test_start
                log(f'Expert check failed, seed {seed}, cost {test_cost:.2f}s, with exception {e}')
                task.clean_cache(result='test_fail')
                seed_status[seed] = False
                with open(seed_path, 'w') as f:
                    json.dump(seed_status, f)
                seed += 1
                continue
        test_num += 1

        succ = False
        eval_start = time.perf_counter()
        task.mode = 'eval'
        try:
            task.reset(seed=seed, instructions=instructions[instruciton_type])
            task.mean_steps = task.cfg.step_lim
            try:
                client.infer(dict(reset=True, prompt=args_cli.prompt))   # = policy.reset()
            except Exception:
                pass
            globals()['_VTA_FIRST'] = True   # after each seed reset, chunk0 skips frame 0 again
            while task.take_action_cnt < task.cfg.step_lim:
                observation = task._get_observations()
                vta_eval(task, observation)                              # = policy.eval()
                if task.eval_success:
                    succ = True
                    break
                if task.check_early_stop():
                    break
        except Exception as e:
            log(f"[{test_num:<3d}] Seed {seed} occurred exception: {e}\n{traceback.format_exc()}")
            succ_status = 'error'
            task.clean_cache(result=succ_status)
            test_num -= 1
            consec_err += 1
            if consec_err >= 10:
                log(f"Aborting: {consec_err} consecutive errors (policy server down?)")
                break
        else:
            eval_cost = time.perf_counter() - eval_start
            if succ:
                succ_num += 1
            consec_err = 0
            succ_status = 'success' if succ else 'failed'
            task.clean_cache(result=succ_status)
            log(f"[{test_num:<3d}] Seed {seed} {succ_status} after {eval_cost:.2f} s.\n"
                f"steps: {task.step_count:<5d}, actions: {task.take_action_cnt:<5d}.\n"
                f"Instruction: {task.instruction}\n"
                f"Total {succ_num}/{test_num}({succ_num/max(1,test_num)*100:.2f}%) success.")
        finally:
            seed += 1

    return {'test_num': test_num, 'succ_num': succ_num}

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
    env_cfg.save_dir = Path('eval_result') / 'lingbot_vta_ee' / task_file_name / task_config_file.stem / (os.environ.get('EVAL_TAG','run')+'_'+curr_time)
    env_cfg.decimation = task_config.get("decimation", env_cfg.decimation)
    env_cfg.obs_data_type = task_config.get("observations", {})
    env_cfg.save_frequency = task_config.get("save_frequency", env_cfg.save_frequency)
    env_cfg.video_frequency = task_config.get("video_frequency", env_cfg.video_frequency)
    env_cfg.random_texture = task_config.get("random_texture", False)
    env_cfg.scene.num_envs = 1
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    import os as _os

    # step_lim now ships data-calibrated per task (max archived success steps x1.05);
    # EVAL_STEP_LIM is an explicit override for experiments.
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
