#!/usr/bin/env python3
"""Dual-arm EE eval client — derived from single-arm eval_twam_ee_cl.py (5 changes only):
  1. _pack_video: 3 cameras head/wrist/wrist_b -> top/wrist_l/wrist_r
  2. _pack_tac: 4 tactile left/right_tactile{,_b} -> tactile_ll/lr/rl/rr
  3. ee_to_rot6d_state: dual-arm 20 dims (arm A s[:10] + arm B s[10:20])
  4. vta_eval action execution: server returns 20 dims -> d[:10]=arm A / d[10:20]=arm B -> drive_dual
  5. drive_dual: per-arm solve_ik+set_arm+set_gripper (mirrors take_action ee_ik), then a single _step
Everything else (eval_policy/main/reset/seeds) matches the single-arm client verbatim.
"""
import sys, os, time, json, yaml, argparse, traceback
from pathlib import Path
from typing import Literal
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root (envs/, task_config/)
from _common import log_line as _log_line, get_config
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dual-arm EE eval")
parser.add_argument("task_name", type=str)
parser.add_argument("task_config", type=str)
parser.add_argument("--server_host", default="127.0.0.1")
parser.add_argument("--server_port", type=int, default=31255)
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
if os.environ.get('UNIVTAC_HEADLESS'): args_cli.livestream = 0; args_cli.headless = True
else: args_cli.livestream = 2
args_cli.num_envs = 1

import numpy as np
from scipy.spatial.transform import Rotation as R
from remote_infer_lib.websocket_client_policy import WebsocketClientPolicy
print(f"[dual-eval] connecting policy server {args_cli.server_host}:{args_cli.server_port} (before Isaac) ...", flush=True)
client = WebsocketClientPolicy(host=str(args_cli.server_host), port=int(args_cli.server_port))
print(f"[dual-eval] connected: {client.get_server_metadata()}", flush=True)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import torch, importlib
from envs._base_task import Pose   # Pose needed by dual-arm solve_ik (same source as take_action)

def to_u8(t):
    x = t[..., :3]; x = x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x); x = np.squeeze(x)
    if x.dtype != np.uint8:
        # float buffers arrive in BOTH [0,1] (cameras) and [0,255] (taxim tactile_rgb) ranges
        _sc = 1.0 if float(x.max()) > 1.5 else 255.0
        x = (x * _sc).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(x)

def _arm_state(rm):
    pose = rm.get_ee_pose()
    p_xyz = np.asarray(pose.p, dtype=np.float64).reshape(-1)[:3]
    q_wxyz = np.asarray(pose.q, dtype=np.float64).reshape(-1)
    Rm = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
    r6 = Rm[:, :2].T.reshape(-1); grip = float(rm.get_gripper_qpos())
    s = np.zeros(10, dtype=np.float32); s[:3] = p_xyz; s[3:9] = r6; s[9] = grip
    return s

def ee_to_rot6d_state_dual(task):   # (3) dual-arm 20 dims
    s = np.zeros(20, dtype=np.float32)
    s[:10] = _arm_state(task._robot_manager)
    s[10:20] = _arm_state(task._robot_manager_b)
    return s

def rot6d10_to_ee(d):
    xyz = np.asarray(d[:3], dtype=np.float64)
    c1 = np.asarray(d[3:6], dtype=np.float64); c2 = np.asarray(d[6:9], dtype=np.float64)
    b1 = c1 / (np.linalg.norm(c1) + 1e-8); c2 = c2 - (b1 @ c2) * b1; b2 = c2 / (np.linalg.norm(c2) + 1e-8)
    b3 = np.cross(b1, b2); Rm = np.stack([b1, b2, b3], axis=1)
    q_xyzw = R.from_matrix(Rm).as_quat(); grip = float(d[9])
    return [float(xyz[0]), float(xyz[1]), float(xyz[2]),
            float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]), grip]

def _drive_one(rm, ee, force=True):   # mirrors take_action ee_ik (direct IK)
    target_pose = Pose(p=ee[:3], q=ee[3:7])
    ik = rm.solve_ik(target_pose)
    if ik['status'] == 'Success': rm.set_arm(ik['position'], force=force)
    rm.set_gripper(ee[7], force=force)

def drive_dual(task, ee_a, ee_b, force=True):   # (5) both arms in one frame + _step + cnt
    # UNIVTAC_TICKS_PER_SLOT: ticks executed per action slot (same semantics as single-arm eval_twam_ee_cl).
    # One data row = 2 sim steps (save_frequency=2); the default 1 tick runs at 2x speed; set 2 to match the real cadence.
    # cnt counts ticks (single-arm take_action increments once per call; ticks=2 likewise counts 2), consistent with step_lim.
    for _t in range(int(os.environ.get('UNIVTAC_TICKS_PER_SLOT', '2'))):  # 2 = one dataset row per action (save_frequency=2)
        _drive_one(task._robot_manager, ee_a, force)
        _drive_one(task._robot_manager_b, ee_b, force)
        task._step()
        task.take_action_cnt += 1
        if getattr(task, 'eval_success', False) or task.take_action_cnt >= task.cfg.step_lim:
            break

def _pack_video(c):   # (1) 3 cameras
    return {'observation.images.top':     to_u8(c["head"]["rgb"]),
            'observation.images.wrist_l':  to_u8(c["wrist"]["rgb"]),
            'observation.images.wrist_r':  to_u8(c["wrist_b"]["rgb"])}

_CFG_TAC_KEY = None  # set in main() from task_config['policy_tac_key']

def _tac_key():
    # Priority: explicit env override > task_config policy_tac_key > legacy default.
    return os.environ.get('UNIVTAC_TAC_KEY') or _CFG_TAC_KEY or 'gel_particle'  # default = NeoSim's particle representation; legacy rgb lines set policy_tac_key in their config

def _pack_tac(t):   # (2) 4 tactile (repr pipeline: policy_tac_key=gel_particle in the config)
    _tk = _tac_key()
    return {'observation.images.tactile_ll': to_u8(t["left_tactile"][_tk]),
            'observation.images.tactile_lr': to_u8(t["right_tactile"][_tk]),
            'observation.images.tactile_rl': to_u8(t["left_tactile_b"][_tk]),
            'observation.images.tactile_rr': to_u8(t["right_tactile_b"][_tk])}

_VTA_FIRST = True
def vta_eval(task, observation):
    global _VTA_FIRST
    cam = observation["observation"]; tac = observation["tactile"]
    chunk_start_state = ee_to_rot6d_state_dual(task).tolist()
    # plain infer sends only the current tactile frame (as single-arm): the server VAE cold seed takes 1 frame; more frames break avg_shortcut
    # (WAN VAE temporal causality needs 1+4k frames). Tactile temporal alignment is carried by the kv_cache tac_frame_list.
    obs = {'obs': _pack_video(cam), 'tactile': _pack_tac(tac),
           'current_state': chunk_start_state, 'prompt': args_cli.prompt}
    ret = client.infer(obs)
    action = np.asarray(ret['action'])   # (C, F, H): C=channels (>=20, [:10]=arm A, [10:20]=arm B), F=frames, H=substeps per frame
    F = int(action.shape[1]); H = int(action.shape[2])
    key_frame_list = []; tac_frame_list = []
    apf = max(1, H // 4)   # 4 keyframes per frame (matches single-arm eval_twam_ee_cl)
    start_f = 1 if _VTA_FIRST else 0; _VTA_FIRST = False
    for i in range(start_f, F):
        for j in range(H):
            d = np.asarray(action[:, i, j]).reshape(-1)   # (C,) 20-dim action: [:10]=arm A, [10:20]=arm B
            ee_a = rot6d10_to_ee(d[:10]); ee_b = rot6d10_to_ee(d[10:20])   # (4) split into the two arms
            drive_dual(task, ee_a, ee_b)
            # dual path bypasses take_action, so check success explicitly (single-arm does it inside take_action): check_success sets eval_success
            if task.check_success():
                task.eval_success = True; return
            if task.take_action_cnt >= task.cfg.step_lim: return
            if (j + 1) % apf == 0:
                kobs = task._get_observations()
                key_frame_list.append(_pack_video(kobs["observation"]))
                tac_frame_list.append(_pack_tac(kobs["tactile"]))
                _hd = os.environ.get('UNIVTAC_HD_DUMP')
                if _hd:
                    try:
                        import cv2 as _cv2
                        os.makedirs(_hd, exist_ok=True)
                        _c = globals().setdefault('_HD_CTR', [0])
                        _idx = _c[0]; _c[0] += 1
                        _cam = kobs["observation"]; _tc = kobs["tactile"]
                        _cv2.imwrite(f"{_hd}/top_{_idx:04d}.png", to_u8(_cam["head"]["rgb"])[:, :, ::-1])
                        _cv2.imwrite(f"{_hd}/wristl_{_idx:04d}.png", to_u8(_cam["wrist"]["rgb"])[:, :, ::-1])
                        _cv2.imwrite(f"{_hd}/wristr_{_idx:04d}.png", to_u8(_cam["wrist_b"]["rgb"])[:, :, ::-1])
                        for _tk1 in os.environ.get('UNIVTAC_HD_TAC_KEY', 'rgb').split(','):
                            _tk1 = _tk1.strip()
                            for _sn, _sk in [("ll", "left_tactile"), ("lr", "right_tactile"),
                                             ("rl", "left_tactile_b"), ("rr", "right_tactile_b")]:
                                if _sk in _tc and _tk1 in _tc[_sk]:
                                    _cv2.imwrite(f"{_hd}/tac{_sn}_{_tk1}_{_idx:04d}.png",
                                                 to_u8(_tc[_sk][_tk1])[:, :, ::-1])
                        with open(f"{_hd}/meta.txt", "a") as _mf:
                            _mf.write(f"{_idx} frame={i} slot={j} act_cnt={task.take_action_cnt}\n")
                    except Exception as _he:
                        print(f"[hd-dump] {_he}", flush=True)
    if key_frame_list:
        try:
            client.infer({'obs': key_frame_list, 'tactile': tac_frame_list, 'state': action,
                          'current_state': chunk_start_state, 'compute_kv_cache': True,
                          'imagine': False, 'prompt': args_cli.prompt})
        except Exception as _e:
            print(f"[cl] compute_kv_cache FAILED -> abort seed: {_e}", flush=True)
            raise RuntimeError(f"compute_kv_cache commit failed: {_e}")

log_path = Path('./log')
def log(msg):
    _log_line(msg, log_path, args_cli.print_only)


def eval_policy(task, expert_check, start_seed, max_seed, test_total_num, instructions, instruciton_type='seen'):
    assert not expert_check, '--expert_check is not implemented in the dual-arm client'
    test_num, succ_num, seed = 0, 0, start_seed
    consec_err = 0
    seed_path = task.save_root.parent / 'seeds.json'; seed_path.parent.mkdir(parents=True, exist_ok=True)
    seed_status = json.load(open(seed_path)) if seed_path.exists() else {}
    while test_num < test_total_num and (max_seed == -1 or seed <= max_seed):
        if not seed_status.get(str(seed), True): seed += 1; continue
        test_num += 1; succ = False; eval_start = time.perf_counter(); task.mode = 'eval'
        try:
            task.reset(seed=seed, instructions=instructions[instruciton_type])
            task.mean_steps = task.cfg.step_lim
            try: client.infer(dict(reset=True, prompt=args_cli.prompt))
            except Exception: pass
            globals()['_VTA_FIRST'] = True   # reset: cold-seed flag (single tactile frame, no history to clear)
            while task.take_action_cnt < task.cfg.step_lim:
                vta_eval(task, task._get_observations())
                if task.eval_success: succ = True; break
                if task.check_early_stop(): break
        except Exception as e:
            log(f"[{test_num:<3d}] Seed {seed} exception: {e}\n{traceback.format_exc()}")
            task.clean_cache(result='error'); test_num -= 1
            consec_err += 1
            if consec_err >= 10:
                log(f"Aborting: {consec_err} consecutive errors (policy server down?)")
                break
        else:
            consec_err = 0
            if succ: succ_num += 1
            task.clean_cache(result='success' if succ else 'failed')
            log(f"[{test_num:<3d}] Seed {seed} {'success' if succ else 'failed'} "
                f"after {time.perf_counter()-eval_start:.1f}s. Total {succ_num}/{test_num}.")
        finally: seed += 1
    return {'test_num': test_num, 'succ_num': succ_num}

def main():
    global log_path
    task_file_name = args_cli.task_name
    task_config, task_config_file = get_config(args_cli.task_config, default_root=Path('./task_config'), type='yaml')
    try: instructions, _ = get_config(task_file_name, default_root=Path('./instructions'), type='json')
    except Exception: instructions = {'seen': ['Empty'], 'unseen': ['Empty']}
    task_module = importlib.import_module(f"envs.{task_file_name}")
    curr_time = time.strftime(r'%Y-%m-%d_%H:%M:%S')
    env_cfg = task_module.TaskCfg()
    env_cfg.save_dir = Path('eval_result') / 'lingbot_vta_dual' / task_file_name / task_config_file.stem / (os.environ.get('EVAL_TAG','run')+'_'+curr_time)
    env_cfg.decimation = task_config.get("decimation", env_cfg.decimation)
    env_cfg.obs_data_type = task_config.get("observations", {})
    env_cfg.save_frequency = task_config.get("save_frequency", env_cfg.save_frequency)
    env_cfg.video_frequency = task_config.get("video_frequency", env_cfg.video_frequency)
    env_cfg.scene.num_envs = 1
    if args_cli.device is not None: env_cfg.sim.device = args_cli.device
    # step_lim ships data-calibrated per task (max archived success steps x1.05);
    # EVAL_STEP_LIM is an explicit override for experiments.
    if os.environ.get('EVAL_STEP_LIM'): env_cfg.step_lim = int(os.environ['EVAL_STEP_LIM'])
    task = task_module.Task(env_cfg, mode='eval')
    log_path = task.save_root / "log.log"
    global _CFG_TAC_KEY
    _CFG_TAC_KEY = task_config.get('policy_tac_key')
    log(f"DUAL Task: {task_file_name} | cfg: {task_config_file.stem} | server {args_cli.server_host}:{args_cli.server_port} | prompt='{args_cli.prompt}' | tac_key={_tac_key()}")
    start_seed = 1000000 if args_cli.start_seed == -1 else args_cli.start_seed  # held-out range, aligned with the single-arm client (collection seeds start at 0)
    results = eval_policy(task, args_cli.expert_check, start_seed, args_cli.max_seed, args_cli.total_num, instructions, 'seen')
    log(f"Final: {results['succ_num']}/{results['test_num']} success.")
    task.close(); simulation_app.close()

if __name__ == "__main__":
    main()
