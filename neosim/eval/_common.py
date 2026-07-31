"""Shared skeleton for the eval clients: byte-identical pieces extracted verbatim.

`eval_policy_full` / `eval_policy_lite` are the two loop variants that were
previously copied across clients; the only client-specific behavior is injected
via `infer_fn` (one policy-server round-trip), `reset_fn` (server-side reset)
and `log`. Loop semantics (expert check, success latching, early stop, seed
bookkeeping, metadata counters) are unchanged — see eval/README.md.
"""
import json
import time
import yaml
import traceback
from pathlib import Path


def get_config(file, default_root, type):
    if type == 'yaml':
        file = Path(file) if (str(file).endswith('.yml') or str(file).endswith('.yaml')) else default_root / f'{file}.yml'
        with open(file, 'r') as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader), file
    else:
        file = Path(file) if str(file).endswith('.json') else default_root / f'{file}.json'
        with open(file, 'r') as f:
            return json.load(f), file


def log_line(msg, log_path, print_only=False):
    msg = f"[{time.strftime(r'%Y-%m-%d %H:%M:%S')}] {msg}"
    if not print_only:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')
    print(msg, flush=True)


def eval_policy_full(task, expert_check, start_seed, max_seed, test_total_num,
                     instructions, instruciton_type='seen', *,
                     infer_fn, reset_fn, log):
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
                    seed_status[str(seed)] = True
                    with open(seed_path, 'w') as f:
                        json.dump(seed_status, f)
                test_cost = time.perf_counter() - test_start
                task.clean_cache(result='test_success')
                log(f'Expert check succ, seed {seed}, cost {test_cost:.2f}s')
            except Exception as e:
                test_cost = time.perf_counter() - test_start
                log(f'Expert check failed, seed {seed}, cost {test_cost:.2f}s, with exception {e}')
                task.clean_cache(result='test_fail')
                seed_status[str(seed)] = False
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
                reset_fn()                                               # = policy.reset()
            except Exception:
                pass
            while task.take_action_cnt < task.cfg.step_lim:
                observation = task._get_observations()
                infer_fn(task, observation)                              # = policy.eval()
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


def eval_policy_lite(task, start_seed, max_seed, test_total_num, instructions,
                     itype='seen', *, infer_fn, reset_fn, log):
    test_num, succ_num, seed = 0, 0, start_seed
    consec_err = 0
    while test_num < test_total_num and (max_seed == -1 or seed <= max_seed):
        test_num += 1; succ = False; eval_start = time.perf_counter(); task.mode = 'eval'
        try:
            task.reset(seed=seed, instructions=instructions[itype])
            task.mean_steps = task.cfg.step_lim
            try: reset_fn()
            except Exception: pass
            while task.take_action_cnt < task.cfg.step_lim:
                infer_fn(task, task._get_observations())
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
