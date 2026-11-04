"""Offline (teacher-forced) check of a served policy on expert demos (N₀-VTLA：仿真实验.md §10.7).

For every QUERY_EVERY-th frame t of a demo, send the recorded observation (3 cameras + 4 tactile pads
+ 16-dim joint state, decoded from the raw HDF5 exactly as the converter does) to serve_zmq.py and
compare the predicted chunk with the expert's next ACTION_HORIZON joint frames (action[t] = state[t+1]).
Errors are reported per phase of arm A's gripper (open approach / closing / holding) so a poor grasp
fit shows up separately from the rest of the trajectory.
  usage: offline_check.py <hdf5_dir> <seed> [<seed> ...] --port P --prompt "..." [--every 4]
"""
import argparse, json, sys
from pathlib import Path
import cv2, h5py, numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2] / "N0-Foundation/neosim/eval"))
from remote_infer_lib.n0vtla_zmq_client import N0VTLAZmqClient, ACTION_HORIZON

CAMS = {"head": "observation/head/rgb", "wrist": "observation/wrist/rgb", "wrist_b": "observation/wrist_b/rgb"}
TACS = {"left_tactile": "tactile/left_tactile/rgb", "right_tactile": "tactile/right_tactile/rgb",
        "left_tactile_b": "tactile/left_tactile_b/rgb", "right_tactile_b": "tactile/right_tactile_b/rgb"}


def img(f, key, t):
    return cv2.imdecode(np.frombuffer(f[key][t], np.uint8), cv2.IMREAD_COLOR)   # simulator RGB order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hdf5_dir"); ap.add_argument("seeds", type=int, nargs="+")
    ap.add_argument("--port", type=int, required=True); ap.add_argument("--prompt", required=True)
    ap.add_argument("--every", type=int, default=4); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    client = N0VTLAZmqClient(host="127.0.0.1", port=a.port); client.wait_for_server()
    rows = []
    for seed in a.seeds:
        with h5py.File(Path(a.hdf5_dir) / f"{seed}.hdf5", "r") as f:
            q = np.concatenate([f["embodiment/joint"][:, :8], f["embodiment_b/joint"][:, :8]], 1).astype(np.float32)
            T = len(q); client.reset()   # the first predict (frame 0) captures the tactile baseline, as in eval
            g = q[:, 7]; closing = np.where(np.diff(g) < -1e-4)[0]
            c0, c1 = (closing[0], closing[-1] + 1) if len(closing) else (T, T)
            for t in range(0, T - ACTION_HORIZON - 1, a.every):
                pred = client.predict(q[t], a.prompt, cams={k: img(f, v, t) for k, v in CAMS.items()},
                                      tacs={k: img(f, v, t) for k, v in TACS.items()})
                gt = q[t + 1:t + 1 + ACTION_HORIZON]
                err = np.abs(np.asarray(pred) - gt)                      # (H, 16)
                phase = "approach" if t < c0 - 20 else "grasp" if t <= c1 + 10 else "after"
                rows.append({"seed": seed, "t": t, "phase": phase,
                             "armA_joint_mae": float(err[:, :7].mean()), "armB_joint_mae": float(err[:, 8:15].mean()),
                             "armA_joint_max": float(err[:, :7].max()),
                             "gripA_mae": float(err[:, 7].mean()), "gripB_mae": float(err[:, 15].mean()),
                             "gripA_pred_end": float(pred[-1][7]), "gripA_gt_end": float(gt[-1, 7]),
                             "step_change_A": float(np.abs(gt[-1, :7] - q[t, :7]).mean())})
            print(f"seed {seed}: {T} frames, A closes at frames {c0}-{c1}", flush=True)
    if a.out: Path(a.out).write_text("\n".join(json.dumps(r) for r in rows))
    print(f"\n{'phase':9s} {'n':>4s} {'A joint MAE':>12s} {'A joint max':>12s} {'B joint MAE':>12s} "
          f"{'gripA MAE':>10s} {'gripB MAE':>10s} {'A motion/chunk':>15s}")
    for ph in ["approach", "grasp", "after", "all"]:
        r = [x for x in rows if ph == "all" or x["phase"] == ph]
        if not r: continue
        m = lambda k: np.mean([x[k] for x in r])
        print(f"{ph:9s} {len(r):4d} {m('armA_joint_mae'):12.4f} {m('armA_joint_max'):12.4f} {m('armB_joint_mae'):12.4f} "
              f"{m('gripA_mae'):10.4f} {m('gripB_mae'):10.4f} {m('step_change_A'):15.4f}")
    print("\ngrasp phase, arm A gripper at chunk end (pred vs expert):")
    for x in [x for x in rows if x["phase"] == "grasp"][:40]:
        print(f"  seed {x['seed']} t={x['t']:4d}  pred {x['gripA_pred_end']:.4f}  expert {x['gripA_gt_end']:.4f}  "
              f"A joint MAE {x['armA_joint_mae']:.4f}")


if __name__ == "__main__":
    main()
