"""Null-space spread of the expert demos (N₀-VTLA：仿真实验.md §10.7): joint std of arm A across episodes at
the end of its first planned motion and at the first gripper-close frame, next to the gripper-center
(ee) std. A 7-DoF arm can reach the same ee pose with different joints; if the demos do (cup task
2026-10-31: j1/j3 std ~0.065 rad at ~1.6 mm ee std), a joint-space policy mixes the branches.
  usage: joint_spread.py <hdf5_dir> [<hdf5_dir> ...]
"""
import glob, sys
import h5py, numpy as np

for d in sys.argv[1:]:
    first, close, ee = [], [], []
    for fn in sorted(glob.glob(f"{d}/*.hdf5")):
        with h5py.File(fn, "r") as f:
            q = f["embodiment/joint"][:, :8]
            e = f["embodiment/ee"][:, :3] if "embodiment/ee" in f else None
        moving = np.abs(np.diff(q[:, :7], axis=0)).sum(1) > 1e-4
        stops = [t for t in range(1, len(moving)) if moving[t - 1] and not moving[t]]
        closing = np.where(np.diff(q[:, 7]) < -1e-4)[0]
        if not stops or not len(closing):
            continue
        first.append(q[stops[0], :7]); close.append(q[closing[0], :7])
        if e is not None:
            ee.append(e[closing[0]])
    print(f"{d}: {len(close)} episodes")
    for name, Q in [("first motion end", first), ("gripper close", close)]:
        Q = np.array(Q)
        print(f"  {name:16s} joint std {np.round(Q.std(0), 3)}  j1 range [{Q[:, 0].min():.3f}, {Q[:, 0].max():.3f}]")
    if ee:
        print(f"  gripper close    ee std (mm) {np.round(np.array(ee).std(0) * 1000, 2)}")
