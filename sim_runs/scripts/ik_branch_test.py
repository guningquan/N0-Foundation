"""Standalone cuRobo check of the null-space branch chosen for Cup Handover's first planned motion
(N₀-VTLA：仿真实验.md §10.7). Start = the demos' home joints; goal = FK of a demo's joints at the end of
that motion, shifted by the cup's ±3 mm reset noise. Plans TRIALS times per method and prints the spread
of the final joints (a consistent branch gives j1/j3 std ~0.01 rad like Stack Bowls):
  default    MotionGen.plan_single as in envs/robot/curobo_planner.py
  rank       same, with the IK null-space ranking weight raised (the removed UNIVTAC_IK_NULLSPACE_RANK; no effect)
  nearest_js IK with all seeds, take the converged solution nearest to the start, then plan_single_js
  usage: ik_branch_test.py <demo.hdf5> [--trials 20]
"""
import argparse
import h5py, numpy as np, torch, yaml
from pathlib import Path
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig, PoseCostMetric

YML = Path(__file__).resolve().parents[2] / "N0-Foundation/neosim/assets/embodiments/franka/curobo.yml"


def load_motion_gen():
    cfg = yaml.safe_load(YML.read_text())
    k = cfg["robot_cfg"]["kinematics"]
    for key in ("urdf_path", "collision_spheres"):
        if not Path(k[key]).is_absolute():
            k[key] = str(YML.parent / k[key])
    world = WorldConfig(cuboid=[Cuboid(name="far", pose=[5, 5, 5, 1, 0, 0, 0], dims=[0.1, 0.1, 0.1])])
    mg = MotionGen(MotionGenConfig.load_from_robot_config(
        robot_cfg=cfg, world_model=world, interpolation_dt=1 / 120, position_threshold=0.001,
        rotation_threshold=0.01, high_precision=True, collision_activation_distance=0.025))
    mg.warmup()
    return mg


def first_motion(fn):
    with h5py.File(fn, "r") as f:
        q = f["embodiment/joint"][:, :7].astype(np.float32)
    moving = np.abs(np.diff(q, axis=0)).sum(1) > 1e-4
    end = next(t for t in range(1, len(moving)) if moving[t - 1] and not moving[t])
    return q[0], q[end]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("demo"); ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--goal-frame", type=int, default=None, help="use this frame's joints as the goal instead of the first motion end")
    ap.add_argument("--start-frame", type=int, default=None, help="start from this frame's joints instead of home")
    ap.add_argument("--constrained", action="store_true", help="hold orientation and x/y like the linear feed (pose_cost_metric)")
    a = ap.parse_args()
    mg = load_motion_gen(); ta = mg.tensor_args
    q0, q_end = first_motion(a.demo)
    if a.goal_frame is not None:
        with h5py.File(a.demo, "r") as f:
            q_end = f["embodiment/joint"][a.goal_frame, :7].astype(np.float32)
    if a.start_frame is not None:
        with h5py.File(a.demo, "r") as f:
            q0 = f["embodiment/joint"][a.start_frame, :7].astype(np.float32)
    start = JointState.from_position(ta.to_device(q0).view(1, -1), joint_names=mg.joint_names)
    goal = mg.compute_kinematics(JointState.from_position(ta.to_device(q_end).view(1, -1))).ee_pose
    rng = np.random.default_rng(0)
    offsets = rng.uniform(-0.003, 0.003, (a.trials, 2))
    w_default = mg.ik_solver.get_all_rollout_instances()[0].null_convergence.weight.clone()
    print(f"start {np.round(q0, 3)}\ndemo end {np.round(q_end, 3)}")
    for method in ["default", "rank", "nearest_js"]:
        for r in mg.ik_solver.get_all_rollout_instances():
            r.null_convergence.weight.copy_(w_default * 1000 if method == "rank" else w_default)
        finals, fails, errs = [], 0, []
        for dx, dy in offsets:
            mg.reset(reset_seed=True)   # as at every episode reset (robot._reset_idx -> planner.reset)
            pos = goal.position.clone(); pos[0, 0] += float(dx); pos[0, 1] += float(dy)
            g = Pose(pos, goal.quaternion.clone())
            if method == "nearest_js":
                ik = mg.solve_ik(g, retract_config=start.position.clone(),
                                 seed_config=start.position.view(1, 1, -1).clone(), return_seeds=32)
                ok = ik.success.view(-1)
                if not ok.any():
                    fails += 1; continue
                sols = ik.solution.view(-1, mg.dof)[ok]
                best = sols[torch.argmin(torch.linalg.norm(sols - start.position, dim=-1))]
                res = mg.plan_single_js(start, JointState.from_position(best.view(1, -1), joint_names=mg.joint_names),
                                        MotionGenPlanConfig(max_attempts=10))
            else:
                pc = MotionGenPlanConfig(max_attempts=10)
                if a.constrained:
                    pc.pose_cost_metric = PoseCostMetric(hold_partial_pose=True,
                                                         hold_vec_weight=ta.to_device([1, 1, 1, 1, 1, 0]))
                res = mg.plan_single(start, g, pc)
            if not res.success.item():
                fails += 1; continue
            qf = res.optimized_plan.position.view(-1, mg.dof)[-1]
            ee = mg.compute_kinematics(JointState.from_position(qf.view(1, -1))).ee_pose
            errs.append(float(torch.linalg.norm(ee.position - g.position)) * 1000)
            finals.append(qf.cpu().numpy())
        F = np.array(finals)
        print(f"{method:10s} ok {len(F)}/{a.trials}  final joint std {np.round(F.std(0), 3)}  "
              f"j1 range [{F[:, 0].min():.3f}, {F[:, 0].max():.3f}]  ee err max {max(errs):.2f} mm\n"
              f"{'':10s} final joint mean {np.round(F.mean(0), 3)}")


if __name__ == "__main__":
    main()
