"""Task `dual_cup_unstack`. Description, action flow, and exact success criteria: docs/Tasks.md#dual_cup_unstack"""
from ._base_task import *
import numpy as np
import torch

# ============================================================================
# Dual Cup UN-STACK: two paper cups start nested (upper cup inside the lower
# one); the robot separates them.
#   cup : CUP.usd (thin-wall conical paper cup, opening at local +Z, closed
#         bottom at -Z; rim outer dia 76 / base outer dia 54 / height 92 mm,
#         rolled lip at the top)
#   Init: cup_a (lower) upright on the table, cup_b (upper) upright nested
#         inside cup_a (center raised by NEST_RISE), coaxial.
#   Flow: arm A grasps the upper cup near its rim from above, lifts it
#         vertically off the lower cup, moves it to the +Y side of the table,
#         lowers it, and releases; the heavy lower cup stays put by weight.
#   End : cups horizontally separated, upper cup standing upright elsewhere.
#   Planner ignores both cups: they sit at the grasp location, and the
#   thin walls voxelize as solid, so planning would otherwise always fail.
# ----------------------------------------------------------------------------
# CUP local frame (origin centered, z in [-0.046, 0.046]): opening at +Z,
# closed bottom at -Z; origin-to-rim / origin-to-bottom = 0.046 m.
# Both arm bases at (0, +/-0.35, 0) face +x: avoid grasp points too far
# forward or too low (z < 0.04 is below the base plane and IK-infeasible).
# ============================================================================

# ---- Geometry (m) ----
CUP_SCALE   = 0.92     # uniform cup scale (slightly reduced, near original size, still graspable); all cup dims and nest rise scale with it
CUP_HALF    = 0.046 * CUP_SCALE   # cup half height (origin to rim/bottom)
CUP_TOP_R   = 0.038 * CUP_SCALE   # rim outer radius
CUP_BASE_R  = 0.027 * CUP_SCALE   # base outer radius
LIP_R       = 0.0402 * CUP_SCALE  # rolled-lip outer radius
OVERLAP     = 0.020 * CUP_SCALE   # nesting overlap (how deep the upper cup's base sits in the lower cup)
NEST_RISE   = 2 * CUP_HALF - OVERLAP   # +Z offset of the nested upper cup's center relative to the lower cup's
TABLE_TOP   = 0.004    # table top height (cup bottom rests here); cup center = TABLE_TOP + CUP_HALF

# ---- Layout [tune] (world frame) ----
UPRIGHT      = [1, 0, 0, 0]
# The nested stack sits at workspace center (y=0, symmetric reach for both
# arms at y=+/-0.35) and fairly close (x) for better reachability.
STACK_POS    = Pose([0.40,  0.00, TABLE_TOP + CUP_HALF], UPRIGHT)   # lower cup center (on table, close for low reach); upper cup nested above
# Assembly hold pose (upright, kept low so lifting and reaching are easy).
HOLD_POSE    = Pose([0.45,  0.00, 0.130], UPRIGHT)
# The extracted upper cup is placed here, on the placing arm's side
# (horizontal separation from the lower cup marks a successful unstack).
# With swapped roles: B (right, -Y) would hold the lower cup, A (left, +Y)
# extracts the upper cup and places it on the +Y side.
PLACE_POSE   = Pose([0.45,  0.22, 0.050], UPRIGHT)

# ---- Grasp [tune] ----
# Lower-cup grasp height is absolute: the small cup sits very low
# (center ~0.038), and points that are too low are IK-infeasible (the gripper
# would hang in mid-air). z ~ 0.060 is near the rim's outer wall (the nested
# upper cup does not block it) and verified reachable.
CUP_A_GRASP_DZ = (TABLE_TOP + 0.060) - (TABLE_TOP + CUP_HALF)   # puts the lower-cup grasp point at world z ~ 0.064 (reachable)
CUP_B_GRASP_DZ = CUP_HALF - 0.006     # top grasp on the upper cup: at the rim (highest, fully exposed part), descending from above
EXTRACT_RISE   = 0.13     # vertical extraction height (> nesting overlap + margin, guarantees full separation)
NEST_GAP       = 0.005    # small initial vertical gap: if the two UIPC thin-shell cups start interpenetrated,
                          # the IPC contact potential explodes (~26 s/step at reset -> timeout). The gap lets
                          # the upper cup settle in gently instead.
SUCCESS_HOLD_STEPS = 8


@configclass
class TaskCfg(BaseTaskCfg):
    dual_arm = True       # auto-creates arm B (/Robot_b + *_b tactile)
    # Default GUI camera: overhead front view covering both arms and the table.
    viewer = ViewerCfg(eye=(1.45, 0.0, 0.95), lookat=(0.45, 0.0, 0.12))
    video_size = (1760, 320)   # head + wrist + wrist_b (3x480) + 4 tactile streams (2 cols x 160) = 1760 wide
    # Wider dual-arm workspace: table enlarged to 2x2 m to cover both arms (bases at y=+/-0.35).
    plate = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ground_plate",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(SCENE_ASSETS_ROOT / "plate.usda"),
            scale=(2.0, 2.0, 1.0),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                kinematic_enabled=True,
            ),
        ),
    )
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(pos=(1.35, 0.0, 0.85), rot=(0.6352, 0.3107, 0.3107, 0.6352), convention="opengl"),
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.5, focus_distance=1.0, horizontal_aperture=3.6, clipping_range=(0.1, 100.0)
            ),
            width=480, height=270, update_period=1/120,
        ),
        CameraCfg(
            name="wrist",
            prim_path="/World/envs/env_.*/Robot/WristCamera/Camera",
            data_types=["rgb", "depth"],
            spawn=None,
            width=480, height=270, update_period=1/120,
        ),
        CameraCfg(
            name="wrist_b",
            prim_path="/World/envs/env_.*/Robot_b/WristCamera/Camera",
            data_types=["rgb", "depth"],
            spawn=None,
            width=480, height=270, update_period=1/120,
        ),
    ]
    step_lim = 1000  # eval step limit: max archived success steps x1.05 (data-calibrated 2026-07-25)
    reset_time_limit = 1200.0   # dual-arm + two nested UIPC thin-shell cups make reset slow (~600-700 s); allow ample margin


class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode: Literal['collect', 'eval'] = 'collect', render_mode: str | None = None, **kwargs):
        # Friction 20 -> 5: with high friction the nested cup walls "weld"
        # together and the lower cup gets pulled up with the upper one
        # (weld = mu * N_wall > lower cup weight m*g). The lower cup is held
        # down by its own weight (~8.8 N, friction-independent), so lowering
        # mu makes weld < gravity and the lower cup stays put during
        # extraction. The top grasp on the upper cup is a pinch
        # (finger force ~70 N x mu, far above the 8.8 N to support), so mu=5
        # still holds without slipping.
        cfg.sim.physics_material.dynamic_friction = 5.0
        cfg.sim.physics_material.static_friction = 5.0
        cfg.uipc_sim.contact.default_friction_ratio = 5.0
        # Both cups are manipulated objects and start nested at the grasp
        # location. The planner's frozen collision world is built inside
        # super().__init__(); without ignores, action 0 would collide with the
        # stack and planning would fail. The task-level ignore set must be
        # assigned before super().__init__() (get_curr_world_cfg reads the
        # task-level set union each arm's owning_manager set).
        self.planner_ignore_actors = {'cup_a', 'cup_b'}
        super().__init__(cfg, mode, render_mode, **kwargs)
        self._robot_manager.ignore_actors = {'cup_a', 'cup_b'}      # A holds the whole stack (lower cup + nested upper cup)
        self._robot_manager_b.ignore_actors = {'cup_b', 'cup_a'}    # B extracts the upper cup, passing through / leaving the lower cup

    # ---------------------------------------------------------------- actors
    def create_actors(self):
        # Two identical cups (same USD, different instance names); the nested
        # initial poses are set in _reset_actors.
        sc = (CUP_SCALE, CUP_SCALE, CUP_SCALE)
        # Thin-shell cups are very light by default (~18 g) and get pushed
        # away when the gripper closes. Higher density (with high table
        # friction) keeps them in place; the stack stays within Franka payload.
        CUP_DENSITY = 2e5
        self.cup_a = self._actor_manager.add_from_usd_file(
            name='cup_a', asset_path="CUP.usd", pose=STACK_POS, scale=sc, density=CUP_DENSITY)
        self.cup_b = self._actor_manager.add_from_usd_file(
            name='cup_b', asset_path="CUP.usd", scale=sc, density=CUP_DENSITY,
            pose=Pose([STACK_POS.p[0], STACK_POS.p[1], STACK_POS.p[2] + NEST_RISE + NEST_GAP], UPRIGHT))

    def _reset_actors(self):
        # Start nested: lower cup on the table, upper cup upright inside it
        # (center raised by NEST_RISE, same xy + small noise).
        noise = self.create_noise([0.002, 0.002, 0.0])   # small noise: the grasp already has tight tolerances
        base = STACK_POS.add_offset(noise)
        self.cup_a.set_pose(base)
        self.cup_b.set_pose(Pose([base.p[0], base.p[1], base.p[2] + NEST_RISE + NEST_GAP], UPRIGHT))
        self._success_hold_count = 0

    # ---------------------------------------------------------------- helpers
    def _grasp(self, actor, rm, atom, arm, dz, grasp_from=(0, 0, 1), camera_up=(1, 0, 0),
               extra_bias=(0.0, 0.0, 0.0), grip_depth=22.0, open_amt=1.0, side=0.13, up=0.10):
        """Two-stage approach grasp (avoids knocking the cup over on descent):

        open gripper -> move to a point beside/above the cup (backed off by
        `side` along the approach direction `grasp_from`, raised by `up`) ->
        descend vertically to cup height (clear of the cup) -> feed along
        -grasp_from to the grasp pose -> close.
        grasp_from: approach direction (side grasp = horizontal +Y; top grasp
        = vertical +Z). camera_up: up vector for construct_grasp_pose.
        """
        gf = np.array(grasp_from, dtype=float); gf = gf / np.linalg.norm(gf)
        self.move(atom.open_gripper(open_amt), arm=arm)
        target = actor.get_pose().add_bias([extra_bias[0], extra_bias[1], extra_bias[2] + dz], coord='world')
        gc = construct_grasp_pose(
            np.array([target.p[0], target.p[1], target.p[2]]),
            gf, np.array(camera_up, dtype=float))                       # gripper-center grasp pose
        gc_side = gc.add_bias((gf * side).tolist(), coord='world')      # backed off by `side` along the approach direction
        ee_grasp = rm.gripper_center_to_ee(gc)
        ee_side = rm.gripper_center_to_ee(gc_side)
        ee_high = ee_side.add_bias([0.0, 0.0, up], coord='world')       # above the backed-off point
        self.move(atom.move_to_pose(ee_high), arm=arm)                  # 1) move beside/above the cup
        self.move(atom.move_to_pose(ee_side), arm=arm)                  # 2) descend beside the cup (no impact)
        # 3) Linear feed of `side` along gripper-local z (= approach axis;
        #    horizontal for a side grasp): constraint locks the other 5 axes.
        #    The gripper center stops at the cup axis (no overshoot, which
        #    would push the cup away).
        self.move(atom.move_by_displacement(z=side + 0.01, xyz_coord='local'), arm=arm,
                  constraint_pose=[1, 1, 1, 1, 1, 0], time_dilation_factor=0.5)
        self.move(atom.close_gripper(depth_threshold=grip_depth), arm=arm)

    def _place_inhand(self, actor, rm, atom, target_pose, arm):
        """Place the in-hand actor exactly at target_pose using the grasp-time in-hand transform."""
        inhand = actor.get_pose().rebase(rm.get_gripper_center_pose())          # actor pose in gripper frame
        gc_mat = np.array(target_pose.to_transformation_matrix()) @ \
            np.linalg.inv(np.array(inhand.to_transformation_matrix()))          # gripper-center target
        ee = rm.gripper_center_to_ee(Pose.from_matrix(gc_mat))
        self.move(atom.move_to_pose(ee), arm=arm)

    def _dbg(self, tag):
        ga = self._robot_manager.get_gripper_qpos()
        gb = self._robot_manager_b.get_gripper_qpos()
        ap, bp = self.cup_a.get_pose(), self.cup_b.get_pose()
        print(f"[DBG] {tag}: gripA={ga:.4f} gripB={gb:.4f}  "
              f"cup_a=({ap.p[0]:.3f},{ap.p[1]:.3f},{ap.p[2]:.3f})  "
              f"cup_b=({bp.p[0]:.3f},{bp.p[1]:.3f},{bp.p[2]:.3f})", flush=True)

    # ---------------------------------------------------------------- script
    def pre_move(self):
        self.delay(15)   # let the nested stack settle; grasp/unstack runs in _play_once (recorded on video)

    def _play_once(self):
        # The heavy lower cup (2e5 density + high friction) stays on the table
        # by itself; no second arm needed. A grasps the upper cup's rim from
        # directly above (the table/lower cup backstops it, so closing cannot
        # push the cup away -- far more stable than a side grasp), lifts it
        # vertically off the lower cup, then moves it to the +Y side and
        # releases. The lower cup stays put by weight.
        self._grasp(self.cup_b, self._robot_manager, self.atom_a, 'a', CUP_B_GRASP_DZ,
                    grasp_from=(0, 0, 1), camera_up=(1, 0, 0), grip_depth=22.0, open_amt=1.0,
                    side=0.10, up=0.06)
        self._dbg("A top-grasped upper cup")

        # A: lift the upper cup straight up (world +Z); constraint_pose locks
        # the other 5 axes for a straight vertical separation.
        self.move(self.atom_a.move_by_displacement(z=EXTRACT_RISE, xyz_coord='world'),
                  arm='a', constraint_pose=[1, 1, 1, 1, 1, 0], time_dilation_factor=0.5)
        self._dbg("A lifted upper cup")

        # A: move the upper cup above the +Y placement point, lower to the table, release.
        hover = PLACE_POSE.add_bias([0.0, 0.0, 0.10])     # hover 10 cm above the placement point
        self._place_inhand(self.cup_b, self._robot_manager, self.atom_a, hover, 'a')
        self._dbg("A above placement point")
        self.move(self.atom_a.move_by_displacement(z=-0.10, xyz_coord='world'),
                  arm='a', time_dilation_factor=0.5)       # descend to PLACE_POSE.z (a former -0.085 released the cup 15 mm up in mid-air)
        self.delay(8, is_save=True)                         # let the cup settle before releasing
        self.move(self.atom_a.open_gripper(1.0), arm='a')  # release the upper cup
        self._dbg("A placed upper cup and released")
        self.delay(25, is_save=True)

    # ---------------------------------------------------------------- success
    def check_success(self):
        ap = self.cup_a.get_pose()
        bp = self.cup_b.get_pose()
        # Success = cups horizontally separated + upper cup placed stably + former lower cup not tipped over.
        horiz = float(np.linalg.norm(np.array(ap.p[:2], dtype=float) - np.array(bp.p[:2], dtype=float)))
        separated = horiz > 0.12
        a_up = float(np.dot(ap.to_transformation_matrix()[:3, 2], np.array([0, 0, 1]))) > 0.7
        b_up = float(np.dot(bp.to_transformation_matrix()[:3, 2], np.array([0, 0, 1]))) > 0.7
        self.metadata['cup_a'] = [float(v) for v in ap.p]
        self.metadata['cup_b'] = [float(v) for v in bp.p]
        self.metadata['horiz_sep'] = horiz
        self.metadata['cup_a_upright'] = bool(a_up)
        self.metadata['cup_b_upright'] = bool(b_up)
        print(f"[UNSTACK] horiz_sep={horiz*1000:.1f}mm a_up={a_up} b_up={b_up} "
              f"-> separated={separated}", flush=True)
        placed = float(bp.p[2]) < (TABLE_TOP + CUP_HALF + 0.035)   # top cup set down on table, not held aloft
        self.metadata['cup_b_z'] = float(bp.p[2])
        pa = self._robot_manager.get_gripper_percentage()
        released = pa > 0.9   # A gripper opened = cup actually let go before success can count
        bottom_ok = a_up
        success_now = separated and bottom_ok and b_up and placed and released
        if success_now:
            self._success_hold_count = getattr(self, '_success_hold_count', 0) + 1
        else:
            self._success_hold_count = 0
        self.metadata['released'] = bool(released)
        self.metadata['bottom_ok'] = bool(bottom_ok)
        self.metadata['success_hold_count'] = int(self._success_hold_count)
        print(f"[UNSTACK-REL] cup_b_z={float(bp.p[2])*1000:.0f}mm gripperA={pa:.2f} "
              f"placed={placed} released={released} bottom_ok={bottom_ok} "
              f"hold={self._success_hold_count}/{SUCCESS_HOLD_STEPS}", flush=True)
        return bool(self._success_hold_count >= SUCCESS_HOLD_STEPS)
