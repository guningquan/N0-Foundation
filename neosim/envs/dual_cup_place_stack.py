"""Task `dual_cup_place_stack`. Description, action flow, and exact success criteria: docs/Tasks.md#dual_cup_place_stack"""
from ._base_task import *
import numpy as np
import torch

# ============================================================================
# Dual Cup PLACE-STACK: dual-arm cup stacking.
#   Init: two cups on the left/right table areas, opening at local +Z, closed
#         bottom at -Z.
#   Flow: A carries cup_a to the central target; B places cup_b directly on
#         top of cup_a to nest them.
#     cup_a/cup_b : assets/objects/CUP.usd
#       - source surface mesh: assets/objects/CUP.obj
#       - embedded tetrahedral mesh attributes: tet_points / tet_indices /
#         tet_surf_points / tet_surf_indices
#       - build scripts: scripts/asset_tools/build_cup.py,
#         scripts/asset_tools/bake_tet.py,
#         scripts/asset_tools/build_cup_usd.py
#   Geometry:
#     CUP.usd is a thin-wall conical paper cup, opening at local +Z, closed
#     bottom at local -Z. Original size: rim outer dia ~76 mm, base outer dia
#     ~54 mm, height ~92 mm; origin at body center, local z in ~[-46, +46] mm.
# ============================================================================

# ---- Geometry (m) ----
CUP_SCALE = 0.92
CUP_HALF = 0.046 * CUP_SCALE
CUP_TOP_R = 0.038 * CUP_SCALE
OVERLAP = 0.040 * CUP_SCALE
NEST_RISE = 2 * CUP_HALF - OVERLAP
TABLE_TOP = 0.004
GAP = 0.006
STACK_GAP = 0.0

# ---- Layout (world frame) ----
UPRIGHT = [1, 0, 0, 0]
CUP_A_START = Pose([0.40, 0.22, TABLE_TOP + CUP_HALF + GAP], UPRIGHT)
CUP_B_START = Pose([0.40, -0.22, TABLE_TOP + CUP_HALF + GAP], UPRIGHT)
STACK_TARGET = Pose([0.48, 0.00, TABLE_TOP + CUP_HALF + GAP], UPRIGHT)
STACK_TOP_TARGET = Pose(
    [STACK_TARGET.p[0], STACK_TARGET.p[1], STACK_TARGET.p[2] + NEST_RISE + STACK_GAP],
    UPRIGHT,
)

# ---- Motion parameters ----
GRASP_R = CUP_TOP_R + 0.002
GRASP_DZ = CUP_HALF - 0.011
GRASP_PRE_DIS = 0.12
PRE_GRIPPER_OPEN = 0.62
PLACE_HOVER = 0.11
RETRACT_Z = 0.10
ARM_A_PARK_POS = [0.36, 0.30, 0.24]
GRIPPER_RELEASE_THRESH = 0.65
SUCCESS_HOLD_STEPS = 8

# ---- Top grasp across the cup (staged descent as in dual_cup_stack/handover), no weld ----
USE_WELD = False               # False: cup held by gripper contact only (no rigid weld)
TOP_GRASP_DZ = CUP_HALF - 0.006  # grasp height: cup rim (relative to cup center)
TOP_GRASP_OPEN = 1.0           # fully open before grasping to clear the rim diameter
TOP_GRASP_CLOSE = 0.5          # closing amount (fingers press the outer wall; full close would crush the soft cup)
TOP_GRASP_SIDE = 0.10          # staged descent: back-off along the approach axis before the constrained linear feed
TOP_GRASP_UP = 0.06            # staged descent: extra height at the staging point


@configclass
class TaskCfg(BaseTaskCfg):
    dual_arm = True
    use_adaptive_grasp = False
    viewer = ViewerCfg(eye=(1.45, 0.0, 0.95), lookat=(0.45, 0.0, 0.12))
    video_size = (1760, 320)
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
            offset=CameraCfg.OffsetCfg(
                pos=(1.35, 0.0, 0.85),
                rot=(0.6352, 0.3107, 0.3107, 0.6352),
                convention="opengl",
            ),
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.5,
                focus_distance=1.0,
                horizontal_aperture=3.6,
                clipping_range=(0.1, 100.0),
            ),
            width=480,
            height=270,
            update_period=1 / 120,
        ),
        CameraCfg(
            name="wrist",
            prim_path="/World/envs/env_.*/Robot/WristCamera/Camera",
            data_types=["rgb", "depth"],
            spawn=None,
            width=480,
            height=270,
            update_period=1 / 120,
        ),
        CameraCfg(
            name="wrist_b",
            prim_path="/World/envs/env_.*/Robot_b/WristCamera/Camera",
            data_types=["rgb", "depth"],
            spawn=None,
            width=480,
            height=270,
            update_period=1 / 120,
        ),
    ]
    # The staged grasp lengthens the trajectory (~1500+ steps). Raise the
    # limits, otherwise record_one (saving every step) forces
    # plan_success=False once save_count>1000, truncating everything after
    # B grasps the second cup.
    step_lim = 1900  # eval step limit: max archived success steps x1.05 (data-calibrated 2026-07-25)
    max_save_frames = 2400
    reset_time_limit = 1200.0


class Task(BaseTask):
    def __init__(
        self,
        cfg: BaseTaskCfg,
        mode: Literal["collect", "eval"] = "collect",
        render_mode: str | None = None,
        **kwargs,
    ):
        cfg.sim.physics_material.dynamic_friction = 5.0
        cfg.sim.physics_material.static_friction = 5.0
        cfg.uipc_sim.contact.default_friction_ratio = 5.0
        # cup_a is A's in-hand/target object and must be ignored when the
        # planning world is initialized; cup_b stays as a static obstacle so
        # cuRobo does not hit the "Primitive Collision has no obstacles"
        # branch with only the ground present. Arm B's runtime ignore set
        # still contains cup_a/cup_b.
        self.planner_ignore_actors = {"cup_a"}
        super().__init__(cfg, mode, render_mode, **kwargs)
        self._robot_manager.ignore_actors = {"cup_a", "cup_b"}
        self._robot_manager_b.ignore_actors = {"cup_a", "cup_b"}

    # ---------------------------------------------------------------- actors
    def create_actors(self):
        sc = (CUP_SCALE, CUP_SCALE, CUP_SCALE)
        cup_density = 2e5
        self.cup_a = self._actor_manager.add_from_usd_file(
            name="cup_a",
            asset_path="CUP.usd",
            pose=CUP_A_START,
            scale=sc,
            density=cup_density,
        )
        self.cup_b = self._actor_manager.add_from_usd_file(
            name="cup_b",
            asset_path="CUP.usd",
            pose=CUP_B_START,
            scale=sc,
            density=cup_density,
        )

    def _reset_actors(self):
        noise_a = self.create_noise([0.003, 0.003, 0.0])
        noise_b = self.create_noise([0.003, 0.003, 0.0])
        self.cup_a.set_pose(CUP_A_START.add_offset(noise_a))
        self.cup_b.set_pose(CUP_B_START.add_offset(noise_b))
        self._success_hold_count = 0
        self.metadata["target_pose"] = [float(v) for v in STACK_TARGET.p]

    # ---------------------------------------------------------------- helpers
    def _register_rim_grasp(self, actor, rim_dir):
        rd = np.array(rim_dir, dtype=float)
        rd /= np.linalg.norm(rd)
        center = actor.get_pose()
        gp = np.array(center.p, dtype=float) + rd * GRASP_R + np.array([0.0, 0.0, GRASP_DZ])
        grasp_pose = construct_grasp_pose(
            gp,
            np.array([0.0, 0.0, 1.0], dtype=float),
            np.array([1.0, 0.0, 0.0], dtype=float),
        )
        return actor.register_point(grasp_pose, type="contact")

    def _grasp_cup(self, actor, atom, rm, arm, rim_dir=None):
        """Top grasp across the cup (staged stable descent as in dual_cup_stack/handover).

        Gripper center aligned with the cup axis. Fully open -> staging point
        above the rim -> vertical descent to the side pose -> 5-axis-locked
        linear press-down feed -> close. No weld by default.
        """
        gf = np.array([0.0, 0.0, 1.0], dtype=float)          # approach from directly above
        self.move(atom.open_gripper(TOP_GRASP_OPEN), arm=arm)
        target = actor.get_pose().add_bias([0.0, 0.0, TOP_GRASP_DZ], coord="world")
        gc = construct_grasp_pose(np.array(target.p, dtype=float), gf,
                                  np.array([1.0, 0.0, 0.0], dtype=float))
        gc_side = gc.add_bias((gf * TOP_GRASP_SIDE).tolist(), coord="world")   # slightly above the rim
        ee_side = rm.gripper_center_to_ee(gc_side)
        ee_high = ee_side.add_bias([0.0, 0.0, TOP_GRASP_UP], coord="world")
        self.move(atom.move_to_pose(ee_high), arm=arm)                        # 1) staging point
        self.move(atom.move_to_pose(ee_side), arm=arm)                        # 2) vertical descent to the side pose
        self.move(atom.move_by_displacement(z=TOP_GRASP_SIDE + 0.005, xyz_coord="local"),
                  arm=arm, constraint_pose=[1, 1, 1, 1, 1, 0], time_dilation_factor=0.5)  # 3) linear press-down
        self.move(atom.close_gripper(TOP_GRASP_CLOSE, depth_threshold=None), arm=arm)     # 4) close
        if USE_WELD:
            self.weld_actor(actor, rm)
        self.delay(8, is_save=True)

    def _place_inhand(self, actor, rm, atom, target_pose, arm):
        inhand = actor.get_pose().rebase(rm.get_gripper_center_pose())
        gc_mat = np.array(target_pose.to_transformation_matrix()) @ np.linalg.inv(
            np.array(inhand.to_transformation_matrix())
        )
        ee = rm.gripper_center_to_ee(Pose.from_matrix(gc_mat))
        self.move(atom.move_to_pose(ee), arm=arm, time_dilation_factor=0.5)

    def _unweld_actor(self, actor):
        self._welds = [w for w in self._welds if w[0] is not actor]

    def _cups_released(self):
        a_open = float(self._robot_manager.get_gripper_percentage()) >= GRIPPER_RELEASE_THRESH
        b_open = float(self._robot_manager_b.get_gripper_percentage()) >= GRIPPER_RELEASE_THRESH
        no_cup_welds = not any(w[0] is self.cup_a or w[0] is self.cup_b for w in self._welds)
        return a_open, b_open, no_cup_welds, bool(a_open and b_open and no_cup_welds)

    def _release_cup(self, actor, atom, arm, park=False, retract=True):
        self._unweld_actor(actor)
        self.delay(5, is_save=True)
        self.move(atom.open_gripper(1.0), arm=arm)
        if retract:
            self.move(
                atom.move_by_displacement(z=RETRACT_Z, xyz_coord="world"),
                arm=arm,
                time_dilation_factor=0.5,
            )
        if park:
            # If arm A stayed above the central target, arm B would keep
            # colliding with it while placing the second cup. After releasing
            # and lifting, A parks at a fixed spot above the left side;
            # back_to_origin() is avoided as it retreats too far visually.
            rm, _ = self._arm(arm)
            gc_now = rm.get_gripper_center_pose()
            park_gc = Pose(ARM_A_PARK_POS, gc_now.q)
            self.move(
                atom.move_to_pose(rm.gripper_center_to_ee(park_gc)),
                arm=arm,
                time_dilation_factor=0.5,
            )

    def _dbg(self, tag):
        ap, bp = self.cup_a.get_pose(), self.cup_b.get_pose()
        print(
            f"[PLACE_STACK] {tag}: "
            f"cup_a=({ap.p[0]:.3f},{ap.p[1]:.3f},{ap.p[2]:.3f}) "
            f"cup_b=({bp.p[0]:.3f},{bp.p[1]:.3f},{bp.p[2]:.3f})",
            flush=True,
        )

    # ---------------------------------------------------------------- script
    def pre_move(self):
        self.delay(15)
        self._dbg("reset settled")

    def _play_once(self):
        # A: carry the lower cup from the +Y side to the central target.
        self._grasp_cup(self.cup_a, self.atom_a, self._robot_manager, "a", rim_dir=(0, 1, 0))
        self._place_inhand(
            self.cup_a,
            self._robot_manager,
            self.atom_a,
            STACK_TARGET.add_bias([0.0, 0.0, PLACE_HOVER], coord="world"),
            "a",
        )
        self._place_inhand(self.cup_a, self._robot_manager, self.atom_a, STACK_TARGET, "a")
        self._release_cup(self.cup_a, self.atom_a, "a", park=True)
        self._dbg("A placed cup_a")

        # B: carry the upper cup from the -Y side to directly above cup_a and nest it.
        self._grasp_cup(self.cup_b, self.atom_b, self._robot_manager_b, "b", rim_dir=(0, -1, 0))
        # Key: align with cup_a's ACTUAL resting position, not the nominal
        # STACK_TARGET. cup_a carries +/-3 mm noise and shifts slightly after
        # placement (measured ~0.473, -0.005); placing at the fixed 0.48,0
        # makes the rims collide and knock the base cup over (main cause of a
        # 73% failure rate).
        ca = self.cup_a.get_pose()
        top_target = Pose(
            [float(ca.p[0]), float(ca.p[1]), float(ca.p[2]) + NEST_RISE + STACK_GAP],
            UPRIGHT,
        )
        self._place_inhand(
            self.cup_b,
            self._robot_manager_b,
            self.atom_b,
            top_target.add_bias([0.0, 0.0, PLACE_HOVER], coord="world"),
            "b",
        )
        self._place_inhand(self.cup_b, self._robot_manager_b, self.atom_b, top_target, "b")
        self._release_cup(self.cup_b, self.atom_b, "b", retract=False)
        self._dbg("B stacked cup_b")
        self.delay(30, is_save=True)

    # ---------------------------------------------------------------- success
    def check_success(self):
        ap = self.cup_a.get_pose()
        bp = self.cup_b.get_pose()
        target_xy = np.array(STACK_TARGET.p[:2], dtype=float)
        a_xy = np.array(ap.p[:2], dtype=float)
        b_xy = np.array(bp.p[:2], dtype=float)
        target_err = float(np.linalg.norm(a_xy - target_xy))
        stack_err = float(np.linalg.norm(b_xy - a_xy))
        dz = float(bp.p[2] - ap.p[2])
        height_err = abs(dz - NEST_RISE)
        a_up = float(np.dot(ap.to_transformation_matrix()[:3, 2], np.array([0, 0, 1]))) > 0.75
        b_up = float(np.dot(bp.to_transformation_matrix()[:3, 2], np.array([0, 0, 1]))) > 0.75
        a_open, b_open, no_cup_welds, released = self._cups_released()
        stacked_upright = target_err < 0.035 and stack_err < 0.025 and height_err < 0.025 and a_up and b_up
        if stacked_upright and released:
            self._success_hold_count = getattr(self, "_success_hold_count", 0) + 1
        else:
            self._success_hold_count = 0

        self.metadata["cup_a"] = [float(v) for v in ap.p]
        self.metadata["cup_b"] = [float(v) for v in bp.p]
        self.metadata["target_err"] = target_err
        self.metadata["stack_err"] = stack_err
        self.metadata["height_err"] = height_err
        self.metadata["gripper_a_open"] = a_open
        self.metadata["gripper_b_open"] = b_open
        self.metadata["cups_unwelded"] = no_cup_welds
        self.metadata["released"] = released
        self.metadata["stacked_upright"] = bool(stacked_upright)
        self.metadata["success_hold_count"] = int(self._success_hold_count)
        print(
            f"[PLACE_STACK] target_err={target_err*1000:.1f}mm "
            f"stack_err={stack_err*1000:.1f}mm dz={dz*1000:.1f}mm "
            f"height_err={height_err*1000:.1f}mm a_up={a_up} b_up={b_up} "
            f"a_open={a_open:.2f} b_open={b_open:.2f} unwelded={no_cup_welds} "
            f"stacked_upright={stacked_upright} hold={self._success_hold_count}/{SUCCESS_HOLD_STEPS}",
            flush=True,
        )
        # Without welding the cups nest naturally under gravity (dz smaller
        # than the rigid NEST_RISE), so success requires cup_b nested above
        # cup_a within a reasonable range (higher than the base, with actual
        # overlap) instead of dz exactly equal to NEST_RISE.
        nested = 0.010 < dz < 0.055
        return bool(target_err < 0.035 and stack_err < 0.025 and nested and a_up and b_up)
