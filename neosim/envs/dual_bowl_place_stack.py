"""Task `dual_bowl_place_stack`. Description, action flow, and exact success criteria: docs/Tasks.md#dual_bowl_place_stack"""
from ._base_task import *
import numpy as np
import torch

# ============================================================================
# Dual Bowl PLACE-STACK: dual-arm bowl stacking (mirrors dual_cup_place_stack
# with bowls instead of cups).
#   Init: two bowls on the left/right table areas, opening at local +Z,
#         closed bottom at -Z.
#   Flow: A carries bowl_a to the central target and sets it down; B places
#         bowl_b directly on top of bowl_a to nest them.
#     bowl_a/bowl_b : assets/objects/BOWL_STACK.usd (130 mm dia x H58 bowl
#       with vertical outer wall and rounded foot ring; opening at local +Z,
#       closed bottom -Z, origin centered; USD is true scale -> scale=1.0).
#       (Note: this is a different asset from dual_bowl_unstack's BOWL.usd,
#       which is the original H39.4 shallow bowl.)
#   Key differences vs the cup version (same as dual_bowl_unstack):
#     The 130 mm opening exceeds the gripper span, so the bowl cannot be
#     grasped across the body; the gripper pinches one side of the rim
#     (one finger on the inner wall, one on the outer wall, clamping ~5 mm
#     of wall ~5 mm below the rim).
#     No adaptive grasp: use_adaptive_grasp=False + close_gripper(0.0) closes
#     fully; optionally weld_actor rigidly binds the bowl to the gripper so
#     it cannot slip during transport.
# ----------------------------------------------------------------------------
# BOWL local frame (origin centered, z in [-half, +half]): opening at +Z,
# closed bottom at -Z; origin-to-rim = BOWL_HALF.
# ============================================================================

# ---- Geometry (m) ----
BOWL_HALF    = 0.029     # half height (origin to rim/bottom); USD is true scale (130 mm dia, 58 mm tall), scale=1.0
BOWL_OUTER_R = 0.065     # rim outer radius (130 mm dia)
RIM_GRASP_R  = 0.0625    # horizontal distance from gripper center (TCP) to bowl axis (vertical-wall midline r ~ 62.5 mm, wall spans 60.5-65)
FOOT_R       = 0.026     # foot ring outer radius
TABLE_TOP    = 0.004     # table top height (bowl bottom rests here); bowl center = TABLE_TOP + BOWL_HALF
GAP          = 0.006     # small initial center-to-table gap (margin for placing/grasping)
# Weld toggle: with the vertical outer wall, a fully closed gripper is blocked
# by the 5 mm wall and forms a strong pinch; contact friction alone carries
# the bowl, so welding is off. Set True to fall back to rigid binding.
USE_WELD     = False
# Nesting rise (as in dual_bowl_unstack): the nested upper bowl's center sits
# NEST_RISE above the lower bowl's. Vertical-wall bowls nest more shallowly
# (the outer wall is no longer sloped), so success uses a range check instead
# of a fixed value (see check_success).
NEST_RISE    = 0.048
STACK_GAP    = 0.0

# ---- Layout (world frame) ----
UPRIGHT      = [1, 0, 0, 0]
BOWL_A_START = Pose([0.42,  0.26, TABLE_TOP + BOWL_HALF + GAP], UPRIGHT)   # lower bowl on the +Y side (arm A)
BOWL_B_START = Pose([0.42, -0.26, TABLE_TOP + BOWL_HALF + GAP], UPRIGHT)   # upper bowl on the -Y side (arm B)
STACK_TARGET = Pose([0.48,  0.00, TABLE_TOP + BOWL_HALF + GAP], UPRIGHT)   # lower bowl target (center)
STACK_TOP_TARGET = Pose(
    [STACK_TARGET.p[0], STACK_TARGET.p[1], STACK_TARGET.p[2] + NEST_RISE + STACK_GAP],
    UPRIGHT,
)

# ---- Motion parameters ----
RIM_GRASP_DZ    = BOWL_HALF - 0.022   # vertical grasp offset (~22 mm below the rim: deeper grip is steadier, the tall wall allows it)
RIM_DOWN_EXTRA  = 0.020              # extra 20 mm after reaching the pose so the pads seat firmly on both walls (as in unstack)
GRASP_SIDE      = 0.05               # hover height component (side)
GRASP_UP        = 0.06               # hover height component (up)
GRASP_LIFT      = 0.10                # vertical lift right after grasping (before lateral moves, so the bowl is not dragged along the table)
GRASP_PRE_DIS   = 0.12                # hover distance along the approach axis before grasping
PRE_GRIPPER_OPEN = 1.0                # rim grasp: fully open first so the fingers straddle the inner/outer walls
PLACE_HOVER     = 0.12                # hover height directly above the placement target
RETRACT_Z       = 0.12                # vertical retreat after releasing
ARM_A_PARK_POS  = [0.40, 0.32, 0.26]  # arm A parks above +Y after placing the lower bowl, clearing the way for arm B
SUCCESS_HOLD_STEPS   = 8               # consecutive steps of (nested upright + both grippers released) required (as in the cup version)
GRIPPER_RELEASE_THRESH = 0.65          # gripper opening percentage >= this counts as released


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
    # Dual-arm + rim grasp (segmented constrained linear feeds) + nesting
    # placement runs longer than the cup version; without welding each grasp
    # adds press-down + forced close + settle steps.
    step_lim = 2800  # eval step limit: max archived success steps x1.05 (data-calibrated 2026-07-25)
    max_save_frames = 2600
    reset_time_limit = 1200.0


class Task(BaseTask):
    def __init__(
        self,
        cfg: BaseTaskCfg,
        mode: Literal["collect", "eval"] = "collect",
        render_mode: str | None = None,
        **kwargs,
    ):
        # Medium-high friction: helps hold the rim (line contact) without the
        # contact-solve cost of very high friction.
        cfg.sim.physics_material.dynamic_friction = 5.0
        cfg.sim.physics_material.static_friction = 5.0
        cfg.uipc_sim.contact.default_friction_ratio = 5.0
        # bowl_a is A's in-hand/target object and must be ignored when the
        # planning world is initialized; bowl_b stays as a static obstacle so
        # cuRobo does not hit the "Primitive Collision has no obstacles"
        # branch with only the ground present.
        self.planner_ignore_actors = {"bowl_a"}
        super().__init__(cfg, mode, render_mode, **kwargs)
        self._robot_manager.ignore_actors = {"bowl_a", "bowl_b"}
        self._robot_manager_b.ignore_actors = {"bowl_a", "bowl_b"}

    # ---------------------------------------------------------------- actors
    def create_actors(self):
        # The lower bowl (bowl_a) later serves as the base that must withstand
        # the upper bowl nesting into it -> denser + high friction to stay
        # put; the upper bowl (bowl_b) stays light for handling. Without
        # welding, the thin-wall pinch must carry the whole bowl: the lower
        # bowl is kept much lighter than before (2e5 would skew the grip or
        # slip out) -- just heavy enough to act as a stable base.
        self.bowl_a = self._actor_manager.add_from_usd_file(
            name="bowl_a",
            asset_path="BOWL_STACK.usd",
            pose=BOWL_A_START,
            density=1e4,
        )
        self.bowl_b = self._actor_manager.add_from_usd_file(
            name="bowl_b",
            asset_path="BOWL_STACK.usd",
            pose=BOWL_B_START,
            density=5e3,
        )

    def _reset_actors(self):
        noise_a = self.create_noise([0.003, 0.003, 0.0])
        noise_b = self.create_noise([0.003, 0.003, 0.0])
        self.bowl_a.set_pose(BOWL_A_START.add_offset(noise_a))
        self.bowl_b.set_pose(BOWL_B_START.add_offset(noise_b))
        self.metadata["target_pose"] = [float(v) for v in STACK_TARGET.p]
        self._success_hold_count = 0

    # ---------------------------------------------------------------- helpers
    def _register_rim_grasp(self, actor, rim_dir):
        # Grasp point = bowl center + radial RIM_GRASP_R (wall midline ~5 mm
        # below the rim) + vertical RIM_GRASP_DZ. camera_up=(1,0,0) makes the
        # finger open/close axis radial (rim_dir): one finger on the inner
        # wall, one on the outer wall.
        rd = np.array(rim_dir, dtype=float)
        rd /= np.linalg.norm(rd)
        center = actor.get_pose()
        gp = np.array(center.p, dtype=float) + rd * RIM_GRASP_R + np.array([0.0, 0.0, RIM_GRASP_DZ])
        grasp_pose = construct_grasp_pose(
            gp,
            np.array([0.0, 0.0, 1.0], dtype=float),
            np.array([1.0, 0.0, 0.0], dtype=float),
        )
        return actor.register_point(grasp_pose, type="contact")

    def _close_gripper_direct(self, rm, percent=0.0, settle_steps=10, is_save=True):
        # Force-close at the joint level and settle stepwise so the fingers
        # actually press on the thin wall (a planned close_gripper alone often
        # fails to seat firmly).
        target = rm.gripper_percent2qpos(percent)
        pos = torch.tensor([target, target], device=rm.device, dtype=rm.robot.data.joint_pos.dtype)
        self.atom_id += 1
        self.atom_tag = "close_direct"
        rm.set_gripper(pos, force=True)
        for _ in range(settle_steps):
            rm.set_gripper(pos, force=True)
            self._step(is_save=is_save)
        self._update_render()

    def _open_gripper_direct(self, rm, percent=1.0, settle_steps=12, is_save=True):
        target = rm.gripper_percent2qpos(percent)
        pos = torch.tensor([target, target], device=rm.device, dtype=rm.robot.data.joint_pos.dtype)
        self.atom_id += 1
        self.atom_tag = "open_direct"
        rm.set_gripper(pos, force=True)
        for _ in range(settle_steps):
            rm.set_gripper(pos, force=True)
            self._step(is_save=is_save)
        self._update_render()

    def _grasp_bowl(self, actor, atom, rm, arm, rim_dir):
        # Same recipe as dual_bowl_unstack._grasp_rim (same bowl asset, holds
        # without welding): hover -> constrained linear descent (extra
        # RIM_DOWN_EXTRA so the pads seat on both walls) -> planned close +
        # forced joint-level clamp.
        rd = np.array(rim_dir, dtype=float); rd = rd / np.linalg.norm(rd)
        self.move(atom.open_gripper(PRE_GRIPPER_OPEN), arm=arm)
        center = actor.get_pose()
        gp = np.array(center.p, dtype=float) + rd * RIM_GRASP_R + np.array([0.0, 0.0, RIM_GRASP_DZ])
        gc = construct_grasp_pose(gp, np.array([0, 0, 1], dtype=float), np.array([1, 0, 0], dtype=float))
        gc_high = gc.add_bias([0.0, 0.0, GRASP_SIDE + GRASP_UP], coord="world")   # hover above the rim point
        self.move(atom.move_to_pose(rm.gripper_center_to_ee(gc_high)), arm=arm)
        down_total = GRASP_SIDE + GRASP_UP + RIM_DOWN_EXTRA
        self.move(atom.move_by_displacement(z=down_total * 0.75, xyz_coord="local"), arm=arm,
                  constraint_pose=[1, 1, 1, 1, 1, 0], time_dilation_factor=0.5)
        self.move(atom.move_by_displacement(z=down_total * 0.25, xyz_coord="local"), arm=arm,
                  constraint_pose=[1, 1, 1, 1, 1, 0], time_dilation_factor=0.5)
        # Close fully + forced clamp so both fingers press hard on the wall.
        self.move(atom.close_gripper(0.0, depth_threshold=None), arm=arm)
        self._close_gripper_direct(rm, 0.0, settle_steps=10, is_save=True)
        if USE_WELD:
            self.weld_actor(actor, rm)
        self.delay(8, is_save=True)
        # Lift vertically first, then let _place_inhand move laterally --
        # dragging along the table would tilt the bowl or break the grip.
        self.move(
            atom.move_by_displacement(z=GRASP_LIFT, xyz_coord="world"),
            arm=arm,
            time_dilation_factor=0.5,
        )
        gc = rm.get_gripper_center_pose()
        bp = actor.get_pose()
        print(
            f"[BOWL_STACK] {arm} grasp+lift: bowl=({bp.p[0]:.3f},{bp.p[1]:.3f},{bp.p[2]:.3f}) "
            f"gc=({gc.p[0]:.3f},{gc.p[1]:.3f},{gc.p[2]:.3f}) "
            f"held={'YES' if bp.p[2] > 0.08 else 'NO(dropped)'}",
            flush=True,
        )

    def _place_inhand(self, actor, rm, atom, target_pose, arm):
        # Solve the gripper-center target from the current in-hand relative
        # pose -> places the bowl center at target even with the off-center
        # rim grasp.
        inhand = actor.get_pose().rebase(rm.get_gripper_center_pose())
        gc_mat = np.array(target_pose.to_transformation_matrix()) @ np.linalg.inv(
            np.array(inhand.to_transformation_matrix())
        )
        ee = rm.gripper_center_to_ee(Pose.from_matrix(gc_mat))
        self.move(atom.move_to_pose(ee), arm=arm, time_dilation_factor=0.5)

    def _unweld_actor(self, actor):
        self._welds = [w for w in self._welds if w[0] is not actor]

    def _bowls_released(self):
        # Both bowls truly released = both grippers open and both bowls
        # unwelded (transport binding removed).
        a_open = float(self._robot_manager.get_gripper_percentage()) >= GRIPPER_RELEASE_THRESH
        b_open = float(self._robot_manager_b.get_gripper_percentage()) >= GRIPPER_RELEASE_THRESH
        no_bowl_welds = not any(w[0] is self.bowl_a or w[0] is self.bowl_b for w in self._welds)
        return a_open, b_open, no_bowl_welds, bool(a_open and b_open and no_bowl_welds)

    def _release_bowl(self, actor, atom, arm, park=False, retract=True):
        rm, _ = self._arm(arm)
        self._unweld_actor(actor)
        self.delay(5, is_save=True)
        self._open_gripper_direct(rm, 1.0, settle_steps=12, is_save=True)
        if retract:
            self.move(
                atom.move_by_displacement(z=RETRACT_Z, xyz_coord="world"),
                arm=arm,
                time_dilation_factor=0.5,
            )
        if park:
            # If arm A stayed above the central target after placing the lower
            # bowl, arm B would keep colliding with it while nesting the upper
            # bowl -> park arm A above the +Y side.
            rm, _ = self._arm(arm)
            gc_now = rm.get_gripper_center_pose()
            park_gc = Pose(ARM_A_PARK_POS, gc_now.q)
            self.move(
                atom.move_to_pose(rm.gripper_center_to_ee(park_gc)),
                arm=arm,
                time_dilation_factor=0.5,
            )

    def _dbg(self, tag):
        ap, bp = self.bowl_a.get_pose(), self.bowl_b.get_pose()
        print(
            f"[BOWL_STACK] {tag}: "
            f"bowl_a=({ap.p[0]:.3f},{ap.p[1]:.3f},{ap.p[2]:.3f}) "
            f"bowl_b=({bp.p[0]:.3f},{bp.p[1]:.3f},{bp.p[2]:.3f})",
            flush=True,
        )

    # ---------------------------------------------------------------- script
    def pre_move(self):
        self.delay(15)
        self._dbg("reset settled")

    def _play_once(self):
        # A: carry the lower bowl from the +Y side to the central target.
        self._grasp_bowl(self.bowl_a, self.atom_a, self._robot_manager, "a", rim_dir=(0, 1, 0))
        self._place_inhand(
            self.bowl_a,
            self._robot_manager,
            self.atom_a,
            STACK_TARGET.add_bias([0.0, 0.0, PLACE_HOVER], coord="world"),
            "a",
        )
        self._place_inhand(self.bowl_a, self._robot_manager, self.atom_a, STACK_TARGET, "a")
        self._release_bowl(self.bowl_a, self.atom_a, "a", park=True)
        self._dbg("A placed bowl_a")

        # B: carry the upper bowl from the -Y side to directly above bowl_a
        # and nest it.
        self._grasp_bowl(self.bowl_b, self.atom_b, self._robot_manager_b, "b", rim_dir=(0, -1, 0))
        self._place_inhand(
            self.bowl_b,
            self._robot_manager_b,
            self.atom_b,
            STACK_TOP_TARGET.add_bias([0.0, 0.0, PLACE_HOVER], coord="world"),
            "b",
        )
        self._place_inhand(self.bowl_b, self._robot_manager_b, self.atom_b, STACK_TOP_TARGET, "b")
        self._release_bowl(self.bowl_b, self.atom_b, "b", retract=False)
        self._dbg("B stacked bowl_b")
        self.delay(30, is_save=True)

    # ---------------------------------------------------------------- success
    def check_success(self):
        ap = self.bowl_a.get_pose()
        bp = self.bowl_b.get_pose()
        target_xy = np.array(STACK_TARGET.p[:2], dtype=float)
        a_xy = np.array(ap.p[:2], dtype=float)
        b_xy = np.array(bp.p[:2], dtype=float)
        target_err = float(np.linalg.norm(a_xy - target_xy))
        stack_err = float(np.linalg.norm(b_xy - a_xy))
        dz = float(bp.p[2] - ap.p[2])
        height_err = abs(dz - NEST_RISE)
        a_up = float(np.dot(ap.to_transformation_matrix()[:3, 2], np.array([0, 0, 1]))) > 0.75
        b_up = float(np.dot(bp.to_transformation_matrix()[:3, 2], np.array([0, 0, 1]))) > 0.75
        a_open, b_open, no_bowl_welds, released = self._bowls_released()
        # Bowls are wider than cups: placement/nesting tolerances slightly relaxed.
        stacked_upright = target_err < 0.04 and stack_err < 0.03 and height_err < 0.025 and a_up and b_up
        if stacked_upright and released:
            self._success_hold_count = getattr(self, "_success_hold_count", 0) + 1
        else:
            self._success_hold_count = 0

        self.metadata["bowl_a"] = [float(v) for v in ap.p]
        self.metadata["bowl_b"] = [float(v) for v in bp.p]
        self.metadata["target_err"] = target_err
        self.metadata["stack_err"] = stack_err
        self.metadata["height_err"] = height_err
        self.metadata["gripper_a_open"] = a_open
        self.metadata["gripper_b_open"] = b_open
        self.metadata["bowls_unwelded"] = no_bowl_welds
        self.metadata["released"] = released
        self.metadata["stacked_upright"] = bool(stacked_upright)
        self.metadata["success_hold_count"] = int(self._success_hold_count)
        print(
            f"[BOWL_STACK] target_err={target_err*1000:.1f}mm "
            f"stack_err={stack_err*1000:.1f}mm dz={dz*1000:.1f}mm "
            f"height_err={height_err*1000:.1f}mm a_up={a_up} b_up={b_up} "
            f"a_open={a_open:.2f} b_open={b_open:.2f} unwelded={no_bowl_welds} "
            f"stacked_upright={stacked_upright} hold={self._success_hold_count}/{SUCCESS_HOLD_STEPS}",
            flush=True,
        )
        # Vertical-wall bowls have no fixed nesting depth -> use a range to
        # decide "upper bowl seated on the lower bowl".
        nested = 0.025 < dz < 0.072
        return bool(target_err < 0.04 and stack_err < 0.03 and nested and a_up and b_up)
