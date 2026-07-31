"""Task `dual_bowl_unstack`. Description, action flow, and exact success criteria: docs/Tasks.md#dual_bowl_unstack"""
from ._base_task import *
import numpy as np
import torch

# ============================================================================
# Dual Bowl UN-STACK: two bowls start nested (upper bowl seated inside the
# lower one). Arm A grasps the upper bowl by its rim, extracts it vertically,
# and places it on a target plate to the side. Mirrors dual_cup_stack
# (cup unstacking), with these differences:
#   1) Object is BOWL.usd (130 mm dia shallow ceramic bowl, ~5 mm wall, foot
#      ring; opening faces local +Z, closed bottom -Z, origin at body center).
#   2) Rim grasp: the 130 mm opening exceeds the gripper span, so unlike the
#      cup the bowl cannot be grasped across the body; the gripper pinches one
#      section of the rim radially (one finger on the inner wall, one on the
#      outer wall, clamping ~5 mm of wall).
#   3) Adaptive grasp disabled: use_adaptive_grasp=False + close_gripper(0.0)
#      closes fully.
# The lower bowl stays on the table by weight (high density + high table
# friction). As with nested cups, wall friction can bind the nested bowls;
# friction is lowered to 5 to avoid it.
# ----------------------------------------------------------------------------
# BOWL local frame (origin centered, z in [-half, +half]): opening at +Z,
# closed bottom at -Z; origin-to-rim / origin-to-bottom = BOWL_HALF.
# ============================================================================

# ---- Geometry (m) ----
BOWL_HALF    = 0.0197    # half height (origin to rim/bottom); USD is true scale (130 mm dia, 39.4 mm tall), scene scale=1.0
BOWL_OUTER_R = 0.065     # outer radius at the very top of the rim (130 mm dia)
# The wall tapers quickly below the rim: very thin at the top (0.5-1 mm,
# r ~ 0.064-0.065), thicker further down (3 mm below rim: r in [0.0588, 0.0633],
# ~4.5 mm thick; 5 mm below: r in [0.0584, 0.0618], ~3.4 mm).
# Grasp ~5 mm below the rim (solid wall, tolerant of height error) at the wall
# midline radius ~0.060: that radius stays inside the wall anywhere 2-8 mm
# below the rim, so a few mm of descent error still lands the wall between the
# fingers.
RIM_GRASP_R  = 0.060     # horizontal distance from gripper center (TCP) to bowl axis (wall midline, 5 mm below rim)
FOOT_R       = 0.026     # foot ring outer radius
TABLE_TOP    = 0.004     # table top height (bowl bottom rests here); bowl center = TABLE_TOP + BOWL_HALF
PLATE_HALF_Z = 0.004     # target plate half height (PLATE.usd, visual placement hint)
PLATE_CZ     = TABLE_TOP + PLATE_HALF_Z
PLATE_R      = 0.10      # target plate outer radius (200 mm dia)

# Nesting: the bowl asset has a raised inner bottom / foot ring, so the nested
# upper bowl naturally sits higher. The initial upper-bowl center is raised
# accordingly, keeping the layout consistent with the geometry and leaving
# more rim exposed for grasping.
NEST_RISE    = 0.038
NEST_GAP     = 0.014     # extra initial nesting gap so the bowls separate easily

# ---- Layout [tune] (world frame) ----
UPRIGHT      = [1, 0, 0, 0]
STACK_POS    = Pose([0.45, 0.00, TABLE_TOP + BOWL_HALF], UPRIGHT)   # lower bowl center (on table, near the arm for low reach)
# The extracted upper bowl is placed on the +Y side of the table (horizontal
# separation from the lower bowl marks a successful unstack).
PLATE_POSE   = Pose([0.45, 0.30, PLATE_CZ], UPRIGHT)
PLACE_POSE   = Pose([PLATE_POSE.p[0], PLATE_POSE.p[1], TABLE_TOP + 2 * PLATE_HALF_Z + BOWL_HALF], UPRIGHT)
PLACE_SETTLE_DROP = 0.008

# ---- Grasp [tune] ----
# Rim grasp: descend from above onto the +Y side of the upper bowl's rim (the
# section closest to arm A) and pinch the wall radially.
# The TCP (gripper center) sits ~1 cm below the pad contact surface: with the
# TCP at the rim the pads would still hover above the bowl (verified in sim).
# The bowl asset exposes more rim, so the grasp point is shifted up toward the
# solid upper wall.
RIM_GRASP_DZ = BOWL_HALF - 0.006
EXTRACT_RISE = 0.16                # vertical extraction height (clear the stack before moving sideways)
RIM_DOWN_EXTRA = 0.020             # extra 20 mm on the descent so the pads seat firmly on the rim walls
RELEASE_HOVER_Z = 0.16             # hover height above the placement area before lowering
SUCCESS_HOLD_STEPS = 8             # consecutive steps of (separated + both upright + on plate + released) required (as in cup_unstack)


@configclass
class TaskCfg(BaseTaskCfg):
    dual_arm = True       # auto-creates arm B (/Robot_b + *_b tactile); this task only uses arm A (as in cup unstack)
    use_adaptive_grasp = False   # rim grasp closes fully via close_gripper(0.0) instead of adaptive grasping
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
    step_lim = 1350  # eval step limit: max archived success steps x1.05 (data-calibrated 2026-07-25)
    reset_time_limit = 1200.0   # dual-arm + two nested UIPC soft bowls make reset slow; allow ample margin


class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode: Literal['collect', 'eval'] = 'collect', render_mode: str | None = None, **kwargs):
        # Higher friction grips the rim better but makes the nested-bowl
        # contact solve much heavier; medium-high friction keeps enough grip
        # while easing the nested-contact computation.
        cfg.sim.physics_material.dynamic_friction = 5.0
        cfg.sim.physics_material.static_friction = 5.0
        cfg.uipc_sim.contact.default_friction_ratio = 5.0
        # Both bowls are manipulated objects and start nested at the grasp
        # location. The planner's frozen collision world is built inside
        # super().__init__(); without ignores, action 0 would collide with the
        # stack and planning would fail, so the task-level ignore set must be
        # assigned before super().__init__() (get_curr_world_cfg reads the
        # task-level set union each arm's owning_manager set).
        self.planner_ignore_actors = {'bowl_a', 'bowl_b', 'dish'}
        super().__init__(cfg, mode, render_mode, **kwargs)
        self._robot_manager.ignore_actors = {'bowl_a', 'bowl_b', 'dish'}      # arm A must pass through / separate from the lower bowl
        self._robot_manager_b.ignore_actors = {'bowl_a', 'bowl_b', 'dish'}    # arm B is idle; ignore to avoid spurious frozen-world collisions

    # NOTE: get_frame_shot is not overridden -- the base class composes the
    # dual-arm 1760x320 frame (head|wrist|wrist_b at 480 each + 4 tactile
    # streams, 2 columns x 160), matching video_size exactly.

    # ---------------------------------------------------------------- actors
    def create_actors(self):
        self.dish = self._actor_manager.add_from_usd_file(
            name='dish', asset_path="PLATE.usd", pose=PLATE_POSE, density=1e4,
            constitution_cfg=UipcObjectCfg.AffineBodyConstitutionCfg(kinematic=True))
        # Two identical bowls (same USD, different instance names); the nested
        # initial poses are set in _reset_actors.
        # Soft-body bowls are light by default and get pushed away when the
        # gripper closes or extracts. The lower bowl gets high density
        # (2e5 -> ~11 kg) plus high friction so it stays firmly on the table
        # during extraction. The upper bowl stays light: the one-sided rim
        # grasp has a large moment arm (~6 cm from bowl center to grip point),
        # and a light bowl keeps the torque low enough for the line contact to
        # hold without slipping or flipping; a heavy bowl would slip out.
        self.bowl_a = self._actor_manager.add_from_usd_file(
            name='bowl_a', asset_path="BOWL.usd", pose=STACK_POS, density=2e5)
        self.bowl_b = self._actor_manager.add_from_usd_file(
            name='bowl_b', asset_path="BOWL.usd", density=3e3,
            pose=Pose([STACK_POS.p[0], STACK_POS.p[1], STACK_POS.p[2] + NEST_RISE + NEST_GAP], UPRIGHT))

    def _reset_actors(self):
        # Start nested: lower bowl on the table, upper bowl seated upright
        # inside it (center raised by NEST_RISE, same xy + small noise).
        noise = self.create_noise([0.002, 0.002, 0.0])   # small noise: the rim grasp already has tight tolerances
        base = STACK_POS.add_offset(noise)
        self.dish.set_pose(PLATE_POSE)
        self.bowl_a.set_pose(base)
        self.bowl_b.set_pose(Pose([base.p[0], base.p[1], base.p[2] + NEST_RISE + NEST_GAP], UPRIGHT))
        self.metadata['plate_xy'] = [float(PLATE_POSE.p[0]), float(PLATE_POSE.p[1])]
        self._success_hold_count = 0

    # ---------------------------------------------------------------- helpers
    def _grasp_rim(self, actor, rm, atom, arm, rim_dir=(0, 1, 0), camera_up=(1, 0, 0),
                   dz=RIM_GRASP_DZ, side=0.05, up=0.06):
        """Grasp one side of the bowl rim from above (radial pinch on the wall).

        rim_dir selects the horizontal direction of the rim section (default +Y,
        the side closest to arm A); the gripper center lands on that side's wall
        midline. camera_up is chosen so the finger open/close axis is radial
        (rim_dir): one finger on the inner wall, one on the outer wall.
        Two-stage approach: hover above the rim point -> descend vertically onto
        the rim -> constrained vertical feed to seat on the rim -> close fully.
        """
        rd = np.array(rim_dir, dtype=float); rd = rd / np.linalg.norm(rd)
        self.move(atom.open_gripper(1.0), arm=arm)
        center = actor.get_pose()
        # Rim grasp point = bowl center + radial RIM_GRASP_R (wall midline 5 mm below rim) + vertical dz
        gp = np.array([center.p[0], center.p[1], center.p[2]], dtype=float) + rd * RIM_GRASP_R + np.array([0, 0, dz])
        gc = construct_grasp_pose(gp, np.array([0, 0, 1], dtype=float), np.array(camera_up, dtype=float))
        gc_high = gc.add_bias([0.0, 0.0, side + up], coord='world')    # hover pose above the rim point
        ee_high = rm.gripper_center_to_ee(gc_high)
        self.move(atom.move_to_pose(ee_high), arm=arm)                 # 1) hover above the rim point
        # 2) Split the final descent into two short constrained linear feeds to
        # avoid the instability of one long single-shot plan.
        down_total = side + up + RIM_DOWN_EXTRA
        self.move(atom.move_by_displacement(z=down_total * 0.75, xyz_coord='local'), arm=arm,
                  constraint_pose=[1, 1, 1, 1, 1, 0], time_dilation_factor=0.5)
        self.move(atom.move_by_displacement(z=down_total * 0.25, xyz_coord='local'), arm=arm,
                  constraint_pose=[1, 1, 1, 1, 1, 0], time_dilation_factor=0.5)
        tcp = rm.get_gripper_center_pose()
        msg = (f"[DBG] grasp-target gp=({gp[0]:.3f},{gp[1]:.3f},{gp[2]:.3f})  "
               f"TCP-actual=({tcp.p[0]:.3f},{tcp.p[1]:.3f},{tcp.p[2]:.3f})  rim_top_z={center.p[2]+BOWL_HALF:.4f}")
        try:  # log finger/pad world z to calibrate the pad-contact vs TCP vertical offset
            ids, names = rm.robot.find_bodies('.*finger.*')
            fz = rm.robot.data.body_link_pos_w[0, ids]
            msg += "  fingers[" + " ".join(f"{names[i]}z={fz[i,2].item():.4f}" for i in range(len(names))) + "]"
        except Exception as e:
            msg += f"  (finger-pos err: {e})"
        print(msg, flush=True)
        # 3) Close fully so both fingers press hard on the rim.
        self.move(atom.close_gripper(0.0), arm=arm)
        self._close_gripper_direct(rm, 0.0, settle_steps=10, is_save=True)

    def _move_gripper_direct(self, rm, atom, target_pose, arm):
        """Move the gripper center to a world position, keeping its current orientation."""
        gc_now = rm.get_gripper_center_pose()
        target_gc = Pose(list(target_pose.p), gc_now.q)
        target_ee = rm.gripper_center_to_ee(target_gc)
        self.move(atom.move_to_pose(target_ee), arm=arm, time_dilation_factor=0.5)

    def _open_gripper_direct(self, rm, percent=1.0, settle_steps=12, is_save=True):
        target = rm.gripper_percent2qpos(percent)
        pos = torch.tensor([target, target], device=rm.device, dtype=rm.robot.data.joint_pos.dtype)
        self.atom_id += 1
        self.atom_tag = 'open_direct'
        rm.set_gripper(pos, force=True)
        for _ in range(settle_steps):
            rm.set_gripper(pos, force=True)
            self._step(is_save=is_save)
        self._update_render()

    def _close_gripper_direct(self, rm, percent=0.0, settle_steps=10, is_save=True):
        target = rm.gripper_percent2qpos(percent)
        pos = torch.tensor([target, target], device=rm.device, dtype=rm.robot.data.joint_pos.dtype)
        self.atom_id += 1
        self.atom_tag = 'close_direct'
        rm.set_gripper(pos, force=True)
        for _ in range(settle_steps):
            rm.set_gripper(pos, force=True)
            self._step(is_save=is_save)
        self._update_render()

    def _dbg(self, tag):
        ap, bp = self.bowl_a.get_pose(), self.bowl_b.get_pose()
        ga = self._robot_manager.get_gripper_qpos()
        print(f"[DBG] {tag}: gripA={ga:.4f}  "
              f"bowl_a=({ap.p[0]:.3f},{ap.p[1]:.3f},{ap.p[2]:.3f})  "
              f"bowl_b=({bp.p[0]:.3f},{bp.p[1]:.3f},{bp.p[2]:.3f})", flush=True)

    # ---------------------------------------------------------------- script
    def pre_move(self):
        self.delay(20)   # let the nested stack settle; grasp/unstack runs in _play_once (recorded on video)
        self._dbg("nested stack settled")

    def _play_once(self):
        # Arm A grasps the upper bowl's +Y rim from above, extracts it
        # vertically off the lower bowl, moves it to the +Y table area, places
        # it, and releases. The heavy, high-friction lower bowl stays on the
        # table by its own weight; no second arm is needed to hold it.
        self._grasp_rim(self.bowl_b, self._robot_manager, self.atom_a, 'a',
                        rim_dir=(0, 1, 0), camera_up=(1, 0, 0))
        self._dbg("A grasped upper bowl rim")

        # A: extract the upper bowl straight up (world +Z); constraint_pose
        # locks the other 5 axes for a straight vertical separation.
        self.move(self.atom_a.move_by_displacement(z=EXTRACT_RISE * 0.7, xyz_coord='world'),
                  arm='a', constraint_pose=[1, 1, 1, 1, 1, 0], time_dilation_factor=0.5)
        self.move(self.atom_a.move_by_displacement(z=EXTRACT_RISE * 0.3, xyz_coord='world'),
                  arm='a', constraint_pose=[1, 1, 1, 1, 1, 0], time_dilation_factor=0.5)
        self._dbg("A extracted upper bowl")

        # A: move the upper bowl above the +Y placement point, lower it to the
        # table, then release.
        hover = Pose([PLATE_POSE.p[0], PLATE_POSE.p[1], PLACE_POSE.p[2] + RELEASE_HOVER_Z], PLACE_POSE.q)
        self._move_gripper_direct(self._robot_manager, self.atom_a, hover, 'a')
        self._dbg("A above placement point")
        # Lower straight down (world -Z) from directly above the plate so the
        # final drop stays centered on the plate.
        gripper_now = self._robot_manager.get_gripper_center_pose()
        drop_z = (PLACE_POSE.p[2] - gripper_now.p[2])   # full descent (a former 0.5x factor released the bowl ~66 mm above the plate)
        self.move(self.atom_a.move_by_displacement(z=float(drop_z), xyz_coord='world'),
                  arm='a', constraint_pose=[1, 1, 1, 1, 1, 0], time_dilation_factor=0.5)
        self.move(self.atom_a.move_by_displacement(z=-PLACE_SETTLE_DROP, xyz_coord='world'),
                  arm='a', constraint_pose=[1, 1, 1, 1, 1, 0], time_dilation_factor=0.5)
        self.delay(8, is_save=True)                         # let the bowl settle before releasing
        self._open_gripper_direct(self._robot_manager, 1.0, settle_steps=14, is_save=True)   # release the upper bowl
        self._dbg("A placed upper bowl and released")
        self.delay(25, is_save=True)

    # ---------------------------------------------------------------- success
    def check_success(self):
        ap = self.bowl_a.get_pose()
        bp = self.bowl_b.get_pose()
        # Success = bowls separated + upper bowl upright and within the target
        # plate + former lower bowl not tipped over + gripper released.
        horiz = float(np.linalg.norm(np.array(ap.p[:2], dtype=float) - np.array(bp.p[:2], dtype=float)))
        separated = horiz > 0.12
        a_up = float(np.dot(ap.to_transformation_matrix()[:3, 2], np.array([0, 0, 1]))) > 0.7
        b_up = float(np.dot(bp.to_transformation_matrix()[:3, 2], np.array([0, 0, 1]))) > 0.6
        bottom_ok = a_up   # former lower bowl still upright (not tipped during extraction)
        plate_horiz = float(np.linalg.norm(np.array(bp.p[:2], dtype=float) - np.array(PLATE_POSE.p[:2], dtype=float)))
        on_plate = plate_horiz < (PLATE_R - BOWL_OUTER_R + 0.01)
        pa = self._robot_manager.get_gripper_percentage()
        released = pa > 0.9   # A gripper opened = bowl actually let go before success can count
        success_now = separated and bottom_ok and b_up and on_plate and released
        if success_now:
            self._success_hold_count = getattr(self, '_success_hold_count', 0) + 1
        else:
            self._success_hold_count = 0
        self.metadata['bowl_a'] = [float(v) for v in ap.p]
        self.metadata['bowl_b'] = [float(v) for v in bp.p]
        self.metadata['horiz_sep'] = horiz
        self.metadata['bowl_b_to_plate_horiz'] = plate_horiz
        self.metadata['bottom_ok'] = bool(bottom_ok)
        self.metadata['released'] = bool(released)
        self.metadata['success_hold_count'] = int(self._success_hold_count)
        print(f"[UNSTACK] horiz_sep={horiz*1000:.1f}mm plate_err={plate_horiz*1000:.1f}mm "
              f"a_up={a_up} b_up={b_up} on_plate={on_plate} separated={separated}", flush=True)
        print(f"[UNSTACK-REL] gripperA={pa:.2f} released={released} bottom_ok={bottom_ok} "
              f"hold={self._success_hold_count}/{SUCCESS_HOLD_STEPS}", flush=True)
        return bool(self._success_hold_count >= SUCCESS_HOLD_STEPS)
