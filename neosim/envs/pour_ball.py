"""Task `pour_ball`. Description, action flow, and exact success criteria: docs/Tasks.md#pour_ball"""
from ._base_task import *
import numpy as np
import torch

# ============================================================================
# Pour Ball -- pour a small ball from a cup onto a plate.
#   Single arm: grasp the cup containing the ball -> lift -> move above the
#   plate -> tilt -> ball lands on the plate.
#   Plate = fixed receiving surface (kinematic), nothing placed on it; the cup
#   is the dynamic object being grasped.
#   Cup: CUP.usd (paper cup, rim local +Z, closed bottom local -Z; centered,
#   rim OD 76 / bottom OD 54 / height 92 mm).
#   Plate: PLATE.usd (shallow dish, OD 200, 8 mm rim). Ball: BALL.usd (OD 16 mm).
# ----------------------------------------------------------------------------
#   Camera/view reuses the centered top-down setup from dual_screw_sleeve.
#   Grasp is a SIDE grasp on the cup body: hover above the cup side, descend
#   vertically next to it, then insert horizontally and close. This gives the
#   wrist a larger moment arm for pouring than pinching the rim from above.
#   After grasping, the cup is welded to the gripper.
#   Scene center is (0.5, 0) (in front of the base), same as the screw task.
# ============================================================================

# ---- Geometry (m) ----
CUP_SCALE  = 0.93                    # cup scaled 1.5x (0.62 -> 0.93), rim OD ~71 mm, near gripper opening limit;
                                     # side grasp inserts at the cup wall rather than reaching over the rim
CUP_HALF   = 0.046 * CUP_SCALE       # cup half-height (origin -> rim/bottom)
CUP_TOP_R  = 0.038 * CUP_SCALE       # cup rim outer radius
BALL_R     = 0.008                   # ball radius (BALL.usd, OD 16 mm)
TABLE_TOP  = 0.004                   # table top (ground_plate) height
PLATE_HALF_Z = 0.004                 # plate half-height
PLATE_CZ   = TABLE_TOP + PLATE_HALF_Z          # plate center z (bottom resting on table)
PLATE_R    = 0.10                    # plate outer radius (OD 200)

# ---- Layout [tune] (world frame) -- scene centered at (0.5, 0), same as the screw task ----
UPRIGHT      = [1, 0, 0, 0]
GAP          = 0.006                                          # small gap to avoid initial UIPC penetration
PLATE_POS    = Pose([0.50, -0.10, PLATE_CZ], UPRIGHT)         # plate (receiving surface / pour target)
CUP_POS      = Pose([0.50,  0.10, TABLE_TOP + CUP_HALF + GAP], UPRIGHT)   # cup (grasped, contains the ball)
BALL_DZ      = -CUP_HALF + 0.006 + BALL_R                     # ball z relative to cup center (rests on inner bottom, no wall penetration)

# ---- Grasp / pour [tune] ----
CUP_GRASP_DZ = 0.015                 # grasp point on the upper cup wall, just below the rim
CUP_SIDE_APPROACH = 0.14             # standoff outside the cup along the approach direction for the side grasp
CUP_SIDE_UP = 0.10                   # height of the side pre-grasp pose above the cup side
PREGRASP_CLEARANCE = 0.12            # lift vertically in place before the pre-grasp pose, so a diagonal path does not knock the cup over
GRIP_CLOSE   = 0.60                  # gripper closure (0=fully closed / 1=open): kept loose to avoid squeezing/deforming the large cup
LIFT_RISE    = 0.16                  # kept as an empirical reference; now move_to_pose goes directly to the pour prep pose
POUR_OVER_Y  = 0.00                  # pour pose directly above the plate center (previous manual offset removed)
POUR_Z       = 0.26                  # extra height for the large cup + large tilt-angle workspace
POUR_PREP_X  = -0.03                 # pull the pour prep pose slightly toward the robot so the wrist flip does not start over-extended
POUR_ANGLE   = 210.0 / 180.0 * np.pi    # over-rotate so the cup does not stop at "almost inverted but not enough"
POUR_LIFT    = 0.00                  # only rotate the wrist while pouring, no simultaneous lift
POUR_JOINT_STEPS = 80                # interpolation steps for rotating the wrist joint directly


@configclass
class TaskCfg(BaseTaskCfg):
    step_lim = 1500  # eval step limit: max archived success steps x1.05 (data-calibrated 2026-07-25)
    use_adaptive_grasp = False  # adaptive grasp only closes to a light-touch threshold and cannot hold the cup (slips/drops the ball); close fully instead
    reset_time_limit = 1200.0   # UIPC contact solve is occasionally slow (shared GPU); allow ample margin
    # GUI view: centered front-top view of the whole workspace (same as dual_screw_sleeve).
    viewer = ViewerCfg(eye=(1.45, 0.0, 0.95), lookat=(0.45, 0.0, 0.12))
    video_size = (1120, 320)   # head + wrist (2x480) + 2 tactile (1 col x 160) = 1120 wide
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
    ]


class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode: Literal['collect', 'eval'] = 'collect', render_mode: str | None = None, **kwargs):
        # Medium friction (2.0): mu=5 makes the ball stick in the cup; mu=0.8 lets it slide out
        # too early mid-flip and get flung off target toward -y. Medium friction plus fully
        # inverting the cup over the plate center drops the ball nearly vertically onto the plate.
        # Grip relies on geometric closure (not friction), so lowering friction does not affect the hold.
        cfg.sim.physics_material.dynamic_friction = 2.0
        cfg.sim.physics_material.static_friction = 2.0
        cfg.uipc_sim.contact.default_friction_ratio = 2.0
        # Thin-shell cup/plate/ball get voxelized into solid obstacles by the planner; unless
        # ignored, action0 always collides and planning fails. Ignore all task objects.
        self.planner_ignore_actors = {'cup', 'ball', 'dish'}
        super().__init__(cfg, mode, render_mode, **kwargs)
        self._robot_manager.ignore_actors = {'cup', 'ball', 'dish'}

    # ---------------------------------------------------------------- actors
    def create_actors(self):
        sc = (CUP_SCALE, CUP_SCALE, CUP_SCALE)
        # Plate: kinematic ABD (fixed receiving surface). self.plate is taken by the base class (ground plate), so store as self.dish.
        self.dish = self._actor_manager.add_from_usd_file(
            name='dish', asset_path="PLATE.usd", pose=PLATE_POS, density=1e4,
            constitution_cfg=UipcObjectCfg.AffineBodyConstitutionCfg(kinematic=True))
        # Cup: dynamic object (grasped). Density matches a real paper cup (~5e3 -> tens of grams);
        # a ~0.9 kg cup (density 2e5) slips between the fingers under gravity torque, drifting the
        # in-hand transform and pushing later planning targets out of reach.
        self.cup = self._actor_manager.add_from_usd_file(
            name='cup', asset_path="CUP.usd", pose=CUP_POS, scale=sc, density=5e3)
        # Ball: dynamic, starts at the inner bottom of the cup.
        self.ball = self._actor_manager.add_from_usd_file(
            name='ball', asset_path="BALL.usd",
            pose=Pose([CUP_POS.p[0], CUP_POS.p[1], CUP_POS.p[2] + BALL_DZ], UPRIGHT),
            density=2000.0)

    def _reset_actors(self):
        noise = self.create_noise([0.003, 0.003, 0.0])
        cup = CUP_POS.add_offset(noise)
        self.dish.set_pose(PLATE_POS)
        self.cup.set_pose(cup)
        self.ball.set_pose(Pose([cup.p[0], cup.p[1], cup.p[2] + BALL_DZ], UPRIGHT))
        self.metadata['plate_xy'] = [float(PLATE_POS.p[0]), float(PLATE_POS.p[1])]

    # ---------------------------------------------------------------- helpers
    def _grasp_cup(self):
        """Grasp the upper cup body from the side.

        Two-stage approach that avoids pressing on the rim like a top grasp:
          1) hover above the cup side;
          2) descend vertically next to the cup;
          3) insert straight along gripper local z (horizontal approach axis) to the cup center;
          4) close the gripper on the cup body.
        """
        grasp_from = np.array([0.0, 1.0, 0.0], dtype=float)   # approach horizontally from the +Y side
        camera_up = np.array([0.0, 0.0, -1.0], dtype=float)   # no 180-deg wrist flip; fingers still open along world-x
        gf = grasp_from / np.linalg.norm(grasp_from)
        self.move(self.atom.open_gripper(1.0))
        self.move(self.atom.move_by_displacement(z=PREGRASP_CLEARANCE, xyz_coord='world'),
                  tag='grasp_side_clearance', time_dilation_factor=0.5)
        target = self.cup.get_pose().add_bias([0.0, 0.0, CUP_GRASP_DZ], coord='world')
        gc = construct_grasp_pose(np.array(target.p, dtype=float), gf, camera_up)
        gc_side = gc.add_bias((gf * CUP_SIDE_APPROACH).tolist(), coord='world')
        ee_side = self._robot_manager.gripper_center_to_ee(gc_side)
        ee_high = ee_side.add_bias([0.0, 0.0, CUP_SIDE_UP], coord='world')
        ee_now = self._robot_manager.get_ee_pose()
        ee_transit = Pose([ee_high.p[0], ee_high.p[1], ee_now.p[2]], ee_now.q)
        self.move(self.atom.move_to_pose(ee_transit), tag='grasp_side_horizontal')
        self.move(self.atom.move_to_pose(ee_high), tag='grasp_side_presolve')
        self.move(self.atom.move_to_pose(ee_side), tag='grasp_side_drop')
        self.move(self.atom.move_by_displacement(z=CUP_SIDE_APPROACH + 0.01, xyz_coord='local'),
                  tag='grasp_side_insert', constraint_pose=[1, 1, 1, 1, 1, 0], time_dilation_factor=0.5)
        self.move(self.atom.close_gripper(GRIP_CLOSE))    # close to GRIP_CLOSE (not 0): full closure crushes the cup into the soft gelpad -> unstable contact, cup wobbles/spins in air
        if self.no_tactile:
            # Motion-test mode (UNIVTAC_NO_TACTILE=1): no gelpad contact, so weld the cup rigidly to the gripper for carrying/pouring
            self.weld_actor(self.cup, self._robot_manager)

    def _pour(self):
        """Pour: move straight to the pour prep pose above the plate, then flip the cup stepwise.

        Skips the intermediate lift-then-translate moves; moving the gripper center directly
        above the plate keeps the path short and avoids swinging between grasp and pour.
        """
        gc0 = self._robot_manager.get_gripper_center_pose()
        gc_prep = Pose([PLATE_POS.p[0] + POUR_PREP_X, PLATE_POS.p[1] + POUR_OVER_Y, POUR_Z], gc0.q)
        ee_prep = self._robot_manager.gripper_center_to_ee(gc_prep)
        self.move(self.atom.move_to_pose(ee_prep), tag='move_over_plate', time_dilation_factor=0.5)

        # Rotate the wrist joint directly in joint space, bypassing end-effector pose planning failures.
        arm_q = self._robot_manager.robot.data.joint_pos[0, :7].clone()
        start_q7 = arm_q[6].item()
        target_q7 = start_q7 - POUR_ANGLE
        q7_seq = torch.linspace(start_q7, target_q7, POUR_JOINT_STEPS, device=self.device)
        self.atom_id += 1
        self.atom_tag = 'pour'
        for q7 in q7_seq:
            q = arm_q.clone()
            q[6] = q7
            self._robot_manager.set_arm(q, force=True)
            self._step(is_save=True)
        self._update_render()
        self.delay(30, is_save=True)   # dwell after pouring so the ball settles on the plate

    # ---------------------------------------------------------------- script
    def pre_move(self):
        self.delay(15)            # let the cup/ball settle
        self.move(self.atom.open_gripper(1.0))

    def _play_once(self):
        self._grasp_cup()
        self._pour()

    # ---------------------------------------------------------------- success
    def check_success(self):
        bp = self.ball.get_pose()
        px, py = PLATE_POS.p[0], PLATE_POS.p[1]
        horiz = float(np.linalg.norm(np.array([bp.p[0] - px, bp.p[1] - py])))
        bz = float(bp.p[2])
        # Success = ball within the plate radius AND resting on the plate (low z) AND out of the cup.
        # Fix (2026-07-14): the original criterion only checked ball-in-plate-region + low, without
        # verifying the ball had left the cup; lowering the cup (ball still inside) over the plate
        # produced a false success + early stop. Require the ball to clear the cup rim.
        cup_p = self.cup.get_pose().p
        ball_to_cup = float(np.linalg.norm(np.array([bp.p[0] - cup_p[0], bp.p[1] - cup_p[1]])))
        on_plate = horiz < PLATE_R - 0.005          # within the plate (5 mm margin)
        low = bz < 0.03                             # resting on the plate (not stuck in the cup / mid-air)
        ball_out = ball_to_cup > CUP_TOP_R          # ball horizontally clear of the cup rim => actually poured out
        self.metadata['ball_xyz'] = [float(v) for v in bp.p]
        self.metadata['ball_to_plate_horiz'] = horiz
        self.metadata['ball_to_cup_horiz'] = ball_to_cup
        print(f"[POUR] ball=({bp.p[0]:.3f},{bp.p[1]:.3f},{bp.p[2]:.3f}) "
              f"horiz={horiz*1000:.1f}mm to_cup={ball_to_cup*1000:.1f}mm "
              f"on_plate={on_plate} low={low} ball_out={ball_out}", flush=True)
        return bool(on_plate and low and ball_out)
