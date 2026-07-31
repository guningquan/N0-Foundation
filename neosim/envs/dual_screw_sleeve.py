"""Task `dual_screw_sleeve`. Description, action flow, and exact success criteria: docs/Tasks.md#dual_screw_sleeve"""
from ._base_task import *
import numpy as np
import torch

# ============================================================================
# Dual Screw-Sleeve: dual-arm assembly. Left arm (A) holds the sleeve up,
# right arm (B) inserts the screw into it (assets scaled 3x).
#   screw : SCREW.usd  (thumbtack-shaped: shaft dia 30 x 105 + disc dia 72 x 12;
#           shaft along local Z, tip at -Z, disc at +Z)
#   sleeve: SLEEVE.usd (hollow cylinder: outer dia 54 / inner dia 36 / length 66;
#           bore axis along local Z, open at both ends)
#   Init: screw standing on the table disc-up (shaft down); sleeve lying on its
#         side (bore axis horizontal).
#   Flow: A grasps the lying sleeve -> lifts it to the assembly pose and holds;
#         B grasps the screw (by the disc, shaft down) -> aligns with the bore
#         -> pushes it in (with tactile sensing).
#   Planner ignores: A ignores the in-hand sleeve; B ignores the in-hand screw
#   plus the sleeve it must pass through (the small bore voxelizes as solid).
# ----------------------------------------------------------------------------
# SCREW local frame (origin centered, z in [-0.0578, 0.0578]): tip at -Z,
# disc at +Z; origin-to-tip = 0.0578 m.
# SLEEVE local frame (origin centered, z in [-0.033, 0.033], outer dia 0.054):
# bore axis along local Z, half length 0.033 m.
# ============================================================================

# ---- Geometry (3x, m) ----
SLEEVE_H     = 0.044    # sleeve length (bore open at both ends)  [asset rescaled to 2/3]
SLEEVE_OUT_R = 0.018    # sleeve outer radius (resting height when lying on its side)
SCREW_TIP    = 0.0385   # screw origin -> shaft tip (local -Z)
INSERT_DEPTH = 0.055    # [tune] shaft insertion depth into the sleeve
SLEEVE_GRASP_AXIS = 0.0    # axial offset of the sleeve grasp (0 = body center; a +Y offset made grasp/lift fail)

# ---- Layout [tune] (world frame) ----
# Objects sit directly in front of their arms (same y as the base) so the
# top-down grasp matches the validated single-arm setup (base origin -> (0.5, 0, table)).
LYING_QUAT = [0.70711, 0.70711, 0, 0]   # 90 deg about X: bore axis local Z -> world -Y (sleeve lying sideways)
# Screw: standing disc-up (identity), shaft tip at the table -> center z = half height + clearance; in front of arm B
SCREW_REST    = Pose([0.50, -0.35, 0.042], [1, 0, 0, 0])
# Sleeve: lying sideways, in front of arm A
SLEEVE_REST   = Pose([0.50,  0.35, 0.021], LYING_QUAT)
# A lifts the sleeve here with an outward tilt (bore opening toward arm B);
# the screw is inserted along that tilted axis. _place_inhand computes the
# gripper target directly (not place_actor, so a tilted pose is placed exactly).
# [tune] world euler (rad): pitch about X downward, yaw about Z toward arm B (-x).
ASSEMBLY_TILT = [0.0, 0.0, -0.3]   # yaw the bore opening outward (~17 deg); no downward pitch
ASSEMBLY_POSE = Pose([0.48,  0.18, 0.22], LYING_QUAT).add_rotation(ASSEMBLY_TILT, coord='world')   # y shifted toward arm A

# ---- Grasp [tune] ----
SLEEVE_GRASP_DZ = 0.0      # lying sleeve: top-down grasp at the outer-diameter center
SCREW_GRASP_DZ  = 0.033    # screw: offset toward +Z (disc end), pinching the disc from above with the shaft hanging below


@configclass
class TaskCfg(BaseTaskCfg):
    dual_arm = True       # auto-creates arm B (/Robot_b + *_b tactile)
    # Default GUI camera: overhead front view covering both arms and the table
    # (replaces the base class's low, offset default).
    viewer = ViewerCfg(eye=(1.45, 0.0, 0.95), lookat=(0.45, 0.0, 0.12))
    video_size = (1760, 320)   # head + wrist + wrist_b (3x480) + 4 tactile streams (2 cols x 160) = 1760 wide
    # Wider dual-arm workspace: table enlarged to 2x2 m to cover both arms
    # (bases at y=+/-0.40). Thickness unchanged (top still z=0.0025).
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
    step_lim = 1300  # eval step limit: max archived success steps x1.05 (data-calibrated 2026-07-25)
    reset_time_limit = 600.0   # heavy dual-arm scene (2 curobo instances + 4 tactile + rendering + high friction); reset is slow


class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode: Literal['collect', 'eval'] = 'collect', render_mode: str | None = None, **kwargs):
        cfg.sim.physics_material.dynamic_friction = 2.5
        cfg.sim.physics_material.static_friction = 2.5
        cfg.uipc_sim.contact.default_friction_ratio = 2.5   # moderate friction (high friction + deep grip stalls the UIPC contact solve on lift); grip retention relies on reasserting frozen joints
        super().__init__(cfg, mode, render_mode, **kwargs)
        self._robot_manager.ignore_actors = {'sleeve'}            # sleeve in A's hand
        self._robot_manager_b.ignore_actors = {'screw', 'sleeve'}  # screw in B's hand + the sleeve it passes through

    # ---------------------------------------------------------------- actors
    def create_actors(self):
        self.sleeve = self._actor_manager.add_from_usd_file(
            name='sleeve', asset_path="SLEEVE.usd", pose=SLEEVE_REST)
        self.screw = self._actor_manager.add_from_usd_file(
            name='screw', asset_path="SCREW.usd", pose=SCREW_REST)

    def _reset_actors(self):
        self.sleeve.set_pose(SLEEVE_REST.add_offset(self.create_noise([0.005, 0.005, 0.0])))
        self.screw.set_pose(SCREW_REST.add_offset(self.create_noise([0.005, 0.005, 0.0])))

    # ---------------------------------------------------------------- helpers
    def _grasp(self, actor, atom, arm, dz, grasp_from=(0, 0, 1), camera_up=(1, 0, 0),
               extra_bias=(0.0, 0.0, 0.0), grip_depth=28.0, open_amt=1.0):
        """Grasp with the given arm (construct_grasp_pose convention).

        Adaptive close to tactile depth grip_depth: stops at the opening that
        gives contact plus moderate pressure, instead of over-penetrating and
        squeezing the object away as a fixed opening would (smaller = deeper,
        tighter). grasp_from: approach direction; camera_up: up vector for
        construct_grasp_pose (panda fingers open along EE-Y); dz/extra_bias:
        grasp-point offsets; open_amt: pre-grasp opening.
        """
        self.move(atom.open_gripper(open_amt), arm=arm)
        target = actor.get_pose().add_bias([extra_bias[0], extra_bias[1], extra_bias[2] + dz], coord='world')
        cpose = construct_grasp_pose(
            np.array([target.p[0], target.p[1], target.p[2]]),
            np.array(grasp_from, dtype=float), np.array(camera_up, dtype=float))
        cid = actor.register_point(cpose, type='contact')
        self.move(atom.grasp_actor(actor, contact_point_id=cid, is_close=False), arm=arm)
        self.move(atom.close_gripper(depth_threshold=grip_depth), arm=arm)

    def _place_inhand(self, actor, rm, atom, target_pose, arm):
        """Place the in-hand actor exactly at target_pose.

        Computes the gripper EE target from the grasp-time in-hand transform
        and calls move_to_pose, bypassing place_actor's alignment semantics
        (which skew tilted targets).
        """
        inhand = actor.get_pose().rebase(rm.get_gripper_center_pose())          # actor pose in gripper frame
        gc_mat = np.array(target_pose.to_transformation_matrix()) @ \
            np.linalg.inv(np.array(inhand.to_transformation_matrix()))          # gripper-center target
        ee = rm.gripper_center_to_ee(Pose.from_matrix(gc_mat))
        self.move(atom.move_to_pose(ee), arm=arm)

    def _hole_axis(self):
        """Sleeve bore axis (world frame), the end pointing toward arm B (-Y side).

        Follows the sleeve's actual orientation, so a tilted assembly pose works.
        """
        sp = self.sleeve.get_pose()
        axis = sp.to_transformation_matrix()[:3, 2]          # sleeve local Z = bore axis (world)
        if axis[1] > 0:
            axis = -axis                                     # point toward -Y (arm B side)
        return np.array(axis, dtype=float)

    def _hole_pose(self):
        """Screw insertion target (screw origin target).

        Orientation = sleeve's actual orientation (screw shaft local -Z then
        points into the bore); origin sits SCREW_TIP outside the arm-B-side
        opening. Follows a tilted assembly pose automatically.
        """
        sp = self.sleeve.get_pose()
        axis = self._hole_axis()
        origin = np.array(sp.p, dtype=float) + axis * (SLEEVE_H / 2 + SCREW_TIP)   # just outside the bore opening
        return Pose(origin.tolist(), sp.q)

    def _dbg(self, tag):
        ga = self._robot_manager.get_gripper_qpos()        # arm A gripper opening (qpos)
        sp = self.sleeve.get_pose()
        print(f"[DBG] {tag}: armA_grip_qpos={ga:.4f}  sleeve=({sp.p[0]:.3f},{sp.p[1]:.3f},{sp.p[2]:.3f})", flush=True)

    # ---------------------------------------------------------------- script
    def pre_move(self):
        self.delay(10)   # just let the objects settle; grasp/insert runs in _play_once (recorded on video)

    def _play_once(self):
        # A: grasp the lying sleeve (top-down across the dia-54 tube) and hold
        # it at the assembly pose (horizontal, bore axis along Y).
        # camera_up=(0,-1,0): fingers still open along world X (gripping the
        # side walls, not blocking the +/-Y openings), but the wrist flips 180
        # deg about the approach axis for a smoother approach and firmer grip.
        # grip_depth clamps tightly so the cylinder cannot slip. Adaptive close
        # (depth 28 = moderate pressure, no over-penetration); grip held by
        # reasserting frozen joints.
        self._grasp(self.sleeve, self.atom_a, 'a', SLEEVE_GRASP_DZ, camera_up=(0, -1, 0),
                    extra_bias=(0.0, SLEEVE_GRASP_AXIS, 0.0), grip_depth=28.0, open_amt=0.55)
        self._place_inhand(self.sleeve, self._robot_manager, self.atom_a, ASSEMBLY_POSE, 'a')
        self._dbg("A holding sleeve, B idle")
        # B: adaptive grasp of the screw disc (shaft hanging below), lift off the table
        self._grasp(self.screw, self.atom_b, 'b', SCREW_GRASP_DZ, camera_up=(1, 0, 0), grip_depth=28.0)
        self._dbg("B grasped screw")
        self.move(self.atom_b.move_by_displacement(z=0.15, xyz_coord='world'), arm='b'); self._dbg("B lifted screw")

        # B: lateral insertion -- rotate the screw shaft horizontal (+Y),
        # align with the bore opening from the -Y side, then push in along +Y.
        sp = self.sleeve.get_pose()
        print(f"[DBG] sleeve pre-insert pose = p({sp.p[0]:.3f},{sp.p[1]:.3f},{sp.p[2]:.3f}) "
              f"q({sp.q[0]:.3f},{sp.q[1]:.3f},{sp.q[2]:.3f},{sp.q[3]:.3f})  "
              f"(assembly pose should be 0.48,0.10,0.22)", flush=True)
        hole = self._hole_pose()
        axis = self._hole_axis()                             # bore axis (toward arm B)
        hover = Pose((np.array(hole.p) + axis * 0.08).tolist(), hole.q)   # hover 8 cm outward along the bore axis
        self._place_inhand(self.screw, self._robot_manager_b, self.atom_b, hover, 'b')
        self._place_inhand(self.screw, self._robot_manager_b, self.atom_b, hole, 'b')   # align with the bore opening
        # Linear insertion: move along gripper-local z (= insertion direction
        # once aligned) with constraint_pose locking the other 5 axes so
        # curobo solves a straight line (same as insert_USB; the earlier
        # world-Y move with wrong constraints neither solved nor stayed straight).
        self.move(self.atom_b.move_by_displacement(z=INSERT_DEPTH * 0.6, xyz_coord='local'),
                  arm='b', constraint_pose=[1, 1, 1, 1, 1, 0], time_dilation_factor=0.5)
        self.move(self.atom_b.move_by_displacement(z=INSERT_DEPTH * 0.5, xyz_coord='local'),
                  arm='b', constraint_pose=[1, 1, 1, 1, 1, 0], time_dilation_factor=0.5)
        self.delay(20, is_save=True)

    # ---------------------------------------------------------------- success
    def check_success(self):
        screw_pose = self.screw.get_pose()
        sleeve_pose = self.sleeve.get_pose()
        rel = screw_pose.rebase(sleeve_pose)             # screw in sleeve frame (bore axis = sleeve local Z)
        self.metadata['rel_pose'] = rel.tolist()
        # Shaft axis collinear with bore axis (lateral insertion, so use abs):
        # screw local Z in the sleeve frame should be ~ +/-Z.
        axis_dot = float(np.dot(rel.to_transformation_matrix()[:3, 2], np.array([0, 0, 1])))
        axis_aligned = abs(axis_dot) > 0.9
        radial = float(np.linalg.norm(rel.p[0:2]))       # offset perpendicular to the bore axis
        radial_ok = radial < 0.008
        depth_ok = bool(abs(rel.p[2]) < SLEEVE_H / 2 + 0.02)  # within the sleeve along the bore axis
        print(f"[DUAL] rel(sleeve frame) radial={radial*1000:.1f}mm along_axis={rel.p[2]*1000:.1f}mm "
              f"|axis_dot|={abs(axis_dot):.3f} -> aligned={axis_aligned} radial={radial_ok} depth={depth_ok}", flush=True)
        return bool(axis_aligned and radial_ok and depth_ok)
