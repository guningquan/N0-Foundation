"""Task `dual_plate_place_stack`. Description, action flow, and exact success criteria: docs/Tasks.md#dual_plate_place_stack"""
from ._base_task import *
import numpy as np
import torch

# ============================================================================
# Dual Plate PLACE-STACK: dual-arm plate stacking.
#   Init: two plates on the left/right table areas, face at local +Z.
#   Flow: A grasps plate_a's +Y rim and places it at the central target; B
#         grasps plate_b's -Y rim and places it directly on top of plate_a.
# ----------------------------------------------------------------------------
#   Object files:
#     plate_a/plate_b : assets/objects/PLATE.usd
#       - The task's PLATE.obj is not kept in the repo; regenerate it with
#         scripts/asset_tools/build_plate.py into /tmp/plate_assets/PLATE.obj.
#       - PLATE.usd carries tet_* attributes used by the UIPC deformable /
#         contact simulation.
#       - Asset rebuild chain: build_plate.py -> bake_tet.py -> write_tet_to_usd.py.
#   Geometry:
#     PLATE.usd is a shallow disc/tray, outer dia ~200 mm, rim height ~8 mm,
#     origin at body center, local z in ~[-4, +4] mm.
#   Tactile:
#     Tactile capture works from the USD alone as long as it carries mesh/tet
#     data loadable by UIPC. This task creates the UIPC actor via
#     add_from_usd_file("PLATE.usd"); GelSight contact with the rim still
#     records tactile rgb/depth/marker/pose.
# ============================================================================

# Weld toggle: with the larger low-wall plate (vertical outer wall), a fully
# closed gripper is blocked by the 4.5 mm wall and forms a strong pinch;
# contact friction alone carries the plate, so welding is off. Set True to
# fall back to rigid binding.
USE_WELD = False

# ---- Geometry (m) ---- (PLATE_STACK.usd = 190 mm dia / 40 mm tall vertical-wall shallow bowl, true scale, scale=1.0)
PLATE_HALF_Z = 0.020     # plate half height (origin to rim/bottom)
PLATE_R = 0.095          # rim outer radius (190 mm dia)
RIM_GRASP_R = 0.091      # horizontal distance from gripper center (TCP) to plate axis (vertical-wall midline r ~ 91 mm, 8 mm wall spans 87-95)
RIM_GRASP_DZ = PLATE_HALF_Z - 0.014   # vertical grasp offset (~14 mm below the rim, on the vertical wall band)
RIM_DOWN_EXTRA = 0.014   # extra 14 mm after reaching the pose so the pads seat firmly (the 40 mm wall allows it without hitting the table)
GRASP_SIDE = 0.05        # hover height component (side)
GRASP_UP = 0.06          # hover height component (up)
GRASP_LIFT = 0.10        # vertical lift right after grasping (before lateral moves)
TABLE_TOP = 0.004
GAP = 0.006
STACK_DZ = 0.028         # nesting rise (H40 vertical wall; check_success uses a range)

# ---- Layout (world frame) ----
UPRIGHT = [1, 0, 0, 0]
PLATE_A_START = Pose([0.42, 0.26, TABLE_TOP + PLATE_HALF_Z + GAP], UPRIGHT)
PLATE_B_START = Pose([0.42, -0.26, TABLE_TOP + PLATE_HALF_Z + GAP], UPRIGHT)
STACK_TARGET = Pose([0.50, 0.00, TABLE_TOP + PLATE_HALF_Z], UPRIGHT)
STACK_TOP_TARGET = Pose(
    [STACK_TARGET.p[0], STACK_TARGET.p[1], STACK_TARGET.p[2] + STACK_DZ],
    UPRIGHT,
)

# ---- Motion parameters ----
GRASP_PRE_DIS = 0.12
PLACE_HOVER = 0.10
RETRACT_Z = 0.10
ARM_A_PARK_POS = [0.36, 0.31, 0.24]
SUCCESS_HOLD_STEPS = 8                 # consecutive steps of (stacked upright + both grippers released) required (as in the cup version)
GRIPPER_RELEASE_THRESH = 0.65          # gripper opening percentage >= this counts as released


@configclass
class TaskCfg(BaseTaskCfg):
    dual_arm = True
    use_adaptive_grasp = False
    viewer = ViewerCfg(eye=(1.45, 0.0, 0.95), lookat=(0.50, 0.0, 0.08))
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
    # Without welding each grasp adds press-down + forced close + settle steps.
    step_lim = 2500  # eval step limit: max archived success steps x1.05 (data-calibrated 2026-07-25)
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
        cfg.sim.physics_material.dynamic_friction = 5.0
        cfg.sim.physics_material.static_friction = 5.0
        cfg.uipc_sim.contact.default_friction_ratio = 5.0
        # Keep plate_b as a static obstacle at init to avoid cuRobo's
        # empty-obstacle branch; both arms ignore the manipulated plates at runtime.
        self.planner_ignore_actors = {"plate_a"}
        super().__init__(cfg, mode, render_mode, **kwargs)
        self._robot_manager.ignore_actors = {"plate_a", "plate_b"}
        self._robot_manager_b.ignore_actors = {"plate_a", "plate_b"}

    # ---------------------------------------------------------------- actors
    def create_actors(self):
        # Without welding, the thin-wall pinch must carry the whole plate: the
        # lower plate is kept light (2e5 would skew the grip or slip out),
        # just heavy enough to act as a stable base.
        self.plate_a = self._actor_manager.add_from_usd_file(
            name="plate_a",
            asset_path="PLATE_STACK.usd",
            pose=PLATE_A_START,
            density=1e4,
        )
        self.plate_b = self._actor_manager.add_from_usd_file(
            name="plate_b",
            asset_path="PLATE_STACK.usd",
            pose=PLATE_B_START,
            density=5e3,
        )

    def _reset_actors(self):
        noise_a = self.create_noise([0.003, 0.003, 0.0])
        noise_b = self.create_noise([0.003, 0.003, 0.0])
        self.plate_a.set_pose(PLATE_A_START.add_offset(noise_a))
        self.plate_b.set_pose(PLATE_B_START.add_offset(noise_b))
        self.metadata["target_pose"] = [float(v) for v in STACK_TARGET.p]
        self._success_hold_count = 0

    # ---------------------------------------------------------------- helpers
    def _register_rim_grasp(self, actor, rim_dir):
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

    def _grasp_plate(self, actor, atom, rm, arm, rim_dir, camera_up=(1, 0, 0)):
        # Same recipe as dual_bowl_place_stack/unstack (holds without welding):
        # hover -> constrained linear descent (extra RIM_DOWN_EXTRA so the
        # pads seat on both walls) -> planned close + forced joint-level clamp
        # -> lift, then move. camera_up sets the wrist orientation: A (+Y arm)
        # uses [1,0,0]; mirrored B (-Y arm) needs [-1,0,0] to reach the wide
        # plate's grasp point (fingers still clamp the wall along Y, only the
        # wrist flips to B's natural side).
        rd = np.array(rim_dir, dtype=float); rd = rd / np.linalg.norm(rd)
        self.move(atom.open_gripper(1.0), arm=arm)
        center = actor.get_pose()
        gp = np.array(center.p, dtype=float) + rd * RIM_GRASP_R + np.array([0.0, 0.0, RIM_GRASP_DZ])
        print(f"[PLATE_STACK] {arm} grasp target gp=({gp[0]:.3f},{gp[1]:.3f},{gp[2]:.3f}) "
              f"center=({center.p[0]:.3f},{center.p[1]:.3f}) rim_dir={tuple(rim_dir)} cam_up={tuple(camera_up)}", flush=True)
        gc = construct_grasp_pose(gp, np.array([0, 0, 1], dtype=float), np.array(camera_up, dtype=float))
        gc_high = gc.add_bias([0.0, 0.0, GRASP_SIDE + GRASP_UP], coord="world")
        self.move(atom.move_to_pose(rm.gripper_center_to_ee(gc_high)), arm=arm)
        down_total = GRASP_SIDE + GRASP_UP + RIM_DOWN_EXTRA
        self.move(atom.move_by_displacement(z=down_total * 0.75, xyz_coord="local"), arm=arm,
                  constraint_pose=[1, 1, 1, 1, 1, 0], time_dilation_factor=0.5)
        self.move(atom.move_by_displacement(z=down_total * 0.25, xyz_coord="local"), arm=arm,
                  constraint_pose=[1, 1, 1, 1, 1, 0], time_dilation_factor=0.5)
        self.move(atom.close_gripper(0.0, depth_threshold=None), arm=arm)
        self._close_gripper_direct(rm, 0.0, settle_steps=10, is_save=True)
        if USE_WELD:
            self.weld_actor(actor, rm)
        self.delay(8, is_save=True)
        self.move(
            atom.move_by_displacement(z=GRASP_LIFT, xyz_coord="world"),
            arm=arm,
            time_dilation_factor=0.5,
        )
        gc_now = rm.get_gripper_center_pose()
        bp = actor.get_pose()
        print(
            f"[PLATE_STACK] {arm} grasp+lift: plate=({bp.p[0]:.3f},{bp.p[1]:.3f},{bp.p[2]:.3f}) "
            f"gc=({gc_now.p[0]:.3f},{gc_now.p[1]:.3f},{gc_now.p[2]:.3f}) "
            f"held={'YES' if bp.p[2] > 0.08 else 'NO(dropped)'}",
            flush=True,
        )

    def _place_inhand(self, actor, rm, atom, target_pose, arm):
        inhand = actor.get_pose().rebase(rm.get_gripper_center_pose())
        gc_mat = np.array(target_pose.to_transformation_matrix()) @ np.linalg.inv(
            np.array(inhand.to_transformation_matrix())
        )
        ee = rm.gripper_center_to_ee(Pose.from_matrix(gc_mat))
        self.move(atom.move_to_pose(ee), arm=arm, time_dilation_factor=0.5)

    def _unweld_actor(self, actor):
        self._welds = [w for w in self._welds if w[0] is not actor]

    def _plates_released(self):
        # Both plates truly released = both grippers open and both plates
        # unwelded (transport binding removed).
        a_open = float(self._robot_manager.get_gripper_percentage()) >= GRIPPER_RELEASE_THRESH
        b_open = float(self._robot_manager_b.get_gripper_percentage()) >= GRIPPER_RELEASE_THRESH
        no_plate_welds = not any(w[0] is self.plate_a or w[0] is self.plate_b for w in self._welds)
        return a_open, b_open, no_plate_welds, bool(a_open and b_open and no_plate_welds)

    def _release_plate(self, actor, atom, arm, park=False):
        rm, _ = self._arm(arm)
        self._unweld_actor(actor)
        self.delay(5, is_save=True)
        self._open_gripper_direct(rm, 1.0, settle_steps=12, is_save=True)
        # NOTE: do NOT add constraint_pose=[1,1,1,1,1,0] to the vertical
        # retreat -- the 5-axis-locked retreat often fails to plan from the
        # post-place configuration, and a single failed move makes the
        # framework skip all subsequent moves (B's whole segment would be
        # skipped and it would never grasp). Use a plain world-frame vertical
        # displacement instead (as in the bowl task's _release_bowl).
        self.move(
            atom.move_by_displacement(z=RETRACT_Z, xyz_coord="world"),
            arm=arm,
            time_dilation_factor=0.5,
        )
        if park:
            rm, _ = self._arm(arm)
            gc_now = rm.get_gripper_center_pose()
            park_gc = Pose(ARM_A_PARK_POS, gc_now.q)
            self.move(
                atom.move_to_pose(rm.gripper_center_to_ee(park_gc)),
                arm=arm,
                time_dilation_factor=0.5,
            )

    def _dbg(self, tag):
        ap, bp = self.plate_a.get_pose(), self.plate_b.get_pose()
        print(
            f"[PLATE_STACK] {tag}: "
            f"plate_a=({ap.p[0]:.3f},{ap.p[1]:.3f},{ap.p[2]:.3f}) "
            f"plate_b=({bp.p[0]:.3f},{bp.p[1]:.3f},{bp.p[2]:.3f})",
            flush=True,
        )

    # ---------------------------------------------------------------- script
    def pre_move(self):
        self.delay(15)
        self._dbg("reset settled")

    def _play_once(self):
        # A: carry the lower plate from the +Y side to the central target.
        self._grasp_plate(self.plate_a, self.atom_a, self._robot_manager, "a", rim_dir=(0, 1, 0))
        self._place_inhand(
            self.plate_a,
            self._robot_manager,
            self.atom_a,
            STACK_TARGET.add_bias([0.0, 0.0, PLACE_HOVER], coord="world"),
            "a",
        )
        self._place_inhand(self.plate_a, self._robot_manager, self.atom_a, STACK_TARGET, "a")
        self._release_plate(self.plate_a, self.atom_a, "a", park=True)
        self._dbg("A placed plate_a")

        # B: carry the upper plate from the -Y side to directly above plate_a,
        # grasping the far -Y rim (mirror of A, same configuration as the bowl task).
        self._grasp_plate(self.plate_b, self.atom_b, self._robot_manager_b, "b",
                          rim_dir=(0, -1, 0), camera_up=(1, 0, 0))
        # Key: align with plate_a's ACTUAL resting position, not the nominal
        # STACK_TARGET. plate_a carries noise/offset; placing at the nominal
        # center would make the rims collide and knock the base plate askew
        # (same failure mode as cup stacking).
        pa = self.plate_a.get_pose()
        top_target = Pose(
            [float(pa.p[0]), float(pa.p[1]), float(pa.p[2]) + STACK_DZ],
            UPRIGHT,
        )
        self._place_inhand(
            self.plate_b,
            self._robot_manager_b,
            self.atom_b,
            top_target.add_bias([0.0, 0.0, PLACE_HOVER], coord="world"),
            "b",
        )
        self._place_inhand(self.plate_b, self._robot_manager_b, self.atom_b, top_target, "b")
        self._release_plate(self.plate_b, self.atom_b, "b")
        self._dbg("B stacked plate_b")
        self.delay(30, is_save=True)

    # ---------------------------------------------------------------- success
    def check_success(self):
        ap = self.plate_a.get_pose()
        bp = self.plate_b.get_pose()
        target_xy = np.array(STACK_TARGET.p[:2], dtype=float)
        a_xy = np.array(ap.p[:2], dtype=float)
        b_xy = np.array(bp.p[:2], dtype=float)
        target_err = float(np.linalg.norm(a_xy - target_xy))
        stack_err = float(np.linalg.norm(b_xy - a_xy))
        dz = float(bp.p[2] - ap.p[2])
        height_err = abs(dz - STACK_DZ)
        a_up = float(np.dot(ap.to_transformation_matrix()[:3, 2], np.array([0, 0, 1]))) > 0.8
        b_up = float(np.dot(bp.to_transformation_matrix()[:3, 2], np.array([0, 0, 1]))) > 0.8
        a_open, b_open, no_plate_welds, released = self._plates_released()
        stacked_upright = target_err < 0.04 and stack_err < 0.03 and height_err < 0.015 and a_up and b_up
        if stacked_upright and released:
            self._success_hold_count = getattr(self, "_success_hold_count", 0) + 1
        else:
            self._success_hold_count = 0

        self.metadata["plate_a"] = [float(v) for v in ap.p]
        self.metadata["plate_b"] = [float(v) for v in bp.p]
        self.metadata["target_err"] = target_err
        self.metadata["stack_err"] = stack_err
        self.metadata["height_err"] = height_err
        self.metadata["gripper_a_open"] = a_open
        self.metadata["gripper_b_open"] = b_open
        self.metadata["plates_unwelded"] = no_plate_welds
        self.metadata["released"] = released
        self.metadata["stacked_upright"] = bool(stacked_upright)
        self.metadata["success_hold_count"] = int(self._success_hold_count)
        print(
            f"[PLATE_STACK] target_err={target_err*1000:.1f}mm "
            f"stack_err={stack_err*1000:.1f}mm dz={dz*1000:.1f}mm "
            f"height_err={height_err*1000:.1f}mm a_up={a_up} b_up={b_up} "
            f"a_open={a_open:.2f} b_open={b_open:.2f} unwelded={no_plate_welds} "
            f"stacked_upright={stacked_upright} hold={self._success_hold_count}/{SUCCESS_HOLD_STEPS}",
            flush=True,
        )
        # Vertical-wall plates have no fixed nesting depth -> use a range to
        # decide "upper plate seated on the lower plate".
        nested = 0.012 < dz < 0.050
        return bool(target_err < 0.04 and stack_err < 0.03 and nested and a_up and b_up)
