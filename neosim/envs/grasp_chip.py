"""Task `grasp_chip`. Description, action flow, and exact success criteria: docs/Tasks.md#grasp_chip"""
from ._base_task import *
import os
import numpy as np
import torch

# ============================================================================
# Grasp Chip -- pick up a fragile thin chip from the table; squeezing too hard
# "crushes" it (fail).
#   chip : CHIP.usd  (thin curved rigid strip, 82.5x18x2.4 mm with an arc)
#   Script (data collection): open gripper -> descend from above and grasp ->
#   adaptive light close -> single direct lift -> hold.
#   "Force" = tactile gelpad indentation pressure = far_plane - get_min_depth() (mm);
#   harder squeeze = larger value.
#   Max indentation over the episode > MAX_PRESSURE => squeezed too hard =>
#   chip "crushed" => fail (no actual fracture needed).
#   Normal adaptive light close is ~7 mm, so MAX_PRESSURE sits above that with margin.
# ----------------------------------------------------------------------------
#   gsmini: tactile_far_plane=34 mm (no contact), adaptive light-close target
#   depth=27.5 (~6.5 mm indentation)
# ============================================================================

CHIP_REST_Z   = 0.01      # [tune] body-center height on the table
CHIP_QUAT     = [-0.70710678, 0.0, 0.0, 0.70710678]  # placement orientation about the vertical axis
GRASP_Z       = 0.004     # [tune] grasp-point z offset from the chip body center (near the raised ends)
LIFT          = 0.08      # single direct lift height (m)
MAX_PRESSURE  = float(os.environ.get('CHIP_MAX_PRESSURE', '8.5'))  # [tune] indentation limit (mm); exceeding = crushed = fail (normal ~7 mm, hard squeeze ~8.7 mm)
HOLD_PRESSURE = 1.0       # [tune] indentation (mm) below this = essentially no contact = dropped
# Grip-force switch: 0 = default adaptive light close; >0 = close to the given tactile depth
# (smaller = tighter). Used to squeeze too hard on purpose: CHIP_GRIP_DEPTH=20
GRIP_DEPTH    = float(os.environ.get('CHIP_GRIP_DEPTH', '0'))


@configclass
class TaskCfg(BaseTaskCfg):
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(pos=(0.74, 0.0, 0.066), rot=(0.512, 0.512, 0.487, 0.487), convention="opengl"),
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.5, focus_distance=1.0, horizontal_aperture=3.6, clipping_range=(0.1, 100.0)
            ),
            width=int(os.environ.get("UNIVTAC_CAM_W",480)), height=int(os.environ.get("UNIVTAC_CAM_H",270)), update_period=1/120,
        ),
        CameraCfg(
            name="wrist",
            prim_path="/World/envs/env_.*/Robot/WristCamera/Camera",
            data_types=["rgb", "depth"],
            spawn=None,
            width=int(os.environ.get("UNIVTAC_CAM_W",480)), height=int(os.environ.get("UNIVTAC_CAM_H",270)), update_period=1/120,
        ),
    ]
    step_lim = 300  # eval step limit: max archived success steps x1.05 (data-calibrated 2026-07-25)


class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode: Literal['collect', 'eval'] = 'collect', render_mode: str | None = None, **kwargs):
        cfg.sim.physics_material.dynamic_friction = 2.5
        cfg.sim.physics_material.static_friction = 2.5
        cfg.uipc_sim.contact.default_friction_ratio = 2.5
        super().__init__(cfg, mode, render_mode, **kwargs)
        # After grasping, the chip moves with the gripper -> ignore it in planning
        # (otherwise the in-hand object occludes the gripper and planning fails)
        self.planner_ignore_actors = {'chip'}
        self.max_pressure_seen = 0.0   # max indentation over the episode (= max grip force)

    # ---------------------------------------------------------------- force
    def _pressure(self):
        return self.cfg.robot.tactile_far_plane - torch.min(self._tactile_manager.get_min_depth()).item()

    def _track(self):
        p = self._pressure()
        self.max_pressure_seen = max(self.max_pressure_seen, p)
        if p > MAX_PRESSURE:           # squeezed too hard -> chip crushed -> episode fails
            self.plan_success = False

    # ---------------------------------------------------------------- actors
    def create_actors(self):
        self.chip = self._actor_manager.add_from_usd_file(
            name='chip', asset_path="CHIP.usd",
            pose=Pose([0.5, 0.0, CHIP_REST_Z], CHIP_QUAT),
        )

    def _reset_actors(self):
        offset = self.create_noise([0.005, 0.005, 0.0])   # xy placement noise, +/-5 mm
        self.chip.set_pose(Pose([0.5, 0.0, CHIP_REST_Z], CHIP_QUAT).add_offset(offset))

    # ---------------------------------------------------------------- script
    def pre_move(self):
        self.delay(10)
        self.start_ee_z = self._robot_manager.get_ee_pose()[2]
        self.max_pressure_seen = 0.0
        self.grasp_ee_z = None   # grasp-point height, reset per episode (recorded in check_success on first hold in eval)

    def _play_once(self):
        # Open fully (grasping along the length direction, wide span)
        self.move(self.atom.open_gripper(1.0))
        # Descend from above to grasp (camera_up = chip Y axis -> pinch the two ends)
        grasp_rotate = self.rng.uniform(-np.pi / 18, np.pi / 18)
        target_pose = self.chip.get_pose().add_bias([0.0, 0.0, GRASP_Z]).add_rotation([0.0, grasp_rotate, 0.0])
        tmat = target_pose.to_transformation_matrix()
        cpose = construct_grasp_pose(target_pose.p, tmat[:3, 2], tmat[:3, 1])
        cid = self.chip.register_point(cpose, type='contact')
        self.move(self.atom.grasp_actor(self.chip, contact_point_id=cid, is_close=False)); self._track()
        # Adaptive close: default is a light grip (no crushing); with CHIP_GRIP_DEPTH set, close deeper = tighter (to squeeze too hard on purpose)
        if GRIP_DEPTH > 0:
            self.move(self.atom.close_gripper(depth_threshold=GRIP_DEPTH)); self._track()
        else:
            self.move(self.atom.close_gripper()); self._track()
        # Record ee height after grasping (before lifting): used to check "has it risen from the
        # grasp point" (the grasp descends first, so comparing against home would be wrong)
        self.grasp_ee_z = self._robot_manager.get_ee_pose()[2]
        # Single direct lift, then hold
        self.move(self.atom.move_by_displacement(z=LIFT, xyz_coord='world')); self._track()
        self.delay(20, is_save=True); self._track()
        print(f"[CHIP] max indentation = {self.max_pressure_seen:.2f} mm  (limit MAX_PRESSURE={MAX_PRESSURE}, exceeding=fail)", flush=True)

    # ---------------------------------------------------------------- success / fail
    def check_early_stop(self):
        # Indentation exceeded the limit at some point -> crushed -> fail
        return self.max_pressure_seen > MAX_PRESSURE

    def check_success(self):
        p = self._pressure()
        ee_z = self._robot_manager.get_ee_pose()[2]
        # Eval does not run _play_once/_track: check_success runs every step, so accumulate the
        # max indentation here (restores crush detection)
        self.max_pressure_seen = max(self.max_pressure_seen, p)
        self.metadata['max_pressure_mm'] = self.max_pressure_seen
        held = p > HOLD_PRESSURE                              # still holding (not dropped)
        not_crushed = self.max_pressure_seen <= MAX_PRESSURE  # never crushed
        # Eval has no expert grasp phase to set grasp_ee_z: record the grasp-point height
        # (~lowest descent point) on first hold (p > HOLD), then check for lifting
        if getattr(self, 'grasp_ee_z', None) is None:
            if held:
                self.grasp_ee_z = ee_z
            return False                                      # not holding yet => not successful
        lifted = (ee_z - self.grasp_ee_z) > LIFT * 0.7        # risen from the grasp point (not compared with home)
        return held and not_crushed and lifted
