"""Task `insert_USB`. Description, action flow, and exact success criteria: docs/Tasks.md#insert_usb"""
from ._base_task import *
import numpy as np
import torch

# ============================================================================
# Insert USB -- pure peg-in-hole: the gripper starts holding the USB and aligns
# it directly with the hole.
#   peg : USB.usd      (rigid, body 19.9x9.5 mm, full length 63 mm, plug 12x4.5 mm
#                       facing -z, protruding 11 mm)
#   slot: USBSlot.usd  (base with hole, 26x16x13 mm, hole 15x7.5 mm, depth 11 mm,
#                       opening on the top face, density=1e5)
#   Script: _reset_actors places the USB at the gripper center (plug down) and
#   nearly closes the gripper instantly -> pre_move tightens the grip and moves
#   above the hole -> _play_once fine-aligns and presses down to insert.
# ----------------------------------------------------------------------------
#   USB local geometry (origin near the body center): plug tip -31.5 mm,
#   body bottom -20.5 mm, body top +31.5 mm
#   => origin->plug tip 0.0315 m; origin->body center +0.0055 m
# ============================================================================

SLOT_HEIGHT = 0.013      # base height (m), hole opening on the top face
HOLE_DEPTH  = 0.011      # hole depth (m)
USB_TIP     = 0.0315     # USB origin -> plug tip (m)
BODY_CENTER = 0.0055     # USB origin -> body center (m, along +z)
# [tune] grip: place the body center at the gripper center -> USB origin sits BODY_CENTER below the gripper center
GRIP_Z      = -BODY_CENTER


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
    step_lim = 1000  # eval step limit: max archived success steps x1.05 (data-calibrated 2026-07-25)


class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode: Literal['collect', 'eval'] = 'collect', render_mode: str | None = None, **kwargs):
        cfg.sim.physics_material.dynamic_friction = 2.5
        cfg.sim.physics_material.static_friction = 2.5
        cfg.uipc_sim.contact.default_friction_ratio = 2.5
        # The USB stays in hand the whole time, so the planner must treat it as attached rather
        # than a static obstacle (otherwise the long peg occludes the gripper -> planning always fails).
        # Must be set before super().__init__(): the planner (and its frozen collision world) is
        # created inside super().__init__(); without the ignore set by then, the USB initial pose
        # (0.45, 0, 0.25, right at the home gripper) is treated as a static obstacle -> the start
        # state is already in world collision -> action 0 planning fails.
        self.planner_ignore_actors = {'usb'}
        super().__init__(cfg, mode, render_mode, **kwargs)

    # ---------------------------------------------------------------- actors
    def create_actors(self):
        # Base: heavy rigid body acting as a fixed socket
        self.slot = self._actor_manager.add_from_usd_file(
            name='slot', asset_path="USBSlot.usd",
            pose=Pose([0.55, 0.0, 0.002], [1, 0, 0, 0]), density=1e5,
        )
        # USB: spawn at a safe free spot; _reset_actors then places it precisely at the gripper center (avoids initial penetration)
        self.usb = self._actor_manager.add_from_usd_file(
            name='usb', asset_path="USB.usd",
            pose=Pose([0.45, 0.0, 0.25], [1, 0, 0, 0]),
        )

    # ---------------------------------------------------------------- reset
    def _reset_actors(self):
        # Base xy noise, +/-5 mm
        base_offset = self.create_noise([0.005, 0.005, 0.0])
        base_pose = Pose([0.55, 0.0, self.slot.get_pose()[2]], [1, 0, 0, 0]).add_offset(base_offset)
        self.slot.set_pose(base_pose)

        # Place the USB at the gripper center: at home the gripper points down, so identity orientation means plug down
        gc = self._robot_manager.get_gripper_center_pose()
        usb_pose = Pose(gc.p, [1, 0, 0, 0]).add_bias([0.0, 0.0, GRIP_Z])
        self.usb.set_pose(usb_pose)
        # Start with the gripper nearly closed: force-snap the fingers close to the USB (tiny gap,
        # no penetration -> no IPC error), removing the visible slow close from the 0.02 opening;
        # then set the target to 0 so the final squeeze establishes the grip while the USB is still constrained.
        n, dev = self.num_envs, self.device
        self._robot_manager.set_gripper(torch.full((n, 2), 0.006, device=dev), force=True)
        self._robot_manager.set_gripper(torch.zeros((n, 2), device=dev), force=False)

    # ---------------------------------------------------------------- script
    def pre_move(self):
        self.delay(10)
        # Close the gripper on the pre-positioned USB
        self.move(self.atom.close_gripper())

        # Hole target (place_actor takes the USB origin target): plug tip at the hole opening -> origin USB_TIP above the opening
        self.hole_pose = self.slot.get_pose().add_bias([0.0, 0.0, SLOT_HEIGHT + USB_TIP])
        noise = self.create_noise([0.004, 0.004, 0.0])
        self.noise_pose = self.hole_pose.add_offset(noise)

        # Move above the hole (with noise; plug tip hovers ~6 cm above the opening, enough clearance so the long peg does not hit the base and fail planning)
        self.move(self.atom.place_actor(
            self.usb, target_pose=self.noise_pose.add_bias([0.0, 0.0, 0.06]),
            pre_dis=0.05, dis=0.02, is_open=False,
        ))

    def _play_once(self):
        # Fine-align to the hole opening
        self.move(self.atom.place_actor(
            self.usb, target_pose=self.hole_pose,
            pre_dis=0.02, dis=0.004, is_open=False,
        ), time_dilation_factor=0.5)
        # Compliant press-down insertion (rotation about z relaxed); total ~1.5x hole depth = fully seated + light press
        self.move(self.atom.move_by_displacement(
            z=HOLE_DEPTH * 0.9, xyz_coord='local',
        ), time_dilation_factor=0.5, constraint_pose=[1, 1, 1, 1, 1, 0])
        self.move(self.atom.move_by_displacement(
            z=HOLE_DEPTH * 0.6, xyz_coord='local',
        ), time_dilation_factor=0.5, constraint_pose=[1, 1, 1, 1, 1, 0])
        self.delay(20, is_save=True)

    # ---------------------------------------------------------------- success
    def check_success(self, z_threshold=-0.008):
        # Success = USB aligned with the hole and actually inserted (z<0 means inserted;
        # z=-0.011 fully seated; -0.008 ~ 73% depth).
        # Fix (2026-07-14): the old z_threshold=+0.006 was too loose; hovering ~5.5 mm above the
        # opening counted as success, and check_success early-stopped on arrival so the policy
        # never pressed down. Negative threshold requires real insertion.
        usb_pose = self.usb.get_pose().rebase(self.hole_pose)
        self.metadata['rel_pose'] = usb_pose.tolist()
        return np.all(np.abs(usb_pose.p[0:2]) < np.array([0.005, 0.005])) \
            and usb_pose.p[2] < z_threshold \
            and np.dot(usb_pose.to_transformation_matrix()[:3, 2], np.array([0, 0, 1])) > 0.95
