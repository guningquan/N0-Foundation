import yaml
import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.utils import configclass

from ..utils.transforms import *
from ..utils.atom import GRASP_DIRECTION_DIC
from .robot_cfg import RobotCfg
from .curobo_planner import CuroboPlanner, CuroboPlannerCfg
from .._global import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from curobo.wrap.reacher.motion_gen import MotionGenResult
    from .._base_task import BaseTask

class RobotManager:
    def __init__(self, robot_cfg:RobotCfg, task:'BaseTask', planner_time_dilation_factor:float=1.0,
                 name:str='robot'):
        self.cfg = robot_cfg
        self.task = task
        self.name = name
        self.device = task.device
        # Actors this arm's planner must ignore (in-hand attachments / targets to pass through);
        # each arm has its own ignore set in dual-arm mode.
        self.ignore_actors:set = set()
        self.sensor_type = task.cfg.tactile_sensor_type
        if self.sensor_type in ['gsmini', 'gf225', 'xensews']: # franka panda
            self.robot_type = 'franka_panda'

        self.robot = Articulation(self.cfg.robot)
        self.task.scene.articulations[name] = self.robot
        self.planner_time_dilation_factor = planner_time_dilation_factor

        self.gripper_max_qpos = 0.039
        self.last_arm_velocity = None
        self.last_gripper_velocity = None

        if self.robot_type == 'franka_panda':
            self.hand_name = 'panda_hand'
            self._arm_joint_names = [
                'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
                'panda_joint5', 'panda_joint6', 'panda_joint7'
            ]
            self._gripper_joint_names = [
                'panda_finger_joint1', 'panda_finger_joint2'
            ]
            self.gripper_max_qpos = self.cfg.gripper_max_qpos
            self.yaml_path = str(EMBODIMENTS_ROOT / 'franka' / 'curobo.yml')
            offset = self.cfg.gripper_offset
        else:
            raise NotImplementedError(f"Robot type {self.robot_type} not implemented.")
 
        # offset from end-effector to gripper center frame
        self._offset = Pose(p=[0, 0, -offset], q=[1, 0, 0, 0])
        self._offset_pos = torch.tensor([0.0, 0.0, offset], device=self.device).repeat(self.task.num_envs, 1)
        self._offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.task.num_envs, 1)

    def setup(self):
        """Set up robot properties."""
        body_ids, body_names = self.robot.find_bodies(self.hand_name)
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]
        self._jacobi_body_idx = self._body_idx - 1

        joint_names = self.robot.joint_names
        self.joint_name_to_id = {name: i for i, name in enumerate(joint_names)}

        self._arm_ids = torch.tensor([
            self.joint_name_to_id[n] for n in self._arm_joint_names
        ], device=self.device)
        self._gripper_ids = torch.tensor([
            self.joint_name_to_id[n] for n in self._gripper_joint_names
        ], device=self.device)
        self.origin_pose = self.get_gripper_center_pose()
        self._all_ids = torch.cat([self._arm_ids, self._gripper_ids], dim=0)
 
        # Base pose (including orientation): the second arm's base has a yaw; using position only would give the planner a wrongly oriented frame.
        self.root_pose = Pose(
            self.robot.data.root_link_pos_w[0].cpu().numpy(),
            self.robot.data.root_link_quat_w[0].cpu().numpy(),
        )
        planner_cfg = CuroboPlannerCfg(
            dt=self.task.cfg.sim.dt,
            all_joints_name=self.robot.joint_names,
            active_joints_name=self._arm_joint_names,
            robot_prime_path=self.cfg.robot.prim_path,
            yaml_path=self.yaml_path
        )
        self.planner = CuroboPlanner(
            task=self.task,
            cfg=planner_cfg,
            robot_origin_pose=self.root_pose,
        )
        # Give the planner access to this arm's ignore set (in-hand objects / pass-through targets)
        self.planner.owning_manager = self
    
    def ee_to_gripper_center(self, ee_pose:Pose) -> Pose:
        """Convert an end-effector pose to the gripper-center pose."""
        return ee_pose.add_offset(self._offset.inv())

    def gripper_center_to_ee(self, gripper_center_pose:Pose) -> Pose:
        """Convert a gripper-center pose to the end-effector target pose."""
        return gripper_center_pose.add_offset(self._offset)
    
    def get_gripper_center_pose(self, env_ids:slice=None) -> Pose:
        """Current gripper-center pose."""
        return self.ee_to_gripper_center(self.get_ee_pose())
    
    def get_inhand_pose(self, actor:'Actor') -> Pose:
        return actor.get_pose().rebase(self.get_gripper_center_pose())
    
    def get_ee_pose(self, env_ids:slice=None) -> Pose:
        """Current end-effector pose in the WORLD frame.

        plan_arm takes world-frame targets (rebasing to this arm's base frame internally), so
        this must return the world frame; otherwise move_by_displacement/place_actor patterns
        ("current ee + displacement") break for non-origin bases (e.g. dual arm). For a single
        arm at the origin with identity orientation, world frame == base frame (backward compatible).
        """
        if env_ids is None:
            env_ids = [0]
        ee_pos_w = self.robot.data.body_link_pos_w[:, self._body_idx]
        ee_quat_w = self.robot.data.body_link_quat_w[:, self._body_idx]
        return Pose(ee_pos_w[0].cpu().numpy(), ee_quat_w[0].cpu().numpy())

    def get_qpos(self):
        return self.robot.data.joint_pos.clone().cpu()
    
    def get_gripper_qpos(self):
        return self.get_qpos()[0, self._gripper_ids[0]].clone().cpu().item()
    def get_gripper_percentage(self):
        return self.get_gripper_qpos() / self.gripper_max_qpos

    def set_arm(self, pos:torch.Tensor, vel:torch.Tensor=None, env_ids:slice=None, force:bool=True):
        '''Set arm joint targets.'''
        self.robot.set_joint_position_target(pos, joint_ids=self._arm_ids, env_ids=env_ids)
        if vel is not None:
            self.robot.set_joint_velocity_target(vel, joint_ids=self._arm_ids, env_ids=env_ids)
        if force:
            self.robot.root_physx_view.set_dof_positions(
                self.robot._data.joint_pos_target,
                self.robot._ALL_INDICES
            )

    def set_gripper(self, pos:torch.Tensor, vel:torch.Tensor=None, env_ids:slice=None, force:bool=True):
        '''Set gripper joint targets.'''
        self.robot.set_joint_position_target(pos, joint_ids=self._gripper_ids, env_ids=env_ids)
        if vel is not None:
            self.robot.set_joint_velocity_target(vel, joint_ids=self._gripper_ids, env_ids=env_ids)
        if force:
            self.robot.root_physx_view.set_dof_positions(
                self.robot._data.joint_pos_target,
                self.robot._ALL_INDICES
            )

    def reassert(self):
        """Force-teleport every dof of this arm to its current joint target.
        In dual-arm mode the passive arm relies on this each step to stay frozen at its grasp
        pose and maintain grip (PD alone would loosen the gripper and drop the object)."""
        self.robot.root_physx_view.set_dof_positions(
            self.robot._data.joint_pos_target, self.robot._ALL_INDICES)

    def plan_arm(self, target_pose:Pose, constraint_pose=None, pre_dis=None, time_dilation_factor=None):
        result:MotionGenResult = self.planner.plan_path(
            curr_joint_pos=self.robot.data.joint_pos[0, :self.robot.num_joints-2],
            curr_joint_vel=self.robot.data.joint_vel[0, :self.robot.num_joints-2],
            target_ee_pose=target_pose,
            real_robot_pose=self.root_pose,
            pre_dis=pre_dis,
            constraint_pose=constraint_pose,
            time_dilation_factor=time_dilation_factor
        )
        
        if result.success.item():
            return {
                'status': 'Success',
                'num_steps': result.interpolated_plan.position.shape[0],
                'position': result.interpolated_plan.position.detach(),
                'velocity': result.interpolated_plan.velocity.detach()
            }
        else:
            return {'status': 'Fail', 'num_steps': 0, 'position': None, 'velocity': None}

    def solve_ik(self, target_pose:Pose):
        # direct single-shot IK (no curobo trajectory) -> joint positions or None
        return self.planner.solve_ik(
            target_pose, self.root_pose,
            curr_joint_pos=self.robot.data.joint_pos[0, :self.robot.num_joints-2])

    def gripper_percent2qpos(self, percentage:float):
        gripper_range = [0, self.gripper_max_qpos]
        target_pos = gripper_range[0] + (gripper_range[1] - gripper_range[0]) * percentage
        return target_pos

    def plan_gripper(self, pos:float, type:Literal['percent', 'qpos'] = 'percent'):
        if type == 'percent':
            target_pos = self.gripper_percent2qpos(pos)
        else:
            target_pos = pos
        gripper_pos = self.robot.data.joint_pos[0, self._gripper_ids][0]
        num_steps = np.ceil(abs(target_pos - gripper_pos.cpu().item()) / 0.0005).astype(int)
        position = torch.linspace(gripper_pos, target_pos, num_steps, device=self.device)
        velocity = torch.clip((position - gripper_pos)/self.task.cfg.sim.dt, -0.0001, 0.0001)

        return {
            'status': 'Success',
            'num_steps': num_steps,
            'position': position.detach(),
            'velocity': velocity.detach()
        }

    def _reset_idx(self, env_ids: torch.Tensor | None=None):
        """Reset the environment."""
        if not hasattr(self, 'origin_pose'):
            self.setup()
        joint_pos = self.robot.data.default_joint_pos.clone()
        joint_vel = torch.zeros_like(joint_pos)
        
        self.planner.reset()
        self.robot.set_joint_position_target(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)
    
    def get_observations(self, data_type:list[str]=['joint', 'ee']) -> dict:
        obs = {}
        if 'ee' in data_type:
            obs['ee'] = self.get_ee_pose().totensor(device=self.device)
        if 'joint' in data_type:
            obs['joint'] = self.robot.data.joint_pos.squeeze(0)
        return obs
    
    def get_grasp_perfect_direction(self):
        return 'top_down'