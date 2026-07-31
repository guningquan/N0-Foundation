from tacex_assets.robots.franka.franka_gsmini_gripper_uipc_high_res import (
    FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG
)
from tacex_assets.robots.franka.franka_xensews_gripper_uipc import (
    FRANKA_PANDA_ARM_XENSEWS_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG
)
from tacex_assets.robots.franka.franka_gf225_gripper_uipc import (
    FRANKA_PANDA_ARM_GF225_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG
)

from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from ..sensors.tactile import TactileCfg, create_tactile_cfg

@configclass
class RobotCfg:
    robot: ArticulationCfg = None
    tactiles: list[TactileCfg] = []

    gripper_offset: float = 0.131 # in m
    gripper_max_qpos: float = 0.039 # in m

    tactile_far_plane: float = 30.0 # in mm
    adaptive_grasp_depth_threshold: float = 27.5 # in mm, used for grasping
    contact_threshold: tuple[float, float] = (27.5, 28.0) # in mm, used in `gravity_rotate` api

def create_franka_gsmini_gripper(
    data_type:list[str],
    prim_path:str="/World/envs/env_.*/Robot",
    base_pos:tuple[float, float, float]|None=None,
    base_rot:tuple[float, float, float, float]|None=None,
    name_suffix:str="",
    dense_gelpad:bool=False,
):
    """Build the RobotCfg for a Franka + gsmini gripper.

    Defaults: prim at /World/envs/env_.*/Robot, base at the env origin (single-arm tasks unchanged).
    For dual arm, pass a different prim_path / base_pos / base_rot / name_suffix to build the
    second arm. Tactile prims are derived from prim_path; tactile names get name_suffix appended
    (must be globally unique, otherwise the two arms' tactile sensors collide by name).
    """
    init_kwargs = dict(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": 0.0,
            "panda_joint3": 0.0,
            "panda_joint4": -2.46,
            "panda_joint5": 0.0,
            "panda_joint6": 2.5,
            "panda_joint7": 0.741,
            "panda_finger.*": 0.02,
        }
    )
    if base_pos is not None:
        init_kwargs["pos"] = base_pos
    if base_rot is not None:
        init_kwargs["rot"] = base_rot
    robot = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG.replace(
        prim_path=prim_path,
        init_state=ArticulationCfg.InitialStateCfg(**init_kwargs),
    )
    # [PATCH-D] opt-in denser gelpad FEM mesh (option B). Point the robot USD at the dense
    # variant (gelpads re-tetrahedralized at edge_length_r=0.05: ~615 nodal / 2276 tets vs
    # the default 169 / 424); generate it with scripts/asset_tools/make_dense_gelpad.py.
    if dense_gelpad:
        from pathlib import Path
        dense_usd = (Path(__file__).resolve().parents[2] / "third_party/TacEx/source/tacex_assets"
                     "/tacex_assets/data/Robots/Franka/GelSight_Mini/Gripper/uipc_gelpads_dense_wrist.usd")
        if not dense_usd.exists():
            raise FileNotFoundError(
                f"dense gelpad USD not found: {dense_usd}\n"
                "generate it once with: python scripts/asset_tools/make_dense_gelpad.py")
        robot = robot.replace(spawn=robot.spawn.replace(usd_path=str(dense_usd)))
    tactiles = [
        create_tactile_cfg(
            prim_path=f"{prim_path}/gelsight_mini_case_left",
            gelpad_prim_path=f"{prim_path}/gelpad_left",
            gelpad_attachment_body_name="gelsight_mini_case_left",
            name=f"left_tactile{name_suffix}",
            sensor_type="gsmini",
            data_type=data_type,
            dense=dense_gelpad,
        ),
        create_tactile_cfg(
            prim_path=f"{prim_path}/gelsight_mini_case_right",
            gelpad_prim_path=f"{prim_path}/gelpad_right",
            gelpad_attachment_body_name="gelsight_mini_case_right",
            name=f"right_tactile{name_suffix}",
            sensor_type="gsmini",
            data_type=data_type,
            dense=dense_gelpad,
        )
    ]
    return RobotCfg(
        robot=robot,
        tactiles=tactiles,
        gripper_offset=0.131,
        gripper_max_qpos=0.039,
        tactile_far_plane=34.0,
        adaptive_grasp_depth_threshold=27.5,
        contact_threshold=(27.5, 28.0)
    )


# Dual-arm (gsmini) layout: two Franka bases separated along y, both facing +x (side by side,
# ALOHA-like), sharing the workspace in front of the table (x ~ 0.5).
# Side-by-side without yaw avoids initial arm/table penetration from angled bases; add a small
# yaw [tune] later if better reachability is needed.
# Quaternion for yaw theta about z = (cos theta/2, 0, 0, sin theta/2).
DUAL_ARM_A_POS = (0.0, 0.35, 0.0)       # left arm (holds the sleeve)
DUAL_ARM_A_ROT = (1.0, 0.0, 0.0, 0.0)   # facing +x
DUAL_ARM_B_POS = (0.0, -0.35, 0.0)      # right arm (inserts the screw)
DUAL_ARM_B_ROT = (1.0, 0.0, 0.0, 0.0)   # facing +x

def create_franka_gsmini_gripper_dual(data_type:list[str]):
    """Return (arm_a_cfg, arm_b_cfg). Arm A uses the default /Robot prim (existing asset/tactile
    paths); arm B uses /Robot_b with a _b suffix on tactile names. Each arm's base pose is offset."""
    arm_a = create_franka_gsmini_gripper(
        data_type, prim_path="/World/envs/env_.*/Robot",
        base_pos=DUAL_ARM_A_POS, base_rot=DUAL_ARM_A_ROT, name_suffix="",
    )
    arm_b = create_franka_gsmini_gripper(
        data_type, prim_path="/World/envs/env_.*/Robot_b",
        base_pos=DUAL_ARM_B_POS, base_rot=DUAL_ARM_B_ROT, name_suffix="_b",
    )
    return arm_a, arm_b

def create_franka_gf225_gripper(data_type:list[str]):
    robot = FRANKA_PANDA_ARM_GF225_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": 0.0,
                "panda_joint3": 0.0,
                "panda_joint4": -2.46,
                "panda_joint5": 0.0,
                "panda_joint6": 2.5,
                "panda_joint7": 0.741,
                "panda_finger.*": 0.02,
            }
        ), 
    )
    tactiles = [
        create_tactile_cfg(
            prim_path="/World/envs/env_.*/Robot/GF225_left",
            gelpad_prim_path="/World/envs/env_.*/Robot/GF225_gelpad_left",
            gelpad_attachment_body_name="GF225_left",
            name="left_tactile",
            sensor_type="gf225",
            data_type=data_type,
        ),
        create_tactile_cfg(
            prim_path="/World/envs/env_.*/Robot/GF225_right",
            gelpad_prim_path="/World/envs/env_.*/Robot/GF225_gelpad_right",
            gelpad_attachment_body_name="GF225_right",
            name="right_tactile",
            sensor_type="gf225",
            data_type=data_type,
        )
    ]
    return RobotCfg(
        robot=robot,
        tactiles=tactiles,
        gripper_offset=0.131,
        gripper_max_qpos=0.039,
        tactile_far_plane=26.5,
        adaptive_grasp_depth_threshold=25.3,
        contact_threshold=(25.5, 26.3)
    )

def create_franka_xensews_gripper(data_type:list[str]):
    robot = FRANKA_PANDA_ARM_XENSEWS_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": 0.0,
                "panda_joint3": 0.0,
                "panda_joint4": -2.46,
                "panda_joint5": 0.0,
                "panda_joint6": 2.5,
                "panda_joint7": 0.741,
                "panda_finger.*": 0.02,
            }
        ),
    )
    tactiles = [
        create_tactile_cfg(
            prim_path="/World/envs/env_.*/Robot/XenseWS_left",
            gelpad_prim_path="/World/envs/env_.*/Robot/XenseWS_gelpad_left",
            gelpad_attachment_body_name="XenseWS_left",
            name="left_tactile",
            sensor_type="xensews",
            data_type=data_type,
        ),
        create_tactile_cfg(
            prim_path="/World/envs/env_.*/Robot/XenseWS_right",
            gelpad_prim_path="/World/envs/env_.*/Robot/XenseWS_gelpad_right",
            gelpad_attachment_body_name="XenseWS_right",
            name="right_tactile",
            sensor_type="xensews",
            data_type=data_type,
        )
    ]
    return RobotCfg(
        robot=robot,
        tactiles=tactiles,
        gripper_offset=0.131,
        gripper_max_qpos=0.039,
        tactile_far_plane=30.0,
        adaptive_grasp_depth_threshold=27.3,
        contact_threshold=(27.5, 27.8)
    )