# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import numpy as np

from isaaclab_assets import FRANKA_PANDA_CFG  # isort: skip
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

from . import mdp
##
# Scene definition
##


def create_cube_configs(num_cubes: int, x_range: tuple = (0.2, 0.6), y_range: tuple = (-0.3, 0.3),
                        z_pos: float = 0.055, cube_size: float = 0.05):
    """Create multiple cube configurations with random positions.

    Args:
        num_cubes: Number of cubes to create
        x_range: Range for x positions (min, max)
        y_range: Range for y positions (min, max)
        z_pos: Fixed z position for all cubes
        cube_size: Size of each cube

    Returns:
        Dictionary of cube configurations
    """
    cube_configs = {}

    # Generate random positions
    np.random.seed(42)  # For reproducibility
    x_positions = np.random.uniform(x_range[0], x_range[1], num_cubes)
    y_positions = np.random.uniform(y_range[0], y_range[1], num_cubes)

    # Color variations for cubes
    colors = [
        (1.0, 0.0, 0.0),  # Red
        (0.0, 1.0, 0.0),  # Green
        (0.0, 0.0, 1.0),  # Blue
        (1.0, 1.0, 0.0),  # Yellow
        (1.0, 0.0, 1.0),  # Magenta
        (0.0, 1.0, 1.0),  # Cyan
        (1.0, 0.5, 0.0),  # Orange
        (0.5, 0.0, 1.0),  # Purple
    ]

    for i in range(num_cubes):
        cube_configs[f'cube_{i}'] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Cube_{i}",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[x_positions[i], y_positions[i], z_pos],
                rot=[1, 0, 0, 0]
            ),
            spawn=sim_utils.CuboidCfg(
                size=[cube_size, cube_size, cube_size],
                semantic_tags=[('color', f'color_{i}')],
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=colors[i % len(colors)],
                    metallic=0.2
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0)
            ),
        )

    return cube_configs


@configclass
class FrankaSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # robots
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # objects - multiple cubes with random positions
    def __post_init__(self):
        """Initialize multiple cubes with random positions."""
        # Create 4 cubes with random positions
        cube_configs = create_cube_configs(num_cubes=4)

        # Add each cube to the scene configuration
        for cube_name, cube_cfg in cube_configs.items():
            setattr(self, cube_name, cube_cfg)

        # Keep the first cube as 'object' for compatibility with existing code
        self.object = cube_configs['cube_0']

    # Listens to the required transforms, adding visualization markers to end-effector frame
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0", # root of robot (base_link)
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_CFG.copy().replace(
            markers={"frame": FRAME_MARKER_CFG.markers["frame"].replace(scale=(0.1, 0.1, 0.1))},
            prim_path="/Visuals/FrameTransformer"
        ),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand", # end-effector link
                name="end_effector",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.1034],
                ),
            ),
        ],
    )


    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    
    ### Added for pick and place task ###
    drop_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),  # Match episode_length_s
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.3, 0.6),      
            pos_y=(-0.4, 0.4),
            pos_z=(0.2, 0.2),  
            roll=(0.0, 0.0), 
            pitch=(0.0, 0.0), 
            yaw=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
    )
    gripper_action: ActionTerm = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        
        # Option 1: Observe all cubes (recommended for learning which cube to pick)
        all_cubes_positions = ObsTerm(func=mdp.all_cubes_positions_in_robot_root_frame, params={"num_cubes": 4})
        
        # Option 2: Only observe the target cube (if you want to pre-select target externally)
        # target_cube_position = ObsTerm(func=mdp.target_cube_position_in_robot_root_frame)
        
        ### Added for pick and place task
        target_drop_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "drop_pose"}) 
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Reset all cubes to random positions and select target cube
    reset_all_cubes = EventTerm(
        func=mdp.reset_all_cubes_to_random_positions,
        mode="reset",
        params={
            "num_cubes": 4,
            "pose_range": {"x": (0.2, 0.6), "y": (-0.3, 0.3), "z": (0.055, 0.055)},
            "selection_mode": "closest",  # Options: "closest", "random", "sequential"
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # PHASE 1: PICKING (using target cube)
    reaching_object = RewTerm(
        func=mdp.object_ee_distance, 
        params={"std": 0.1, "use_target_cube": True, "num_cubes": 4}, 
        weight=1.0
    )
    lifting_object = RewTerm(
        func=mdp.object_is_lifted, 
        params={"minimal_height": 0.04, "use_target_cube": True, "num_cubes": 4}, 
        weight=15.0
    )

    # PHASE 2: MOVING TO TARGET
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.3, 
            "minimal_height": 0.2, 
            "command_name": "drop_pose",
            "use_target_cube": True,
            "num_cubes": 4
        },
        weight=16.0,
    )

    # When close to the goal, provide a finer-grained reward to encourage precise placement
    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.05, 
            "minimal_height": 0.2, 
            "command_name": "drop_pose",
            "use_target_cube": True,
            "num_cubes": 4
        },
        weight=5.0,
    )

    # PHASE 3: SUCCESS BONUS - Reward for successfully placing cube at goal
    object_placement_success = RewTerm(
        func=mdp.object_reached_goal,
        params={
            "command_name": "drop_pose",
            "threshold": 0.02,
            "use_target_cube": True,
            "num_cubes": 4
        },
        weight=50.0,  # Large bonus for successful placement
    )
    
    # --- PENALTIES --- #
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-5e-5)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-5e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_reaching_goal = DoneTerm(
        func=mdp.object_reached_goal, 
        params={
            "command_name": "drop_pose",
            "threshold": 0.02,
            "use_target_cube": True,
            "num_cubes": 4
        }
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-3, "num_steps": 100000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-3, "num_steps": 100000}
    )


##
# Environment configuration
##
@configclass
class FrankPickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Franka pick-and-place environment."""

    # Scene settings
    scene: FrankaSceneCfg = FrankaSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 5.0  # Increased from 5.0 to allow time for pick and place
        # simulation settings
        self.sim.dt = 0.01
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        
@configclass
class FrankPickPlaceCfgEnvCfg_PLAY(FrankPickPlaceEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False