# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# -----------------------------------------------------------------------

def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    use_target_cube: bool = False,
    num_cubes: int = 4,
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel.
    
    Args:
        use_target_cube: If True, use the target cube from env.target_cube_idx.
                        If False, use the object_cfg (backward compatible).
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
<<<<<<< HEAD
    ee_w = ee_frame.data.target_pos_w[..., 0, :]  # (num_envs, 3)
    
    if use_target_cube:
        # Get target cube position based on env.target_cube_idx
        if not hasattr(env, 'target_cube_idx'):
            env.target_cube_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        
        target_idx = env.target_cube_idx
        cube_positions_w = []
        for i in range(num_cubes):
            cube_name = f"cube_{i}"
            if cube_name in env.scene.keys():
                cube: RigidObject = env.scene[cube_name]
                cube_positions_w.append(cube.data.root_pos_w[:, :3].unsqueeze(1))
        
        all_cubes_w = torch.cat(cube_positions_w, dim=1)  # (num_envs, num_cubes, 3)
        cube_pos_w = all_cubes_w[torch.arange(env.num_envs, device=env.device), target_idx]
    else:
        # Original behavior
        object: RigidObject = env.scene[object_cfg.name]
        cube_pos_w = object.data.root_pos_w[:, :3]
    
    # Distance of the end-effector to the object
=======
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
>>>>>>> 4ced5ec5c64190a74d6b135520d0e7bd6e256168
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    return 1 - torch.tanh(object_ee_distance / std)

<<<<<<< HEAD
def object_is_lifted(
    env: ManagerBasedRLEnv, 
    minimal_height: float, 
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    use_target_cube: bool = False,
    num_cubes: int = 4,
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height.
    
    Args:
        use_target_cube: If True, use the target cube from env.target_cube_idx.
                        If False, use the object_cfg (backward compatible).
    """
    if use_target_cube:
        # Get target cube position based on env.target_cube_idx
        if not hasattr(env, 'target_cube_idx'):
            env.target_cube_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        
        target_idx = env.target_cube_idx
        cube_positions_w = []
        for i in range(num_cubes):
            cube_name = f"cube_{i}"
            if cube_name in env.scene.keys():
                cube: RigidObject = env.scene[cube_name]
                cube_positions_w.append(cube.data.root_pos_w[:, :3].unsqueeze(1))
        
        all_cubes_w = torch.cat(cube_positions_w, dim=1)  # (num_envs, num_cubes, 3)
        cube_pos_w = all_cubes_w[torch.arange(env.num_envs, device=env.device), target_idx]
        cube_height = cube_pos_w[:, 2]
    else:
        object: RigidObject = env.scene[object_cfg.name]
        cube_height = object.data.root_pos_w[:, 2]
    
    return torch.where(cube_height > minimal_height, 1.0, 0.0)
=======
>>>>>>> 4ced5ec5c64190a74d6b135520d0e7bd6e256168

def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    use_target_cube: bool = False,
    num_cubes: int = 4,
) -> torch.Tensor:
<<<<<<< HEAD
    """Reward the agent for tracking the goal pose using tanh-kernel.
    
    Args:
        minimal_height: If provided, only reward when object is above this height.
                       If None, reward movement to goal without lifting requirement.
        use_target_cube: If True, use the target cube from env.target_cube_idx.
                        If False, use the object_cfg (backward compatible).
    """
=======
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
>>>>>>> 4ced5ec5c64190a74d6b135520d0e7bd6e256168
    robot: RigidObject = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)
<<<<<<< HEAD
    
    # Compute target position in world frame
    target_pos_b = command[:, :3]
    target_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, target_pos_b)
    
    if use_target_cube:
        # Get target cube position based on env.target_cube_idx
        if not hasattr(env, 'target_cube_idx'):
            env.target_cube_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        
        target_idx = env.target_cube_idx
        cube_positions_w = []
        for i in range(num_cubes):
            cube_name = f"cube_{i}"
            if cube_name in env.scene.keys():
                cube: RigidObject = env.scene[cube_name]
                cube_positions_w.append(cube.data.root_pos_w[:, :3].unsqueeze(1))
        
        all_cubes_w = torch.cat(cube_positions_w, dim=1)  # (num_envs, num_cubes, 3)
        object_pos_w = all_cubes_w[torch.arange(env.num_envs, device=env.device), target_idx]
    else:
        object: RigidObject = env.scene[object_cfg.name]
        object_pos_w = object.data.root_pos_w[:, :3]
    
    # Distance of object to goal
    distance = torch.norm(target_pos_w - object_pos_w, dim=1)
    reward = 1 - torch.tanh(distance / std)
    
    # Apply lifting constraint if specified
    if minimal_height is not None:
        reward = (object_pos_w[:, 2] > minimal_height) * reward
    
    return reward
=======
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))
>>>>>>> 4ced5ec5c64190a74d6b135520d0e7bd6e256168


def drop_object_reward(
    env: ManagerBasedRLEnv,
    distance_threshold: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:  
    """Reward the agent for dropping the object at the target location."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    # check if within threshold
    within_threshold = (distance < distance_threshold).float()
    # gripper opening reward
    gripper_joints = robot.data.joint_pos[:, -2:]
    gripper_opened = torch.where(torch.mean(gripper_joints, dim=1) >= 0.04, 1.0, 0.0)
    return gripper_opened * within_threshold


def grasp_reward(
    env: ManagerBasedRLEnv,
    distance_threshold: float = 0.04,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward closing the gripper when near the object."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]
    
    # Get distance from EE to object
    object_pos_w = object.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(object_pos_w - ee_pos_w, dim=1)
    
    # Only reward gripper closing when close to object
    near_object = (distance < distance_threshold).float()
    
    # Gripper closure: 0.04 is fully open, 0.0 is fully closed
    gripper_joints = robot.data.joint_pos[:, -2:]
    gripper_closure = 1.0 - torch.clamp(torch.mean(gripper_joints, dim=1) / 0.04, 0.0, 1.0)
    
    return near_object * gripper_closure


def placement_height_reward(
    env: ManagerBasedRLEnv,
    xy_threshold: float = 0.08,
    target_height_offset: float = 0.05,
    command_name: str = "drop_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward lowering the object when it's near the target XY position."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # Get target position in world frame
    target_pos_b = command[:, :3]
    target_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, target_pos_b
    )
    
    object_pos_w = object.data.root_pos_w
    
    # Check if object is near target in XY plane
    xy_distance = torch.norm(object_pos_w[:, :2] - target_pos_w[:, :2], dim=1)
    near_target_xy = (xy_distance < xy_threshold).float()
    
    # Reward being at the right height (target height + small offset)
    target_height = target_pos_w[:, 2] + target_height_offset
    height_error = torch.abs(object_pos_w[:, 2] - target_height)
    height_reward = torch.exp(-height_error / 0.05)  # Sharp peak at correct height
    
    return near_target_xy * height_reward


def release_reward(
    env: ManagerBasedRLEnv,
    xy_threshold: float = 0.05,
    height_threshold: float = 0.08,
    command_name: str = "drop_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward opening the gripper when object is correctly positioned above target."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # Get target position in world frame
    target_pos_b = command[:, :3]
    target_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, target_pos_b
    )
    
    object_pos_w = object.data.root_pos_w
    
    # Check if object is well-positioned
    xy_distance = torch.norm(object_pos_w[:, :2] - target_pos_w[:, :2], dim=1)
    height_distance = torch.abs(object_pos_w[:, 2] - target_pos_w[:, 2])
    
    well_positioned = ((xy_distance < xy_threshold) & (height_distance < height_threshold)).float()
    
    # Gripper opening: 0.0 is closed, 0.04 is open
    gripper_joints = robot.data.joint_pos[:, -2:]
    gripper_opening = torch.clamp(torch.mean(gripper_joints, dim=1) / 0.04, 0.0, 1.0)
    
    return well_positioned * gripper_opening

# -----------------------------------------------------------------------


# -----------------------------------------------------------------------
# Separated Staged Rewards
# -----------------------------------------------------------------------


def reward_stage_reach(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    use_target_cube: bool = False,
    num_cubes: int = 4,
) -> torch.Tensor:
<<<<<<< HEAD
    """
    A staged reward that guides the agent through Reach -> Lift -> Move -> Place.
    The later stages are only rewarded if the previous stages are satisfied.
    
    Args:
        use_target_cube: If True, use the target cube from env.target_cube_idx.
                        If False, use the object_cfg (backward compatible).
    """
    # 1. Get Assets
    robot: RigidObject = env.scene[robot_cfg.name]
=======
    """Stage 1: Reward for reaching the object."""
    object: RigidObject = env.scene[object_cfg.name]
>>>>>>> 4ced5ec5c64190a74d6b135520d0e7bd6e256168
    ee_frame = env.scene[ee_frame_cfg.name]

<<<<<<< HEAD
    # 2. Calculate Transforms
    # Target in World Frame
    target_pos_b = command[:, :3]
    target_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, target_pos_b)
    
    # Get object position (either specific object or target cube)
    if use_target_cube:
        if not hasattr(env, 'target_cube_idx'):
            env.target_cube_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        
        target_idx = env.target_cube_idx
        cube_positions_w = []
        for i in range(num_cubes):
            cube_name = f"cube_{i}"
            if cube_name in env.scene.keys():
                cube: RigidObject = env.scene[cube_name]
                cube_positions_w.append(cube.data.root_pos_w[:, :3].unsqueeze(1))
        
        all_cubes_w = torch.cat(cube_positions_w, dim=1)  # (num_envs, num_cubes, 3)
        object_pos_w = all_cubes_w[torch.arange(env.num_envs, device=env.device), target_idx]
    else:
        object: RigidObject = env.scene[object_cfg.name]
        object_pos_w = object.data.root_pos_w[:, :3]
    
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]

    # 3. Calculate Distances
    # Distance: EE to Object
    d_ee_obj = torch.norm(object_pos_w - ee_pos_w, dim=1)
    
    # Distance: Object to Target (XY plane only)
    d_obj_target_xy = torch.norm(object_pos_w[:, :2] - target_pos_w[:, :2], dim=1)
    
    # Distance: Object to Target (Z axis / Height)
    d_obj_target_z = torch.abs(object_pos_w[:, 2] - target_pos_w[:, 2])
=======
    object_pos_w = object.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]

    # as the robot approaches the object, it should lower its speed
    object_speed = torch.norm(object.data.root_vel_w, dim=1)
>>>>>>> 4ced5ec5c64190a74d6b135520d0e7bd6e256168

    distance = torch.norm(object_pos_w - ee_pos_w, dim=1)
    return (1 - torch.tanh(distance / std)) * torch.exp(-object_speed)

# def reward_stage_reach_linear(
#     env: ManagerBasedRLEnv,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
# ) -> torch.Tensor:
#     # 1. Get EE and Object positions
#     robot: RigidObject = env.scene[robot_cfg.name]
#     # NOTE: Ensure index -1 is your hand/gripper. If unsure, check your asset.
#     ee_pos_w = robot.data.body_pos_w[:, -1, :] 
#     object: RigidObject = env.scene[object_cfg.name]
#     object_pos_w = object.data.root_pos_w

#     # 2. Calculate Euclidean Distance
#     distance = torch.norm(object_pos_w - ee_pos_w, dim=1)
    
#     # 3. The Fix: Negative Distance Reward
#     # This has a constant gradient. 1cm closer is ALWAYS +0.01 reward points.
#     # No saturation.
#     return -distance

def reward_stage_lift(
    env: ManagerBasedRLEnv,
    min_height: float = 0.04,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Stage 2: Reward ONLY for lifting height."""
    object: RigidObject = env.scene[object_cfg.name]

    return torch.where(
        object.data.root_pos_w[:, 2] > min_height,
        1.0,
        0.0,
    )


def reward_stage_transport(
    env: ManagerBasedRLEnv,
    std: float = 0.2,
    min_height: float = 0.04,
    command_name: str = "drop_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Stage 3: Reward for XY alignment ONLY when lifted."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    is_lifted = object.data.root_pos_w[:, 2] > min_height

    target_pos_b = command[:, :3]
    target_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, target_pos_b
    )
    object_pos_w = object.data.root_pos_w
    d_obj_target = torch.norm(object_pos_w - target_pos_w, dim=1)

    return torch.where(
        is_lifted,
        1 - torch.tanh(d_obj_target / std),
        torch.zeros_like(d_obj_target),
    )


def reward_stage_place(
    env: ManagerBasedRLEnv,
    distance_threshold: float = 0.03,  # within 3 cm to target
    command_name: str = "drop_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Stage 4: Reward for lowering and releasing. Only active if above target."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    target_pos_b = command[:, :3]
    target_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, target_pos_b
    )
    object_pos_w = object.data.root_pos_w

    # 1. Check if distance to target XY is within threshold
    d_obj_target = torch.norm(object_pos_w - target_pos_w, dim=1)
    can_drop = (d_obj_target < distance_threshold).float()

    # 2. Gripper Release Reward
    gripper_joints = robot.data.joint_pos[:, -2:]
    gripper_norm = torch.clamp(torch.mean(gripper_joints, dim=1) / 0.04, 0.0, 1.0)

    return gripper_norm * can_drop
