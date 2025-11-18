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

def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3), absolute world position
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3), absolute world position
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)

def object_is_lifted(
    env: ManagerBasedRLEnv, 
    minimal_height: float, 
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)

def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    minimal_height: float = None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel.
    
    Args:
        minimal_height: If provided, only reward when object is above this height.
                       If None, reward movement to goal without lifting requirement.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # Compute target position in world frame
    target_pos_b = command[:, :3]
    target_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, target_pos_b)
    
    # Distance of object to goal
    distance = torch.norm(target_pos_w - object.data.root_pos_w, dim=1)
    reward = 1 - torch.tanh(distance / std)
    
    # Apply lifting constraint if specified
    if minimal_height is not None:
        reward = (object.data.root_pos_w[:, 2] > minimal_height) * reward
    
    return reward

def placing_object_gentle(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for placing the object gently at the target position."""

    object: RigidObject = env.scene[object_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    target_pos_b = command[:, :3]
    target_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, 
        robot.data.root_quat_w, 
        target_pos_b
    )
    
    # Distance to target
    pos_error = torch.norm(target_pos_w - object.data.root_pos_w, dim=1)
    
    # Object velocity
    vel_norm = torch.norm(object.data.root_lin_vel_w, dim=1)

    # Position Reward (get to the spot)
    rew_pos = 1 - torch.tanh(pos_error / std)
    
    # Velocity Reward (be slow)
    # We want velocity to be 0, but we prioritize this only when close to target
    # If pos_error is large, rew_pos is small, so the velocity penalty matters less
    rew_vel = 1 - torch.tanh(vel_norm / 0.5) # 0.5 is velocity sensitivity

    # If we are far away, we care mostly about position.
    # If we are close, we care about position AND being slow.
    return rew_pos * rew_vel

# def object_drop_goal_distance(
#     env: ManagerBasedRLEnv,
#     std: float,
#     command_name: str,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
# ) -> torch.Tensor:
#     """
#     Reward the agent for tracking the goal pose (drop pose) in 3D space.
#     Unlike object_goal_distance, this DOES NOT require the object to be lifted.
#     """
#     # extract the used quantities (to enable type-hinting)
#     robot: RigidObject = env.scene[robot_cfg.name]
#     object: RigidObject = env.scene[object_cfg.name]
#     command = env.command_manager.get_command(command_name)
    
#     # compute the target position in the world frame
#     # command is (num_envs, 7) [pos, quat] or (num_envs, 3) [pos]
#     target_pos_b = command[:, :3] 
    
#     # transform command from target_pos from base frame to world frame
#     target_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, target_pos_b)

#     # distance of the object to the goal: (num_envs,)
#     distance = torch.norm(target_pos_w - object.data.root_pos_w, dim=1)
    
#     # Standard tanh kernel reward
#     return 1 - torch.tanh(distance / std)

# def object_reached_goal(
#     env: ManagerBasedRLEnv,
#     command_name: str,
#     threshold: float,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
# ) -> torch.Tensor:
#     """Reward the agent for successfully reaching the goal position with the object within a threshold."""
#     # extract the used quantities (to enable type-hinting)
#     robot: RigidObject = env.scene[robot_cfg.name]
#     object: RigidObject = env.scene[object_cfg.name]
#     command = env.command_manager.get_command(command_name)
#     # compute the desired position in the world frame
#     des_pos_b = command[:, :3]
#     des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
#     # distance of the object to the goal: (num_envs,)
#     distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
#     # return 1.0 if the object is within threshold of the goal, 0.0 otherwise
#     return torch.where(distance < threshold, torch.ones_like(distance), torch.zeros_like(distance))


# --- STAGED REWARDS FOR PICK-AND-PLACE TASK --- #
def staged_pick_and_place_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.25,
    command_name: str = "drop_pose",
    lift_height: float = 0.2,
    target_xy_threshold: float = 0.05,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    A staged reward that guides the agent through Reach -> Lift -> Move -> Place.
    The later stages are only rewarded if the previous stages are satisfied.
    """
    # 1. Get Assets
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    command = env.command_manager.get_command(command_name)

    # 2. Calculate Transforms
    # Target in World Frame
    target_pos_b = command[:, :3]
    target_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, target_pos_b)
    
    # Object and EE positions
    object_pos_w = object.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]

    # 3. Calculate Distances
    # Distance: EE to Object
    d_ee_obj = torch.norm(object_pos_w - ee_pos_w, dim=1)
    
    # Distance: Object to Target (XY plane only)
    d_obj_target_xy = torch.norm(object_pos_w[:, :2] - target_pos_w[:, :2], dim=1)
    
    # Distance: Object to Target (Z axis / Height)
    # We assume target Z is likely 0 or close to ground for placement
    d_obj_target_z = torch.abs(object_pos_w[:, 2] - target_pos_w[:, 2])

    # 4. Define Criteria (The "Levels")
    # Is the object lifted high enough to transition to transport mode?
    is_lifted = object_pos_w[:, 2] > lift_height
    
    # Is the object physically close to the target XY coordinates?
    is_above_target = d_obj_target_xy < target_xy_threshold

    # -----------------------------------------------------------------------
    # 5. Calculate Staged Rewards
    # -----------------------------------------------------------------------
    
    # STAGE 1: Reaching (Always active, but small weight relative to others)
    # Use a tighter std for reaching to encourage precision grasp
    rew_reach = 1 - torch.tanh(d_ee_obj / std)

    # STAGE 2: Lifting
    # Progressive reward for lifting: reward increases as object approaches lift_height
    # Once lifted, give maximum reward to maintain lifted state
    height_error = torch.abs(lift_height - object_pos_w[:, 2])
    rew_lift = torch.where(
        is_lifted,
        torch.ones_like(height_error),  # Max reward if lifted
        1 - torch.tanh(height_error / std)  # Progressive lifting reward
    )

    # STAGE 3: Transport (XY Plane)
    # This reward is ONLY active (or heavily weighted) if the object is lifted.
    # Otherwise, if we reward this while on ground, the robot will drag it.
    transport_reward = 1 - torch.tanh(d_obj_target_xy / std)
    rew_transport = transport_reward * is_lifted

    # STAGE 4: Placing (Z Axis)
    # Only reward Z-axis placement when object is positioned above target XY
    # Use soft gating to allow gradual transition from transport to placement
    rew_place = 1 - torch.tanh(d_obj_target_z / std)
    
    # Gradual activation of placement reward based on XY proximity
    
    # Allow placement when lifted OR when very close to target (hysteresis)
    can_place = is_lifted & is_above_target
    
    rew_place_final = torch.where(
        can_place,
        rew_place,  # Focus on Z placement if above target
        torch.zeros_like(rew_place)
    )

    # -----------------------------------------------------------------------
    # 6. Composition
    # -----------------------------------------------------------------------
    
    # Balanced weights to create smooth learning progression:
    # - Reach: Foundation skill, always active
    # - Lift: Critical transition, moderate weight
    # - Transport: Equal importance to lifting
    # - Place: Highest reward for task completion
    
    total_reward = (
        1.0 * rew_reach +      # Base reaching skill
        1.5 * rew_lift +       # Lifting encouragement
        1.5 * rew_transport +  # Transport when lifted
        2.0 * rew_place_final  # Completion bonus
    )

    # Clamp reward to prevent numerical instability
    total_reward = torch.clamp(total_reward, min=0, max=10.0)

    return total_reward