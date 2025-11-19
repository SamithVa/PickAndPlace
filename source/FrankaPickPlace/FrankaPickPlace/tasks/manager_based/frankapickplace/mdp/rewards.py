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

# def object_is_lifted(
#     env: ManagerBasedRLEnv, 
#     minimal_height: float, 
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object")
# ) -> torch.Tensor:
#     """Reward the agent for lifting the object above the minimal height."""
#     object: RigidObject = env.scene[object_cfg.name]
#     return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)

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

### --- New reward functions for Franka Pick and Place --- ###

def object_ee_distance_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    Penalize distance between end-effector and object.
    We use a penalty (negative cost) rather than a reward for reaching to encourage
    optimality (shortest path).
    """
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    
    # Target object position: (num_envs, 3), absolute world position
    object_pos = object.data.root_pos_w
    # End-effector position: (num_envs, 3), absolute world position
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    
    distance = torch.norm(ee_w - object_pos, dim=1)
    return distance # 

def object_drag_penalty(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    limit_vel: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """
    Penalize the agent if the object is moving fast in XY plane while 
    being close to the ground (dragging/sliding).
    """
    object: RigidObject = env.scene[object_cfg.name]
    
    # 1. Object Velocity in XY
    vel_xy = torch.norm(object.data.root_lin_vel_w[:, :2], dim=1)
    
    # 2. Object Height
    height = object.data.root_pos_w[:, 2]
    
    # 3. Define "Dragging"
    # It is dragging if: Height is LOW (< minimal) AND Velocity is HIGH (> limit)
    is_dragging = (height < minimal_height) & (vel_xy > limit_vel)
    
    # Return 1.0 if dragging (we will multiply by a negative weight in config)
    return is_dragging.float()

def object_is_lifted(
    env: ManagerBasedRLEnv, 
    minimal_height: float, 
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    
    # Retrieve the object's Z-position
    object_z = object.data.root_pos_w[:, 2]
    
    # 1. Dense Reward Component:
    # Use tanh to give stronger gradient at the bottom (initial lift).
    # We want tanh(z / std) to be ~1.0 when z == minimal_height.
    # tanh(3.0) ~ 0.995. So std = minimal_height / 3.0
    std = minimal_height / 3.0
    dense_reward = torch.tanh(object_z / std)
    
    # 2. Sparse Reward Component:
    # Bonus for meeting the strict requirement.
    sparse_reward = (object_z > minimal_height).float()
    
    # Combine them. 
    return 0.5 * dense_reward + 0.5 * sparse_reward

from isaaclab.utils.math import quat_error_magnitude, quat_mul
def object_orientation_grasp_reward(
    env: ManagerBasedRLEnv,
    std: float,
    distance_threshold: float = 0.10, # Gate: Only active within 10cm
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    Reward for aligning gripper Z (down) with object Z (up), 
    BUT weighted by how close the robot is to the object.
    """
    ee_frame = env.scene[ee_frame_cfg.name]
    object = env.scene[object_cfg.name]

    # --- 1. Calculate Orientation Error (Same as before) ---
    ee_quat = ee_frame.data.target_quat_w[..., 0, :]
    obj_quat = object.data.root_quat_w
    
    # 180-degree flip for top-down grasp
    num_envs = env.num_envs
    flip_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=env.device).repeat(num_envs, 1)
    target_grasp_quat = quat_mul(obj_quat, flip_quat)
    
    dot_prod = torch.sum(ee_quat * target_grasp_quat, dim=1)
    rot_error = 1.0 - torch.square(dot_prod)
    
    rot_reward = 1.0 - torch.tanh(rot_error / std)

    # --- 2. Calculate Distance (The Gate) ---
    # We need to know how close the EE is to the Object
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    obj_pos = object.data.root_pos_w
    
    dist = torch.norm(ee_pos - obj_pos, dim=1)

    # --- 3. Create the Scaling Factor ---
    # If dist is 0 (touching), scale is 1.0 --> Full Orientation Reward
    # If dist is large, scale is 0.0 --> No Orientation Reward
    
    # We use a slightly wider std for the gate so it activates smoothly as we approach
    gate_std = distance_threshold 
    dist_scale = 1.0 - torch.tanh(dist / gate_std)

    # --- 4. Combine ---
    return rot_reward * dist_scale

def object_transport_reward(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    minimal_height: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """
    Reward moving the object to the target, BUT ONLY IF the object is lifted.
    This prevents the robot from just pushing the cube on the table.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # 1. Check if lifted (Gating Condition)
    is_lifted = object.data.root_pos_w[:, 2] > minimal_height

    # 2. Calculate Distance to Target
    target_pos_b = command[:, :3]
    target_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, target_pos_b)
    
    # We focus on XY distance first for transport
    dist_xy = torch.norm(target_pos_w[:, :2] - object.data.root_pos_w[:, :2], dim=1)
    
    # 3. Calculate Reward
    rew = 1.0 - torch.tanh(dist_xy / std)
    
    # 4. Apply Gate: If not lifted, reward is 0.0
    return torch.where(is_lifted, rew, torch.zeros_like(rew))

def object_placement_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """
    A sparse 'bonus' reward for successfully placing the object within a threshold.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    target_pos_b = command[:, :3]
    target_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, target_pos_b)
    
    distance = torch.norm(target_pos_w - object.data.root_pos_w, dim=1)
    
    # Return 1.0 if close, else 0.0
    return (distance < threshold).float()

def object_placed_retreat_reward(
    env: ManagerBasedRLEnv,
    target_threshold: float,
    retreat_dist_threshold: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    Reward the hand for moving AWAY from the object, 
    but only if the object remains at the target.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    command = env.command_manager.get_command(command_name)

    # 1. Check if object is at target
    target_pos_b = command[:, :3]
    target_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, target_pos_b)
    dist_obj_target = torch.norm(target_pos_w - object.data.root_pos_w, dim=1)
    
    is_at_target = dist_obj_target < target_threshold

    # 2. Calculate distance between EE and Object
    # We want this to be LARGE (retreating)
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    dist_ee_obj = torch.norm(ee_pos_w - object.data.root_pos_w, dim=1)

    # 3. Create the Reward
    # We use tanh to cap the reward. 
    # As dist_ee_obj increases (robot moves away), reward goes to 1.0
    retreat_reward = torch.tanh(dist_ee_obj / retreat_dist_threshold)

    # 4. Gating
    # If the object leaves the target (because the robot dragged it away),
    # is_at_target becomes False, and reward drops to 0.
    return torch.where(is_at_target, retreat_reward, torch.zeros_like(retreat_reward))