from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    return object_pos_b


def all_cubes_positions_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    num_cubes: int = 4,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The positions of all cubes in the robot's root frame.
    
    Returns:
        torch.Tensor: Shape (num_envs, num_cubes * 3) - flattened positions of all cubes
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    
    all_positions = []
    for i in range(num_cubes):
        cube_name = f"cube_{i}"
        if cube_name in env.scene.keys():
            cube: RigidObject = env.scene[cube_name]
            cube_pos_w = cube.data.root_pos_w[:, :3]
            cube_pos_b, _ = subtract_frame_transforms(
                robot.data.root_pos_w, 
                robot.data.root_quat_w, 
                cube_pos_w
            )
            all_positions.append(cube_pos_b)
    
    # Concatenate all cube positions: (num_envs, num_cubes * 3)
    return torch.cat(all_positions, dim=-1)


def target_cube_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The position of the currently targeted cube in the robot's root frame.
    
    This function uses the target_cube_idx stored in the environment buffer.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    
    # Get target cube index from environment (stored in extras)
    if not hasattr(env, 'target_cube_idx'):
        # Initialize with cube 0 if not set
        env.target_cube_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    
    target_idx = env.target_cube_idx
    
    # Gather positions of all cubes
    num_cubes = 4
    cube_positions_w = []
    for i in range(num_cubes):
        cube_name = f"cube_{i}"
        if cube_name in env.scene.keys():
            cube: RigidObject = env.scene[cube_name]
            cube_positions_w.append(cube.data.root_pos_w[:, :3].unsqueeze(1))  # (num_envs, 1, 3)
    
    # Stack: (num_envs, num_cubes, 3)
    all_cubes_w = torch.cat(cube_positions_w, dim=1)
    
    # Select target cube for each environment
    target_pos_w = all_cubes_w[torch.arange(env.num_envs, device=env.device), target_idx]
    
    # Transform to robot root frame
    target_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, 
        robot.data.root_quat_w, 
        target_pos_w
    )
    
    return target_pos_b