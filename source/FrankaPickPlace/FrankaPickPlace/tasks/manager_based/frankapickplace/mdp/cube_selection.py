# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions for selecting which cube to pick in multi-cube pick and place tasks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def select_closest_cube_to_ee(env: ManagerBasedRLEnv, num_cubes: int = 4) -> torch.Tensor:
    """Select the cube closest to the end-effector for each environment.
    
    Args:
        env: The environment
        num_cubes: Number of cubes in the scene
        
    Returns:
        torch.Tensor: Indices of selected cubes (num_envs,)
    """
    ee_frame: FrameTransformer = env.scene["ee_frame"]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]  # (num_envs, 3)
    
    # Collect all cube positions
    cube_positions = []
    for i in range(num_cubes):
        cube_name = f"cube_{i}"
        if cube_name in env.scene.keys():
            cube: RigidObject = env.scene[cube_name]
            cube_positions.append(cube.data.root_pos_w[:, :3].unsqueeze(1))  # (num_envs, 1, 3)
    
    # Stack all positions: (num_envs, num_cubes, 3)
    all_cubes_pos = torch.cat(cube_positions, dim=1)
    
    # Calculate distances from EE to all cubes
    # ee_pos_w: (num_envs, 3) -> (num_envs, 1, 3)
    distances = torch.norm(all_cubes_pos - ee_pos_w.unsqueeze(1), dim=-1)  # (num_envs, num_cubes)
    
    # Select closest cube index for each environment
    closest_idx = torch.argmin(distances, dim=-1)  # (num_envs,)
    
    return closest_idx


def select_random_cube(env: ManagerBasedRLEnv, num_cubes: int = 4) -> torch.Tensor:
    """Randomly select a cube for each environment.
    
    Args:
        env: The environment
        num_cubes: Number of cubes in the scene
        
    Returns:
        torch.Tensor: Randomly selected cube indices (num_envs,)
    """
    return torch.randint(0, num_cubes, (env.num_envs,), device=env.device)


def select_sequential_cube(env: ManagerBasedRLEnv, num_cubes: int = 4) -> torch.Tensor:
    """Select cubes sequentially (cycle through 0, 1, 2, 3, ...).
    
    This function maintains a counter that increments each time a cube is successfully placed.
    
    Args:
        env: The environment
        num_cubes: Number of cubes in the scene
        
    Returns:
        torch.Tensor: Sequentially selected cube indices (num_envs,)
    """
    if not hasattr(env, 'cube_sequence_counter'):
        env.cube_sequence_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    
    return env.cube_sequence_counter % num_cubes


def update_target_cube(env: ManagerBasedRLEnv, selection_mode: str = "closest", num_cubes: int = 4):
    """Update the target cube index based on selection mode.
    
    This should be called during reset events.
    
    Args:
        env: The environment
        selection_mode: One of "closest", "random", or "sequential"
        num_cubes: Number of cubes in the scene
    """
    if selection_mode == "closest":
        target_idx = select_closest_cube_to_ee(env, num_cubes)
    elif selection_mode == "random":
        target_idx = select_random_cube(env, num_cubes)
    elif selection_mode == "sequential":
        target_idx = select_sequential_cube(env, num_cubes)
    else:
        raise ValueError(f"Unknown selection mode: {selection_mode}")
    
    env.target_cube_idx = target_idx
    return target_idx
