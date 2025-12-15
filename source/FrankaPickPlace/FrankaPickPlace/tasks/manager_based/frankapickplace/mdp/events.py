# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event functions for multi-cube pick and place tasks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_all_cubes_to_random_positions(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    num_cubes: int = 4,
    pose_range: dict = None,
    selection_mode: str = "closest",
):
    """Reset all cubes to random positions and update target cube selection.
    
    Args:
        env: The environment
        env_ids: Environment indices to reset
        num_cubes: Number of cubes in the scene
        pose_range: Dictionary with position ranges {"x": (min, max), "y": (min, max), "z": (min, max)}
        selection_mode: How to select target cube ("closest", "random", or "sequential")
    """
    if pose_range is None:
        pose_range = {"x": (0.2, 0.6), "y": (-0.3, 0.3), "z": (0.055, 0.055)}
    
    # Reset each cube to a random position
    for i in range(num_cubes):
        cube_name = f"cube_{i}"
        if cube_name in env.scene.keys():
            cube: RigidObject = env.scene[cube_name]
            
            # Generate random positions for this cube
            num_resets = len(env_ids)
            x_positions = torch.rand(num_resets, device=env.device) * (pose_range["x"][1] - pose_range["x"][0]) + pose_range["x"][0]
            y_positions = torch.rand(num_resets, device=env.device) * (pose_range["y"][1] - pose_range["y"][0]) + pose_range["y"][0]
            z_positions = torch.rand(num_resets, device=env.device) * (pose_range["z"][1] - pose_range["z"][0]) + pose_range["z"][0]
            
            # Set new positions
            cube.data.root_pos_w[env_ids, 0] = x_positions
            cube.data.root_pos_w[env_ids, 1] = y_positions
            cube.data.root_pos_w[env_ids, 2] = z_positions
            
            # Reset velocities
            cube.data.root_lin_vel_w[env_ids] = 0.0
            cube.data.root_ang_vel_w[env_ids] = 0.0
            
            # Write changes to simulation
            cube.write_root_state_to_sim(cube.data.root_state_w[env_ids], env_ids)
    
    # Update target cube selection after resetting positions
    from .cube_selection import update_target_cube
    update_target_cube(env, selection_mode=selection_mode, num_cubes=num_cubes)


def update_target_cube_on_success(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    selection_mode: str = "sequential",
    num_cubes: int = 4,
):
    """Update target cube when current cube is successfully placed.
    
    This is typically called when termination condition is met (cube reached goal).
    
    Args:
        env: The environment
        env_ids: Environment indices where task was completed
        selection_mode: How to select next target cube
        num_cubes: Number of cubes in the scene
    """
    if selection_mode == "sequential":
        # Increment counter for sequential selection
        if hasattr(env, 'cube_sequence_counter'):
            env.cube_sequence_counter[env_ids] += 1
    
    # Update target cube for these environments
    from .cube_selection import update_target_cube
    update_target_cube(env, selection_mode=selection_mode, num_cubes=num_cubes)
