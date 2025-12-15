# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "target_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    use_target_cube: bool = False,
    num_cubes: int = 4,
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold for the object to reach the goal position. Defaults to 0.02.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").
        use_target_cube: If True, use the target cube from env.target_cube_idx.
        num_cubes: Number of cubes in the scene.

    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    
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
    
    # distance of the object to the goal: (num_envs,)
    distance = torch.norm(des_pos_w - object_pos_w, dim=1)

    # return True if the object is within threshold of the goal
    return distance < threshold
