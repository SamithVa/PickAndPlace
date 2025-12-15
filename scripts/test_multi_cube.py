#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to test the multi-cube pick and place environment.

Usage:
    # Run with visualization
    python scripts/test_multi_cube.py --num_envs 4 --enable_cameras
    
    # Test different selection modes
    python scripts/test_multi_cube.py --selection_mode random
    python scripts/test_multi_cube.py --selection_mode sequential
    python scripts/test_multi_cube.py --selection_mode closest
"""

import argparse
import torch

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Test multi-cube pick and place environment")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--selection_mode", type=str, default="closest", 
                   choices=["closest", "random", "sequential"],
                   help="Cube selection strategy")
parser.add_argument("--enable_cameras", action="store_true", help="Enable camera rendering")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after app is launched
from FrankaPickPlace.tasks.manager_based.frankapickplace import FrankPickPlaceEnvCfg
from isaaclab.envs import ManagerBasedRLEnv


def main():
    """Test the multi-cube environment with random actions."""
    
    # Create environment configuration
    env_cfg = FrankPickPlaceEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # Update selection mode in event configuration
    env_cfg.events.reset_all_cubes.params["selection_mode"] = args_cli.selection_mode
    
    # Create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    print(f"\n{'='*80}")
    print(f"Multi-Cube Pick and Place Test")
    print(f"{'='*80}")
    print(f"Number of environments: {env.num_envs}")
    print(f"Selection mode: {args_cli.selection_mode}")
    print(f"Number of cubes: 4")
    print(f"Observation space shape: {env.observation_space}")
    print(f"Action space shape: {env.action_space}")
    print(f"{'='*80}\n")
    
    # Reset environment
    obs, _ = env.reset()
    print(f"✓ Environment reset successful")
    print(f"  Observation shape: {obs['policy'].shape}")
    
    # Initialize target cube tracking
    if not hasattr(env, 'target_cube_idx'):
        env.target_cube_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    
    # Print initial target cubes
    print(f"\nInitial target cube indices: {env.target_cube_idx.cpu().numpy()}")
    
    # Simulation loop
    episode_count = 0
    step_count = 0
    success_count = 0
    
    try:
        while simulation_app.is_running() and episode_count < 10:
            # Random actions for testing
            with torch.inference_mode():
                actions = 2 * torch.rand(env.num_envs, env.action_space.shape[0], device=env.device) - 1
            
            # Step the environment
            obs, rewards, terminated, truncated, info = env.step(actions)
            step_count += 1
            
            # Check for episode completion
            dones = terminated | truncated
            if dones.any():
                episode_count += torch.sum(dones).item()
                success_count += torch.sum(terminated).item()
                
                print(f"\nEpisode(s) completed at step {step_count}:")
                print(f"  Environments done: {torch.where(dones)[0].cpu().numpy()}")
                print(f"  Success (terminated): {torch.where(terminated)[0].cpu().numpy()}")
                print(f"  Timeout (truncated): {torch.where(truncated)[0].cpu().numpy()}")
                
                if hasattr(env, 'target_cube_idx'):
                    print(f"  New target cubes: {env.target_cube_idx.cpu().numpy()}")
            
            # Print periodic status
            if step_count % 100 == 0:
                avg_reward = rewards.mean().item()
                print(f"Step {step_count}: Avg reward = {avg_reward:.4f}")
    
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
    
    finally:
        # Print summary
        print(f"\n{'='*80}")
        print(f"Simulation Summary")
        print(f"{'='*80}")
        print(f"Total steps: {step_count}")
        print(f"Episodes completed: {episode_count}")
        print(f"Successful placements: {success_count}")
        if episode_count > 0:
            success_rate = success_count / episode_count * 100
            print(f"Success rate: {success_rate:.2f}%")
        print(f"{'='*80}\n")
        
        # Close environment
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
