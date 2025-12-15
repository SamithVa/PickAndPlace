#!/usr/bin/env python3
"""
Record autonomous demonstrations using the existing ArmPickController.

This script runs the pick and place task autonomously using your controller
and records the state-action trajectories for imitation learning.
"""

from isaacsim import SimulationApp
import numpy as np
import h5py
import os
import argparse
from collections import defaultdict
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Record autonomous demonstrations")
    parser.add_argument("--num_demos", type=int, default=10, help="Number of demonstrations to record")
    parser.add_argument("--cube_num", type=int, default=6, help="Number of cubes in scene")
    parser.add_argument("--output", type=str, default="./datasets/auto_demos.hdf5", help="Output file path")
    parser.add_argument("--max_steps", type=int, default=2000, help="Maximum steps per episode")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    return parser.parse_args()

args = parse_args()

# Create output directory
os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

# Start Isaac Sim
simulation_app = SimulationApp({"headless": args.headless, "livestream": 2})

from isaacsim.core.api import World
from class_taskEnv import taskEnv_SceneSetup
from class_controller import ArmPickController

# Storage for demonstrations
demonstrations = []
current_demo = defaultdict(list)

def save_demonstrations(demos, filepath):
    """Save demonstrations to HDF5 file."""
    print(f"\nSaving {len(demos)} demonstrations to {filepath}...")
    
    with h5py.File(filepath, 'w') as f:
        for i, demo in enumerate(demos):
            demo_group = f.create_group(f'demo_{i}')
            
            # Save observations and actions
            for key in ['observations', 'actions', 'robot_states']:
                if key in demo and len(demo[key]) > 0:
                    # Convert to numpy array with explicit dtype
                    try:
                        data = np.array(demo[key], dtype=np.float32)
                        demo_group.create_dataset(key, data=data)
                    except (ValueError, TypeError) as e:
                        print(f"Warning: Could not save {key} for demo {i}: {e}")
                        # Try to convert each element individually
                        converted_data = []
                        for item in demo[key]:
                            if hasattr(item, 'flatten'):
                                converted_data.append(np.array(item).flatten().astype(np.float32))
                            else:
                                converted_data.append(np.array(item, dtype=np.float32).flatten())
                        data = np.array(converted_data, dtype=np.float32)
                        demo_group.create_dataset(key, data=data)
            
            # Add metadata
            demo_group.attrs['num_steps'] = len(demo['actions'])
            demo_group.attrs['success'] = bool(demo.get('success', False))
        
        # Global metadata
        f.attrs['num_demos'] = len(demos)
        f.attrs['creation_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        f.attrs['cube_num'] = args.cube_num
    
    print(f"✓ Successfully saved to {filepath}")

# Initialize world and scene
my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()

# Create task
my_task = taskEnv_SceneSetup(name="env_armPick", cube_num=args.cube_num)
my_world.add_task(my_task)

# Initialize scene
my_world.reset()

# Get robot
task_params = my_task.get_params()
robot_name = task_params["robot_name"]["value"]
my_franka = my_world.scene.get_object(robot_name)

# Create controller
my_controller = ArmPickController(
    name="stacking_controller",
    gripper=my_franka.gripper,
    articulation=my_franka,
    picking_order_cube_names=my_task.get_cube_names(),
    robot_observation_name=robot_name,
)

articulation_controller = my_franka.get_articulation_controller()

print("=" * 70)
print(f"RECORDING {args.num_demos} AUTONOMOUS DEMONSTRATIONS")
print("=" * 70)
print(f"Cubes: {args.cube_num}")
print(f"Max steps per demo: {args.max_steps}")
print(f"Output: {args.output}")
print("=" * 70)

# Main recording loop
demo_count = 0
step_count = 0
reset_needed = False
recording = True

# Start the simulation
print("\nStarting simulation and recording demo 1...")
my_world.play()

try:
    while simulation_app.is_running() and demo_count < args.num_demos:
        my_world.step(render=True)
        
        if my_world.is_stopped() and not reset_needed:
            reset_needed = True
            
        if my_world.is_playing():
            if reset_needed:
                # Save current demonstration if it has data
                if len(current_demo['actions']) > 0:
                    # Check if task was successful (you can add your own success criteria)
                    metrics = my_task.calculate_metrics() if hasattr(my_task, 'calculate_metrics') else {}
                    current_demo['success'] = True  # Modify based on your success criteria
                    
                    demonstrations.append(dict(current_demo))
                    demo_count += 1
                    
                    print(f"\n✓ Demo {demo_count}/{args.num_demos} recorded ({len(current_demo['actions'])} steps)")
                    if metrics:
                        print(f"  Metrics: {metrics}")
                
                # Reset for next demo
                if demo_count < args.num_demos:
                    current_demo = defaultdict(list)
                    my_world.reset()
                    # Reset controller without arguments
                    try:
                        my_controller.reset()
                    except TypeError:
                        # If reset() doesn't work, recreate the controller
                        my_controller = ArmPickController(
                            name="stacking_controller",
                            gripper=my_franka.gripper,
                            articulation=my_franka,
                            picking_order_cube_names=my_task.get_cube_names(),
                            robot_observation_name=robot_name,
                        )
                    step_count = 0
                    reset_needed = False
                    print(f"\nStarting demo {demo_count + 1}...")
                else:
                    break
            
            # Get observations
            observations = my_task.get_observations()
            
            # Compute actions from controller
            actions = my_controller.forward(observations=observations)
            
            # Record step
            if recording:
                # Store observations (flatten if dict)
                if isinstance(observations, dict):
                    obs_flat = []
                    for key, value in observations.items():
                        if isinstance(value, (np.ndarray, list)):
                            obs_flat.extend(np.array(value, dtype=np.float32).flatten().tolist())
                    current_demo['observations'].append(np.array(obs_flat, dtype=np.float32))
                else:
                    current_demo['observations'].append(np.array(observations, dtype=np.float32).flatten())
                
                # Store actions
                if hasattr(actions, 'cpu'):
                    actions_np = actions.cpu().numpy().astype(np.float32)
                else:
                    actions_np = np.array(actions, dtype=np.float32)
                current_demo['actions'].append(actions_np.flatten())
                
                # Store robot state
                robot_state = my_franka.get_joint_positions()
                if hasattr(robot_state, 'cpu'):
                    robot_state = robot_state.cpu().numpy().astype(np.float32)
                else:
                    robot_state = np.array(robot_state, dtype=np.float32)
                current_demo['robot_states'].append(robot_state.flatten())
            
            # Apply actions to robot
            articulation_controller.apply_action(actions)
            
            step_count += 1
            
            # Progress indicator
            if step_count % 100 == 0:
                print(f"  Step {step_count}/{args.max_steps}", end='\r')
            
            # Check for maximum steps
            if step_count >= args.max_steps:
                reset_needed = True

except KeyboardInterrupt:
    print("\n\nInterrupted by user")

finally:
    # Save any remaining demonstration
    if len(current_demo['actions']) > 0:
        current_demo['success'] = False  # Incomplete demo
        demonstrations.append(dict(current_demo))
        demo_count += 1
        print(f"\n✓ Saved incomplete demo {demo_count}")
    
    # Save all demonstrations to file
    if len(demonstrations) > 0:
        save_demonstrations(demonstrations, args.output)
        print(f"\n{'=' * 70}")
        print(f"✓ Recording complete: {len(demonstrations)} demonstrations saved")
        print(f"{'=' * 70}")
    else:
        print("\n✗ No demonstrations recorded")
    
    simulation_app.close()
