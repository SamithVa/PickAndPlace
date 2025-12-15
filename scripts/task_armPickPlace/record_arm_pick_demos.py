#!/usr/bin/env python3
"""
Script to record demonstrations for the custom arm pick and place task.

This script allows manual control via keyboard to demonstrate the pick and place task.
Demonstrations are saved in HDF5 format for imitation learning.

Usage:
    python record_arm_pick_demos.py --num_demos 10 --output datasets/arm_pick_demos.hdf5
"""

import argparse
import h5py
import numpy as np
import os
import sys
import time
from collections import defaultdict

# Add parent directory to path to import custom modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from arm_pick_place_env import ArmPickPlaceEnv
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Record demonstrations for arm pick and place task"
    )
    parser.add_argument(
        "--num_demos",
        type=int,
        default=10,
        help="Number of demonstrations to record (0 for infinite)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./datasets/arm_pick_demos.hdf5",
        help="Output HDF5 file path"
    )
    parser.add_argument(
        "--cube_num",
        type=int,
        default=10,
        help="Number of cubes in the scene"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no GUI)"
    )
    parser.add_argument(
        "--episode_length",
        type=int,
        default=1000,
        help="Maximum episode length"
    )
    return parser.parse_args()


class DemonstrationRecorder:
    """Records demonstrations and saves them to HDF5 format."""
    
    def __init__(self, output_path, max_demos=10):
        """Initialize the recorder.
        
        Args:
            output_path: Path to save HDF5 file
            max_demos: Maximum number of demonstrations to record
        """
        self.output_path = output_path
        self.max_demos = max_demos
        self.demos = []
        self.current_demo = defaultdict(list)
        self.recording = False
        self.demo_count = 0
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    def start_recording(self):
        """Start recording a new demonstration."""
        self.recording = True
        self.current_demo = defaultdict(list)
        print(f"\n[RECORDING DEMO {self.demo_count + 1}] Press 'S' to stop and save, 'D' to discard")
    
    def stop_recording(self, save=True):
        """Stop recording and optionally save the demonstration.
        
        Args:
            save: Whether to save the demonstration
        """
        self.recording = False
        
        if save and len(self.current_demo['observations']) > 0:
            # Convert lists to numpy arrays
            demo = {
                'observations': np.array(self.current_demo['observations']),
                'actions': np.array(self.current_demo['actions']),
                'rewards': np.array(self.current_demo['rewards']),
                'dones': np.array(self.current_demo['dones']),
            }
            self.demos.append(demo)
            self.demo_count += 1
            print(f"✓ Demo {self.demo_count} saved ({len(demo['actions'])} steps)")
        else:
            print("✗ Demo discarded")
        
        self.current_demo = defaultdict(list)
    
    def record_step(self, obs, action, reward, done):
        """Record a single step.
        
        Args:
            obs: Observation
            action: Action taken
            reward: Reward received
            done: Whether episode is done
        """
        if self.recording:
            # Extract observation data (flatten if dict)
            if isinstance(obs, dict):
                obs_flat = np.concatenate([
                    np.array(v).flatten() for v in obs.values()
                ])
            else:
                obs_flat = np.array(obs).flatten()
            
            self.current_demo['observations'].append(obs_flat)
            self.current_demo['actions'].append(np.array(action))
            self.current_demo['rewards'].append(reward)
            self.current_demo['dones'].append(done)
    
    def save_to_hdf5(self):
        """Save all demonstrations to HDF5 file."""
        if len(self.demos) == 0:
            print("No demonstrations to save!")
            return
        
        print(f"\nSaving {len(self.demos)} demonstrations to {self.output_path}...")
        
        with h5py.File(self.output_path, 'w') as f:
            # Create group for each demonstration
            for i, demo in enumerate(self.demos):
                demo_group = f.create_group(f'demo_{i}')
                demo_group.create_dataset('observations', data=demo['observations'])
                demo_group.create_dataset('actions', data=demo['actions'])
                demo_group.create_dataset('rewards', data=demo['rewards'])
                demo_group.create_dataset('dones', data=demo['dones'])
                
                # Add metadata
                demo_group.attrs['num_steps'] = len(demo['actions'])
                demo_group.attrs['total_reward'] = np.sum(demo['rewards'])
            
            # Add global metadata
            f.attrs['num_demos'] = len(self.demos)
            f.attrs['creation_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"✓ Successfully saved {len(self.demos)} demonstrations")


def main():
    """Main function to run demonstration recording."""
    args = parse_arguments()
    
    print("=" * 70)
    print("ARM PICK AND PLACE DEMONSTRATION RECORDER")
    print("=" * 70)
    print(f"Recording up to {args.num_demos if args.num_demos > 0 else '∞'} demonstrations")
    print(f"Output file: {args.output_path}")
    print("\nControls:")
    print("  R - Start recording new demonstration")
    print("  S - Stop and save current demonstration")
    print("  D - Discard current demonstration")
    print("  Q - Quit (will save all recorded demos)")
    print("  Use arrow keys/WASD for robot control")
    print("=" * 70)
    
    # Create environment
    print("\nInitializing environment...")
    env = ArmPickPlaceEnv(
        headless=args.headless,
        cube_num=args.cube_num,
        render_mode="human" if not args.headless else None
    )
    
    # Create keyboard interface
    keyboard = Se3Keyboard(Se3KeyboardCfg(
        pos_sensitivity=0.2,
        rot_sensitivity=0.5
    ))
    
    # Create recorder
    recorder = DemonstrationRecorder(args.output, max_demos=args.num_demos)
    
    # Reset environment
    obs, info = env.reset()
    keyboard.reset()
    
    print("\n✓ Environment ready! Press 'R' to start recording first demo\n")
    
    try:
        step_count = 0
        episode_step = 0
        
        while True:
            # Get keyboard input
            delta_action = keyboard.advance()
            
            # Convert Se3 action to joint action
            # Note: You may need to adjust this based on your controller
            action = delta_action.cpu().numpy() if hasattr(delta_action, 'cpu') else delta_action
            
            # Check for control keys
            # Note: You'll need to add keyboard callbacks for R, S, D, Q
            # This is simplified - in practice, use keyboard.add_callback()
            
            # Step environment if recording
            if recorder.recording:
                obs, reward, terminated, truncated, info = env.step(action)
                recorder.record_step(obs, action, reward, terminated or truncated)
                
                episode_step += 1
                
                # Print progress
                if episode_step % 50 == 0:
                    print(f"  Recording... Step {episode_step}/{args.episode_length}")
                
                # Auto-stop if episode ends
                if terminated or truncated or episode_step >= args.episode_length:
                    recorder.stop_recording(save=True)
                    obs, info = env.reset()
                    episode_step = 0
                    
                    # Check if we've recorded enough demos
                    if args.num_demos > 0 and recorder.demo_count >= args.num_demos:
                        print(f"\n✓ Recorded target number of demonstrations ({args.num_demos})")
                        break
            else:
                # Just render when not recording
                env.render()
                time.sleep(0.01)
            
            step_count += 1
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        # Save all recorded demonstrations
        recorder.save_to_hdf5()
        
        # Clean up
        env.close()
        print("\n✓ Recording session completed")


if __name__ == "__main__":
    main()
