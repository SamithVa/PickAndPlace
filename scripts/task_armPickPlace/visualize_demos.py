#!/usr/bin/env python3
"""
Visualize and analyze recorded demonstrations.

This script loads and displays information about recorded demonstrations,
helping you verify the data quality before training.
"""

import h5py
import numpy as np
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze recorded demonstrations")
    parser.add_argument("dataset", type=str, help="Path to HDF5 dataset file")
    parser.add_argument("--demo_id", type=int, default=None, help="Show details for specific demo")
    parser.add_argument("--stats", action="store_true", help="Show detailed statistics")
    return parser.parse_args()


def print_dataset_summary(filepath):
    """Print summary of the entire dataset."""
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return False
    
    print("=" * 70)
    print(f"DATASET SUMMARY: {filepath}")
    print("=" * 70)
    
    with h5py.File(filepath, 'r') as f:
        # Global info
        num_demos = f.attrs.get('num_demos', len(f.keys()))
        print(f"\n📊 Total Demonstrations: {num_demos}")
        
        if 'creation_time' in f.attrs:
            print(f"🕐 Created: {f.attrs['creation_time']}")
        
        if 'cube_num' in f.attrs:
            print(f"🎲 Cubes: {f.attrs['cube_num']}")
        
        print("\n" + "-" * 70)
        print("Individual Demonstrations:")
        print("-" * 70)
        
        total_steps = 0
        successful_demos = 0
        
        for i in range(num_demos):
            demo_key = f'demo_{i}'
            if demo_key not in f:
                continue
            
            demo = f[demo_key]
            num_steps = demo.attrs.get('num_steps', len(demo['actions']))
            success = demo.attrs.get('success', 'Unknown')
            
            total_steps += num_steps
            if success == True or success == 'True':
                successful_demos += 1
                status = "✓"
            elif success == False or success == 'False':
                status = "✗"
            else:
                status = "?"
            
            # Get shapes
            obs_shape = demo['observations'].shape if 'observations' in demo else 'N/A'
            action_shape = demo['actions'].shape if 'actions' in demo else 'N/A'
            
            print(f"{status} Demo {i:3d}: {num_steps:4d} steps | "
                  f"Obs: {str(obs_shape):20s} | Actions: {str(action_shape):20s}")
        
        print("-" * 70)
        print(f"\n📈 Statistics:")
        print(f"  Successful: {successful_demos}/{num_demos} ({100*successful_demos/num_demos:.1f}%)")
        print(f"  Total steps: {total_steps}")
        print(f"  Avg steps/demo: {total_steps/num_demos:.1f}")
        print("=" * 70)
    
    return True


def print_demo_details(filepath, demo_id):
    """Print detailed information for a specific demonstration."""
    with h5py.File(filepath, 'r') as f:
        demo_key = f'demo_{demo_id}'
        
        if demo_key not in f:
            print(f"Error: Demo {demo_id} not found in dataset")
            return
        
        demo = f[demo_key]
        
        print("=" * 70)
        print(f"DEMO {demo_id} DETAILS")
        print("=" * 70)
        
        # Attributes
        print("\n📋 Attributes:")
        for attr_name, attr_value in demo.attrs.items():
            print(f"  {attr_name}: {attr_value}")
        
        # Datasets
        print("\n📊 Data Arrays:")
        for key in demo.keys():
            data = demo[key]
            print(f"\n  {key}:")
            print(f"    Shape: {data.shape}")
            print(f"    Dtype: {data.dtype}")
            
            # Statistics for numeric data
            if np.issubdtype(data.dtype, np.number):
                print(f"    Min:   {np.min(data):.4f}")
                print(f"    Max:   {np.max(data):.4f}")
                print(f"    Mean:  {np.mean(data):.4f}")
                print(f"    Std:   {np.std(data):.4f}")
                
                # Show first few values
                print(f"    First 3 steps:")
                for i in range(min(3, len(data))):
                    print(f"      [{i}]: {data[i][:5]}..." if len(data[i]) > 5 else f"      [{i}]: {data[i]}")
        
        print("=" * 70)


def print_statistics(filepath):
    """Print detailed statistics across all demonstrations."""
    with h5py.File(filepath, 'r') as f:
        num_demos = f.attrs.get('num_demos', len(f.keys()))
        
        print("=" * 70)
        print("DETAILED STATISTICS")
        print("=" * 70)
        
        # Collect all data
        all_obs = []
        all_actions = []
        all_rewards = []
        episode_lengths = []
        
        for i in range(num_demos):
            demo_key = f'demo_{i}'
            if demo_key not in f:
                continue
            
            demo = f[demo_key]
            
            if 'observations' in demo:
                all_obs.append(demo['observations'][:])
            if 'actions' in demo:
                all_actions.append(demo['actions'][:])
            if 'rewards' in demo:
                all_rewards.append(demo['rewards'][:])
            
            episode_lengths.append(len(demo['actions']))
        
        # Observations statistics
        if all_obs:
            all_obs_concat = np.concatenate(all_obs, axis=0)
            print("\n🔍 Observation Statistics:")
            print(f"  Total timesteps: {len(all_obs_concat)}")
            print(f"  Dimension: {all_obs_concat.shape[1]}")
            print(f"  Range: [{np.min(all_obs_concat):.4f}, {np.max(all_obs_concat):.4f}]")
            print(f"  Mean: {np.mean(all_obs_concat):.4f}")
            print(f"  Std:  {np.std(all_obs_concat):.4f}")
        
        # Actions statistics
        if all_actions:
            all_actions_concat = np.concatenate(all_actions, axis=0)
            print("\n🎮 Action Statistics:")
            print(f"  Total timesteps: {len(all_actions_concat)}")
            print(f"  Dimension: {all_actions_concat.shape[1]}")
            print(f"  Range: [{np.min(all_actions_concat):.4f}, {np.max(all_actions_concat):.4f}]")
            print(f"  Mean: {np.mean(all_actions_concat):.4f}")
            print(f"  Std:  {np.std(all_actions_concat):.4f}")
        
        # Episode length statistics
        if episode_lengths:
            print("\n📏 Episode Length Statistics:")
            print(f"  Min:    {np.min(episode_lengths)} steps")
            print(f"  Max:    {np.max(episode_lengths)} steps")
            print(f"  Mean:   {np.mean(episode_lengths):.1f} steps")
            print(f"  Median: {np.median(episode_lengths):.1f} steps")
            print(f"  Std:    {np.std(episode_lengths):.1f} steps")
        
        # Rewards statistics (if available)
        if all_rewards:
            all_rewards_concat = np.concatenate(all_rewards, axis=0)
            print("\n🏆 Reward Statistics:")
            print(f"  Total reward: {np.sum(all_rewards_concat):.2f}")
            print(f"  Mean/step:    {np.mean(all_rewards_concat):.4f}")
            print(f"  Mean/episode: {np.sum(all_rewards_concat)/len(all_rewards):.4f}")
        
        print("=" * 70)


def main():
    args = parse_args()
    
    # Check file exists
    if not os.path.exists(args.dataset):
        print(f"❌ Error: Dataset file not found: {args.dataset}")
        print("\nMake sure you've recorded demonstrations first:")
        print("  python record_autonomous_demos.py --num_demos 10 --output datasets/demos.hdf5")
        return
    
    # Print summary
    if not print_dataset_summary(args.dataset):
        return
    
    # Print specific demo details if requested
    if args.demo_id is not None:
        print("\n")
        print_demo_details(args.dataset, args.demo_id)
    
    # Print detailed statistics if requested
    if args.stats:
        print("\n")
        print_statistics(args.dataset)


if __name__ == "__main__":
    main()
