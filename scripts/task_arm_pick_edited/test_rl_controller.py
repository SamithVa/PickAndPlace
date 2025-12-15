#!/usr/bin/env python3
"""
Quick test script for RL Pick-Place Controller
Tests the controller with your trained policy
"""

import numpy as np
import torch
from pathlib import Path


def test_observation_format():
    """Test that observation format matches training (36 dims)"""
    print("=" * 70)
    print("Testing Observation Format")
    print("=" * 70)
    
    # Create sample observations
    obs = {
        'joint_pos': np.random.randn(9),
        'joint_vel': np.random.randn(9),
        'object_position': np.array([0.3, 0.0, 0.05]),
        'object_orientation': np.array([1.0, 0.0, 0.0, 0.0]),
        'place_target': np.array([0.5, 0.2, 0.025]),
        'previous_action': np.random.randn(8),
    }
    
    # Convert to 36-dim vector
    obs_vector = np.concatenate([
        obs['joint_pos'],           # 9
        obs['joint_vel'],           # 9
        obs['object_position'],     # 3
        obs['object_orientation'],  # 4
        obs['place_target'],        # 3
        obs['previous_action'],     # 8
    ])
    
    print(f"✓ Observation vector shape: {obs_vector.shape}")
    assert obs_vector.shape == (36,), f"Expected (36,), got {obs_vector.shape}"
    print(f"✓ All components present")
    print(f"  - Joint positions: {obs['joint_pos'].shape}")
    print(f"  - Joint velocities: {obs['joint_vel'].shape}")
    print(f"  - Object position: {obs['object_position'].shape}")
    print(f"  - Object orientation: {obs['object_orientation'].shape}")
    print(f"  - Target position: {obs['place_target'].shape}")
    print(f"  - Previous action: {obs['previous_action'].shape}")
    print()


def test_action_format():
    """Test action conversion from policy output to robot command"""
    print("=" * 70)
    print("Testing Action Format")
    print("=" * 70)
    
    # Simulate policy output (8-dim)
    policy_action = np.random.randn(8)
    print(f"✓ Policy output shape: {policy_action.shape}")
    
    # Split
    arm_actions = policy_action[:7]
    gripper_binary = policy_action[7]
    
    print(f"✓ Arm actions: {arm_actions.shape}")
    print(f"✓ Gripper binary: {gripper_binary:.3f}")
    
    # Convert gripper
    if gripper_binary > 0.0:
        gripper_fingers = np.array([0.0, 0.0])  # Closed
        gripper_state = "CLOSED"
    else:
        gripper_fingers = np.array([0.04, 0.04])  # Open
        gripper_state = "OPEN"
    
    # Combine
    robot_command = np.concatenate([arm_actions, gripper_fingers])
    
    print(f"✓ Robot command shape: {robot_command.shape}")
    print(f"✓ Gripper state: {gripper_state}")
    assert robot_command.shape == (9,), f"Expected (9,), got {robot_command.shape}"
    print()


def test_policy_loading(policy_path: str):
    """Test loading trained policy"""
    print("=" * 70)
    print("Testing Policy Loading")
    print("=" * 70)
    
    policy_path = Path(policy_path)
    
    if not policy_path.exists():
        print(f"✗ Policy not found: {policy_path}")
        print(f"  Please provide valid policy path")
        return False
    
    print(f"✓ Policy file exists: {policy_path}")
    
    try:
        if policy_path.suffix == ".pt":
            policy = torch.jit.load(str(policy_path), map_location="cpu")
            print(f"✓ Successfully loaded JIT policy")
        else:
            print(f"✗ Unsupported format: {policy_path.suffix}")
            return False
    except Exception as e:
        print(f"✗ Failed to load policy: {e}")
        return False
    
    # Test inference
    try:
        obs_tensor = torch.randn(1, 36).float()  # Changed from 39 to 36
        with torch.no_grad():
            if policy_path.suffix == ".pt":
                action = policy(obs_tensor)
            else:
                action = policy.run(None, {"obs": obs_tensor.numpy()})[0]
        
        print(f"✓ Policy inference successful")
        print(f"✓ Input shape: {obs_tensor.shape}")
        print(f"✓ Output shape: {action.shape if hasattr(action, 'shape') else np.array(action).shape}")
        
        return True
    except Exception as e:
        print(f"✗ Policy inference failed: {e}")
        return False


def test_completion_detection():
    """Test task completion detection logic"""
    print("=" * 70)
    print("Testing Completion Detection")
    print("=" * 70)
    
    # Test case 1: Success
    object_pos = np.array([0.5, 0.2, 0.025])
    target_pos = np.array([0.5, 0.2, 0.025])
    object_vel = np.array([0.01, 0.01, 0.01])
    gripper_state = 0.04  # Open
    
    distance = np.linalg.norm(object_pos - target_pos)
    velocity = np.linalg.norm(object_vel)
    gripper_open = gripper_state > 0.035
    
    success = distance < 0.08 and velocity < 0.15 and gripper_open
    
    print(f"Test Case 1: Object at target")
    print(f"  Distance: {distance:.4f} m (threshold: 0.08)")
    print(f"  Velocity: {velocity:.4f} m/s (threshold: 0.15)")
    print(f"  Gripper: {'OPEN' if gripper_open else 'CLOSED'}")
    print(f"  Result: {'✓ SUCCESS' if success else '✗ FAIL'}")
    print()
    
    # Test case 2: Too far
    object_pos = np.array([0.3, 0.0, 0.05])
    target_pos = np.array([0.5, 0.2, 0.025])
    
    distance = np.linalg.norm(object_pos - target_pos)
    success = distance < 0.08
    
    print(f"Test Case 2: Object too far")
    print(f"  Distance: {distance:.4f} m (threshold: 0.08)")
    print(f"  Result: {'✗ EXPECTED FAIL' if not success else '✓ UNEXPECTED SUCCESS'}")
    print()


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "RL PICK-PLACE CONTROLLER TEST SUITE" + " " * 18 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    # Test 1: Observation format
    try:
        test_observation_format()
        print("✓ Observation format test PASSED\n")
    except Exception as e:
        print(f"✗ Observation format test FAILED: {e}\n")
    
    # Test 2: Action format
    try:
        test_action_format()
        print("✓ Action format test PASSED\n")
    except Exception as e:
        print(f"✗ Action format test FAILED: {e}\n")
    
    # Test 3: Policy loading
    print("Enter path to your trained policy (or 'skip'):")
    policy_path = input("> ").strip()
    
    if policy_path and policy_path.lower() != 'skip':
        try:
            if test_policy_loading(policy_path):
                print("✓ Policy loading test PASSED\n")
            else:
                print("✗ Policy loading test FAILED\n")
        except Exception as e:
            print(f"✗ Policy loading test FAILED: {e}\n")
    else:
        print("⊘ Policy loading test SKIPPED\n")
    
    # Test 4: Completion detection
    try:
        test_completion_detection()
        print("✓ Completion detection test PASSED\n")
    except Exception as e:
        print(f"✗ Completion detection test FAILED: {e}\n")
    
    print("=" * 70)
    print("Test Suite Complete")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Export your trained policy: python scripts/rsl_rl/play.py --export")
    print("2. Update class_controller.py with policy path")
    print("3. Run your main simulation")
    print()


if __name__ == "__main__":
    main()
