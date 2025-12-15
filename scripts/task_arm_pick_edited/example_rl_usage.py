"""
Example usage of RL-based Pick and Place Controller
Demonstrates how to integrate with your existing IsaacSim setup
"""

import numpy as np
from isaacsim.core.prims import SingleArticulation
from isaacsim.robot.manipulators.grippers import ParallelGripper
from rl_pick_place_controller import RLPickPlaceController, MultiObjectRLController


def example_single_object():
    """Example: Control single pick-place with RL policy"""
    
    # Setup (your existing code)
    robot = SingleArticulation(prim_path="/World/Franka")
    gripper = ParallelGripper(
        end_effector_prim_path="/World/Franka/panda_hand",
        joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
        joint_opened_positions=np.array([0.04, 0.04]),
        joint_closed_positions=np.array([0.0, 0.0]),
    )
    
    # Create RL controller
    controller = RLPickPlaceController(
        name="rl_controller",
        policy_path="/path/to/your/model.pt",  # From IsaacLab training
        gripper=gripper,
        articulation=robot,
        device="cuda",
        action_scale=0.5,  # Match training
        use_safety_constraints=True,
    )
    
    # Initialize for task
    controller._robot_observation_name = "franka"
    controller._current_cube_name = "cube_0"
    
    # Main loop
    for step in range(500):
        # Get observations (your existing observation system)
        observations = {
            'franka': {
                'joint_positions': robot.get_joint_positions(),
                'joint_velocities': robot.get_joint_velocities(),
            },
            'cube_0': {
                'position': np.array([0.3, 0.0, 0.05]),
                'velocity': np.array([0.0, 0.0, 0.0]),
                'color': 0,
            },
            'target_positions': np.array([
                [0.5, 0.2, 0.025],  # Red target
                [0.5, 0.0, 0.025],  # Green target  
                [0.5, -0.2, 0.025], # Blue target
            ])
        }
        
        # Get action from RL policy
        action = controller.forward(observations)
        
        # Apply action
        robot.apply_action(action)
        
        # Step simulation
        # world.step()
        
        # Check completion
        if controller.is_done():
            print(f"Task completed in {step} steps!")
            break
    
    # Get statistics
    stats = controller.get_stats()
    print(f"Stats: {stats}")


def example_multiple_objects():
    """Example: Handle multiple objects with single trained policy"""
    
    # Setup
    robot = SingleArticulation(prim_path="/World/Franka")
    gripper = ParallelGripper(
        end_effector_prim_path="/World/Franka/panda_hand",
        joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
        joint_opened_positions=np.array([0.04, 0.04]),
        joint_closed_positions=np.array([0.0, 0.0]),
    )
    
    # Create multi-object controller
    controller = MultiObjectRLController(
        name="multi_rl_controller",
        policy_path="/path/to/your/model.pt",
        gripper=gripper,
        articulation=robot,
        picking_order_cube_names=["cube_0", "cube_1", "cube_2", "cube_3"],
        robot_observation_name="franka",
        device="cuda",
    )
    
    # Main loop
    for step in range(2000):  # Longer for multiple objects
        # Get observations (example structure)
        observations = get_observations()  # Your observation function
        
        # Get action - controller manages object switching internally
        action = controller.forward(observations)
        
        # Apply and step
        robot.apply_action(action)
        # world.step()
        
        if controller.is_done():
            print(f"All objects completed in {step} steps!")
            break


def get_observations():
    """Example observation structure - replace with your actual code"""
    return {
        'franka': {
            'joint_positions': np.zeros(9),
            'joint_velocities': np.zeros(9),
        },
        'cube_0': {'position': np.array([0.3, 0, 0.05]), 'velocity': np.zeros(3), 'color': 0, 'size': np.array([0.05, 0.05, 0.05])},
        'cube_1': {'position': np.array([0.2, 0.1, 0.05]), 'velocity': np.zeros(3), 'color': 1, 'size': np.array([0.05, 0.05, 0.05])},
        'cube_2': {'position': np.array([0.2, -0.1, 0.05]), 'velocity': np.zeros(3), 'color': 2, 'size': np.array([0.05, 0.05, 0.05])},
        'cube_3': {'position': np.array([0.1, 0, 0.05]), 'velocity': np.zeros(3), 'color': 0, 'size': np.array([0.05, 0.05, 0.05])},
        'target_positions': np.array([
            [0.5, 0.2, 0.025],
            [0.5, 0.0, 0.025],
            [0.5, -0.2, 0.025],
        ])
    }


def integrate_with_your_code():
    """
    How to integrate with your existing class_controller.py
    """
    
    # REPLACE THIS in your ArmPickController.__init__:
    # OLD:
    # self._pick_place_controller = PickPlaceController(...)
    
    # NEW:
    from rl_pick_place_controller import MultiObjectRLController
    
    self._pick_place_controller = MultiObjectRLController(
        name="rl_multi_controller",
        policy_path=policy_path,  # Your trained policy
        gripper=gripper,
        articulation=articulation,
        picking_order_cube_names=picking_order_cube_names,
        robot_observation_name=robot_observation_name,
        device="cuda",
        action_scale=0.5,
    )
    
    # Then in forward(), just call:
    # actions = self._pick_place_controller.forward(observations)
    # That's it! The RL controller handles everything.


if __name__ == "__main__":
    print("=" * 60)
    print("RL Pick-Place Controller Examples")
    print("=" * 60)
    
    print("\n1. Single Object Example:")
    print("   - Loads trained policy")
    print("   - Processes one pick-place cycle")
    print("   - Auto-detects completion")
    
    print("\n2. Multiple Objects Example:")
    print("   - Same policy, multiple objects")
    print("   - Automatic object switching")
    print("   - Height stacking support")
    
    print("\n3. Integration Example:")
    print("   - Drop-in replacement for PickPlaceController")
    print("   - No state machine needed")
    print("   - Uses your trained IsaacLab policy")
    
    print("\n" + "=" * 60)
    print("To run: python example_rl_usage.py")
    print("=" * 60)
