# arm_pick_place_env.py
"""Gym environment wrapper for the custom arm pick and place task."""

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from isaacsim import SimulationApp
from isaacsim.core.api import World

from class_taskEnv import taskEnv_SceneSetup
from class_controller import ArmPickController


class ArmPickPlaceEnv(gym.Env):
    """Gym wrapper for the custom arm pick and place environment."""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, headless=False, cube_num=10, render_mode="human"):
        """Initialize the environment.
        
        Args:
            headless: Whether to run without GUI
            cube_num: Number of cubes in the scene
            render_mode: Rendering mode for the environment
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.cube_num = cube_num
        
        # Initialize Isaac Sim
        self.simulation_app = SimulationApp({"headless": headless})
        
        # Create world and scene
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
        
        # Create task
        self.task = taskEnv_SceneSetup(name="env_armPick", cube_num=cube_num)
        self.world.add_task(self.task)
        
        # Initialize scene
        self.world.reset()
        
        # Get robot
        task_params = self.task.get_params()
        robot_name = task_params["robot_name"]["value"]
        self.franka = self.world.scene.get_object(robot_name)
        
        # Create controller
        self.controller = ArmPickController(
            name="stacking_controller",
            gripper=self.franka.gripper,
            articulation=self.franka,
            picking_order_cube_names=self.task.get_cube_names(),
            robot_observation_name=robot_name,
        )
        
        # Get articulation controller
        self.articulation_controller = self.franka.get_articulation_controller()
        
        # Define action and observation spaces
        # Actions: 7 DOF joint positions + 1 gripper action
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )
        
        # Observations: will be defined based on task observations
        # For now, use a placeholder
        self.observation_space = spaces.Dict({
            "policy": spaces.Box(
                low=-np.inf, high=np.inf, shape=(50,), dtype=np.float32
            )
        })
        
        self._step_count = 0
        self._episode_length = 1000  # Maximum episode length
        
    def reset(self, seed=None, options=None):
        """Reset the environment.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Reset world and controller
        self.world.reset()
        self.controller.reset()
        self._step_count = 0
        
        # Get initial observations
        observations = self.task.get_observations()
        
        # Convert observations to gym format
        obs = self._convert_observations(observations)
        info = {}
        
        return obs, info
    
    def step(self, action):
        """Take a step in the environment.
        
        Args:
            action: Action to take (8D: 7 joint positions + 1 gripper)
            
        Returns:
            observation: New observation
            reward: Reward for this step
            terminated: Whether episode is terminated
            truncated: Whether episode is truncated
            info: Additional information
        """
        # Apply action to robot
        # Note: You may need to convert the action format
        # to match what your articulation_controller expects
        self.articulation_controller.apply_action(action)
        
        # Step the world
        self.world.step(render=True)
        
        # Get observations
        observations = self.task.get_observations()
        obs = self._convert_observations(observations)
        
        # Calculate reward (you'll need to implement this in your task)
        reward = self._calculate_reward()
        
        # Check termination conditions
        self._step_count += 1
        terminated = self._check_termination()
        truncated = self._step_count >= self._episode_length
        
        # Get info
        info = {
            "step_count": self._step_count,
            "metrics": self.task.calculate_metrics() if hasattr(self.task, 'calculate_metrics') else {}
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            # Isaac Sim handles rendering automatically
            pass
        return None
    
    def close(self):
        """Close the environment."""
        if hasattr(self, 'simulation_app'):
            self.simulation_app.close()
    
    def _convert_observations(self, observations):
        """Convert task observations to gym observation format.
        
        Args:
            observations: Raw observations from task
            
        Returns:
            Gym-formatted observations
        """
        # Convert to numpy/tensor format expected by gym
        # This depends on your observation structure
        if isinstance(observations, dict):
            # Flatten or structure as needed
            obs_dict = {}
            for key, value in observations.items():
                if isinstance(value, (np.ndarray, torch.Tensor)):
                    obs_dict[key] = np.array(value, dtype=np.float32)
            return {"policy": np.concatenate([v.flatten() for v in obs_dict.values()])}
        else:
            return {"policy": np.array(observations, dtype=np.float32)}
    
    def _calculate_reward(self):
        """Calculate reward for current state.
        
        Returns:
            Reward value
        """
        # Implement your reward function here
        # This should match your task objectives
        # For now, return 0
        return 0.0
    
    def _check_termination(self):
        """Check if episode should terminate.
        
        Returns:
            Whether episode is terminated
        """
        # Implement termination conditions
        # e.g., task completion, failure states
        return False


# Register the environment with gymnasium
gym.register(
    id="ArmPickPlace-v0",
    entry_point="arm_pick_place_env:ArmPickPlaceEnv",
    max_episode_steps=1000,
)
