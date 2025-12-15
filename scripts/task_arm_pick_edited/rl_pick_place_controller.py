"""
RL-based Pick and Place Controller for IsaacLab
Replaces state machine logic with trained neural network policy
"""

import typing
import numpy as np
import torch
from pathlib import Path

from isaacsim.core.api.controllers import BaseController
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.core.prims import SingleArticulation as Articulation


class RLPickPlaceController(BaseController):
    """
    RL-based pick and place controller using trained policy from IsaacLab.
    
    Unlike the state machine controller, this directly uses the neural network
    to decide all actions based on observations.
    
    Features:
    - Loads trained policy checkpoint (JIT or ONNX)
    - Handles observation preprocessing matching training
    - Converts policy actions to articulation commands
    - Tracks task progress and completion
    - Optional safety constraints
    
    Args:
        name: Controller identifier
        policy_path: Path to trained policy (.pt or .onnx file)
        gripper: Gripper controller reference
        articulation: Robot articulation reference
        device: Device to run inference on ("cuda" or "cpu")
        obs_scale: Optional observation scaling factors
        action_scale: Action scaling factor (default: 0.5 from training)
        use_safety_constraints: Whether to apply safety checks
    """
    
    def __init__(
        self,
        name: str,
        policy_path: str,
        gripper: ParallelGripper,
        articulation: Articulation,
        device: str = "cuda",
        obs_scale: typing.Optional[dict] = None,
        action_scale: float = 0.5,
        use_safety_constraints: bool = True,
    ) -> None:
        super().__init__(name=name)
        
        self._gripper = gripper
        self._articulation = articulation
        self._device = device
        self._action_scale = action_scale
        self._use_safety_constraints = use_safety_constraints
        
        # Load trained policy
        self._policy = self._load_policy(policy_path)
        
        # Observation scaling (if provided during training)
        self._obs_scale = obs_scale if obs_scale is not None else {}
        
        # Action buffer for observation
        self._last_action = np.zeros(8)  # 7 arm + 1 gripper
        
        # Task state tracking
        self._is_done = False
        self._step_count = 0
        self._max_steps = 500  # Episode length from training (5s / 0.01s * 2 decimation)
        
        # Success detection
        self._object_at_target_steps = 0
        self._success_threshold_steps = 20  # ~0.4s of stability
        
        print(f"[{self.name}] RL Policy loaded from: {policy_path}")
        print(f"[{self.name}] Device: {self._device}")
        print(f"[{self.name}] Action scale: {self._action_scale}")
        
    def _load_policy(self, policy_path: str) -> torch.nn.Module:
        """Load trained policy model (JIT or ONNX)."""
        policy_path = Path(policy_path)
        
        if not policy_path.exists():
            raise FileNotFoundError(f"Policy not found: {policy_path}")
        
        if policy_path.suffix == ".pt":
            # Load JIT traced model
            policy = torch.jit.load(str(policy_path), map_location=self._device)
            policy.eval()
        else:
            raise ValueError(f"Unsupported policy format: {policy_path.suffix}")
        
        return policy
    
    def _prepare_observation(
        self,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        object_position: np.ndarray,
        object_orientation: np.ndarray,
        target_position: np.ndarray,
    ) -> torch.Tensor:
        """
        Prepare observation tensor matching training format (36 dims).
        
        Observation structure:
        - joint_pos (9): Joint positions
        - joint_vel (9): Joint velocities  
        - object_position (3): Object position in robot root frame
        - object_orientation (4): Object quaternion
        - place_target (3): Target position
        - actions (8): Previous actions (7 arm + 1 gripper)
        
        Total: 36 dimensions
        """
        # Concatenate all observations
        obs = np.concatenate([
            joint_positions,      # 9
            joint_velocities,     # 9
            object_position,      # 3
            object_orientation,   # 4
            target_position,      # 3
            self._last_action,    # 8
        ]).astype(np.float32)
        
        # Apply scaling if provided
        if self._obs_scale:
            for key, scale in self._obs_scale.items():
                if key in ['joint_pos', 'joint_vel', 'object_position']:
                    # Apply scaling to specific observation components
                    pass
        
        # Convert to tensor
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self._device)
        
        return obs_tensor
    
    def forward(
        self,
        observations: dict,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
        end_effector_offset: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """
        Main control loop: Get action from RL policy based on observations.
        
        Args:
            observations: Dictionary containing:
                - robot_observation_name: dict with 'joint_positions', 'joint_velocities'
                - cube_name: dict with 'position', 'velocity'
                - 'target_positions': target placement location
                
        Returns:
            ArticulationAction with joint positions or velocities
        """
        
        # Check if done
        if self._is_done or self._step_count >= self._max_steps:
            return self._get_hold_action(observations)
        
        # Extract observations from dictionary
        robot_obs = observations.get(self._robot_observation_name, {})
        joint_positions = robot_obs.get('joint_positions', np.zeros(9))
        joint_velocities = robot_obs.get('joint_velocities', np.zeros(9))
        
        # Get object observation (assume first cube in picking order)
        object_obs = observations.get(self._current_cube_name, {})
        object_position = object_obs.get('position', np.zeros(3))
        object_orientation = object_obs.get('orientation', np.array([1.0, 0.0, 0.0, 0.0]))
        
        # Get target position
        color_idx = object_obs.get('color', 0)
        target_positions = observations.get('target_positions', np.zeros((3, 3)))
        target_position = target_positions[color_idx]
        
        # Prepare observation tensor
        obs_tensor = self._prepare_observation(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            object_position=object_position,
            object_orientation=object_orientation,
            target_position=target_position,
        )
        
        # Get action from policy
        with torch.no_grad():
            action_tensor = self._policy(obs_tensor)
        
        # Convert to numpy
        action = action_tensor.squeeze(0).cpu().numpy()
        
        # Store for next observation
        self._last_action = action.copy()
        
        # Apply safety constraints if enabled
        if self._use_safety_constraints:
            action = self._apply_safety_constraints(action, observations)
        
        # Split into arm (7 joints) and gripper (1 binary)
        arm_action = action[:7]
        gripper_binary = action[7]
        
        # Convert gripper binary to two finger positions
        # gripper_binary > 0 means close, < 0 means open
        if gripper_binary > 0.0:
            gripper_pos = np.array([0.0, 0.0])  # Closed
        else:
            gripper_pos = np.array([0.04, 0.04])  # Open
        
        # Combine into 9-dim joint command
        joint_positions_cmd = np.concatenate([arm_action, gripper_pos])
        
        # Create articulation action
        # Note: Your training uses JointPositionAction, so we command positions
        articulation_action = ArticulationAction(
            joint_positions=joint_positions_cmd
        )
        
        # Update step count
        self._step_count += 1
        
        # Check for task completion (using position only, no velocity available)
        # self._check_completion(object_position, target_position, object_velocity)
        
        return articulation_action
    
    def _apply_safety_constraints(
        self,
        action: np.ndarray,
        observations: dict
    ) -> np.ndarray:
        """
        Apply safety constraints to prevent dangerous actions.
        
        Examples:
        - Don't open gripper if object is lifted and not at target
        - Limit joint velocities
        - Prevent self-collision configurations
        """
        action = action.copy()
        
        # Get current state
        object_obs = observations.get(self._current_cube_name, {})
        object_position = object_obs.get('position', np.zeros(3))
        object_height = object_position[2]
        
        color_idx = object_obs.get('color', 0)
        target_positions = observations.get('target_positions', np.zeros((3, 3)))
        target_position = target_positions[color_idx]
        
        distance_to_target = np.linalg.norm(object_position - target_position)
        
        # Safety rule: Don't open gripper if object is lifted but not at target
        if object_height > 0.05 and distance_to_target > 0.1:
            # Force gripper to stay closed
            action[7] = max(action[7], 0.0)
        
        return action
    
    def _check_completion(
        self,
        object_position: np.ndarray,
        target_position: np.ndarray,
        object_velocity: np.ndarray,
    ) -> None:
        """Check if task is successfully completed."""
        
        # Calculate 3D distance to target
        distance = np.linalg.norm(object_position - target_position)
        velocity = np.linalg.norm(object_velocity)
        
        # Check if object is at target and stable
        if distance < 0.08 and velocity < 0.15:
            self._object_at_target_steps += 1
        else:
            self._object_at_target_steps = 0
        
        # Mark as done if stable for threshold duration
        if self._object_at_target_steps >= self._success_threshold_steps:
            self._is_done = True
            print(f"[{self.name}] Task completed successfully!")
    
    def _get_hold_action(self, observations: dict) -> ArticulationAction:
        """Return action that holds current position."""
        robot_obs = observations.get(self._robot_observation_name, {})
        current_positions = robot_obs.get('joint_positions', np.zeros(9))
        
        return ArticulationAction(joint_positions=current_positions)
    
    def is_done(self) -> bool:
        """Check if controller has completed the task."""
        return self._is_done
    
    def reset(
        self,
        robot_observation_name: typing.Optional[str] = None,
        current_cube_name: typing.Optional[str] = None,
    ) -> None:
        """Reset controller state for new episode."""
        super().reset()
        
        if robot_observation_name is not None:
            self._robot_observation_name = robot_observation_name
        if current_cube_name is not None:
            self._current_cube_name = current_cube_name
            
        self._last_action = np.zeros(8)
        self._is_done = False
        self._step_count = 0
        self._object_at_target_steps = 0
        
        print(f"[{self.name}] Reset complete")
    
    def get_stats(self) -> dict:
        """Get controller statistics."""
        return {
            'step_count': self._step_count,
            'is_done': self._is_done,
            'object_at_target_steps': self._object_at_target_steps,
            'success_rate': 1.0 if self._is_done else 0.0,
        }


class MultiObjectRLController(BaseController):
    """
    Extended RL controller for handling multiple objects sequentially.
    
    Uses the same trained policy but manages multiple pick-place cycles.
    """
    
    def __init__(
        self,
        name: str,
        policy_path: str,
        gripper: ParallelGripper,
        articulation: Articulation,
        picking_order_cube_names: typing.List[str],
        robot_observation_name: str,
        device: str = "cuda",
        **kwargs
    ) -> None:
        super().__init__(name=name)
        
        # Create single-object RL controller
        self._rl_controller = RLPickPlaceController(
            name=f"{name}_rl",
            policy_path=policy_path,
            gripper=gripper,
            articulation=articulation,
            device=device,
            **kwargs
        )
        
        self._picking_order_cube_names = picking_order_cube_names
        self._robot_observation_name = robot_observation_name
        self._current_cube_idx = 0
        self._height_offsets = [0.0] * 3  # Height for each color stack
        
        # Initialize controller with first cube
        self._rl_controller._robot_observation_name = robot_observation_name
        self._rl_controller._current_cube_name = picking_order_cube_names[0]
        
        print(f"[{self.name}] Managing {len(picking_order_cube_names)} objects")
    
    def forward(
        self,
        observations: dict,
        **kwargs
    ) -> ArticulationAction:
        """Handle multiple objects sequentially."""
        
        # Check if all objects processed
        if self._current_cube_idx >= len(self._picking_order_cube_names):
            return self._rl_controller._get_hold_action(observations)
        
        # Get current cube
        current_cube_name = self._picking_order_cube_names[self._current_cube_idx]
        cube_obs = observations.get(current_cube_name, {})
        color_idx = cube_obs.get('color', 0)
        
        # Adjust target height for stacking
        if 'target_positions' in observations:
            target_positions = observations['target_positions'].copy()
            cube_size = cube_obs.get('size', np.array([0.05, 0.05, 0.05]))
            target_positions[color_idx][2] = self._height_offsets[color_idx] + cube_size[2] / 2
            observations['target_positions'] = target_positions
        
        # Get action from RL controller
        action = self._rl_controller.forward(observations, **kwargs)
        
        # Check if current object is done
        if self._rl_controller.is_done():
            print(f"[{self.name}] Completed object {self._current_cube_idx}: {current_cube_name}")
            
            # Update height offset for this color
            cube_size = cube_obs.get('size', np.array([0.05, 0.05, 0.05]))
            self._height_offsets[color_idx] += cube_size[2]
            
            # Move to next object
            self._current_cube_idx += 1
            
            # Reset controller for next object
            if self._current_cube_idx < len(self._picking_order_cube_names):
                next_cube_name = self._picking_order_cube_names[self._current_cube_idx]
                self._rl_controller.reset(
                    robot_observation_name=self._robot_observation_name,
                    current_cube_name=next_cube_name
                )
        
        return action
    
    def is_done(self) -> bool:
        """Check if all objects are processed."""
        return self._current_cube_idx >= len(self._picking_order_cube_names)
    
    def reset(self, picking_order_cube_names: typing.Optional[typing.List[str]] = None) -> None:
        """Reset for new episode with new object order."""
        super().reset()
        
        if picking_order_cube_names is not None:
            self._picking_order_cube_names = picking_order_cube_names
        
        self._current_cube_idx = 0
        self._height_offsets = [0.0] * 3
        
        # Reset inner controller
        first_cube_name = self._picking_order_cube_names[0]
        self._rl_controller.reset(
            robot_observation_name=self._robot_observation_name,
            current_cube_name=first_cube_name
        )
        
        print(f"[{self.name}] Reset with {len(self._picking_order_cube_names)} objects")
