# 在 Isaac Sim 中，对控制器的编写，初始化阶段通常包括定义控制器的基本框架、加载必要的模块，以及设置任务所需的初始参数和对象
import typing
import numpy as np

from isaacsim.core.api.controllers import BaseController
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka.controllers.stacking_controller import StackingController
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import PickPlaceController
from isaacsim.robot.manipulators.controllers import pick_place_controller
from isaacsim.robot.manipulators.controllers import stacking_controller
#from omni.isaac.franka.controllers import FrankaPickPlaceController
import torch 

class ArmPickController(BaseController):
    def __init__(
        self,
        name: str,
        gripper: ParallelGripper,
        articulation: Articulation,
        picking_order_cube_names: typing.List[str],
        robot_observation_name: str,
        policy_path: str = None,
    ) -> None:
        # Use the Franka StackingController as base
        super().__init__(name=name )
        self._pick_place_controller = PickPlaceController(
            name="pick_place_controller",
            gripper=gripper,
            robot_articulation=articulation)
        
        self._picking_order_cube_names = picking_order_cube_names
        self._robot_observation_name = robot_observation_name
        self._current_cube_numth = 0    
        self._current_height = [0.0] * 3  # 对应3种颜色的目标位置高度
        # new add
        self._last_completed_cube_numth = -1  # 跟踪上一个完成的方块
        self._current_cube_start_time = 0  # 当前方块开始处理的时间
        self._max_cube_time = 1000  # 每个方块的最大处理时间（步数）
        self._policy = torch.jit.load(policy_path) if policy_path is not None else None

    def forward(
        self,
        observations: dict,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
        end_effector_offset: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """ 控制器主循环：根据当前观测，决定机械臂的动作，并增加夹爪轨迹控制。"""

        # 1. 检查是否所有方块都已处理完成
        if self._current_cube_numth >= len(self._picking_order_cube_names):
            # 所有方块已完成，输出空动作
            target_joint_positions = [None] * observations[self._robot_observation_name]['joint_positions'].shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)

        # 2. 获取当前方块的颜色索引
        current_cube_name = self._picking_order_cube_names[self._current_cube_numth]
        color_idx = observations[current_cube_name]['color']
        placing_target_postion = observations['target_positions'][color_idx]
        placing_target_postion[2] = self._current_height[color_idx]+ observations[current_cube_name]['size'][2]/2  # 计算放置位置的高度

        # 调用PickPlaceController的forward方法       
        cube_current_position = observations[current_cube_name]['position']       # 通过事件状态或空间距离判断夹爪动作
        cube_current_orientation = observations[current_cube_name]['orientation']
        robot_current_joint_position = observations[self._robot_observation_name]['joint_positions']



        # ===== RL Policy Inference =====
        if self._policy is not None:
            # Get all required observations
            joint_vel = observations[self._robot_observation_name]['joint_velocities']
            
            # Get previous action (if available, else use zeros)
            previous_action = observations[self._robot_observation_name].get('action', np.zeros(8))
            
            # Prepare observation matching training format (36 dims total):
            # - joint_pos (9)
            # - joint_vel (9)  
            # - object_position (3)
            # - object_orientation (4)
            # - place_target (3)
            # - previous_actions (8: 7 arm + 1 gripper)
            
            obs = np.concatenate([
                robot_current_joint_position,  # 9
                joint_vel,                     # 9
                cube_current_position,         # 3
                cube_current_orientation,      # 4
                placing_target_postion,        # 3
                previous_action,               # 8
            ]).astype(np.float32)  # Total: 36
            
            # Convert to tensor and run inference
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            
            with torch.no_grad():
                action = self._policy(obs_tensor).squeeze(0).cpu().numpy()
            
            # Split: 7 arm joints + 1 gripper binary
            arm_actions = action[:7]
            gripper_binary = action[7]
            
            # === Manual Gripper Override ===
            # Force gripper to open when object is near target position
            object_lifted = cube_current_position[2] > 0.04  # Object is lifted
            distance_to_target = np.linalg.norm(cube_current_position - placing_target_postion)
            near_target = distance_to_target < 0.12  # Within 12cm of target
            
            # Override: Open gripper if object is lifted and near target
            if object_lifted and near_target:
                gripper_fingers = np.array([0.04, 0.04])  # Force open
                print(f"[Manual Override] Opening gripper - distance: {distance_to_target:.3f}m")
            else:
                # Use policy decision
                if gripper_binary > 0.0:
                    gripper_fingers = np.array([0.0, 0.0])  # Closed
                else:
                    gripper_fingers = np.array([0.04, 0.04])  # Open
            
            # Combine to 9-dim joint command
            actions_9dim = np.concatenate([arm_actions, gripper_fingers])
            
            # Store action for next observation
            observations[self._robot_observation_name]['action'] = action
        else:
            # Fallback: use pick_place_controller if no policy
            actions_9dim = self._pick_place_controller.forward(
                picking_position=cube_current_position,
                placing_position=placing_target_postion,
                current_joint_positions=robot_current_joint_position,
                end_effector_orientation=end_effector_orientation,
                end_effector_offset=end_effector_offset,
            ).joint_positions
            
        # Create articulation action (positions, not velocities)
        actions = ArticulationAction(joint_positions=actions_9dim)
        # ===== Check Task Completion =====
        # For RL policy: check if object is at target and stable
        is_done = self._check_task_completion(observations, current_cube_name, color_idx)
        
        if is_done:
            print(f"[ArmPickController] Completed cube {self._current_cube_numth} (color {color_idx})")
            
            # Update height for stacking
            cube_size = observations[current_cube_name]['size']
            self._current_height[color_idx] += cube_size[2]
            self._last_completed_cube_numth = self._current_cube_numth
            
            # Move to next cube
            self._current_cube_numth += 1
            
            # Reset pick_place_controller if using state machine fallback
            if self._policy is None:
                self._pick_place_controller.reset()
            
            # Reset timing
            self._current_cube_start_time = 0
            
        return actions
    
    def _check_task_completion(
        self,
        observations: dict,
        cube_name: str,
        color_idx: int,
    ) -> bool:
        """
        Check if current pick-place task is complete.
        
        Criteria:
        - Object is at target position (< 8cm)
        - Object is stable (velocity < 0.15 m/s)
        - Gripper is open (released object)
        - Sustained for 20+ steps (~0.4s)
        """
        # If using state machine fallback, use its completion check
        if self._policy is None:
            return self._pick_place_controller.is_done()
        
        # RL policy completion check
        cube_obs = observations[cube_name]
        cube_position = cube_obs['position']
        cube_velocity = cube_obs.get('velocity', np.zeros(3))
        
        target_position = observations['target_positions'][color_idx]
        target_position[2] = self._current_height[color_idx] + cube_obs['size'][2] / 2
        
        # Check conditions
        distance = np.linalg.norm(cube_position - target_position)
        velocity = np.linalg.norm(cube_velocity)
        
        # Get gripper state
        robot_obs = observations[self._robot_observation_name]
        gripper_joints = robot_obs['joint_positions'][-2:]
        gripper_open = np.mean(gripper_joints) > 0.035  # Open threshold
        
        # Object is at target, stable, and gripper is open
        conditions_met = (distance < 0.08 and velocity < 0.15 and gripper_open)
        
        # Track sustained completion
        if not hasattr(self, '_completion_steps'):
            self._completion_steps = 0
            
        if conditions_met:
            self._completion_steps += 1
        else:
            self._completion_steps = 0
        
        # Require 20 consecutive steps of completion (prevents false positives)
        is_complete = self._completion_steps >= 20
        
        # Also check timeout
        self._current_cube_start_time += 1
        timed_out = self._current_cube_start_time >= self._max_cube_time
        
        if timed_out:
            print(f"[ArmPickController] Cube {self._current_cube_numth} timed out after {self._max_cube_time} steps")
            self._completion_steps = 0  # Reset for next cube
            return True
        
        if is_complete:
            self._completion_steps = 0  # Reset for next cube
            
        return is_complete

    def reset(self, picking_order_cube_names: typing.Optional[typing.List[str]] = None) -> None:
        """Reset the controller state including height tracking."""
        super().reset(picking_order_cube_names)
        self._current_height = [0.0] * 3
        self._last_completed_cube_numth = -1
        self._current_cube_start_time = 0
        self._completion_steps = 0