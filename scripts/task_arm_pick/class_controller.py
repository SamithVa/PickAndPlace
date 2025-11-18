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



        # use RL model here to decide actions
        # actions = self._pick_place_controller.forward(
        #     picking_position=cube_current_position, 
        #     placing_position=placing_target_postion,
        #     current_joint_positions=robot_current_joint_position,
        #     end_effector_orientation=end_effector_orientation,
        #     end_effector_offset=end_effector_offset,
        # )
        # robot_joint_vel
        joint_vel = observations[self._robot_observation_name]['joint_velocities']
        previous_action = observations[self._robot_observation_name]['action']
        
        # print("robot_current_joint_position", robot_current_joint_position.shape)
        # print("joint_vel", joint_vel.shape)
        # print("placing_target_postion", placing_target_postion.shape)
        # print("cube_current_position", cube_current_position.shape)
        # print("cube_orientation", observations[current_cube_name]['orientation'].shape)
        # print("previous_action", previous_action.shape)
        
        obs = torch.tensor(
            np.concatenate(( # total 36
                robot_current_joint_position,  # 9
                joint_vel,  # 9
                placing_target_postion, # 3 
                cube_current_position, # 3
                cube_current_orientation,  # 4
                previous_action,  # 8
                )
            ), dtype=torch.float32
        ).unsqueeze(0)
        
        actions = self._policy(obs).squeeze(0).detach().numpy() if self._policy is not None else None
        # actions to dict
        actions = ArticulationAction(joint_velocities=actions)
        print(actions)

        if self._pick_place_controller.is_done():
            print(f"PickPlaceController reports done for cube {self._current_cube_numth} (color index {color_idx})")
                        # 更新该颜色位置的高度（加上方块高度）
            cube_size = observations[current_cube_name]['size']
            self._current_height[color_idx] += cube_size[2]
            self._last_completed_cube_numth = self._current_cube_numth
            
            # 移动到下一个方块
            self._current_cube_numth += 1
            
            # 重置PickPlaceController状态以准备下一个方块
            self._pick_place_controller.reset()
            
            # 重置当前方块处理时间
            self._current_cube_time = 0
            
        return actions

    def reset(self, picking_order_cube_names: typing.Optional[typing.List[str]] = None) -> None:
        """Reset the controller state including height tracking."""
        super().reset(picking_order_cube_names)
        self._current_height = [0.0] * 3
        self._last_completed_cube_numth = -1
        self._current_cube_time = 0