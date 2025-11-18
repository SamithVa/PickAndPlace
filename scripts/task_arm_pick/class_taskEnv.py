# Task类核心内容，通过继承BaseTask类来组织任务逻辑，实现机械臂抓取与分类任务
# 可通过修改以下内容来实现不同的任务：
#（1）场景初始化（set_up_scene）
#     负责定义任务的吃初始场景。如加载资源和配置物理参数
#（2）观察获取（get_observations）
#     负责获取任务执行过程中的观察数据，如机械臂末端位置、方块位置等
#（3）指标计算（calculate_metrics）
#     定义任务
#（4）重置（reset）
import argparse
import sys

import carb
from isaacsim.core.api.tasks import BaseTask

from isaacsim.core.api.objects import GroundPlane,DynamicCuboid
from isaacsim.core.api.scenes import Scene

import numpy as np
from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.prims import is_prim_path_valid

import isaacsim.core.utils.stage as stage_utils

class taskEnv_SceneSetup(BaseTask):
    """机械臂抓取与分类任务"""
    def __init__(
        self, 
        name="task_arm_pickplace",
        cube_num=8,
        cube_scale=None,
    )->None:
        super().__init__(name,offset=None)
        self._robot = None
        self._cubes = []            # 立方体列表
        self._cube_positions = []
        self._cube_colors = []

        self._cube_num = cube_num
        self._cube_scale = cube_scale
        if self._cube_scale is None:
            self._cube_scale = np.array([0.0515, 0.0515, 0.0515])/get_stage_units()  # 默认方块尺寸
        self._target_positions = np.array([[0.2, -0.2, 0.0],     # Red  分类目标位置 (3个目标位置对应3种颜色)
                                           [0.4, -0.2, 0.0],     # Green
                                           [0.6, -0.2, 0.0]])    # Blue 
        self._target_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        self._target_colors = np.array([[1, 0, 0],   # Red
                                        [0, 1, 0],   # Green
                                        [0, 0, 1]])  # Blue
        
        # Initialize missing attributes
        self._task_objects = {}

    # （1）场景初始化
    def set_up_scene(self, scene: Scene) -> None:
        super().set_up_scene(scene)
        
        # 添加随机方块
        for i in range(self._cube_num):
            cube_position, cube_orientation, cube_color = self.add_random_cube()
            cube_prim_path = find_unique_string_name(
                initial_name="/World/Cube", 
                is_unique_fn=lambda x: not is_prim_path_valid(x)
            )
            cube_name = find_unique_string_name(
                initial_name="cube", 
                is_unique_fn=lambda x: not scene.object_exists(x)
            )
            # 创建方块对象添加到_cubes列表中，便于后续统一管理和访问
            self._cubes.append(
                scene.add(            #将创建的DynamicCuboid对象添加到物理场景中，使得立方体参与物理仿真，返回一个场景对象的引用
                    DynamicCuboid(    #创建具有物理属性的立方体（具有质量、惯性等物理属性），可以参与碰撞检测，能够被机器人抓取和移动
                        name=cube_name,
                        prim_path=cube_prim_path,
                        position=cube_position,
                        orientation=cube_orientation,
                        scale=self._cube_scale,
                        size=1.0,    # 立方体的边长         
                        color=cube_color,
                    )
                )
            )
            # 创建一个名称到对象的映射字典
            self._task_objects[self._cubes[-1].name] = self._cubes[-1]  # self._cubes[-1] 获取刚添加的最后一个立方体

        # 添加Franka机械臂
        try:
            from isaacsim.robot.manipulators.examples.franka import Franka
            self._robot = scene.add(Franka(prim_path="/World/Franka", name="myfranka"))
            self._task_objects[self._robot.name] = self._robot
        except Exception as e:
            print(f"Failed to load Franka robot: {e}")
            print("Continuing without robot for now...")
            self._robot = None
        print("finished adding cubes and robot")
        print(f"创建了 {len(self._cubes)} 个方块")
        print(f"方块名称: {[cube.name for cube in self._cubes]}")
        print(f"方块颜色索引: {self._cube_colors}")
        print(f"目标位置: {self._target_positions}")
   
    def calculate_metrics(self)-> dict:
        # 计算方块与目标位置距离作为指标
        observations = self.get_observations()
        distances = []
        cube_count = 0
        for key, value in observations.items():
            if key != "target_positions" and key != self._robot.name if self._robot else "franka":
                if "position" in value:
                    cube_count += 1
                    target_idx = (cube_count - 1) % len(self._target_positions)  # 简单分配策略
                    distances.append(np.linalg.norm(value["position"] - self._target_positions[target_idx]))
        return {"avg_distance": np.mean(distances) if distances else 0.0}

    # （2）观察获取（get_observations）      
    def get_observations(self)->dict:
        
        observations = {}

        # 机械臂观测：关节状态、末端执行器位姿 
        if self._robot is not None:
            joints_state = self._robot.get_joints_state()
            joint_currentpos_01 = joints_state.positions
            joint_currentpos_02 = self._robot.get_joint_positions()
            end_effector_position_01, _ = self._robot.end_effector.get_local_pose()
            end_effector_position_02 = self._robot.end_effector.get_world_pose()
            #print("end_effector_position_01", end_effector_position_01)
            #print("end_effector_position_02", end_effector_position_02)

            # NOTE
            joint_velocities = joints_state.velocities
      
            observations[self._robot.name] = {
                "joint_positions": joint_currentpos_02, # 9 DOF
                "joint_velocities": joint_velocities, # 9 DOF
                "action": self.last_action if hasattr(self, 'last_action') else np.zeros(joint_currentpos_02.shape[0] - 1), # joint_currentpos_02.shape[0] - 1 (gripper is only considered as open/close) 
                "end_effector_position": end_effector_position_02,
            }
        
        # 方块观测：位姿、颜色索引 
        for i in range(len(self._cubes)):
            cube_position_local, cube_orientation_local = self._cubes[i].get_local_pose()
            cube_position_world, cube_orientation_world = self._cubes[i].get_world_pose()

            observations[self._cubes[i].name] = {
                "position": cube_position_world, # 3
                "orientation": cube_orientation_world, # 4
                "size": self._cube_scale,
                "color": self._cube_colors[i] if i < len(self._cube_colors) else 0
            }
        self._target_positions /= get_stage_units()
        # 目标位置观测
        observations["target_positions"] = self._target_positions
        return observations 
    
    def reset(self):
        # 重置方块位置
        if hasattr(self, '_cubes') and self._cubes:
            for i, cube in enumerate(self._cubes):
                cube.set_world_pose(position=np.array([0, 0.1*i, 0.1]))
        return True
    
    def post_reset(self) -> None:
        from isaacsim.robot.manipulators.grippers import ParallelGripper
        if self._robot is not None and isinstance(self._robot.gripper, ParallelGripper):
            #self._robot.gripper.set_gripper_positions(self._robot.gripper.joint_opened_positions)  # 张开夹爪
            print("Gripper opened after reset.")

        return

    def get_params(self) -> dict:
        """Get the parameters of the task."""
        params_representation = dict()
        if self._robot is not None:
            params_representation["robot_name"] = {"value": self._robot.name, "modifiable": False}
        return params_representation
    
    def is_done(self) -> bool:
        super().is_done()
        # 任务完成条件：所有方块都接近各自目标位置
        observations = self.get_observations()
        for i in range(len(self._cubes)):
            cube_name = self._cubes[i].name
            if cube_name in observations:
                target_idx = i % len(self._target_positions)
                distance = np.linalg.norm(observations[cube_name]["position"] - self._target_positions[target_idx])
                if distance > 0.05:  # 设定一个距离阈值
                   return False
        return True

    #（3）随机生成方块位置、颜色（add_random_cube）
    def add_random_cube(self):
        while True:
            # 在指定范围内随机采样一个位置
            position = np.random.uniform(0.1, 0.5, 3)
            position /= get_stage_units()    # 转换为仿真场景单位

            # 检查新位置与所有已存在方块的距离，避免重叠
            if all(np.linalg.norm(position - pos) >= 2 * self._cube_scale[0]
                   for pos in self._cube_positions):
                orientation = None  # 默认朝向
                self._cube_positions.append(position)
                rand_color_idx = np.random.choice(len(self._target_colors))
                cube_color = self._target_colors[rand_color_idx]
                self._cube_colors.append(rand_color_idx)

                return position, orientation, cube_color

    def get_cube_names(self):
        """Return the names of all cubes in the task."""
        return [cube.name for cube in self._cubes]
