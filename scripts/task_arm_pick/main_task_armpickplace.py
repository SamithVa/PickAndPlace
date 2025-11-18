# main_task_armpickplace.py

from isaacsim import SimulationApp
import numpy as np
# 启动Isaac Sim
simulation_app = SimulationApp({"headless": True})

from isaacsim.core.api import World
my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()

from class_taskEnv import taskEnv_SceneSetup
from class_controller import ArmPickController
my_task  = taskEnv_SceneSetup(name="env_armPick",cube_num= 10)  
my_world.add_task(my_task)    #当添加任务时，框架会自动调用set_up_scene，将world的scene传入task场景的创建（set_up_scene(scene)）

# 初始化场景（World协调所有任务初始化Scene）
my_world.reset()

# 获取任务参数，确保机器人已正确加载
task_params = my_task.get_params()
robot_name = task_params["robot_name"]["value"]
my_franka = my_world.scene.get_object(robot_name)

# 创建自定义的ArmPickController控制器
# 该控制器基于StackingController，负责机械臂的抓取与堆叠逻辑
# 参数说明：
#   - name: 控制器名称
#   - gripper: 机械臂末端夹爪对象
#   - articulation: 机械臂本体对象
#   - picking_order_cube_names: 方块抓取顺序的名称列表
#   - robot_observation_name: 机器人观测数据的键名
my_controller = ArmPickController(
    name="stacking_controller",
    gripper=my_franka.gripper,
    articulation=my_franka,
    picking_order_cube_names=my_task.get_cube_names(),  # 获取所有方块的名称，定义抓取顺序
    robot_observation_name=robot_name,
    policy_path="./logs/rsl_rl/franka_pick_place/2025-11-18_18-04-11_lift_to_air/exported/policy.pt"
)

# 获取机械臂的关节控制器，用于后续将控制器输出的动作应用到机器人
articulation_controller = my_franka.get_articulation_controller()

# 主仿真循环
step_count = 0
reset_needed = False
while simulation_app.is_running():
    my_world.step(render=True)    # World更新物理仿真，Scene管理对象状态
    
    if my_world.is_stopped() and not reset_needed: # 监测仿真状态，如果仿真停止且未需要重置，则开始仿真
        reset_needed = True
        
    if my_world.is_playing():

        if reset_needed:
            my_world.reset()
            my_controller.reset()
            reset_needed = False
            
        # Get observations and apply controller actions
        observations = my_task.get_observations()
        actions = my_controller.forward(observations=observations) # 根据观测数据计算动作指令
        my_task.last_action = actions  # NOTE: 保存最后的动作指令到任务中，供观测使用
        articulation_controller.apply_action(actions) #将动作指令应用到机器人仿真中
        
        step_count += 1
        # 每100步打印一次指标
        if step_count % 100 == 0:
           #print("Metrics:", my_task.calculate_metrics())
           pass

simulation_app.close()