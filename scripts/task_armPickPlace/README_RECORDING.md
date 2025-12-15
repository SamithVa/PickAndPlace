# Recording Demonstrations for Imitation Learning

This guide explains how to record demonstrations from your pick and place task for imitation learning.

## Overview

You have **two options** for recording demonstrations:

1. **Autonomous Recording** (Easiest) - Use your existing `ArmPickController` to automatically generate demonstrations
2. **Manual Teleoperation** (More flexible) - Manually control the robot using keyboard/spacemouse

## Option 1: Record Autonomous Demonstrations (Recommended)

This method uses your existing controller to automatically execute the task and record the trajectories.

### Usage

```bash
cd /data/wanshan/Desktop/learning/isaaclab/FrankaPickPlace/scripts/task_armPickPlace

# Record 10 demonstrations with 5 cubes
python record_autonomous_demos.py --num_demos 10 --cube_num 5 --output ../../datasets/auto_demos.hdf5

# Run in headless mode (faster, no GUI)
python record_autonomous_demos.py --num_demos 50 --cube_num 10 --output ../../datasets/auto_demos.hdf5 --headless
```

### Parameters

- `--num_demos`: Number of demonstrations to record (default: 10)
- `--cube_num`: Number of cubes in the scene (default: 10)
- `--output`: Output HDF5 file path (default: ./datasets/auto_demos.hdf5)
- `--max_steps`: Maximum steps per episode (default: 2000)
- `--headless`: Run without GUI (faster)

### What Gets Recorded

Each demonstration contains:
- **observations**: Flattened observation vectors at each timestep
- **actions**: Actions computed by the controller
- **robot_states**: Joint positions of the robot
- **success**: Whether the demonstration succeeded
- **num_steps**: Total steps in the demonstration

## Option 2: Record Manual Demonstrations

This method allows you to manually control the robot using keyboard/device input.

### Prerequisites

First, you need to complete the gym environment wrapper:

1. **Edit `arm_pick_place_env.py`** to properly implement:
   - `_calculate_reward()` - Define your reward function
   - `_check_termination()` - Define success/failure conditions
   - `_convert_observations()` - Ensure observations are properly formatted

2. **Make sure your `class_taskEnv.py` and `class_controller.py` are accessible**

### Usage

```bash
cd /data/wanshan/Desktop/learning/isaaclab/FrankaPickPlace/scripts/task_armPickPlace

# Record demonstrations manually
python record_arm_pick_demos.py --num_demos 10 --output ../../datasets/manual_demos.hdf5
```

### Controls

During recording:
- **R** - Start recording a new demonstration
- **S** - Stop and save current demonstration
- **D** - Discard current demonstration
- **Q** - Quit and save all recorded demos
- **Arrow keys/WASD** - Control robot movement

## Data Format

Both methods save demonstrations in **HDF5 format** with the following structure:

```
dataset.hdf5
├── demo_0/
│   ├── observations: [num_steps, obs_dim]
│   ├── actions: [num_steps, action_dim]
│   ├── robot_states: [num_steps, joint_dim]
│   └── (attributes: num_steps, success, total_reward)
├── demo_1/
│   └── ...
└── (attributes: num_demos, creation_time)
```

## Using Recorded Data for Imitation Learning

After recording, you can use the demonstrations with imitation learning algorithms:

### Example: Loading Data

```python
import h5py
import numpy as np

# Load demonstrations
with h5py.File('datasets/auto_demos.hdf5', 'r') as f:
    num_demos = f.attrs['num_demos']
    
    for i in range(num_demos):
        demo = f[f'demo_{i}']
        observations = demo['observations'][:]
        actions = demo['actions'][:]
        
        print(f"Demo {i}: {len(actions)} steps")
```

### Compatible Algorithms

The recorded data can be used with:
- **Behavioral Cloning (BC)** - Supervised learning from demonstrations
- **DAgger** - Dataset Aggregation for iterative improvement
- **GAIL** - Generative Adversarial Imitation Learning
- **BC-Z, Diffusion Policy, ACT** - Modern imitation learning methods

## Tips for Good Demonstrations

### For Autonomous Recording:
1. Ensure your controller reliably completes the task
2. Add success criteria to filter out failed attempts
3. Record with varying initial conditions (cube positions)
4. Use `--headless` mode to speed up recording

### For Manual Recording:
1. Practice the task before recording
2. Be consistent in your motions
3. Complete the task smoothly without pausing
4. Record multiple successful attempts
5. Vary your approach slightly for diversity

## Troubleshooting

### Issue: "Module not found: class_taskEnv"
**Solution**: Make sure you're running from the correct directory with access to your custom modules

### Issue: Controller not working properly
**Solution**: Check that your `ArmPickController` is properly initialized and robot is loaded

### Issue: Observations have wrong shape
**Solution**: Edit `_convert_observations()` in `arm_pick_place_env.py` to properly flatten/structure your observations

### Issue: Recording too slow
**Solution**: Use `--headless` flag to disable rendering during recording

## Next Steps

1. **Record demonstrations** using one of the methods above
2. **Visualize the data** to ensure quality:
   ```python
   import h5py
   with h5py.File('datasets/auto_demos.hdf5', 'r') as f:
       print(f"Recorded {f.attrs['num_demos']} demonstrations")
       for i in range(f.attrs['num_demos']):
           demo = f[f'demo_{i}']
           print(f"  Demo {i}: {demo.attrs['num_steps']} steps, Success: {demo.attrs['success']}")
   ```

3. **Train an imitation learning model** using the recorded data
4. **Evaluate** the learned policy on the task

## Additional Resources

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [Imitation Learning Guide](https://imitation.readthedocs.io/)
- [Behavioral Cloning Tutorial](https://spinningup.openai.com/en/latest/algorithms/bc.html)
