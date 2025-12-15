# Multi-Cube Pick and Place Configuration

This document explains the multi-cube pick and place task implementation for the Franka robot.

## Overview

The environment now supports training a robot to pick and place multiple cubes (default: 4 cubes) in the scene. The robot learns to:
1. Select/identify which cube to pick
2. Pick the selected cube
3. Transport it to the target drop location
4. Place it precisely

## Key Features

### 1. Multiple Cubes in Scene
- **Location**: `frankapickplace_env_cfg.py` - `FrankaSceneCfg.__post_init__()`
- Creates 4 cubes with random initial positions
- Each cube has a unique color for visual identification
- Cube positions randomized within range: x=(0.2, 0.6), y=(-0.3, 0.3)

### 2. Cube Selection Strategies
- **Location**: `mdp/cube_selection.py`

Three selection modes available:

#### a) **Closest Mode** (Default)
```python
"selection_mode": "closest"
```
- Selects the cube nearest to the end-effector
- Good for learning efficient picking strategies
- Robot naturally gravitates to nearby objects

#### b) **Random Mode**
```python
"selection_mode": "random"
```
- Randomly selects a cube each episode
- Provides diverse training scenarios
- Helps generalization

#### c) **Sequential Mode**
```python
"selection_mode": "sequential"
```
- Cycles through cubes in order (0→1→2→3→0...)
- Useful for systematic testing
- Counter increments on successful placement

### 3. Observations

The robot observes:
- **Joint positions and velocities** (9 DoF arm + 2 DoF gripper)
- **All cube positions** (4 cubes × 3 coordinates = 12 values)
  - Allows the policy to learn which cube to approach
  - All positions in robot root frame for easier learning
- **Target drop position** (3 coordinates)
- **Previous actions** (for temporal coherence)

**Alternative observation mode** (commented out):
```python
target_cube_position = ObsTerm(func=mdp.target_cube_position_in_robot_root_frame)
```
Use this if you want to pre-select the target cube externally and only observe it.

### 4. Reward Functions

All reward functions updated to support multi-cube tracking:

#### Modified Functions:
- `object_ee_distance()` - Distance from EE to target cube
- `object_is_lifted()` - Whether target cube is lifted
- `object_goal_distance()` - Distance from target cube to goal

Each function has `use_target_cube=True` parameter to track the selected cube instead of a fixed object.

#### Reward Structure:
```python
reaching_object:        1.0  × (1 - tanh(d_ee_cube / 0.1))
lifting_object:        15.0  × is_lifted_above_4cm
object_goal_tracking:  16.0  × (1 - tanh(d_cube_goal / 0.3)) * is_lifted
fine_grained_tracking:  5.0  × (1 - tanh(d_cube_goal / 0.05)) * is_lifted
action_rate_penalty:   -5e-5 × ||Δaction||²
joint_vel_penalty:     -5e-5 × ||joint_vel||²
```

### 5. Events (Resets)

#### On Episode Reset:
```python
reset_all_cubes = EventTerm(
    func=mdp.reset_all_cubes_to_random_positions,
    mode="reset",
    params={
        "num_cubes": 4,
        "pose_range": {"x": (0.2, 0.6), "y": (-0.3, 0.3), "z": (0.055, 0.055)},
        "selection_mode": "closest",
    },
)
```

**What happens**:
1. All 4 cubes reset to new random positions
2. Target cube is selected based on `selection_mode`
3. Velocities set to zero
4. Environment stores `target_cube_idx` for tracking

### 6. Termination Conditions

Episode ends when:
- **Time out**: 5 seconds (500 steps at dt=0.01, decimation=2)
- **Success**: Target cube reaches goal position (within 2cm threshold)

## Configuration Options

### Change Number of Cubes

In `frankapickplace_env_cfg.py`:
```python
# In FrankaSceneCfg.__post_init__()
cube_configs = create_cube_configs(num_cubes=8)  # Change from 4 to 8

# Update all params in EventCfg, RewardsCfg, ObservationsCfg
"num_cubes": 8
```

### Change Selection Strategy

In `EventCfg`:
```python
"selection_mode": "random"  # or "closest" or "sequential"
```

### Modify Cube Spawn Range

In `EventCfg`:
```python
"pose_range": {
    "x": (0.3, 0.7),    # Further from robot
    "y": (-0.4, 0.4),   # Wider spread
    "z": (0.055, 0.055) # Fixed height
}
```

## Training Tips

1. **Start with "closest" mode** - Easier for initial learning
2. **Monitor which cubes are being selected** - Check if robot learns to approach efficiently
3. **Adjust observation space**:
   - Use `all_cubes_positions` to let robot learn selection
   - Use `target_cube_position` if you want pre-selection
4. **Curriculum learning**:
   - Start with 2 cubes, gradually increase to 4+
   - Start with "closest", move to "random" for robustness

## Implementation Details

### Target Cube Tracking
The environment maintains `env.target_cube_idx` (shape: `[num_envs]`) to track which cube each parallel environment should target.

### Cube Indexing
- Cubes named as: `cube_0`, `cube_1`, `cube_2`, `cube_3`
- Indices stored as tensor: `[0, 1, 2, 3]`
- Each environment can target different cubes simultaneously

### Observation Shape
With 4 cubes and default setup:
- Joint pos: 11 values
- Joint vel: 11 values  
- All cubes: 12 values (4 × 3)
- Target drop: 7 values (pos + quat, but only first 3 used)
- Actions: 11 values

**Total**: ~52 observations per step

## Future Extensions

1. **Task Sequencing**: Pick and place all cubes in sequence
2. **Sorting**: Place cubes in specific orders or locations
3. **Stacking**: Stack cubes on top of each other
4. **Obstacle Avoidance**: Add obstacles between cubes and target
5. **Dynamic Targets**: Moving drop locations

## Troubleshooting

### Issue: Robot always picks the same cube
**Solution**: Change `selection_mode` to "random" or ensure cube positions are well distributed

### Issue: Low success rate
**Solutions**:
- Increase episode length: `episode_length_s = 10.0`
- Reduce spawn area to make task easier
- Increase lifting reward weight
- Check if gripper is closing properly

### Issue: Robot drags cubes instead of lifting
**Solutions**:
- Increase `lifting_object` reward weight (try 30.0)
- Ensure `minimal_height` in `object_goal_tracking` is enforced (0.2m)
- Add penalty for low cube height

### Issue: Observations too large
**Solution**: Use `target_cube_position` instead of `all_cubes_positions`

## File Structure

```
mdp/
├── __init__.py              # Imports all modules
├── observations.py          # All cubes and target cube observations
├── rewards.py              # Multi-cube compatible rewards
├── terminations.py         # Goal reaching termination
├── cube_selection.py       # NEW: Cube selection strategies
└── events.py               # NEW: Multi-cube reset logic
```

## Example: Changing to 2 Cubes

```python
# 1. In FrankaSceneCfg.__post_init__()
cube_configs = create_cube_configs(num_cubes=2)

# 2. In ObservationsCfg.PolicyCfg
all_cubes_positions = ObsTerm(
    func=mdp.all_cubes_positions_in_robot_root_frame, 
    params={"num_cubes": 2}
)

# 3. In EventCfg
reset_all_cubes = EventTerm(
    func=mdp.reset_all_cubes_to_random_positions,
    params={"num_cubes": 2, ...}
)

# 4. In all RewardsCfg terms
params={"...", "num_cubes": 2}
```

---

**Ready to train!** Run your training script as usual. The robot will now learn to handle multiple cubes.
