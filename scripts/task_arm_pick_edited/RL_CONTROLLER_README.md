# RL-Based Pick and Place Controller Documentation

## Overview

This implementation replaces the hardcoded state machine controller with a **learned neural network policy** trained in IsaacLab. The RL policy directly maps observations to actions without explicit phase management.

---

## Architecture Comparison

### Original State Machine Controller
```
┌─────────────────────────────────────────┐
│  PickPlaceController                    │
│                                         │
│  Phase 0: Move above object             │
│  Phase 1: Lower to object              │
│  Phase 2: Wait                         │
│  Phase 3: Close gripper                │
│  Phase 4: Lift                         │
│  Phase 5: Transport                    │
│  Phase 6: Lower to goal                │
│  Phase 7: Open gripper                 │
│  Phase 8: Lift up                      │
│  Phase 9: Return                       │
│                                         │
│  Uses: IK solver + time triggers       │
└─────────────────────────────────────────┘
```

### New RL Policy Controller
```
┌─────────────────────────────────────────┐
│  RLPickPlaceController                  │
│                                         │
│  Observation → Neural Network → Action │
│                                         │
│  • No phases                           │
│  • No time triggers                    │
│  • End-to-end learning                 │
│  • Reactive to state                   │
│                                         │
│  Uses: Trained policy (PyTorch)        │
└─────────────────────────────────────────┘
```

---

## Key Components

### 1. `RLPickPlaceController` (Single Object)

**Purpose**: Control one pick-and-place cycle using trained policy

**Key Methods**:
- `forward()`: Get action from policy for current observation
- `_prepare_observation()`: Format obs to match training (39-dim)
- `_apply_safety_constraints()`: Optional safety checks
- `_check_completion()`: Detect successful placement

**Usage**:
```python
controller = RLPickPlaceController(
    name="rl_controller",
    policy_path="model.pt",  # Your trained policy
    gripper=gripper,
    articulation=robot,
    device="cuda",
)

action = controller.forward(observations)
```

### 2. `MultiObjectRLController` (Multiple Objects)

**Purpose**: Handle multiple pick-place cycles with single policy

**Key Features**:
- Automatic object switching
- Height stacking support
- Progress tracking

**Usage**:
```python
controller = MultiObjectRLController(
    name="multi_controller",
    policy_path="model.pt",
    gripper=gripper,
    articulation=robot,
    picking_order_cube_names=["cube_0", "cube_1", "cube_2"],
    robot_observation_name="franka",
)

action = controller.forward(observations)
```

### 3. `ArmPickController` (Updated)

**Purpose**: Your existing controller, now RL-enabled

**What Changed**:
- ✅ Proper 39-dim observation format (matches IsaacLab training)
- ✅ Gripper binary → two-finger conversion
- ✅ Joint position commands (not velocities)
- ✅ Task completion detection for RL
- ✅ Fallback to state machine if no policy

---

## Observation Format

The controller expects observations matching your IsaacLab training:

| Component | Dims | Description |
|-----------|------|-------------|
| `joint_pos` | 9 | Current joint positions |
| `joint_vel` | 9 | Joint velocities |
| `object_position` | 3 | Object position (robot frame) |
| `object_velocity` | 3 | Object linear velocity |
| `place_target` | 7 | Target pose (pos + quat) |
| `actions` | 8 | Previous action (7 arm + 1 gripper) |
| **Total** | **39** | Full observation vector |

### Example Observation Dictionary
```python
observations = {
    'franka': {
        'joint_positions': np.array([...]),  # 9 values
        'joint_velocities': np.array([...]), # 9 values
        'action': np.array([...]),           # 8 values (previous)
    },
    'cube_0': {
        'position': np.array([x, y, z]),
        'velocity': np.array([vx, vy, vz]),
        'color': 0,  # Color index for target
        'size': np.array([0.05, 0.05, 0.05]),
    },
    'target_positions': np.array([
        [0.5, 0.2, 0.025],  # Red target
        [0.5, 0.0, 0.025],  # Green target
        [0.5, -0.2, 0.025], # Blue target
    ])
}
```

---

## Action Format

### Policy Output
- **Shape**: `(8,)`
- **Components**: 
  - First 7: Arm joint position deltas
  - Last 1: Gripper binary (-1 = open, +1 = close)

### Converted to Robot Command
- **Shape**: `(9,)`
- **Components**:
  - First 7: Arm joint positions
  - Last 2: Finger positions (both = 0.04 for open, 0.0 for closed)

### Code
```python
arm_actions = policy_output[:7]
gripper_binary = policy_output[7]

if gripper_binary > 0.0:
    gripper_fingers = [0.0, 0.0]  # Closed
else:
    gripper_fingers = [0.04, 0.04]  # Open

joint_cmd = np.concatenate([arm_actions, gripper_fingers])
action = ArticulationAction(joint_positions=joint_cmd)
```

---

## Task Completion Detection

### Criteria (All Must Be True)
1. **Position**: Object within 8cm of target
2. **Stability**: Object velocity < 0.15 m/s  
3. **Released**: Gripper is open (> 0.035)
4. **Sustained**: Conditions met for 20+ steps (~0.4 seconds)

### Why Sustained Check?
Prevents false positives during:
- Gripper opening while moving
- Brief contact during transport
- Oscillations near target

### Timeout Fallback
- Max time per object: 1000 steps (~20 seconds)
- Auto-advances to next object if stuck

---

## Integration Guide

### Step 1: Export Your Trained Policy

```bash
# In your IsaacLab workspace
cd FrankaPickPlace
python scripts/rsl_rl/play.py \
    --task Isaac-Franka-Pick-Place-v0 \
    --checkpoint /path/to/checkpoint.pt \
    --export
```

This creates: `model_*.pt` (JIT traced)

### Step 2: Update Your Controller

Replace in `class_controller.py`:

```python
# OLD:
from isaacsim.robot.manipulators.controllers import PickPlaceController

# NEW:
from rl_pick_place_controller import MultiObjectRLController

# In __init__:
self._pick_place_controller = MultiObjectRLController(
    name="rl_controller",
    policy_path="/path/to/model.pt",  # Your exported policy
    gripper=gripper,
    articulation=articulation,
    picking_order_cube_names=picking_order_cube_names,
    robot_observation_name=robot_observation_name,
    device="cuda",
)
```

### Step 3: Ensure Observations Match

Make sure your observation dict includes:
- ✅ `joint_positions` (9-dim)
- ✅ `joint_velocities` (9-dim)
- ✅ Object `position` and `velocity`
- ✅ `target_positions` array
- ✅ Previous `action` (8-dim)

### Step 4: Run and Monitor

```python
for step in range(max_steps):
    observations = get_observations()
    action = controller.forward(observations)
    robot.apply_action(action)
    world.step()
    
    if controller.is_done():
        print("All objects placed!")
        break
```

---

## Advantages Over State Machine

| Feature | State Machine | RL Policy |
|---------|--------------|-----------|
| **Adaptability** | Fixed sequence | Reactive to state |
| **Robustness** | Fails on timing | Handles variations |
| **Efficiency** | Fixed phases | Optimized paths |
| **Generalization** | Single scenario | Multiple configs |
| **Timing** | Hardcoded | Learned |
| **Recovery** | No error recovery | Can recover |

---

## Troubleshooting

### Issue: Policy outputs random actions
**Solution**: Check observation normalization matches training
```python
# If you used observation scaling during training:
controller = RLPickPlaceController(
    ...,
    obs_scale={'joint_pos': 1.0, 'joint_vel': 0.1}
)
```

### Issue: Gripper not closing/opening
**Solution**: Check gripper binary threshold
```python
# In controller code, adjust threshold:
if gripper_binary > 0.0:  # Try 0.5 or -0.5 if needed
    gripper_fingers = [0.0, 0.0]
```

### Issue: Task never completes
**Solutions**:
1. Lower completion threshold:
   ```python
   distance < 0.10  # Instead of 0.08
   ```
2. Reduce sustained steps:
   ```python
   self._completion_steps >= 10  # Instead of 20
   ```
3. Check timeout:
   ```python
   self._max_cube_time = 2000  # Increase limit
   ```

### Issue: Policy path not found
**Solution**: Use absolute path or check export location
```python
import os
policy_path = os.path.expanduser("~/models/model.pt")
assert os.path.exists(policy_path), f"Policy not found: {policy_path}"
```

---

## Performance Expectations

### Single Object (Trained Policy)
- **Time to grasp**: 50-100 steps (1-2 seconds)
- **Time to lift**: 20-30 steps (0.4-0.6 seconds)
- **Time to transport**: 50-100 steps (1-2 seconds)
- **Time to place**: 30-50 steps (0.6-1 second)
- **Total**: ~150-280 steps (3-5.6 seconds)

### Multiple Objects
- **Per object**: ~200 steps (4 seconds)
- **4 objects**: ~800 steps (16 seconds)
- Includes stacking height adjustments

---

## Safety Constraints (Optional)

Enable safety checks:
```python
controller = RLPickPlaceController(
    ...,
    use_safety_constraints=True
)
```

### Built-in Constraints
1. **Don't drop during transport**: Forces gripper closed if object lifted but not at target
2. **Joint limits**: (Add custom limits as needed)
3. **Velocity limits**: (Add custom limits as needed)

### Custom Constraints
Add in `_apply_safety_constraints()`:
```python
def _apply_safety_constraints(self, action, obs):
    action = action.copy()
    
    # Example: Limit max joint velocity
    max_vel = 1.0
    action[:7] = np.clip(action[:7], -max_vel, max_vel)
    
    # Example: Prevent collision (custom logic)
    if self_collision_detected(obs):
        action = safe_action
    
    return action
```

---

## Next Steps

1. ✅ **Export your trained policy** from IsaacLab
2. ✅ **Test single object** with `RLPickPlaceController`
3. ✅ **Scale to multiple objects** with `MultiObjectRLController`
4. ✅ **Add safety constraints** if needed
5. ✅ **Monitor performance** and adjust thresholds
6. ✅ **Fine-tune completion detection** for your setup

---

## Files Reference

- `rl_pick_place_controller.py`: Main controller implementations
- `example_rl_usage.py`: Usage examples and integration guide
- `class_controller.py`: Your updated controller (RL-enabled)

---

## Questions?

Common questions:

**Q: Can I use the state machine as fallback?**  
A: Yes! If `policy_path=None`, controller uses `PickPlaceController`

**Q: How do I retrain with better rewards?**  
A: Use the improved reward design in `frankapickplace_env_cfg.py`

**Q: Does this work with other robots?**  
A: Yes, just train a policy for your robot and export

**Q: Can I use this in real-world?**  
A: Sim-to-real requires domain randomization during training and careful testing
