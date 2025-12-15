# Quick Start Guide: Multi-Cube Pick and Place

## What's New?

Your environment now supports **multi-cube pick and place** training! The robot can learn to:
- Observe 4 different cubes in the scene
- Select which cube to pick (or have it selected automatically)
- Pick the selected cube
- Place it at the target drop location

## Quick Test

Test the environment with random actions:

```bash
# Basic test with 4 environments
python scripts/test_multi_cube.py --num_envs 4

# Test with visualization
python scripts/test_multi_cube.py --num_envs 4 --enable_cameras

# Test different selection modes
python scripts/test_multi_cube.py --selection_mode random
python scripts/test_multi_cube.py --selection_mode sequential
python scripts/test_multi_cube.py --selection_mode closest
```

## Training

Train the agent using your existing training script:

```bash
# Example with RSL-RL
python scripts/rsl_rl/train.py --task FrankaPickPlace --num_envs 4096 --headless

# With fewer environments for debugging
python scripts/rsl_rl/train.py --task FrankaPickPlace --num_envs 512

# With video recording
python scripts/rsl_rl/train.py --task FrankaPickPlace --video --video_interval 5000
```

## Key Configuration Changes

### 1. Observations (Auto-configured)
The robot now observes:
- ✅ All 4 cube positions (12 values: 4 cubes × 3 coordinates)
- ✅ Joint states (22 values)
- ✅ Target drop location (7 values)
- ✅ Previous actions (11 values)

**Total observation size: ~52 values** (increased from ~40)

### 2. Cube Selection Modes

Edit `frankapickplace_env_cfg.py` → `EventCfg` → `reset_all_cubes`:

```python
"selection_mode": "closest"  # Robot picks nearest cube
"selection_mode": "random"   # Random cube each episode
"selection_mode": "sequential"  # Cycles through 0→1→2→3
```

**Recommendation**: Start with `"closest"` for easier initial learning.

### 3. Customizing Number of Cubes

To change from 4 to a different number:

1. **In `frankapickplace_env_cfg.py`**:

```python
# In FrankaSceneCfg.__post_init__()
cube_configs = create_cube_configs(num_cubes=8)  # Change here

# In ObservationsCfg.PolicyCfg
all_cubes_positions = ObsTerm(
    func=mdp.all_cubes_positions_in_robot_root_frame, 
    params={"num_cubes": 8}  # Change here
)

# In EventCfg.reset_all_cubes
params={
    "num_cubes": 8,  # Change here
    ...
}

# In ALL reward terms in RewardsCfg
params={"...", "num_cubes": 8}  # Change here
```

## Understanding the System

### How Target Selection Works

1. **At episode start**: System selects a target cube based on `selection_mode`
2. **During episode**: Robot receives observations of all cubes but rewards track only the target
3. **On success**: If using "sequential" mode, next cube becomes target

### Environment State

The environment tracks:
- `env.target_cube_idx`: Which cube each parallel environment should target (shape: [num_envs])
- `env.cube_sequence_counter`: For sequential mode (shape: [num_envs])

### Observation Space Breakdown

```python
# With 4 cubes:
joint_pos:           11 values  # Arm (7) + Gripper (2) joints × 2
joint_vel:           11 values
all_cubes_positions: 12 values  # 4 cubes × (x, y, z)
target_drop:          7 values  # (x, y, z, qw, qx, qy, qz)
actions:             11 values  # Previous action
# Total:             52 values
```

## Monitoring Training

### Check Target Cube Selection

Add this to your training loop to log which cubes are being targeted:

```python
# In your training/evaluation script
print(f"Target cubes: {env.target_cube_idx.cpu().numpy()}")
```

### Expected Behaviors

**Early Training (0-1000 iterations)**:
- Random movements
- May drag cubes instead of lifting
- Low success rate

**Mid Training (1000-5000 iterations)**:
- Starts reaching toward cubes
- Learns to close gripper
- Occasional successful lifts

**Late Training (5000+ iterations)**:
- Consistent picking
- Smooth transport to target
- Higher success rate
- May specialize on certain cube positions

## Troubleshooting

### Issue: "KeyError: cube_0"
**Cause**: Cubes not created in scene
**Fix**: Ensure `FrankaSceneCfg.__post_init__()` is being called

### Issue: Robot ignores some cubes
**Possible causes**:
1. Using "closest" mode and some cubes are always further
2. Cube spawn range too large
3. Policy hasn't learned to explore

**Solutions**:
- Switch to "random" mode for more diverse experience
- Reduce spawn area: `"pose_range": {"x": (0.3, 0.5), "y": (-0.2, 0.2)}`
- Train longer

### Issue: Observation size mismatch
**Cause**: Changed num_cubes but not all config params
**Fix**: Search for `num_cubes` in config and update all occurrences

### Issue: Robot drags cubes
**Cause**: Not learning to lift before transport
**Solutions**:
- Increase `lifting_object` reward: `weight=30.0`
- Check gripper is closing: add observation for gripper state
- Ensure `minimal_height` is enforced in `object_goal_tracking`

## Advanced: Curriculum Learning

Gradually increase task difficulty:

```python
# Stage 1: Single cube, close range (0-10k iterations)
num_cubes=1, pose_range={"x": (0.4, 0.5), "y": (-0.1, 0.1)}

# Stage 2: Two cubes, medium range (10k-20k iterations)
num_cubes=2, pose_range={"x": (0.3, 0.6), "y": (-0.2, 0.2)}

# Stage 3: Four cubes, full range (20k+ iterations)
num_cubes=4, pose_range={"x": (0.2, 0.6), "y": (-0.3, 0.3)}
```

## Files Modified

- ✅ `mdp/observations.py` - Added multi-cube observation functions
- ✅ `mdp/rewards.py` - Updated rewards to track target cube
- ✅ `mdp/cube_selection.py` - NEW: Cube selection strategies
- ✅ `mdp/events.py` - NEW: Multi-cube reset logic
- ✅ `mdp/__init__.py` - Import new modules
- ✅ `frankapickplace_env_cfg.py` - Updated config for multi-cube

## Next Steps

1. **Test the environment**: Run `scripts/test_multi_cube.py`
2. **Start training**: Use your existing training command
3. **Monitor progress**: Watch for increasing success rate
4. **Tune if needed**: Adjust reward weights, selection mode, or spawn range

## Need Help?

- Check `MULTI_CUBE_SETUP.md` for detailed technical documentation
- Review observation space: Are all cubes visible?
- Check target cube selection: Is it working as expected?
- Monitor rewards: Which reward terms are dominant?

---

**Happy Training! 🤖🎯**
