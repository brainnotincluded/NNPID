# Using Trained Models

This guide explains how to load and use trained yaw tracking models in your own control systems.

## Quick Start

### Install (Inference-Only)

```bash
pip install -e .
```

```python
from src.deployment.trained_yaw_tracker import TrainedYawTracker

# Load model
tracker = TrainedYawTracker.from_path("runs/<run_name>/best_model")

# In your control loop
yaw_rate_cmd = tracker.predict(observation, deterministic=True)
```

### Important Notes

- **VecNormalize is required for correct inference**: make sure
  `runs/<run_name>/vec_normalize.pkl` exists next to the model directory.
- **Python version mismatch**: `TrainedYawTracker` uses SB3 `custom_objects`
  for `clip_range` / `lr_schedule`, so models trained on 3.12 can be loaded on 3.10/3.11.

## TrainedYawTracker Class

The `TrainedYawTracker` class provides a simple wrapper around trained neural network models, handling model loading, observation normalization, and action prediction.

### Class Overview

```python
class TrainedYawTracker:
    """Wrapper for trained yaw tracking neural network model.
    
    Encapsulates a trained model and provides a simple interface
    for getting yaw rate commands based on current observations.
    """
```

### Methods

#### `from_path(model_path, config=None) -> TrainedYawTracker`

**Class method** to load a trained model from a file or directory.

**Parameters:**
- `model_path` (str | Path): Path to model file or directory
  - If directory: searches for `best_model.zip` or `final_model.zip`
  - Also supports `runs/<run_name>/best_model/` (directory with `best_model.zip`)
  - If file: loads directly
- `config` (YawTrackingConfig, optional): Environment configuration (uses training config if None)

**Returns:**
- `TrainedYawTracker`: Loaded controller instance

**Raises:**
- `FileNotFoundError`: If model file not found
- `ValueError`: If model cannot be loaded

**Example:**
```python
# Load from directory (searches for best_model.zip)
tracker = TrainedYawTracker.from_path("runs/<run_name>/best_model")

# Load from specific file
tracker = TrainedYawTracker.from_path("runs/model_12345/final_model.zip")
```

#### `predict(observation, deterministic=True, dead_zone=None) -> float`

Get yaw rate command from observation vector.

**Parameters:**
- `observation` (np.ndarray): Observation vector (11 elements)
  - `[0]` target_dir_x: X component of target direction
  - `[1]` target_dir_y: Y component of target direction
  - `[2]` target_angular_vel: Target angular velocity (rad/s)
  - `[3]` current_yaw_rate: Current yaw rate (rad/s)
  - `[4]` yaw_error: Yaw error angle (rad)
  - `[5]` roll: Current roll angle (rad)
  - `[6]` pitch: Current pitch angle (rad)
  - `[7]` altitude_error: Altitude error (m)
  - `[8]` velocity_x: X velocity (m/s)
  - `[9]` velocity_y: Y velocity (m/s)
  - `[10]` previous_action: Previous action value

- `deterministic` (bool): Use deterministic policy (True for deployment, False for exploration)
- `dead_zone` (float | None): Zero out small commands to reduce jitter
  - `None`: uses `config.action_dead_zone`
  - Set to `0.0` to disable

**Returns:**
- `float`: Yaw rate command in range [-1, 1]
  - `+1.0`: Maximum positive yaw rate
  - `-1.0`: Maximum negative yaw rate
  - `0.0`: No yaw command

**Raises:**
- `ValueError`: If observation shape is incorrect

**Example:**
```python
# Get command
yaw_cmd = tracker.predict(obs, deterministic=True)

# Scale to actual yaw rate (use training config value)
max_yaw_rate = tracker.config.max_yaw_rate
actual_yaw_rate = yaw_cmd * max_yaw_rate
```

#### `reset() -> None`

Reset internal state.

Currently a no-op, but can be extended for models with internal state (e.g., RNNs).

**Example:**
```python
tracker.reset()  # Reset before new episode
```

#### `get_info() -> dict[str, Any]`

Get controller information.

**Returns:**
- `dict` with keys:
  - `model_type`: Type of model (e.g., "PPO")
  - `observation_space`: Size of observation vector
  - `has_normalization`: Whether VecNormalize is loaded
  - `config`: Environment configuration dictionary

**Example:**
```python
info = tracker.get_info()
print(f"Model: {info['model_type']}")
print(f"Observation space: {info['observation_space']}")
print(f"Has normalization: {info['has_normalization']}")
```

## Integration Example

Here's how to integrate `TrainedYawTracker` into your own control system:

```python
from src.deployment.trained_yaw_tracker import TrainedYawTracker
import numpy as np

# 1. Load model once at startup
tracker = TrainedYawTracker.from_path("runs/<run_name>/best_model")

# 2. In your main control loop
while running:
    # Get current state from sensors/simulator
    state = get_current_state()  # Your function
    
    # Build observation vector (11 elements)
    obs = build_observation(state, target)  # Your function
    
    # Get command from trained model
    yaw_rate_cmd = tracker.predict(obs, deterministic=True)
    
    # Use command with your stabilizer/controller
    # Scale: yaw_rate_cmd in [-1, 1] -> actual speed in rad/s
max_yaw_rate = tracker.config.max_yaw_rate
    actual_yaw_rate = yaw_rate_cmd * max_yaw_rate
    
    motors = stabilizer.compute_motors(
        state,
        yaw_rate_cmd=actual_yaw_rate,
        dt=dt
    )
    
    # Apply motor commands
    apply_motors(motors)
```

## Observation Format

The model expects a vector of exactly 11 elements:

```python
observation = np.array([
    target_dir_x,        # [0] X component of target direction
    target_dir_y,        # [1] Y component of target direction
    target_angular_vel,  # [2] Target angular velocity (rad/s)
    current_yaw_rate,    # [3] Current yaw rate (rad/s)
    yaw_error,           # [4] Yaw error angle (rad)
    roll,                # [5] Roll angle (rad)
    pitch,               # [6] Pitch angle (rad)
    altitude_error,      # [7] Altitude error (m)
    velocity_x,          # [8] X velocity (m/s)
    velocity_y,          # [9] Y velocity (m/s)
    previous_action,     # [10] Previous action value
], dtype=np.float32)
```

### Building Observations

To build observations from your state:

```python
def build_observation(state, target):
    """Build observation vector from state and target.
    
    Args:
        state: Your state object (position, attitude, etc.)
        target: Target position or direction
        
    Returns:
        Observation vector (11 elements)
    """
    # Calculate target direction (normalized)
    target_dir = target - state.position
    target_dir_xy = target_dir[:2] / np.linalg.norm(target_dir[:2])
    
    # Calculate yaw error
    yaw_error = calculate_yaw_error(state.yaw, target_dir)
    
    # Build observation
    obs = np.array([
        target_dir_xy[0],           # target_dir_x
        target_dir_xy[1],           # target_dir_y
        target.angular_velocity,    # target_angular_vel
        state.yaw_rate,             # current_yaw_rate
        yaw_error,                  # yaw_error
        state.roll,                 # roll
        state.pitch,               # pitch
        abs(state.altitude - 1.0), # altitude_error
        state.velocity[0],          # velocity_x
        state.velocity[1],          # velocity_y
        state.previous_action,      # previous_action
    ], dtype=np.float32)
    
    return obs
```

## Important Notes

1. **Normalization**: The model automatically applies VecNormalize if it was saved during training. This is handled transparently.

2. **Determinism**: Use `deterministic=True` for deployment (consistent behavior), `False` for exploration/testing.

3. **Scaling**: The command is in [-1, 1] range. Multiply by `max_yaw_rate` to get actual yaw rate in rad/s:
   ```python
   actual_yaw_rate = yaw_cmd * max_yaw_rate  # e.g., 2.0 rad/s
   ```

4. **Observation Format**: Must exactly match training format (11 elements, float32).

5. **Error Handling**: The class validates observation shape and raises `ValueError` if incorrect.

## Complete Example

See `examples/use_trained_model.py` for a complete working example with the environment.

## API Reference

For detailed API documentation, see the class docstring:

```python
from src.deployment.trained_yaw_tracker import TrainedYawTracker
help(TrainedYawTracker)
```

## Troubleshooting

### Model Not Found

```python
# Error: FileNotFoundError
# Solution: Check path and ensure model file exists
tracker = TrainedYawTracker.from_path("runs/<run_name>/best_model")  # Correct
```

### Observation Shape Mismatch

```python
# Error: ValueError: Observation size mismatch
# Solution: Ensure observation has exactly 11 elements
obs = np.array([...], dtype=np.float32)  # Must be 11 elements
assert len(obs) == 11
```

### VecNormalize Not Found

If VecNormalize file is missing, the tracker will work but without normalization. This may cause performance degradation if the model was trained with normalization.

## See Also

- [Training Guide](TRAINING.md) - How to train models
- [Architecture Documentation](../ARCHITECTURE.md) - System architecture
- [Getting Started](GETTING_STARTED.md) - Project setup
