# Deployment to Webots via SITL

This guide explains how to deploy the trained NN model from MuJoCo to control a drone in Webots via ArduPilot SITL.

## Overview

The complete pipeline:

```
┌─────────────────────────────────────────────────────────────┐
│                      SERVER SETUP                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Webots Simulator         ArduPilot SITL        NN Bridge  │
│  (Drone physics)    ←─────────────────→  (MAVLink)         │
│        ↑                                        ↑           │
│        │                                        │           │
│        └────────← Motor Commands ←──────────────┤           │
│        ├─→ Sensor Data (IMU/GPS) ──────────────┤           │
│        │                                        │           │
│        │                    Observation  ┌──────┴─────────┐ │
│        │                    ←────────────┤                │ │
│        │                                  │  Best Model   │ │
│        │                        Action    │  (PPO [512,   │ │
│        │                        ────────→ │   512, 256,   │ │
│        │                                  │   128])       │ │
│        │                                  └────────────────┘ │
│        │                                  + VecNormalize    │
│        │                                  (normalization)   │
│        └─────────────────────────────────────────────────────┘
│
```

## Prerequisites

### 1. Install ArduPilot SITL

See `docs/SITL_INTEGRATION.md` for full setup steps on macOS and Linux.
Ensure `sim_vehicle.py` is available on your PATH (or use the full path from `~/ardupilot`).

### 2. Install Python Dependencies

```bash
# Clone NNPID and install
git clone https://github.com/brainnotincluded/NNPID.git
cd NNPID
pip install -e ".[dev]"
pip install pymavlink webots-controller  # For Webots
```

### 3. Webots Setup

- Download Webots from https://cyberbotics.com/
- Ensure ArduCopter simulation model is available (pre-installed or custom)
- Configure Webots-SITL connection (usually localhost:5760)

## VecNormalize: Critical for Deployment

### What is VecNormalize?

VecNormalize is a wrapper from Stable-Baselines3 that normalizes observations before feeding them to the policy. It stores:
- **mean_obs**: Running mean of observations (11 dimensions for yaw tracking)
- **var_obs**: Running variance of observations
- **count**: Number of steps seen during training

During training, observations are normalized as:
```
obs_normalized = (obs - mean_obs) / sqrt(var_obs + 1e-8)
```

**This is CRITICAL**: The neural network was trained with normalized observations. If you don't normalize at deployment, the model will receive out-of-distribution inputs and fail.

### Loading VecNormalize

The normalization wrapper is saved as a pickle file alongside the model:

```
runs/<run_name>/
├── best_model/
│   └── best_model.zip          # PPO model
└── vec_normalize.pkl           # ← REQUIRED for inference!
```

Use `runs/<run_name>/best_model` as the model path, and ensure
`runs/<run_name>/vec_normalize.pkl` exists next to it.

### How to Use VecNormalize

```python
import pickle
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO

# Load model and VecNormalize
model = PPO.load("path/to/best_model.zip")

with open("path/to/vec_normalize.pkl", "rb") as f:
    vec_normalize = pickle.load(f)

# Get observation from environment
obs = env.reset()[0]  # Shape: (11,)

# IMPORTANT: Normalize before passing to model
obs_normalized = vec_normalize.normalize_obs(obs.reshape(1, -1))[0]

# Get action
action, _ = model.predict(obs_normalized, deterministic=True)
```

### Auto-Loading

The evaluation and deployment scripts automatically load VecNormalize if it exists:

```python
# Automatically checks for vec_normalize.pkl in parent directory
model, vec_normalize = load_model_and_vecnorm("path/to/best_model")

# If vec_normalize is None, normalization is skipped
if vec_normalize is not None:
    obs_normalized = vec_normalize.normalize_obs(obs.reshape(1, -1))
else:
    obs_normalized = obs
```

## Phase 1: Verify Locally

Run the best model in MuJoCo to verify it works:

```bash
# Run 3 episodes with detailed logging
python scripts/run_model_mujoco.py \
    --model runs/<run_name>/best_model \
    --episodes 3 \
    --steps 500

# Save detailed step-by-step log for analysis
python scripts/run_model_mujoco.py \
    --model runs/<run_name>/best_model \
    --episodes 1 \
    --output flight_log.json
```

Output log includes:
- Observations (11 dimensions)
- Actions (yaw rate commands)
- Drone state (position, yaw, attitude)
- Target state
- Rewards and tracking metrics

## Phase 2: Setup on Server

### Terminal 1: Start ArduPilot SITL

```bash
sim_vehicle.py -v ArduCopter -f JSON --console
```

Options:
- `-v ArduCopter`: Vehicle type
- `-f JSON`: Use JSON backend (for physics simulation)
- `--console`: Show MAVLink console
- `--map`: Show map (optional)

### Terminal 2: Start Webots Simulator

```bash
# Open Webots and load the scene
# Or run headless
webots --mode=fast --minimize --batch iris_camera_human.wbt
```

### Terminal 3: Run NN Bridge (Deployment)

```bash
python scripts/run_yaw_tracker_sitl.py \
    --model runs/<run_name>/best_model \
    --connection tcp:127.0.0.1:5760 \
    --altitude 2.0 \
    --duration 120 \
    --target-speed 0.5 \
    --target-pattern circular
```

Options:
- `--model`: Path to trained model (VecNormalize auto-loaded)
- `--connection`: MAVLink connection string
- `--altitude`: Takeoff altitude (meters)
- `--duration`: How long to run (seconds)
- `--target-speed`: Target angular velocity (rad/s)
- `--target-pattern`: "circular" or "sinusoidal"

## Phase 3: Custom Webots Bridge (if needed)

If you need tighter integration with Webots, create a custom bridge:

```python
# pseudocode
class WebotsSITLBridge:
    def __init__(self, model_path, sitl_address):
        self.model, self.vec_normalize = load_model_and_vecnorm(model_path)
        self.mav = mavutil.mavlink_connection(f'tcp:{sitl_address}:5760')
        self.webots_supervisor = Supervisor()
        
    def run(self):
        while True:
            # 1. Get Webots drone state
            drone_state = self.webots_supervisor.get_drone_state()
            
            # 2. Convert to observation
            obs = self.state_to_observation(drone_state)
            
            # 3. Normalize
            obs_normalized = self.vec_normalize.normalize_obs(obs.reshape(1, -1))
            
            # 4. Get action from NN
            action, _ = self.model.predict(obs_normalized, deterministic=True)
            
            # 5. Send to SITL
            self.mav.send_yaw_rate_command(action[0])
            
            # 6. Read SITL motor commands
            motor_commands = self.mav.read_motor_commands()
            
            # 7. Apply to Webots drone
            self.webots_supervisor.apply_motor_commands(motor_commands)
```

## Model Specifications

**Current Best Model:**
- Algorithm: PPO (Proximal Policy Optimization)
- Network Architecture: [512, 512, 256, 128] (fully connected)
- Training Steps: ~1M / 20M (curriculum learning)
- Best Tracking: 87.7%
- Observation Space: 11 dimensions
  - target_direction_body (2D)
  - angular_velocity (3D)
  - yaw_rate (1D)
  - yaw_error (1D)
  - roll_pitch (2D)
  - altitude_error (1D)
  - previous_action (1D)
- Action Space: 1 dimension (yaw rate command [-1, 1])

**VecNormalize Statistics:**
- Observation mean: Stored in `vec_normalize.mean_obs`
- Observation variance: Stored in `vec_normalize.var_obs`

## Troubleshooting

### Model gives bad actions on server

**Cause**: VecNormalize not loaded or mismatched normalization

**Fix**:
```python
# Verify VecNormalize exists
vec_norm_path = model_dir / "vec_normalize.pkl"
assert vec_norm_path.exists(), f"Missing {vec_norm_path}"

# Verify dimensions
assert vec_normalize.mean_obs.shape == (11,), "Wrong obs dimension"
```

### Drone doesn't stabilize

**Cause**: SITL-Webots physics mismatch or motor command scaling

**Fix**:
- Verify timestep synchronization (500Hz physics, 50Hz control)
- Check motor command conversion: `thrust = (pwm - 1000) / 1000`
- Ensure coordinate frame matches (ENU vs NED)

### Observations out of bounds

**Cause**: Webots state values differ from training distribution

**Fix**:
- Log observations: `print(vec_normalize.normalize_obs(obs.reshape(1,-1)))`
- Should be mostly in [-2, 2] range
- If values are large (>5), check sensor scaling

## References

- [Stable-Baselines3 VecNormalize](https://stable-baselines3.readthedocs.io/en/master/common/vec_env.html#vecnormalize)
- [ArduPilot SITL Documentation](https://ardupilot.org/dev/docs/sitl-simulator-software-in-the-loop.html)
- [Webots Documentation](https://cyberbotics.com/doc/)
- `scripts/run_yaw_tracker_sitl.py` - Full deployment script
- `scripts/run_model_mujoco.py` - Local verification script
