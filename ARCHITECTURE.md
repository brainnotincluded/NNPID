# NNPID Architecture Guide

> **For LLM Agents**: This document explains the codebase structure to help you navigate and modify the code effectively.

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         NNPID System                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │   MuJoCo    │◄──►│  Gymnasium  │◄──►│  Neural Network     │ │
│  │  Simulator  │    │ Environment │    │  (PPO/SAC)          │ │
│  └─────────────┘    └─────────────┘    └─────────────────────┘ │
│         │                                        │              │
│         │                                        │              │
│         ▼                                        ▼              │
│  ┌─────────────┐                        ┌─────────────────────┐ │
│  │  ArduPilot  │◄───── Deploy ─────────►│  Trained Model      │ │
│  │    SITL     │                        │  (.zip)             │ │
│  └─────────────┘                        └─────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
src/
├── core/           # Physics simulation layer
├── environments/   # RL environment definitions
├── controllers/    # Control algorithms
├── communication/  # SITL protocols
├── deployment/     # Model deployment
├── utils/          # Shared utilities
└── visualization/  # Rendering & plots
```

## Core Modules

### 1. `src/core/` - Physics Simulation

**Purpose**: Wraps MuJoCo for quadrotor physics simulation.

#### `mujoco_sim.py` - Main Simulator Class

```python
class MuJoCoSimulator:
    """High-fidelity quadrotor physics simulation.
    
    Key Methods:
    - reset(position, velocity, quaternion, angular_velocity)
    - step(motor_commands: np.ndarray) -> None
    - get_state() -> QuadrotorState
    - render(renderer) -> np.ndarray (RGB image)
    - set_mocap_pos(name, pos) -> None  # Move markers
    
    Key Properties:
    - dt: float (timestep, default 0.002s = 500Hz)
    - mass: float (quadrotor mass)
    - gravity: np.ndarray ([0, 0, -9.81])
    """
```

#### `QuadrotorState` - State Container

```python
@dataclass
class QuadrotorState:
    position: np.ndarray      # [x, y, z] in world frame
    velocity: np.ndarray      # [vx, vy, vz] in world frame
    quaternion: np.ndarray    # [w, x, y, z] orientation
    angular_velocity: np.ndarray  # [wx, wy, wz] body rates
```

### 2. `src/environments/` - Gymnasium Environments

**Purpose**: Provides standard RL interface for training.

#### Environment Hierarchy

```
BaseDroneEnv (base_drone_env.py)
    │
    ├── HoverEnv (hover_env.py)
    │       Task: Maintain position at setpoint
    │
    ├── YawTrackingEnv (yaw_tracking_env.py)  ◄── MAIN TASK
    │       Task: Face a moving target
    │       Action: yaw_rate [-1, 1]
    │       Observation: [target_dir, angular_vel, yaw_error, ...]
    │
    ├── WaypointEnv (waypoint_env.py)
    │       Task: Navigate through waypoints
    │
    └── SetpointEnv (setpoint_env.py)
            Task: Track position setpoints
```

#### `YawTrackingEnv` - Key Training Environment

```python
class YawTrackingEnv(gym.Env):
    """Train NN to track moving target with yaw control.
    
    Observation Space (11 dims):
    - target_direction_body: [2] direction to target in body frame
    - angular_velocity: [3] body angular rates
    - yaw_rate: [1] current yaw rate
    - yaw_error: [1] angle to target
    - roll_pitch: [2] attitude angles
    - altitude_error: [1] height error
    - previous_action: [1] last yaw command
    
    Action Space (1 dim):
    - yaw_rate_command: [-1, 1] normalized
    
    Reward Function:
    - Exponential bonus for small yaw error
    - Penalty for action magnitude
    - Penalty for action rate of change
    - Bonus for sustained tracking
    
    Config: config/yaw_tracking.yaml
    """
```

### 3. `src/controllers/` - Control Algorithms

**Purpose**: PID and NN controllers for drone control.

#### `hover_stabilizer.py` - SITL-Style Stabilizer

```python
class HoverStabilizer:
    """Direct PID stabilizer that guarantees stable hover.
    
    Architecture (inspired by ArduPilot):
    - Altitude PID: maintains hover height
    - Attitude PID: maintains level flight (roll/pitch = 0)
    - Yaw rate P: follows NN yaw commands
    
    Key Features:
    - Safety mode: ignores yaw if tilt > threshold
    - Anti-windup on all integrators
    - Correct motor mixing for MuJoCo coordinate system
    
    Config:
    - altitude_kp/ki/kd: altitude PID gains
    - attitude_kp/ki/kd: roll/pitch PID gains  
    - yaw_rate_kp: yaw rate P gain
    - safety_tilt_threshold: max tilt before yaw is disabled
    - yaw_authority: max yaw torque
    """
```

#### `base_controller.py` - PID Controller

```python
class PIDController:
    """Classical PID for attitude/position control.
    
    Used by YawTrackingEnv for internal stabilization:
    - Altitude hold (PD)
    - Roll/pitch stabilization (PD)
    - NN controls only yaw rate
    """
```

#### `nn_controller.py` - Neural Network Controller

```python
class NNController:
    """Wraps trained Stable-Baselines3 model.
    
    Methods:
    - load(path) -> NNController
    - predict(obs) -> action
    - save(path)
    """
```

### 4. `src/communication/` - SITL Protocols

**Purpose**: Connect MuJoCo to ArduPilot SITL.

#### Protocol Flow

```
MuJoCo Sim ◄──── JSON (UDP:9002) ────► ArduPilot SITL
    │                                        │
    │           PWM Commands                 │
    ◄────── JSON (UDP:9003) ──────────►      │
    │                                        │
    │         MAVLink Commands               │
    ◄────── TCP:5760 ─────────────────►      │
```

#### `mavlink_bridge.py`

```python
class MAVLinkBridge:
    """MAVLink protocol for high-level commands.
    
    Methods:
    - connect(address)
    - arm()
    - set_mode(mode: str)
    - send_attitude_target(roll, pitch, yaw, thrust)
    - get_state() -> DroneState
    """
```

### 5. `src/deployment/` - Model Deployment

**Purpose**: Deploy trained models to real/simulated drones.

#### `yaw_tracker_sitl.py`

```python
class YawTrackerSITL:
    """Deploy yaw tracking NN to ArduPilot SITL.
    
    Flow:
    1. Connect to SITL via MAVLink
    2. Arm and takeoff
    3. Load trained model
    4. Loop: get state → predict action → send yaw command
    """
```

### 6. `src/utils/` - Utilities

#### `rotations.py` - Quaternion Math

```python
class Rotations:
    """Quaternion and rotation utilities.
    
    Key Methods:
    - quaternion_to_euler(q) -> (roll, pitch, yaw)
    - euler_to_quaternion(roll, pitch, yaw) -> q
    - quaternion_rotate_vector(q, v) -> v_rotated
    - quaternion_multiply(q1, q2) -> q
    """
```

#### `transforms.py` - Coordinate Transforms

```python
class CoordinateTransforms:
    """Convert between coordinate frames.
    
    Frames:
    - MuJoCo: Z-up, X-forward
    - NED: North-East-Down (ArduPilot)
    - Body: Drone-relative
    """
```

## Configuration System

### Config Files (`config/`)

```yaml
# config/yaw_tracking.yaml
environment:
  hover_height: 1.0           # Target altitude (m)
  max_episode_steps: 1000     # Episode length
  max_yaw_rate: 2.0           # Max yaw rate (rad/s)
  target_patterns:            # Target motion types
    - circular
    - random
  target_speed_min: 0.1       # Min target angular velocity
  target_speed_max: 0.3       # Max target angular velocity
  
training:
  algorithm: PPO              # RL algorithm
  total_timesteps: 500000     # Training steps
  n_envs: 4                   # Parallel environments
  learning_rate: 0.0003
  policy_kwargs:
    net_arch: [64, 64]        # Hidden layers
    
curriculum:                   # Curriculum learning
  enabled: true
  stages:
    - timesteps: 0
      target_speed_max: 0.1
    - timesteps: 100000
      target_speed_max: 0.2
```

## MuJoCo Model (`models/quadrotor_x500.xml`)

### Coordinate System

MuJoCo uses **X-forward, Y-left, Z-up** coordinate system (ENU-like).

```
        Y (left)
        ^
        |
        |
   -----+-----> X (forward)
        |
        |
        Z (up, out of page)
```

### Motor Layout

```
         Front (X+)
           ▲
    M1 ◄───┼───► M4
   (FL)    │    (FR)
   CCW     │     CW
           │
    M2 ◄───┼───► M3
   (BL)    │    (BR)
    CW     │    CCW
           ▼
         Back (X-)
```

**IMPORTANT**: Motor positions in the model (Y-left convention):

| Motor | Position            | Location    | Direction |
|-------|---------------------|-------------|-----------|
| M1    | (+0.1768, +0.1768)  | Front-Left  | CCW (+yaw)|
| M2    | (-0.1768, +0.1768)  | Back-Left   | CW (-yaw) |
| M3    | (-0.1768, -0.1768)  | Back-Right  | CCW (+yaw)|
| M4    | (+0.1768, -0.1768)  | Front-Right | CW (-yaw) |

### Motor Mixing

The motor mixer converts attitude commands to individual motor thrusts:

```python
# Correct mixer for MuJoCo X-forward, Y-left coordinate system
m1 = thrust + roll_torque - pitch_torque + yaw_torque  # Front-Left CCW
m2 = thrust + roll_torque + pitch_torque - yaw_torque  # Back-Left CW
m3 = thrust - roll_torque + pitch_torque + yaw_torque  # Back-Right CCW
m4 = thrust - roll_torque - pitch_torque - yaw_torque  # Front-Right CW
```

Where:
- **Positive roll** = right side down → increase right motors (M3, M4)
- **Positive pitch** = nose down → increase front motors (M1, M4)
- **Positive yaw** = CCW rotation → increase CCW motors (M1, M3)

```xml
<!-- Model structure -->
<body name="quadrotor">
    <freejoint/>              <!-- 6-DOF motion -->
    <inertial mass="2.0"/>    <!-- 2kg drone -->
    
    <!-- Motors at corners -->
    <body name="motor1" pos="0.1768 0.1768 0.02"/>   <!-- Front-Left -->
    <body name="motor2" pos="-0.1768 0.1768 0.02"/>  <!-- Back-Left -->
    <body name="motor3" pos="-0.1768 -0.1768 0.02"/> <!-- Back-Right -->
    <body name="motor4" pos="0.1768 -0.1768 0.02"/>  <!-- Front-Right -->
</body>

<!-- Target marker (moveable) -->
<body name="target" mocap="true">
    <geom type="sphere" size="0.15"/>
</body>

<!-- Motor actuators: thrust (8N max) + yaw torque (0.08 Nm) -->
<actuator>
    <general name="motor1" gear="0 0 8 0 0 0.08"/>   <!-- CCW, +yaw -->
    <general name="motor2" gear="0 0 8 0 0 -0.08"/>  <!-- CW, -yaw -->
    <general name="motor3" gear="0 0 8 0 0 0.08"/>   <!-- CCW, +yaw -->
    <general name="motor4" gear="0 0 8 0 0 -0.08"/>  <!-- CW, -yaw -->
</actuator>
```

## Training Scripts

### `scripts/train_yaw_tracker.py`

```python
# Main training script flow:
1. Load config from YAML
2. Create vectorized environments
3. Initialize PPO model
4. Setup callbacks:
   - EvalCallback (evaluation during training)
   - CheckpointCallback (save periodically)
   - CurriculumCallback (adjust difficulty)
5. Train model.learn(total_timesteps)
6. Save final model
```

### `scripts/evaluate_yaw_tracker.py`

```python
# Evaluation script flow:
1. Load trained model
2. Create test environment
3. Run N episodes
4. Compute metrics:
   - Mean reward
   - Episode length
   - Yaw error
   - Tracking percentage
5. Render video (optional)
```

## Common Patterns

### Adding a New Environment

1. Create `src/environments/new_env.py`
2. Inherit from `BaseDroneEnv`
3. Define `observation_space` and `action_space`
4. Implement `_get_observation()`, `_compute_reward()`, `_check_termination()`
5. Register with `gym.register()`
6. Add config in `config/new_env.yaml`

### Adding a New Controller

1. Create `src/controllers/new_controller.py`
2. Inherit from `BaseController` or create standalone
3. Implement `compute_action(state) -> np.ndarray`
4. Test with environment

### Modifying Physics

1. Edit `models/quadrotor_x500.xml` for:
   - Mass/inertia changes
   - Motor placement
   - New sensors
2. Update `src/core/mujoco_sim.py` if new data access needed

## Key Data Flows

### Training Flow

```
Environment.reset()
    │
    ▼
┌──►Environment.step(action)
│       │
│       ├── _compute_stabilized_motors(state, yaw_cmd)
│       │       └── PID for altitude, roll, pitch
│       │       └── NN action for yaw
│       │
│       ├── sim.step(motor_commands)
│       │
│       ├── _compute_reward(state)
│       │
│       └── _get_observation()
│               │
│               ▼
│           observation
│               │
└───── PPO.predict(observation) ◄──┘
```

### Deployment Flow

```
ArduPilot SITL
    │
    ▼
YawTrackerSITL.get_state()
    │
    ├── Convert to observation
    │
    ▼
NNController.predict(obs)
    │
    ├── Yaw rate command
    │
    ▼
MAVLink.send_attitude_target()
    │
    ▼
ArduPilot executes
```

## Testing

```bash
# Run all tests
pytest tests/

# Test specific module
pytest tests/test_mujoco_sim.py

# Test with coverage
pytest --cov=src tests/
```

## Performance Tips

1. **Parallel Training**: Use `--n-envs 8` for faster training
2. **GPU**: PyTorch uses GPU automatically if available
3. **Rendering**: Disable `render_mode` during training
4. **Physics Steps**: Adjust `physics_steps_per_control` for speed/accuracy

## Debugging

### Common Issues

1. **Drone crashes immediately**
   - Check `base_thrust` in config (should be ~0.62 for 2kg drone)
   - Verify motor mixing signs match the coordinate system (see Motor Mixing section)
   - Ensure HoverStabilizer is using correct gains

2. **Drone spins uncontrollably**
   - Check motor positions match FL/BL/BR/FR layout
   - Verify yaw signs in motor mixing (CCW motors = +yaw)
   - Reduce `yaw_authority` in stabilizer config

3. **Training doesn't converge**
   - Reduce target speed in curriculum
   - Increase episode length
   - Check reward scaling

4. **SITL connection fails**
   - Ensure ArduPilot SITL is running with `--json`
   - Check UDP ports 9002/9003 are free

### Motor Mixing Debug

If the drone behaves unexpectedly, check the motor mixing:

```python
# Expected behavior for each torque command:
# roll_torque > 0: left motors increase, right decrease
# pitch_torque > 0: back motors increase, front decrease  
# yaw_torque > 0: CCW motors (M1, M3) increase
```

### Logging

```python
from src.utils.logger import get_logger
logger = get_logger(__name__)
logger.info("Training started")
logger.debug(f"State: {state}")
```
