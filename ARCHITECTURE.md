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
├── visualization/  # Rendering, HUD & overlays
└── perturbations/  # Realistic disturbances
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
    
    IMPORTANT: Requires control_frequency >= 100Hz for stability.
    See docs/issues/003-hover-pid-instability.md for details.
    
    Tuned Default Gains (for 100Hz):
    - attitude_kp: 15.0 (reduced from 40.0)
    - attitude_ki: 0.5 (reduced from 2.0)
    - attitude_kd: 5.0 (reduced from 15.0)
    - yaw_authority: 0.03 (balanced for tracking)
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

#### `trained_yaw_tracker.py` - Model Wrapper

```python
class TrainedYawTracker:
    """Wrapper for trained yaw tracking neural network model.
    
    Provides simple interface for loading and using trained models:
    - from_path(): Load model from file/directory
    - predict(): Get yaw rate command from observation
    - Automatic VecNormalize handling
    
    Usage:
        tracker = TrainedYawTracker.from_path("runs/best_model")
        yaw_cmd = tracker.predict(obs, deterministic=True)
    """
```

**Key Features:**
- Encapsulates model loading and normalization
- Handles VecNormalize automatically if available
- Validates observation format
- Simple API for integration into custom control loops

See [Trained Model Usage Guide](docs/TRAINED_MODEL_USAGE.md) for details.

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

### 7. `src/visualization/` - Visualization System

**Purpose**: Comprehensive visualization with 3D objects, HUD, and neural network display.

#### Components

```
visualization/
├── viewer.py           # MuJoCo viewer wrapper
├── scene_objects.py    # 3D geometries for scene
├── nn_visualizer.py    # Neural network display
├── telemetry_hud.py    # Real-time HUD
├── mujoco_overlay.py   # Combined overlay system
└── dashboard.py        # Matplotlib telemetry plots
```

#### `scene_objects.py` - 3D Scene Objects

```python
class SceneObjectManager:
    """Injects 3D geometries into MuJoCo scene.
    
    Objects:
    - WindArrow3D: Directional wind indicator
    - ForceVectors3D: Force vector visualization
    - TrajectoryTrail3D: Path history display
    - TargetBeam3D: Target tracking indicator
    - VRSDangerZone3D: Vortex ring state warning
    """
```

#### `mujoco_overlay.py` - Combined Overlay

```python
class MegaVisualizer:
    """Combines all visualization components.
    
    Integrates:
    - SceneObjectManager (3D objects)
    - NNVisualizer (network diagram)
    - TelemetryHUD (graphs & gauges)
    - FrameAnnotator (status bar)
    
    Usage:
        viz = create_full_visualizer()
        viz.set_model(trained_model)
        viz.set_perturbation_manager(perturbation_manager)
        frame = viz.render_overlay(base_frame)
    """
```

### 8. `src/perturbations/` - Realistic Perturbations

**Purpose**: Add realistic disturbances for robust controller training.

#### Module Structure

```
perturbations/
├── base.py             # Base classes & manager
├── wind.py             # Wind effects
├── delays.py           # Sensor/actuator delays
├── sensor_noise.py     # Sensor perturbations
├── physics.py          # Physical perturbations
├── aerodynamics.py     # Aerodynamic effects
├── external_forces.py  # External disturbances
└── visualization.py    # Perturbation visual effects
```

#### `PerturbationManager` - Central Controller

```python
class PerturbationManager:
    """Manages all perturbation types.
    
    Methods:
    - load_config(path): Load from YAML
    - reset(rng): Reset with new random state
    - update(state, dt): Update perturbations
    - get_total_force(): Combined external force
    - apply_to_observation(obs): Add sensor noise
    - apply_to_action(action): Add actuator delays
    
    Config: config/perturbations.yaml
    """
```

#### Available Perturbations

| Category | Types |
|----------|-------|
| Wind | Steady, Gusts, Turbulence (Dryden), Thermals |
| Delays | Sensor (IMU/GPS), Actuator, Jitter, Dropout |
| Sensor Noise | Gaussian, Drift, Outliers, GPS Loss, EMI |
| Physics | CoM Offset, Motor Variation, Ground Effect |
| Aerodynamics | Air Drag, Blade Flapping, VRS |
| External | Impulses, Vibrations, Periodic Forces |

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
│       ├── HoverStabilizer.compute_motors(state, yaw_cmd, dt)
│       │       └── PID for altitude, roll, pitch
│       │       └── NN yaw rate command applied with limited authority
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

**Using TrainedYawTracker (Recommended):**

```
Trained Model (.zip)
    │
    ▼
TrainedYawTracker.from_path()
    │
    ├── Loads PPO model
    ├── Loads VecNormalize (if available)
    │
    ▼
Your Control Loop:
    │
    ├── Get state from sensors/simulator
    ├── Build observation vector (11 elements)
    │
    ▼
tracker.predict(obs, deterministic=True)
    │
    ├── Normalize observation (if VecNormalize loaded)
    ├── Model.predict() → yaw_rate_cmd [-1, 1]
    │
    ▼
Scale to actual yaw rate: yaw_cmd * max_yaw_rate
    │
    ▼
Use with stabilizer/controller
```

**Using YawTrackerSITL (ArduPilot SITL):**

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

### `scripts/run_mega_viz.py`

```python
# Full visualization script flow:
1. Load trained model (optional)
2. Load perturbation config (optional)
3. Create MegaVisualizer
4. Run environment loop:
   - Get action from model
   - Update visualizer with state
   - Render overlay on frame
   - Handle keyboard (q=quit, r=reset, p=pause, s=screenshot)
5. Save video (optional)
```

### `scripts/model_inspector.py`

```bash
# CLI commands:
arch      - Show network architecture
weights   - Visualize weight matrices
activations - Analyze activations on test episodes
stats     - Export model statistics to JSON
compare   - Compare two models
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
   - Ensure control_frequency >= 100Hz (50Hz causes instability!)
   - Check attitude PID gains (Kp=15, Kd=5 for 100Hz)
   - See docs/issues/003-hover-pid-instability.md

2. **Drone spins uncontrollably**
   - Check motor positions match FL/BL/BR/FR layout
   - Verify yaw signs in motor mixing (CCW motors = +yaw)
   - Reduce `yaw_authority` in stabilizer config
   - See docs/issues/001-yaw-sign-inversion.md

3. **Training doesn't converge**
   - Reduce target speed in curriculum
   - Increase episode length
   - Check reward scaling
   - See docs/tickets/001-reward-system-upgrade.md

4. **Hover oscillates or becomes unstable**
   - Reduce attitude_kp and attitude_kd gains
   - Increase control_frequency (100-200Hz recommended)
   - Check max_integral for anti-windup
   - See docs/issues/003-hover-pid-instability.md

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
